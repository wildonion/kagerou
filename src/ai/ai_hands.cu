// ============================================================================
// Kagerou AI Hand Tracking — BlazePalm detect + hand landmark (21 joints)
// GPU preprocess → ORT inference → CPU decode + NMS → joints in frame pixels.
// Palm convention probed: RGB [0,1] input, scores = sigmoid(outputs[1]),
// boxes = outputs[0][0..3] as (x, y, w, h).
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

namespace kagerou {
namespace ai {

void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h,
                       cudaStream_t s);
void launch_rgb8_to_nchw(const uint8_t* d_rgb, float* d_nchw,
                         uint32_t w, uint32_t h, uint32_t frame_idx,
                         cudaStream_t s);
void launch_crop_rgb(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t src_w, uint32_t src_h,
                     uint32_t crop_x, uint32_t crop_y,
                     uint32_t crop_w, uint32_t crop_h,
                     cudaStream_t s);
void launch_draw_circle(uint8_t* rgb, uint32_t w, uint32_t h,
                        float cx, float cy, float radius,
                        uint8_t r, uint8_t g, uint8_t b,
                        cudaStream_t s);

static inline float psigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static bool g_hands_init = false;

bool ai_hands_init(const char* palm_path, const char* landmark_path, int device_id) {
    if (g_hands_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    if (!eng.load("palm", palm_path, device_id)) {
        fprintf(stderr, "[AI Hands] Cannot load %s\n", palm_path); return false;
    }
    if (!eng.load("handlm", landmark_path, device_id)) {
        fprintf(stderr, "[AI Hands] Cannot load %s\n", landmark_path); return false;
    }
    fprintf(stderr, "[AI Hands] Models ready.\n");
    g_hands_init = true;
    return true;
}

struct PalmDet { float x, y, w, h, score; };

static float palm_iou(const PalmDet& a, const PalmDet& b) {
    float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.w, b.x + b.w), iy2 = std::min(a.y + a.h, b.y + b.h);
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;
    float inter = iw * ih;
    float ua = a.w * a.h + b.w * b.h - inter;
    return ua > 0 ? inter / ua : 0;
}

int ai_hands(const uint8_t* d_src, uint32_t w, uint32_t h,
             HandJoints* hands_out, int max_hands, cudaStream_t stream) {
    if (!g_hands_init || max_hands <= 0) return 0;
    auto& eng = AiInference::get();
    auto* palm = eng.find("palm");
    auto* lm = eng.find("handlm");
    if (!palm || !lm) return 0;

    // ---- stage 1: palm detect at 192 ----
    const uint32_t PD = 192;
    GpuTensor d_rs, d_in;
    d_rs.alloc(PD * PD * 3);
    d_in.alloc(3 * PD * PD * sizeof(float));
    if (!d_rs.ok() || !d_in.ok()) return 0;
    launch_resize_rgb(d_src, d_rs.data, w, h, PD, PD, stream);
    launch_rgb8_to_nchw(d_rs.data, reinterpret_cast<float*>(d_in.data),
                        PD, PD, 0, stream);
    cudaStreamSynchronize(stream);
    eng.gpu_free(d_rs);

    std::vector<int64_t> psh = {1, 3, PD, PD};
    std::vector<std::vector<int64_t>> pshs = {psh};
    std::vector<GpuTensor> pins = {d_in};
    std::vector<GpuTensor> pouts;
    if (!palm->run(pins, pouts, pshs)) {
        eng.gpu_free(d_in);
        for (auto& o : pouts) eng.gpu_free(o);
        return 0;
    }
    eng.gpu_free(d_in);
    if (pouts.size() < 2) {
        for (auto& o : pouts) eng.gpu_free(o);
        return 0;
    }
    std::vector<float> pbox(2016 * 18), pscore(2016);
    cudaMemcpyAsync(pbox.data(), pouts[0].data, pbox.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pscore.data(), pouts[1].data, pscore.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (auto& o : pouts) eng.gpu_free(o);

    // decode + NMS in 192-space
    std::vector<PalmDet> dets;
    for (int i = 0; i < 2016; i++) {
        float s = psigm(pscore[i]);
        if (s < 0.5f) continue;
        PalmDet d;
        d.x = pbox[i * 18 + 0]; d.y = pbox[i * 18 + 1];
        d.w = pbox[i * 18 + 2]; d.h = pbox[i * 18 + 3];
        d.score = s;
        if (d.w > 0 && d.h > 0) dets.push_back(d);
    }
    std::sort(dets.begin(), dets.end(),
              [](const PalmDet& a, const PalmDet& b) { return a.score > b.score; });
    std::vector<PalmDet> kept;
    for (auto& d : dets) {
        bool drop = false;
        for (auto& k : kept) {
            if (palm_iou(d, k) > 0.3f) { drop = true; break; }
        }
        if (!drop) {
            kept.push_back(d);
            if ((int)kept.size() >= max_hands) break;
        }
    }
    if (kept.empty()) return 0;

    // ---- stage 2: landmark per hand (square crop -> 224 -> [0,1]) ----
    const uint32_t LD = 224;
    float sx = (float)w / PD, sy = (float)h / PD;
    int found = 0;
    for (auto& kd : kept) {
        // square crop around palm box, expanded, clamped to frame
        float cx = (kd.x + kd.w / 2) * sx, cy = (kd.y + kd.h / 2) * sy;
        int side = (int)(1.6f * std::max(kd.w * sx, kd.h * sy));
        if (side < 32) side = 32;
        int x1 = (int)(cx - side / 2), y1 = (int)(cy - side / 2);
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x1 + side > (int)w) x1 = w - side;
        if (y1 + side > (int)h) y1 = h - side;
        if (x1 < 0 || y1 < 0) continue;

        GpuTensor d_sq, d_crop, d_lin;
        d_sq.alloc((size_t)side * side * 3);
        d_crop.alloc(LD * LD * 3);
        d_lin.alloc(3 * LD * LD * sizeof(float));
        if (!d_sq.ok() || !d_crop.ok() || !d_lin.ok()) {
            eng.gpu_free(d_sq); eng.gpu_free(d_crop); eng.gpu_free(d_lin);
            continue;
        }
        launch_crop_rgb(d_src, d_sq.data, w, h, (uint32_t)x1, (uint32_t)y1,
                        (uint32_t)side, (uint32_t)side, stream);
        launch_resize_rgb(d_sq.data, d_crop.data, side, side, LD, LD, stream);
        eng.gpu_free(d_sq);
        launch_rgb8_to_nchw(d_crop.data, reinterpret_cast<float*>(d_lin.data),
                            LD, LD, 0, stream);
        cudaStreamSynchronize(stream);
        eng.gpu_free(d_crop);

        std::vector<int64_t> lsh = {1, 3, LD, LD};
        std::vector<std::vector<int64_t>> lshs = {lsh};
        std::vector<GpuTensor> lins = {d_lin};
        std::vector<GpuTensor> louts;
        bool ok = lm->run(lins, louts, lshs);
        eng.gpu_free(d_lin);
        if (!ok || louts.empty()) {
            for (auto& o : louts) eng.gpu_free(o);
            continue;
        }
        std::vector<float> joints(63);
        cudaMemcpyAsync(joints.data(), louts[0].data, 63 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (auto& o : louts) eng.gpu_free(o);

        HandJoints& hj = hands_out[found];
        hj.score = kd.score;
        for (int j = 0; j < 21; j++) {
            float jx = joints[j * 3 + 0], jy = joints[j * 3 + 1];
            // normalize: MediaPipe-style outputs may be pixels or [0,1]
            if (jx > 1.5f) jx /= (float)LD;
            if (jy > 1.5f) jy /= (float)LD;
            hj.x[j] = (float)x1 + jx * (float)side;
            hj.y[j] = (float)y1 + jy * (float)side;
        }
        found++;
    }
    return found;
}

} // namespace ai
} // namespace kagerou
