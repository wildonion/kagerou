// ============================================================================
// Kagerou AI Face Detection — YuNet (OpenCV Zoo 2023mar)
// GPU preprocess → ORT inference → CPU decode + NMS → face boxes
// Decode math: libfacedetection.train (strides 8/16/32, point priors,
// score=sigmoid(cls)*sigmoid(obj), exp box decode, IoU NMS).
// NOTE: input channel order/normalization verified by probe (see below).
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
void launch_letterbox_resize_rgb(const uint8_t* src, uint8_t* dst,
                                 uint32_t src_w, uint32_t src_h,
                                 uint32_t dst_w, uint32_t dst_h,
                                 float& out_scale, float& out_pad_x, float& out_pad_y,
                                 cudaStream_t s);

// PROBE-SELECTED input convention: BGR order, RAW [0,255] values
// (no /255). Variants with /255 found nothing; raw255 found the face.
#define YUNET_SWAP_RB 1
#define YUNET_SCALE 1.0f
#define YUNET_SCORE_THRESH 0.4f

__global__ void yunet_pack_kernel(const uint8_t* __restrict__ rgb,
                                  float* __restrict__ nchw,
                                  uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t plane = w * h, pixel = y * w + x, si = pixel * 3;
    float r = rgb[si + 0], g = rgb[si + 1], b = rgb[si + 2];
#if YUNET_SWAP_RB
    nchw[0 * plane + pixel] = b * YUNET_SCALE;
    nchw[1 * plane + pixel] = g * YUNET_SCALE;
    nchw[2 * plane + pixel] = r * YUNET_SCALE;
#else
    nchw[0 * plane + pixel] = r * YUNET_SCALE;
    nchw[1 * plane + pixel] = g * YUNET_SCALE;
    nchw[2 * plane + pixel] = b * YUNET_SCALE;
#endif
}

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static bool g_face_init = false;

bool ai_face_init(const char* model_path, int device_id) {
    if (g_face_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("face", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Face] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Face] Model ready.\n");
    g_face_init = true;
    return true;
}

struct YunetDet { float x1, y1, x2, y2, score; };

static float iou(const YunetDet& a, const YunetDet& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0;
}

int ai_face(const uint8_t* d_src, uint32_t w, uint32_t h,
            FaceBox* boxes_out, int max_boxes, cudaStream_t stream) {
    if (!g_face_init || max_boxes <= 0) return 0;
    auto& eng = AiInference::get();
    auto* s = eng.find("face");
    if (!s) return 0;

    const uint32_t YUNET_DIM = 640;
    // 1. letterbox to 640x640 (aspect preserved — stretching portrait
    // video into a square makes faces undetectable)
    GpuTensor d_rs, d_in;
    d_rs.alloc(YUNET_DIM * YUNET_DIM * 3);
    d_in.alloc(3 * YUNET_DIM * YUNET_DIM * sizeof(float));
    if (!d_rs.ok() || !d_in.ok()) return 0;
    float lb_scale = 1, lb_px = 0, lb_py = 0;
    launch_letterbox_resize_rgb(d_src, d_rs.data, w, h,
                                YUNET_DIM, YUNET_DIM,
                                lb_scale, lb_px, lb_py, stream);
    {
        dim3 b(16, 16), g((YUNET_DIM + 15) / 16, (YUNET_DIM + 15) / 16);
        yunet_pack_kernel<<<g, b, 0, stream>>>(
            d_rs.data, reinterpret_cast<float*>(d_in.data), YUNET_DIM, YUNET_DIM);
    }
    cudaStreamSynchronize(stream);
    eng.gpu_free(d_rs);

    // 2. run (single input)
    std::vector<int64_t> shape = {1, 3, YUNET_DIM, YUNET_DIM};
    std::vector<std::vector<int64_t>> shapes = {shape};
    std::vector<GpuTensor> inputs = {d_in};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs, shapes)) {
        eng.gpu_free(d_in);
        for (auto& o : outputs) eng.gpu_free(o);
        return 0;
    }
    eng.gpu_free(d_in);
    if (outputs.size() < 12) {
        for (auto& o : outputs) eng.gpu_free(o);
        return 0;
    }

    // 3. download tensors. Graph order groups by TYPE:
    // [cls8,cls16,cls32, obj8,obj16,obj32, bbox8,bbox16,bbox32, kps...]
    const int CNT[3] = {6400, 1600, 400};
    const int ST[3] = {8, 16, 32};
    std::vector<float> cls[3], obj[3], box[3];
    for (int l = 0; l < 3; l++) {
        cls[l].resize(CNT[l]); obj[l].resize(CNT[l]); box[l].resize(CNT[l] * 4);
        cudaMemcpyAsync(cls[l].data(), outputs[l].data, CNT[l] * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(obj[l].data(), outputs[3 + l].data, CNT[l] * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(box[l].data(), outputs[6 + l].data, CNT[l] * 4 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    for (auto& o : outputs) eng.gpu_free(o);

    // 4. decode: point priors, score, exp boxes, back to frame dims
    // (undo the letterbox: unpad, then unscale)
    std::vector<YunetDet> dets;
    dets.reserve(256);
    for (int l = 0; l < 3; l++) {
        int g = YUNET_DIM / ST[l];
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                int k = i * g + j;
                float score = sigm(cls[l][k]) * sigm(obj[l][k]);
                if (score < YUNET_SCORE_THRESH) continue;
                float px = (j + 0.5f) * ST[l], py = (i + 0.5f) * ST[l];
                float pw = (float)ST[l], ph = (float)ST[l];
                float cx = px + box[l][k * 4 + 0] * pw;
                float cy = py + box[l][k * 4 + 1] * ph;
                float bw = expf(box[l][k * 4 + 2]) * pw;
                float bh = expf(box[l][k * 4 + 3]) * ph;
                YunetDet d;
                d.x1 = (cx - bw / 2 - lb_px) / lb_scale;
                d.y1 = (cy - bh / 2 - lb_py) / lb_scale;
                d.x2 = (cx + bw / 2 - lb_px) / lb_scale;
                d.y2 = (cy + bh / 2 - lb_py) / lb_scale;
                if (d.x1 < 0) d.x1 = 0; if (d.y1 < 0) d.y1 = 0;
                if (d.x2 > (float)w) d.x2 = (float)w;
                if (d.y2 > (float)h) d.y2 = (float)h;
                d.score = score;
                if (d.x2 > d.x1 && d.y2 > d.y1) dets.push_back(d);
            }
        }
    }

    // 5. NMS
    std::sort(dets.begin(), dets.end(),
              [](const YunetDet& a, const YunetDet& b) { return a.score > b.score; });
    std::vector<YunetDet> kept;
    for (auto& d : dets) {
        bool drop = false;
        for (auto& k : kept) {
            if (iou(d, k) > 0.3f) { drop = true; break; }
        }
        if (!drop) {
            kept.push_back(d);
            if ((int)kept.size() >= max_boxes) break;
        }
    }
    for (size_t i = 0; i < kept.size(); i++) {
        boxes_out[i].x1 = kept[i].x1; boxes_out[i].y1 = kept[i].y1;
        boxes_out[i].x2 = kept[i].x2; boxes_out[i].y2 = kept[i].y2;
        boxes_out[i].score = kept[i].score;
    }
    return (int)kept.size();
}

} // namespace ai
} // namespace kagerou
