// ============================================================================
// Kagerou AI Gaze — YuNet face box + MediaPipe iris per eye.
// Runs face detection internally, crops both eye regions, runs the iris
// model ([-1,1] NCHW 64x64, validated by probe) and reports iris centers
// + gaze direction. No app-side face state needed.
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>
#include <cmath>

namespace kagerou {
namespace ai {

void launch_crop_rgb(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t src_w, uint32_t src_h,
                     uint32_t crop_x, uint32_t crop_y,
                     uint32_t crop_w, uint32_t crop_h,
                     cudaStream_t s);

__global__ void rgb8_to_nchw_m11_eye_kernel(const uint8_t* __restrict__ rgb,
                                            float* __restrict__ nchw,
                                            uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t plane = w * h, pixel = y * w + x, si = pixel * 3;
    nchw[0 * plane + pixel] = rgb[si + 0] * (2.0f / 255.0f) - 1.0f;
    nchw[1 * plane + pixel] = rgb[si + 1] * (2.0f / 255.0f) - 1.0f;
    nchw[2 * plane + pixel] = rgb[si + 2] * (2.0f / 255.0f) - 1.0f;
}

static bool g_gaze_init = false;

bool ai_gaze_init(const char* iris_path, int device_id) {
    if (g_gaze_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    // face session must exist (demo inits it via the Face button path too)
    if (!eng.find("face")) {
        fprintf(stderr, "[AI Gaze] face session missing (init face first)\n");
        return false;
    }
    if (!eng.load("iris", iris_path, device_id)) {
        fprintf(stderr, "[AI Gaze] Cannot load %s\n", iris_path); return false;
    }
    fprintf(stderr, "[AI Gaze] Model ready.\n");
    g_gaze_init = true;
    return true;
}

int ai_gaze(const uint8_t* d_src, uint32_t w, uint32_t h,
            GazeEye* eyes_out, int max_eyes, cudaStream_t stream) {
    if (!g_gaze_init || max_eyes <= 0) return 0;
    auto& eng = AiInference::get();
    auto* iris = eng.find("iris");
    if (!iris) return 0;

    // 1. face box (full YuNet pass)
    FaceBox fb[2];
    int nface = ai_face(d_src, w, h, fb, 2, stream);
    if (nface <= 0) return 0;

    // 2. eye crops (64x64 fixed windows at heuristic eye centers)
    const uint32_t ED = 64;
    int found = 0;
    for (int f = 0; f < nface && found < max_eyes; f++) {
        float fw = fb[f].x2 - fb[f].x1, fh = fb[f].y2 - fb[f].y1;
        float exs[2] = {fb[f].x1 + 0.30f * fw, fb[f].x1 + 0.70f * fw};
        float ey = fb[f].y1 + 0.42f * fh;
        for (int e = 0; e < 2 && found < max_eyes; e++) {
            int x1 = (int)(exs[e] - ED / 2), y1 = (int)(ey - ED / 2);
            if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
            if (x1 + (int)ED > (int)w) x1 = w - ED;
            if (y1 + (int)ED > (int)h) y1 = h - ED;
            if (x1 < 0 || y1 < 0) continue;

            GpuTensor d_crop, d_in;
            d_crop.alloc(ED * ED * 3);
            d_in.alloc(3 * ED * ED * sizeof(float));
            if (!d_crop.ok() || !d_in.ok()) {
                eng.gpu_free(d_crop); eng.gpu_free(d_in);
                continue;
            }
            launch_crop_rgb(d_src, d_crop.data, w, h, (uint32_t)x1, (uint32_t)y1,
                            ED, ED, stream);
            {
                dim3 b(16, 16), g((ED + 15) / 16, (ED + 15) / 16);
                rgb8_to_nchw_m11_eye_kernel<<<g, b, 0, stream>>>(
                    d_crop.data, reinterpret_cast<float*>(d_in.data), ED, ED);
            }
            cudaStreamSynchronize(stream);
            eng.gpu_free(d_crop);

            std::vector<int64_t> shape = {1, 3, ED, ED};
            std::vector<std::vector<int64_t>> shapes = {shape};
            std::vector<GpuTensor> inputs = {d_in};
            std::vector<GpuTensor> outputs;
            if (!iris->run(inputs, outputs, shapes)) {
                eng.gpu_free(d_in);
                for (auto& o : outputs) eng.gpu_free(o);
                continue;
            }
            eng.gpu_free(d_in);
            if (outputs.size() < 2) {
                for (auto& o : outputs) eng.gpu_free(o);
                continue;
            }
            // iris center = mean of the 5 iris points (order-proof)
            std::vector<float> ipts(15);
            cudaMemcpyAsync(ipts.data(), outputs[1].data, 15 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (auto& o : outputs) eng.gpu_free(o);

            float mx = 0, my = 0;
            for (int j = 0; j < 5; j++) { mx += ipts[j * 3]; my += ipts[j * 3 + 1]; }
            mx /= 5; my /= 5;
            if (mx < 0 || mx >= (float)ED || my < 0 || my >= (float)ED) continue;
            GazeEye& ge = eyes_out[found];
            ge.cx = (float)x1 + mx;
            ge.cy = (float)y1 + my;
            float dx = mx - ED / 2.0f, dy = my - ED / 2.0f;
            float n = sqrtf(dx * dx + dy * dy);
            ge.dx = (n > 1e-3f) ? dx / n : 0;
            ge.dy = (n > 1e-3f) ? dy / n : 0;
            ge.er = 0.16f * (fb[f].x2 - fb[f].x1);
            if (ge.er < 6.0f) ge.er = 6.0f;
            ge.score = fb[f].score;
            found++;
        }
    }
    return found;
}

// ---- Eye-contact redirect -------------------------------------------------
// Shifts eyeball pixels toward the lens inside a feathered ellipse.
// Reads ONLY d_src, writes ONLY d_dst pixels inside its own ellipse;
// untouched pixels are copied through. Buffers must differ.
__global__ void eye_redirect_kernel(const uint8_t* __restrict__ src,
                                    uint8_t* __restrict__ dst,
                                    uint32_t w, uint32_t h,
                                    float ex, float ey, float er,
                                    float dx, float dy, float strength) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t idx = (y * w + x) * 3;

    float rx = ((float)x - ex) / er;
    float ry = ((float)y - ey) / er;
    float d2 = rx * rx + ry * ry;
    if (d2 >= 1.0f) {
        dst[idx + 0] = src[idx + 0];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 2];
        return;
    }
    // feathered mask: full shift at center, fading to the rim
    float d = sqrtf(d2);
    float m = 1.0f - (d * d * (3.0f - 2.0f * d)); // smoothstep falloff
    float ax = dx * strength * m, ay = dy * strength * m;
    float sx = (float)x - ax, sy = (float)y - ay;
    if (sx < 0) sx = 0; if (sy < 0) sy = 0;
    if (sx > (float)(w - 1)) sx = (float)(w - 1);
    if (sy > (float)(h - 1)) sy = (float)(h - 1);
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = x0 + 1 < (int)w ? x0 + 1 : x0;
    int y1 = y0 + 1 < (int)h ? y0 + 1 : y0;
    float fx = sx - x0, fy = sy - y0;
    for (int c = 0; c < 3; c++) {
        float v = (1 - fx) * (1 - fy) * src[(y0 * w + x0) * 3 + c]
                + fx * (1 - fy) * src[(y0 * w + x1) * 3 + c]
                + (1 - fx) * fy * src[(y1 * w + x0) * 3 + c]
                + fx * fy * src[(y1 * w + x1) * 3 + c];
        dst[idx + c] = (uint8_t)(v + 0.5f);
    }
}

bool ai_eye_correct(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    const GazeEye* eyes, int neyes, float strength,
                    cudaStream_t stream) {
    if (neyes <= 0 || strength <= 0.0f) return false;
    if (strength > 1.0f) strength = 1.0f;
    dim3 b(16, 16), g((w + 15) / 16, (h + 15) / 16);
    int done = 0;
    for (int i = 0; i < neyes; i++) {
        if (eyes[i].score < 0.4f) continue;
        // shift opposite the gaze offset: pull iris back to eye center
        eye_redirect_kernel<<<g, b, 0, stream>>>(
            d_src, d_dst, w, h,
            eyes[i].cx, eyes[i].cy, eyes[i].er,
            -eyes[i].dx * eyes[i].er * 0.45f,
            -eyes[i].dy * eyes[i].er * 0.45f,
            strength);
        d_src = d_dst; // chain eyes through the same output
        done++;
    }
    return done > 0;
}

} // namespace ai
} // namespace kagerou
