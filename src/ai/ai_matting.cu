// ============================================================================
// Kagerou AI Video Matting — RVM MobileNetV3 (stateless mode)
// GPU preprocess → ORT inference → alpha composite (bg blur) on GPU.
// Stateless: zero recurrence every frame (robust, no shape bookkeeping).
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include "kagerou/filters.h"
#include <cstdio>
#include <vector>

namespace kagerou {
namespace ai {

void launch_rgb8_to_nchw(const uint8_t* d_rgb, float* d_nchw,
                         uint32_t w, uint32_t h, uint32_t frame_idx,
                         cudaStream_t s);
void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h,
                       cudaStream_t s);

__global__ void composite_alpha_kernel(const uint8_t* __restrict__ fg,
                                       const uint8_t* __restrict__ bg,
                                       const float* __restrict__ alpha,
                                       uint8_t* __restrict__ out,
                                       uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = fminf(fmaxf(alpha[i], 0.0f), 1.0f);
    a = a * a * (3.0f - 2.0f * a); // smoothstep: crisp matte, feathered edge
    out[i * 3 + 0] = (uint8_t)(a * fg[i * 3 + 0] + (1.0f - a) * bg[i * 3 + 0] + 0.5f);
    out[i * 3 + 1] = (uint8_t)(a * fg[i * 3 + 1] + (1.0f - a) * bg[i * 3 + 1] + 0.5f);
    out[i * 3 + 2] = (uint8_t)(a * fg[i * 3 + 2] + (1.0f - a) * bg[i * 3 + 2] + 0.5f);
}

static bool g_matting_init = false;

// Internal working res. MUST match the static TRT build
// (models/matting/rvm_mobilenetv3_static.onnx is baked [1,3,480,640]).
// Any input size is resized here; the alpha is resized back for
// full-res compositing. Recurrence uses true feedback states.
static const uint32_t MAT_W = 640, MAT_H = 480;
static const int MAT_REC_CH[4] = {16, 20, 40, 64};
static const float MAT_DS = 0.5f;

// Defined in ai_depth.cu (shared TU).
__global__ void bilinear_resize_float_kernel(
    const float* __restrict__ d_in, float* __restrict__ d_out,
    uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h);

bool ai_matting_init(const char* model_path, int device_id, bool use_trt) {
    if (g_matting_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("matting", model_path, device_id, use_trt);
    if (!s) { fprintf(stderr, "[AI Matting] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Matting] Model ready.\n");
    g_matting_init = true;
    return true;
}

bool ai_matting(const uint8_t* d_src, uint8_t* d_dst,
                uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_matting_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("matting");
    if (!s) return false;

    // 1. Working-size RGB + NCHW [0,1] (identity when already 640x480)
    GpuTensor d_rs, d_in;
    d_rs.alloc(MAT_W * MAT_H * 3);
    d_in.alloc(3 * MAT_W * MAT_H * sizeof(float));
    if (!d_rs.ok() || !d_in.ok()) return false;
    if (w == MAT_W && h == MAT_H) {
        cudaMemcpyAsync(d_rs.data, d_src, MAT_W * MAT_H * 3,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_resize_rgb(d_src, d_rs.data, w, h, MAT_W, MAT_H, stream);
    }
    launch_rgb8_to_nchw(d_rs.data, reinterpret_cast<float*>(d_in.data),
                        MAT_W, MAT_H, 0, stream);
    eng.gpu_free(d_rs);

    // 2. Recurrence: feed back the previous frame's states (true RVM
    // temporal mode). First call uses [1,C,1,1] zeros; afterwards the
    // persistent buffers carry real states (allocated to output shapes).
    // Degenerate 1x1 recurrency is what poisoned TRT's shape analysis.
    static GpuTensor s_fb[4];
    static std::vector<int64_t> s_fb_shapes[4];
    static bool s_fb_valid = false;
    GpuTensor d_zero[4];
    bool have_fb = s_fb_valid;
    if (have_fb) {
        for (int k = 0; k < 4; k++) {
            if (!s_fb[k].ok()) { have_fb = false; break; }
        }
    }
    GpuTensor d_ds;
    d_ds.alloc(sizeof(float));
    if (!d_ds.ok()) {
        eng.gpu_free(d_in);
        return false;
    }
    float ds_host = MAT_DS;
    cudaMemcpyAsync(d_ds.data, &ds_host, sizeof(float), cudaMemcpyHostToDevice, stream);
    if (!have_fb) {
        for (int i = 0; i < 4; i++) {
            d_zero[i].alloc(MAT_REC_CH[i] * sizeof(float)); // [1,C,1,1] zeros
            if (!d_zero[i].ok()) {
                eng.gpu_free(d_in);
                for (int k = 0; k < i; k++) eng.gpu_free(d_zero[k]);
                eng.gpu_free(d_ds);
                return false;
            }
        }
    }
    cudaStreamSynchronize(stream);

    // 3. Run: src, r1i..r4i, downsample_ratio (shapes = working res)
    std::vector<int64_t> src_sh = {1, 3, (int64_t)MAT_H, (int64_t)MAT_W};
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back(src_sh);
    if (have_fb) {
        for (int k = 0; k < 4; k++) shapes.push_back(s_fb_shapes[k]);
    } else {
        shapes.push_back({1, 16, 1, 1}); shapes.push_back({1, 20, 1, 1});
        shapes.push_back({1, 40, 1, 1}); shapes.push_back({1, 64, 1, 1});
    }
    shapes.push_back({1});
    std::vector<GpuTensor> inputs;
    inputs.push_back(d_in);
    for (int k = 0; k < 4; k++) inputs.push_back(have_fb ? s_fb[k] : d_zero[k]);
    inputs.push_back(d_ds);
    std::vector<GpuTensor> outputs;
    std::vector<std::vector<int64_t>> out_shapes;
    if (!s->run(inputs, outputs, shapes, &out_shapes)) {
        eng.gpu_free(d_in);
        for (int k = 0; k < 4; k++) eng.gpu_free(d_zero[k]);
        eng.gpu_free(d_ds);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    eng.gpu_free(d_in);
    for (int k = 0; k < 4; k++) eng.gpu_free(d_zero[k]);
    eng.gpu_free(d_ds);
    if (outputs.size() < 6 || out_shapes.size() < 6) {
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    // 3b. Refresh persistent feedback states from r1o..r4o.
    {
        bool shape_ok = true;
        for (int k = 0; k < 4; k++) {
            if (out_shapes[2 + k].size() != 4) { shape_ok = false; break; }
        }
        if (shape_ok) {
            bool same = s_fb_valid;
            for (int k = 0; k < 4 && same; k++)
                same = (s_fb_shapes[k] == out_shapes[2 + k]);
            if (!same) {
                for (int k = 0; k < 4; k++) {
                    s_fb[k].free();
                    size_t n = 1;
                    for (auto d : out_shapes[2 + k]) n *= (size_t)d;
                    s_fb[k].alloc(n * sizeof(float));
                    s_fb_shapes[k] = out_shapes[2 + k];
                    if (!s_fb[k].ok()) same = false;
                }
            }
            if (same) {
                for (int k = 0; k < 4; k++) {
                    size_t n = 1;
                    for (auto d : out_shapes[2 + k]) n *= (size_t)d;
                    cudaMemcpyAsync(s_fb[k].data, outputs[2 + k].data,
                                    n * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream);
                }
                s_fb_valid = true;
            } else {
                s_fb_valid = false;
            }
        }
    }

    // 4. pha is working-res (MAT 640x480 via guided refiner);
    // upscale to full res, blur bg, composite. outputs[0]=fgr (unused).
    GpuTensor d_pha, d_bg;
    d_pha.alloc((size_t)w * h * sizeof(float));
    d_bg.alloc((size_t)w * h * 3);
    if (!d_pha.ok() || !d_bg.ok()) {
        eng.gpu_free(d_pha); eng.gpu_free(d_bg);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    if (w == MAT_W && h == MAT_H) {
        cudaMemcpyAsync(d_pha.data, outputs[1].data,
                        (size_t)w * h * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        bilinear_resize_float_kernel<<<((size_t)w*h+255)/256, 256, 0, stream>>>(
            reinterpret_cast<const float*>(outputs[1].data),
            reinterpret_cast<float*>(d_pha.data),
            MAT_W, MAT_H, w, h);
    }
    filters::gaussian_blur(d_src, d_bg.data, w, h, 3, 6.0f, stream);
    {
        uint32_t n = w * h;
        dim3 b(256), g((n + 255) / 256);
        composite_alpha_kernel<<<g, b, 0, stream>>>(
            d_src, d_bg.data, reinterpret_cast<const float*>(d_pha.data),
            d_dst, n);
    }
    eng.gpu_free(d_pha);
    eng.gpu_free(d_bg);
    for (auto& o : outputs) eng.gpu_free(o);
    // Feedback buffers are shared across caller streams (warmup vs frames).
    cudaStreamSynchronize(stream);
    return true;
}

} // namespace ai
} // namespace kagerou
