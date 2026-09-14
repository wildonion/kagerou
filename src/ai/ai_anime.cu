// ============================================================================
// Kagerou AI Style Transfer — AnimeGANv3 Hayao
// GPU preprocess → ORT inference → stylized RGB on GPU
// Model is NHWC with [-1,1] input and tanh [-1,1] output.
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>

namespace kagerou {
namespace ai {

void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h,
                       cudaStream_t s);

__global__ void rgb8_to_nhwc_m11_kernel(const uint8_t* __restrict__ rgb,
                                        float* __restrict__ nhwc,
                                        uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t si = (y * w + x) * 3;
    uint32_t di = (y * w + x) * 3;
    nhwc[di + 0] = rgb[si + 0] * (2.0f / 255.0f) - 1.0f;
    nhwc[di + 1] = rgb[si + 1] * (2.0f / 255.0f) - 1.0f;
    nhwc[di + 2] = rgb[si + 2] * (2.0f / 255.0f) - 1.0f;
}

__global__ void nhwc_tanh_to_rgb8_kernel(const float* __restrict__ nhwc,
                                         uint8_t* __restrict__ rgb,
                                         uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t si = (y * w + x) * 3;
    float r = nhwc[si + 0] * 0.5f + 0.5f;
    float g = nhwc[si + 1] * 0.5f + 0.5f;
    float b = nhwc[si + 2] * 0.5f + 0.5f;
    rgb[si + 0] = (uint8_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
    rgb[si + 1] = (uint8_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
    rgb[si + 2] = (uint8_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
}

static bool g_anime_init = false;

bool ai_anime_init(const char* model_path, int device_id) {
    if (g_anime_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("anime", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Anime] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Anime] Model ready.\n");
    g_anime_init = true;
    return true;
}

bool ai_anime(const uint8_t* d_src, uint8_t* d_dst,
              uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_anime_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("anime");
    if (!s) return false;

    const uint32_t D = 256;
    GpuTensor d_rs, d_in;
    d_rs.alloc(D * D * 3);
    d_in.alloc(3 * D * D * sizeof(float));
    if (!d_rs.ok() || !d_in.ok()) return false;
    launch_resize_rgb(d_src, d_rs.data, w, h, D, D, stream);
    {
        dim3 b(16, 16), g((D + 15) / 16, (D + 15) / 16);
        rgb8_to_nhwc_m11_kernel<<<g, b, 0, stream>>>(
            d_rs.data, reinterpret_cast<float*>(d_in.data), D, D);
    }
    cudaStreamSynchronize(stream);
    eng.gpu_free(d_rs);

    std::vector<int64_t> shape = {1, (int64_t)D, (int64_t)D, 3};
    std::vector<std::vector<int64_t>> shapes = {shape};
    std::vector<GpuTensor> inputs = {d_in};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs, shapes)) {
        eng.gpu_free(d_in);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    eng.gpu_free(d_in);

    // Upscale stylized 256 output back to frame size
    GpuTensor d_small;
    d_small.alloc(D * D * 3);
    if (!d_small.ok()) {
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    {
        dim3 b(16, 16), g((D + 15) / 16, (D + 15) / 16);
        nhwc_tanh_to_rgb8_kernel<<<g, b, 0, stream>>>(
            reinterpret_cast<const float*>(outputs[0].data), d_small.data, D, D);
    }
    for (auto& o : outputs) eng.gpu_free(o);
    launch_resize_rgb(d_small.data, d_dst, D, D, w, h, stream);
    eng.gpu_free(d_small);
    return true;
}

} // namespace ai
} // namespace kagerou
