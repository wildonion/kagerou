// ============================================================================
// Kagerou AI Depth Estimation — Depth Anything V2
// GPU preprocess → ORT inference → depth map on GPU
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>

namespace kagerou {
namespace ai {

void launch_rgb8_to_nchw(const uint8_t* d_rgb, float* d_nchw,
                         uint32_t w, uint32_t h, uint32_t frame_idx,
                         cudaStream_t s);
void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h, cudaStream_t s);
void launch_normalize_01_to_11(float* d_data, uint32_t count, cudaStream_t s);

__global__ void bilinear_resize_float_kernel(
    const float* __restrict__ d_in, float* __restrict__ d_out,
    uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;
    float fx = (float)x * in_w / out_w;
    float fy = (float)y * in_h / out_h;
    int x0 = min((int)fx, (int)in_w - 1), y0 = min((int)fy, (int)in_h - 1);
    int x1 = min(x0 + 1, (int)in_w - 1), y1 = min(y0 + 1, (int)in_h - 1);
    float wx = fx - x0, wy = fy - y0;
    d_out[y * out_w + x] = d_in[y0*in_w+x0]*(1-wx)*(1-wy) + d_in[y0*in_w+x1]*wx*(1-wy)
                          + d_in[y1*in_w+x0]*(1-wx)*wy + d_in[y1*in_w+x1]*wx*wy;
}

static bool g_depth_init = false;

bool ai_depth_init(const char* model_path, int device_id) {
    if (g_depth_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("depth", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Depth] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Depth] Model ready. Input: [");
    auto sh = s->input_shape(0);
    for (size_t i = 0; i < sh.size(); i++)
        fprintf(stderr, "%s%lld", i ? "," : "", sh[i]);
    fprintf(stderr, "]\n");
    g_depth_init = true;
    return true;
}

bool ai_depth(const uint8_t* d_src, float* d_depth,
              uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_depth_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("depth");
    if (!s) return false;

    const uint32_t MODEL_DIM = 518;
    auto out_shape = s->output_shape(0);
    int64_t out_h = out_shape.size() > 1 ? out_shape[1] : MODEL_DIM;
    int64_t out_w = out_shape.size() > 2 ? out_shape[2] : MODEL_DIM;

    // 1. Resize to MODEL_DIM x MODEL_DIM
    GpuTensor d_resized;
    d_resized.alloc(MODEL_DIM * MODEL_DIM * 3);
    launch_resize_rgb(d_src, d_resized.data, w, h, MODEL_DIM, MODEL_DIM, stream);

    // 2. NCHW float [0,1] → normalize [-1,1]
    size_t in_floats = 3 * MODEL_DIM * MODEL_DIM;
    GpuTensor d_input = eng.gpu_alloc(in_floats * sizeof(float));
    if (!d_input.ok()) { d_resized.free(); return false; }
    launch_rgb8_to_nchw(d_resized.data, reinterpret_cast<float*>(d_input.data),
                        MODEL_DIM, MODEL_DIM, 0, stream);
    launch_normalize_01_to_11(reinterpret_cast<float*>(d_input.data), in_floats, stream);

    // 3. Sync pipeline stream so ORT (which uses its own stream) sees the input
    cudaStreamSynchronize(stream);

    // 4. Run inference with explicit shape [1, 3, 518, 518]
    std::vector<int64_t> shape = {1, 3, (int64_t)MODEL_DIM, (int64_t)MODEL_DIM};
    std::vector<std::vector<int64_t>> shapes = {shape};
    std::vector<GpuTensor> inputs = {d_input};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs, shapes)) {
        d_resized.free();
        eng.gpu_free(d_input);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }

    // 4. Output is [1, out_h, out_w] — resize to original dims
    if (out_w == (int64_t)w && out_h == (int64_t)h) {
        cudaMemcpyAsync(d_depth, outputs[0].data, w * h * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        dim3 b(16, 16), g((w + 15) / 16, (h + 15) / 16);
        bilinear_resize_float_kernel<<<g, b, 0, stream>>>(
            reinterpret_cast<const float*>(outputs[0].data), d_depth,
            out_w, out_h, w, h);
    }

    d_resized.free();
    eng.gpu_free(d_input);
    for (auto& o : outputs) eng.gpu_free(o);
    return true;
}

} // namespace ai
} // namespace kagerou
