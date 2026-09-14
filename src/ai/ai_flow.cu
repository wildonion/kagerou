// ============================================================================
// Kagerou AI Optical Flow — RAFT Small
// GPU preprocess → ORT inference → flow field on GPU
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
                       uint32_t dst_w, uint32_t dst_h,
                       cudaStream_t s);
void launch_deinterleave_flow(const float* d_planar, float* d_interleaved,
                              uint32_t n, cudaStream_t s);

static bool g_flow_init = false;

__global__ void resize_flow_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dw * dh) return;
    uint32_t dx = i % dw, dy = i / dw;
    uint32_t sx = min((uint32_t)((float)dx * sw / dw), sw - 1);
    uint32_t sy = min((uint32_t)((float)dy * sh / dh), sh - 1);
    uint32_t si = sy * sw + sx;
    dst[i * 2 + 0] = src[si * 2 + 0] * (float)dw / (float)sw;
    dst[i * 2 + 1] = src[si * 2 + 1] * (float)dh / (float)sh;
}

bool ai_flow_init(const char* model_path, int device_id) {
    if (g_flow_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("raft", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Flow] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Flow] Model ready. Inputs: ");
    for (size_t i = 0; i < s->num_inputs(); i++) {
        auto sh = s->input_shape(i);
        fprintf(stderr, "in%zu=[", i);
        for (size_t j = 0; j < sh.size(); j++)
            fprintf(stderr, "%s%lld", j ? "," : "", sh[j]);
        fprintf(stderr, "] ");
    }
    fprintf(stderr, "\n");
    g_flow_init = true;
    return true;
}

bool ai_flow(const uint8_t* d_frame1, const uint8_t* d_frame2,
             float* d_flow, uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_flow_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("raft");
    if (!s) return false;

    const uint32_t RAFT_H = 360, RAFT_W = 480;

    // 1. Resize both frames to 360x480 RGB, then convert to NCHW float
    GpuTensor d_rgb1, d_rgb2;
    d_rgb1.alloc(RAFT_W * RAFT_H * 3);
    d_rgb2.alloc(RAFT_W * RAFT_H * 3);
    if (!d_rgb1.ok() || !d_rgb2.ok()) return false;

    launch_resize_rgb(d_frame1, d_rgb1.data, w, h, RAFT_W, RAFT_H, stream);
    launch_resize_rgb(d_frame2, d_rgb2.data, w, h, RAFT_W, RAFT_H, stream);

    size_t in_floats = 3 * RAFT_W * RAFT_H;
    GpuTensor d_in1, d_in2;
    d_in1.alloc(in_floats * sizeof(float));
    d_in2.alloc(in_floats * sizeof(float));
    if (!d_in1.ok() || !d_in2.ok()) {
        eng.gpu_free(d_rgb1); eng.gpu_free(d_rgb2);
        return false;
    }

    launch_rgb8_to_nchw(d_rgb1.data, reinterpret_cast<float*>(d_in1.data), RAFT_W, RAFT_H, 0, stream);
    launch_rgb8_to_nchw(d_rgb2.data, reinterpret_cast<float*>(d_in2.data), RAFT_W, RAFT_H, 0, stream);

    eng.gpu_free(d_rgb1); eng.gpu_free(d_rgb2);

    cudaStreamSynchronize(stream);

    // 2. Run inference
    std::vector<GpuTensor> inputs = {d_in1, d_in2};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs)) {
        eng.gpu_free(d_in1); eng.gpu_free(d_in2);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }

    // 3. Get full-res flow [1,2,360,480] NCHW planar, deinterleave, resize to original dims
    float* d_model_flow = reinterpret_cast<float*>(outputs[1].data);

    // Deinterleave planar [U_ch | V_ch] → interleaved [u0,v0, u1,v1, ...]
    const uint32_t flow_n = RAFT_W * RAFT_H;
    GpuTensor d_deint;
    d_deint.alloc(flow_n * 2 * sizeof(float));
    if (!d_deint.ok()) {
        eng.gpu_free(d_in1); eng.gpu_free(d_in2);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }
    launch_deinterleave_flow(d_model_flow, reinterpret_cast<float*>(d_deint.data),
                             flow_n, stream);
    cudaStreamSynchronize(stream);

    // Resize interleaved flow from RAFT dims to original dims using nearest-neighbor
    {
        dim3 b(256);
        dim3 g((w * h + 255) / 256);
        resize_flow_kernel<<<g, b, 0, stream>>>(
            reinterpret_cast<float*>(d_deint.data), d_flow, RAFT_W, RAFT_H, w, h);
    }

    eng.gpu_free(d_in1); eng.gpu_free(d_in2); eng.gpu_free(d_deint);
    for (auto& o : outputs) eng.gpu_free(o);
    return true;
}

} // namespace ai
} // namespace kagerou
