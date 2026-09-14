// ============================================================================
// Kagerou AI Denoise — FastDVDNet Temporal Noise Reduction
// FULL GPU PATH: preprocess (CUDA kernels) → TRT inference → postprocess
// Zero CPU copies. Zero staging. Frames stay on GPU the entire time.
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>

namespace kagerou {
namespace ai {

// ============================================================================
// GPU preprocess kernels (declared in ai_preprocess.cu)
// ============================================================================
void launch_pack_5frames(const uint8_t* frames[5], float* d_batch,
                         uint32_t w, uint32_t h, cudaStream_t s);
void launch_unpack_nchw(const float* d_nchw, uint8_t* d_rgb,
                        uint32_t w, uint32_t h, cudaStream_t s);

// ============================================================================
__global__ void fill_const_kernel(float* d, uint32_t n, float v) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}

static bool g_denoise_init = false;

bool ai_denoise_init(const char* model_path, int device_id) {
    if (g_denoise_init) return true;

    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;

    auto* s = eng.load("fastdvdnet", model_path, device_id);
    if (!s) {
        fprintf(stderr, "[AI Denoise] Cannot load %s\n", model_path);
        return false;
    }

    fprintf(stderr, "[AI Denoise] Model ready. Input shape: [");
    auto sh = s->input_shape(0);
    for (size_t i = 0; i < sh.size(); i++)
        fprintf(stderr, "%s%lld", i ? "," : "", sh[i]);
    fprintf(stderr, "]\n");

    g_denoise_init = true;
    return true;
}

bool ai_denoise(const uint8_t* d_prev4, const uint8_t* d_prev3,
                const uint8_t* d_prev2, const uint8_t* d_prev1,
                const uint8_t* d_curr, uint8_t* d_dst,
                uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_denoise_init) return false;

    auto& eng = AiInference::get();
    auto* s = eng.find("fastdvdnet");
    if (!s) return false;

    // ================================================================
    // ALL ON GPU. No cudaMemcpy. No staging. No CPU.
    // ================================================================

    // 1. Pack 5 RGB frames → NCHW float [1, 15, H, W] on GPU
    size_t batch_floats = 5 * 3 * w * h;
    size_t batch_bytes  = batch_floats * sizeof(float);
    GpuTensor d_batch = eng.gpu_alloc(batch_bytes);

    const uint8_t* frames[5] = {d_prev4, d_prev3, d_prev2, d_prev1, d_curr};
    launch_pack_5frames(frames, reinterpret_cast<float*>(d_batch.data), w, h, stream);

    // 2. Create noise_map [1, 1, H, W] — estimated noise sigma.
    // Must be small (typical sensor noise is ~0.02-0.08). The old 0x3F
    // byte-fill produced ~0.75, i.e. "extremely noisy", so the model
    // melted the picture into a waxy orange smear, worse on motion.
    GpuTensor d_noise_map = eng.gpu_alloc(w * h * sizeof(float));
    if (d_noise_map.ok()) {
        fill_const_kernel<<<((w*h)+255)/256, 256, 0, stream>>>(
            reinterpret_cast<float*>(d_noise_map.data), w * h, 0.05f);
    }

    // 3. Sync pipeline stream so ORT sees the input
    cudaStreamSynchronize(stream);

    // 4. Run inference with explicit shapes for both inputs
    std::vector<int64_t> frames_shape = {1, 15, (int64_t)h, (int64_t)w};
    std::vector<int64_t> noise_shape = {1, 1, (int64_t)h, (int64_t)w};
    std::vector<std::vector<int64_t>> shapes = {frames_shape, noise_shape};
    std::vector<GpuTensor> inputs  = {d_batch, d_noise_map};
    std::vector<GpuTensor> outputs;

    if (!s->run(inputs, outputs, shapes)) {
        eng.gpu_free(d_batch);
        return false;
    }

    // 3. Unpack output NCHW → RGB uint8 on GPU
    //    outputs[0] is [1, 3, H, W] float
    launch_unpack_nchw(reinterpret_cast<const float*>(outputs[0].data), d_dst, w, h, stream);

    // 4. Cleanup
    eng.gpu_free(d_batch);
    eng.gpu_free(d_noise_map);
    for (auto& o : outputs) eng.gpu_free(o);

    return true;
}

} // namespace ai
} // namespace kagerou
