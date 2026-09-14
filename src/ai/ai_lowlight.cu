// ============================================================================
// Kagerou AI Low-Light Enhancement — Zero-DCE
// GPU preprocess → ORT inference → enhanced RGB on GPU
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
void launch_nchw_to_rgb8(const float* d_nchw, uint8_t* d_rgb,
                         uint32_t w, uint32_t h,
                         cudaStream_t s);

static bool g_lowlight_init = false;

bool ai_lowlight_init(const char* model_path, int device_id) {
    if (g_lowlight_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("lowlight", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI LowLight] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI LowLight] Model ready.\n");
    g_lowlight_init = true;
    return true;
}

bool ai_lowlight(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h, cudaStream_t stream) {
    if (!g_lowlight_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("lowlight");
    if (!s) return false;

    // NCHW float [0,1]
    size_t in_floats = 3 * (size_t)w * h;
    GpuTensor d_input = eng.gpu_alloc(in_floats * sizeof(float));
    if (!d_input.ok()) return false;
    launch_rgb8_to_nchw(d_src, reinterpret_cast<float*>(d_input.data),
                        w, h, 0, stream);
    cudaStreamSynchronize(stream);

    std::vector<int64_t> shape = {1, 3, (int64_t)h, (int64_t)w};
    std::vector<std::vector<int64_t>> shapes = {shape};
    std::vector<GpuTensor> inputs = {d_input};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs, shapes)) {
        eng.gpu_free(d_input);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }

    launch_nchw_to_rgb8(reinterpret_cast<const float*>(outputs[0].data),
                        d_dst, w, h, stream);

    eng.gpu_free(d_input);
    for (auto& o : outputs) eng.gpu_free(o);
    return true;
}

} // namespace ai
} // namespace kagerou
