// ============================================================================
// Kagerou AI Pose Estimation — MediaPipe Pose (33 landmarks)
// GPU preprocess → ORT inference → PoseLandmark[33]
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <vector>

namespace kagerou {
namespace ai {

void launch_rgb8_to_nhwc_float(const uint8_t* d_rgb, float* d_out,
                               uint32_t w, uint32_t h, cudaStream_t s);
void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h, cudaStream_t s);

static bool g_pose_init = false;

bool ai_pose_init(const char* model_path, int device_id) {
    if (g_pose_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("pose", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Pose] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Pose] Model ready. Input: [");
    auto sh = s->input_shape(0);
    for (size_t i = 0; i < sh.size(); i++)
        fprintf(stderr, "%s%lld", i ? "," : "", sh[i]);
    fprintf(stderr, "]\n");
    g_pose_init = true;
    return true;
}

bool ai_pose(const uint8_t* d_src, uint32_t w, uint32_t h,
             PoseLandmark* landmarks_out, cudaStream_t stream) {
    if (!g_pose_init) return false;
    auto& eng = AiInference::get();
    auto* s = eng.find("pose");
    if (!s) return false;

    // MediaPipe Pose expects [1, 256, 256, 3] — NHWC float [0,1]
    const uint32_t POSE_DIM = 256;

    // 1. Resize input to 256x256
    GpuTensor d_resized;
    d_resized.alloc(POSE_DIM * POSE_DIM * 3);
    launch_resize_rgb(d_src, d_resized.data, w, h, POSE_DIM, POSE_DIM, stream);

    // 2. Convert to NHWC float [0,1] (NOT NCHW — model expects NHWC)
    GpuTensor d_input;
    d_input.alloc(POSE_DIM * POSE_DIM * 3 * sizeof(float));
    launch_rgb8_to_nhwc_float(d_resized.data, reinterpret_cast<float*>(d_input.data),
                              POSE_DIM, POSE_DIM, stream);

    // 3. Sync pipeline stream so ORT sees the input
    cudaStreamSynchronize(stream);

    // 4. Run inference
    std::vector<GpuTensor> inputs = {d_input};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs)) {
        eng.gpu_free(d_resized); eng.gpu_free(d_input);
        for (auto& o : outputs) eng.gpu_free(o);
        return false;
    }

    // 4. Output: 'Identity' = [1, 195] = 33 landmarks * 5 values (x,y,z,visibility,presence)
    std::vector<float> h_lmarks(195);
    cudaMemcpyAsync(h_lmarks.data(), outputs[0].data, 195 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. Parse 33 landmarks, 5 values each.
    // The model emits x/y in 256px input space and raw visibility/presence
    // logits (measured ~3-4 on real frames), so normalize here: x/y to
    // [0,1] and sigmoided visibility*presence. Consumers can rely on
    // x,y in [0,1] and visibility as a true probability.
    for (int i = 0; i < 33; i++) {
        float x = h_lmarks[i * 5 + 0], y = h_lmarks[i * 5 + 1];
        if (x > 1.5f) x /= 256.0f;
        if (y > 1.5f) y /= 256.0f;
        landmarks_out[i].x = x;
        landmarks_out[i].y = y;
        landmarks_out[i].z = h_lmarks[i * 5 + 2];
        float vv = h_lmarks[i * 5 + 3], pp = h_lmarks[i * 5 + 4];
        landmarks_out[i].visibility =
            (1.0f / (1.0f + expf(-vv))) * (1.0f / (1.0f + expf(-pp)));
    }

    eng.gpu_free(d_resized); eng.gpu_free(d_input);
    for (auto& o : outputs) eng.gpu_free(o);
    return true;
}

} // namespace ai
} // namespace kagerou
