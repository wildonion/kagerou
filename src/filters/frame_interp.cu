// Kagerou SDK — frame interpolation (alpha blending).

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>

namespace kagerou {
namespace filters {

// Blend two frames: out = (1-alpha)*a + alpha*b
__global__ void frame_blend_kernel(const uint8_t* __restrict__ frame_a,
                                   const uint8_t* __restrict__ frame_b,
                                   uint8_t* __restrict__ out,
                                   uint32_t count, float alpha) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float va = (1.0f - alpha) * (float)frame_a[i];
    float vb = alpha * (float)frame_b[i];
    out[i] = (uint8_t)(va + vb + 0.5f);
}

// ---- Host wrapper ----------------------------------------------------------
void frame_blend(const uint8_t* d_frame_a, const uint8_t* d_frame_b,
                 uint8_t* d_out, uint32_t w, uint32_t h,
                 int channels, float alpha, cudaStream_t s) {
    uint32_t count = w * h * channels;
    uint32_t threads = 256;
    uint32_t blocks = (count + threads - 1) / threads;
    frame_blend_kernel<<<blocks, threads, 0, s>>>(d_frame_a, d_frame_b, d_out, count, alpha);
}

} // namespace filters
} // namespace kagerou
