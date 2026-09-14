// Kagerou SDK — super resolution (bicubic upscale + unsharp mask sharpen).

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <math.h>

namespace kagerou {
namespace filters {

// Bicubic sample from device memory (same as scale.cu)
__device__ float sr_cubic_weight(float t, float a = -0.5f) {
    float t2 = t * t;
    float t3 = t2 * t;
    if (t <= 1.0f)      return (a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f;
    else if (t <= 2.0f) return a * t3 - 5.0f * a * t2 + 8.0f * a * t - 4.0f * a;
    return 0.0f;
}

__device__ float sr_bicubic_sample(const uint8_t* __restrict__ src,
                                   uint32_t sw, uint32_t sh,
                                   float fx, float fy, int c, int ch) {
    int x0 = (int)fx - 1;
    int y0 = (int)fy - 1;
    float result = 0.0f;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            int sx = x0 + i; if (sx < 0) sx = 0; if (sx >= (int)sw) sx = sw - 1;
            int sy = y0 + j; if (sy < 0) sy = 0; if (sy >= (int)sh) sy = sh - 1;
            float wx = sr_cubic_weight(fabsf(fx - (float)sx));
            float wy = sr_cubic_weight(fabsf(fy - (float)sy));
            result += src[(sy * sw + sx) * ch + c] * wx * wy;
        }
    }
    return fminf(fmaxf(result, 0.0f), 255.0f);
}

// 2x upscale: one thread per output pixel, bicubic sample from input
__global__ void super_res_2x_kernel(const uint8_t* __restrict__ src,
                                    uint8_t* __restrict__ dst,
                                    uint32_t sw, uint32_t sh,
                                    uint32_t dw, uint32_t dh,
                                    int channels,
                                    float sharpen_strength) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    // source coordinate
    float fx = (float)dx * (float)sw / (float)dw;
    float fy = (float)dy * (float)sh / (float)dh;

    for (int c = 0; c < channels; ++c) {
        float center = sr_bicubic_sample(src, sw, sh, fx, fy, c, channels);

        // unsharp mask: sharpen = center + strength * (center - blur)
        // approximate blur with 3x3 box average
        float blur_sum = 0.0f;
        float blur_count = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                float sx = fx + (float)kx;
                float sy = fy + (float)ky;
                if (sx >= 0.0f && sx < (float)sw && sy >= 0.0f && sy < (float)sh) {
                    blur_sum += sr_bicubic_sample(src, sw, sh, sx, sy, c, channels);
                    blur_count += 1.0f;
                }
            }
        }
        float blur = blur_sum / blur_count;
        float val = center + sharpen_strength * (center - blur);
        val = fminf(fmaxf(val, 0.0f), 255.0f);
        dst[(dy * dw + dx) * channels + c] = (uint8_t)(val + 0.5f);
    }
}

// ---- Host wrapper ----------------------------------------------------------
void super_res_2x(const uint8_t* d_src, uint8_t* d_dst,
                  uint32_t sw, uint32_t sh,
                  int channels, float sharpen_strength,
                  cudaStream_t s) {
    uint32_t dw = sw * 2;
    uint32_t dh = sh * 2;
    super_res_2x_kernel<<<grid_2d(dw, dh), block_2d(dw, dh), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh, channels, sharpen_strength);
}

} // namespace filters
} // namespace kagerou
