// Kagerou SDK — bilateral denoise filter kernel.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <math.h>

namespace kagerou {
namespace filters {

// Bilateral filter: preserves edges while smoothing noise
// Each thread handles one output pixel.
__global__ void denoise_bilateral_kernel(const uint8_t* __restrict__ src,
                                         uint8_t* __restrict__ dst,
                                         uint32_t w, uint32_t h,
                                         int channels,
                                         float sigma_spatial,
                                         float sigma_color,
                                         int kernel_size) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int half_k = kernel_size / 2;
    float inv_spatial2 = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    float inv_color2   = 1.0f / (2.0f * sigma_color * sigma_color);

    for (int c = 0; c < channels; ++c) {
        float sum_w = 0.0f;
        float sum_v = 0.0f;

        for (int ky = -half_k; ky <= half_k; ++ky) {
            for (int kx = -half_k; kx <= half_k; ++kx) {
                int sx = (int)x + kx;
                int sy = (int)y + ky;
                if (sx < 0 || sx >= (int)w || sy < 0 || sy >= (int)h) continue;

                float spatial_dist = (float)(kx * kx + ky * ky);
                float color_dist = (float)(src[(sy * w + sx) * channels + c])
                                 - (float)(src[(y * w + x) * channels + c]);

                float w_spatial = expf(-spatial_dist * inv_spatial2);
                float w_color   = expf(-color_dist * color_dist * inv_color2);
                float wt = w_spatial * w_color;

                sum_w += wt;
                sum_v += wt * src[(sy * w + sx) * channels + c];
            }
        }

        dst[(y * w + x) * channels + c] = (uint8_t)(sum_v / sum_w + 0.5f);
    }
}

// ---- Host wrapper ----------------------------------------------------------
void denoise_bilateral(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h, int channels,
                       float sigma_spatial, float sigma_color,
                       int kernel_size, cudaStream_t s) {
    // kernel_size must be odd
    if (kernel_size < 3) kernel_size = 3;
    if ((kernel_size & 1) == 0) kernel_size += 1;

    denoise_bilateral_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, channels, sigma_spatial, sigma_color, kernel_size);
}

// ---- NV12 bilateral denoise (Y filtered, UV light blur) --------------------
// NV12 layout: [Y: w*h bytes] [UV: w*(h/2) bytes interleaved]
// Y plane: full bilateral filter (single channel, grayscale)
// UV plane: simple 3x3 box blur (color channels, less aggressive)

__global__ void denoise_nv12_y_bilateral_kernel(const uint8_t* __restrict__ src,
                                                 uint8_t* __restrict__ dst,
                                                 uint32_t w, uint32_t h,
                                                 float sigma_spatial,
                                                 float sigma_color,
                                                 int kernel_size) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int half_k = kernel_size / 2;
    float inv_spatial2 = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    float inv_color2   = 1.0f / (2.0f * sigma_color * sigma_color);

    float center_val = (float)src[y * w + x];
    float sum_w = 0.0f;
    float sum_v = 0.0f;

    for (int ky = -half_k; ky <= half_k; ++ky) {
        for (int kx = -half_k; kx <= half_k; ++kx) {
            int sx = (int)x + kx;
            int sy = (int)y + ky;
            if (sx < 0 || sx >= (int)w || sy < 0 || sy >= (int)h) continue;

            float spatial_dist = (float)(kx * kx + ky * ky);
            float color_dist = (float)src[sy * w + sx] - center_val;

            float w_spatial = expf(-spatial_dist * inv_spatial2);
            float w_color   = expf(-color_dist * color_dist * inv_color2);
            float wt = w_spatial * w_color;

            sum_w += wt;
            sum_v += wt * (float)src[sy * w + sx];
        }
    }

    dst[y * w + x] = (uint8_t)(sum_v / sum_w + 0.5f);
}

__global__ void denoise_nv12_uv_blur_kernel(const uint8_t* __restrict__ src,
                                             uint8_t* __restrict__ dst,
                                             uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t uv_w = w;
    uint32_t uv_h = h / 2;
    if (x >= uv_w || y >= uv_h) return;

    uint32_t uv_offset = w * h;
    // 3x3 box blur on UV interleaved (reads 3 U,V pairs = 6 bytes)
    float sum_u = 0, sum_v = 0;
    int count = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int sx = (int)x + kx;
            int sy = (int)y + ky;
            if (sx < 0 || sx >= (int)uv_w || sy < 0 || sy >= (int)uv_h) continue;
            uint32_t off = uv_offset + sy * uv_w + (sx & ~1);
            sum_u += src[off];
            sum_v += src[off + 1];
            count++;
        }
    }
    uint32_t dst_off = uv_offset + y * uv_w + (x & ~1);
    if ((x & 1) == 0) {
        dst[dst_off]     = (uint8_t)(sum_u / count + 0.5f);
        dst[dst_off + 1] = (uint8_t)(sum_v / count + 0.5f);
    }
}

void denoise_nv12_bilateral(const uint8_t* d_src, uint8_t* d_dst,
                            uint32_t w, uint32_t h,
                            float sigma_spatial, float sigma_color,
                            int kernel_size, cudaStream_t s) {
    if (kernel_size < 3) kernel_size = 3;
    if ((kernel_size & 1) == 0) kernel_size += 1;

    // Y plane: bilateral filter
    denoise_nv12_y_bilateral_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, sigma_spatial, sigma_color, kernel_size);

    // UV plane: light 3x3 box blur
    denoise_nv12_uv_blur_kernel<<<grid_2d(w, h / 2), block_2d(w, h / 2), 0, s>>>(
        d_src, d_dst, w, h);
}

} // namespace filters
} // namespace kagerou
