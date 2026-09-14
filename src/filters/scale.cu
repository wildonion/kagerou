// Kagerou SDK — resize kernels (bilinear + bicubic).

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>

namespace kagerou {
namespace filters {

// ---- Bilinear resize -------------------------------------------------------
__global__ void resize_bilinear_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       uint32_t sw, uint32_t sh,
                                       uint32_t dw, uint32_t dh,
                                       int channels) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    float fx = (float)dx * (float)sw / (float)dw;
    float fy = (float)dy * (float)sh / (float)dh;

    uint32_t x0 = (uint32_t)fx;
    uint32_t y0 = (uint32_t)fy;
    uint32_t x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
    uint32_t y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * sw + x0) * channels + c];
        float v10 = src[(y0 * sw + x1) * channels + c];
        float v01 = src[(y1 * sw + x0) * channels + c];
        float v11 = src[(y1 * sw + x1) * channels + c];

        float val = v00 * (1-wx) * (1-wy)
                  + v10 * wx * (1-wy)
                  + v01 * (1-wx) * wy
                  + v11 * wx * wy;

        dst[(dy * dw + dx) * channels + c] = (uint8_t)(val + 0.5f);
    }
}

// ---- Bicubic interpolation helpers ----------------------------------------
__device__ float cubic_weight(float t, float a = -0.5f) {
    float t2 = t * t;
    float t3 = t2 * t;
    float w;
    if (t <= 1.0f)
        w = (a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f;
    else if (t <= 2.0f)
        w = a * t3 - 5.0f * a * t2 + 8.0f * a * t - 4.0f * a;
    else
        w = 0.0f;
    return w;
}

__device__ float bicubic_sample(const uint8_t* __restrict__ src,
                                uint32_t sw, uint32_t sh,
                                float fx, float fy, int c, int channels) {
    int x0 = (int)fx - 1;
    int y0 = (int)fy - 1;

    float result = 0.0f;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            int sx = x0 + i;
            int sy = y0 + j;
            if (sx < 0) sx = 0;
            if (sx >= (int)sw) sx = sw - 1;
            if (sy < 0) sy = 0;
            if (sy >= (int)sh) sy = sh - 1;

            float wx = cubic_weight(fabsf(fx - (float)sx));
            float wy = cubic_weight(fabsf(fy - (float)sy));
            result += src[(sy * sw + sx) * channels + c] * wx * wy;
        }
    }
    return fminf(fmaxf(result, 0.0f), 255.0f);
}

__global__ void resize_bicubic_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      uint32_t sw, uint32_t sh,
                                      uint32_t dw, uint32_t dh,
                                      int channels) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    float fx = (float)dx * (float)sw / (float)dw;
    float fy = (float)dy * (float)sh / (float)dh;

    for (int c = 0; c < channels; ++c) {
        float val = bicubic_sample(src, sw, sh, fx, fy, c, channels);
        dst[(dy * dw + dx) * channels + c] = (uint8_t)(val + 0.5f);
    }
}

// ---- Host wrappers ---------------------------------------------------------
void resize_bilinear(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t sw, uint32_t sh,
                     uint32_t dw, uint32_t dh,
                     int channels, cudaStream_t s) {
    resize_bilinear_kernel<<<grid_2d(dw, dh), block_2d(dw, dh), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh, channels);
}

void resize_bicubic(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t sw, uint32_t sh,
                    uint32_t dw, uint32_t dh,
                    int channels, cudaStream_t s) {
    resize_bicubic_kernel<<<grid_2d(dw, dh), block_2d(dw, dh), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh, channels);
}

// ---- NV12 bilinear resize (Y + UV separately) ------------------------------
// Y plane: full-res bilinear
// UV plane: half-res bilinear on interleaved U,V pairs

__global__ void resize_nv12_y_bilinear_kernel(const uint8_t* __restrict__ src,
                                               uint8_t* __restrict__ dst,
                                               uint32_t sw, uint32_t sh,
                                               uint32_t dw, uint32_t dh) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    float fx = (float)dx * (float)sw / (float)dw;
    float fy = (float)dy * (float)sh / (float)dh;

    uint32_t x0 = (uint32_t)fx;
    uint32_t y0 = (uint32_t)fy;
    uint32_t x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
    uint32_t y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float v00 = src[y0 * sw + x0];
    float v10 = src[y0 * sw + x1];
    float v01 = src[y1 * sw + x0];
    float v11 = src[y1 * sw + x1];

    float val = v00 * (1-wx) * (1-wy)
              + v10 * wx * (1-wy)
              + v01 * (1-wx) * wy
              + v11 * wx * wy;

    dst[dy * dw + dx] = (uint8_t)(val + 0.5f);
}

__global__ void resize_nv12_uv_bilinear_kernel(const uint8_t* __restrict__ src,
                                                uint8_t* __restrict__ dst,
                                                uint32_t sw, uint32_t sh,
                                                uint32_t dw, uint32_t dh) {
    // UV is interleaved U,V at half resolution in both dimensions
    // src: sw * (sh/2) bytes, dst: dw * (dh/2) bytes
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t uv_dw = dw;
    uint32_t uv_dh = dh / 2;
    uint32_t uv_sw = sw;
    uint32_t uv_sh = sh / 2;
    if (dx >= uv_dw / 2 || dy >= uv_dh) return;

    uint32_t px = dx * 2;  // source pixel x (even)
    uint32_t py = dy;

    float fx = (float)px * (float)uv_sw / (float)uv_dw;
    float fy = (float)py * (float)uv_sh / (float)uv_dh;

    uint32_t x0 = (uint32_t)fx;
    uint32_t y0 = (uint32_t)fy;
    uint32_t x0e = x0 & ~1u;  // align to even
    uint32_t x1e = ((x0e + 2) < uv_sw) ? x0e + 2 : x0e;
    uint32_t y1 = (y0 + 1 < uv_sh) ? y0 + 1 : y0;
    float wx = fx - (float)x0e;
    float wy = fy - (float)y0;

    uint32_t src_offset = uv_sw * sh;  // offset to UV plane in NV12
    for (int ch = 0; ch < 2; ++ch) {
        float v00 = src[src_offset + y0 * uv_sw + x0e + ch];
        float v10 = src[src_offset + y0 * uv_sw + x1e + ch];
        float v01 = src[src_offset + y1 * uv_sw + x0e + ch];
        float v11 = src[src_offset + y1 * uv_sw + x1e + ch];

        float val = v00 * (1-wx) * (1-wy)
                  + v10 * wx * (1-wy)
                  + v01 * (1-wx) * wy
                  + v11 * wx * wy;

        uint32_t dst_offset = dw * dh;  // offset to UV plane in dst
        dst[dst_offset + dy * uv_dw + px + ch] = (uint8_t)(val + 0.5f);
    }
}

void resize_nv12_bilinear(const uint8_t* d_src, uint8_t* d_dst,
                          uint32_t sw, uint32_t sh,
                          uint32_t dw, uint32_t dh,
                          cudaStream_t s) {
    // Y plane
    resize_nv12_y_bilinear_kernel<<<grid_2d(dw, dh), block_2d(dw, dh), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh);
    // UV plane
    resize_nv12_uv_bilinear_kernel<<<grid_2d(dw / 2, dh / 2), block_2d(dw / 2, dh / 2), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh);
}

// ---- NV12 bicubic resize (Y + UV separately) -------------------------------
__global__ void resize_nv12_y_bicubic_kernel(const uint8_t* __restrict__ src,
                                              uint8_t* __restrict__ dst,
                                              uint32_t sw, uint32_t sh,
                                              uint32_t dw, uint32_t dh) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    float fx = (float)dx * (float)sw / (float)dw;
    float fy = (float)dy * (float)sh / (float)dh;

    // Bicubic sample Y plane (single channel)
    int x0 = (int)fx - 1;
    int y0 = (int)fy - 1;
    float result = 0.0f;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            int sx = x0 + i;
            int sy = y0 + j;
            if (sx < 0) sx = 0;
            if (sx >= (int)sw) sx = sw - 1;
            if (sy < 0) sy = 0;
            if (sy >= (int)sh) sy = sh - 1;
            float wx = cubic_weight(fabsf(fx - (float)sx));
            float wy = cubic_weight(fabsf(fy - (float)sy));
            result += src[sy * sw + sx] * wx * wy;
        }
    }
    dst[dy * dw + dx] = (uint8_t)(fminf(fmaxf(result, 0.0f), 255.0f) + 0.5f);
}

__global__ void resize_nv12_uv_bicubic_kernel(const uint8_t* __restrict__ src,
                                               uint8_t* __restrict__ dst,
                                               uint32_t sw, uint32_t sh,
                                               uint32_t dw, uint32_t dh) {
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t uv_dw = dw;
    uint32_t uv_dh = dh / 2;
    uint32_t uv_sw = sw;
    uint32_t uv_sh = sh / 2;
    if (dx >= uv_dw / 2 || dy >= uv_dh) return;

    uint32_t px = dx * 2;
    uint32_t py = dy;
    float fx = (float)px * (float)uv_sw / (float)uv_dw;
    float fy = (float)py * (float)uv_sh / (float)uv_dh;

    uint32_t src_offset = uv_sw * sh;
    uint32_t dst_offset = dw * dh;

    for (int ch = 0; ch < 2; ++ch) {
        int x0 = (int)fx - 1;
        int y0 = (int)fy - 1;
        float result = 0.0f;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                int sx = x0 + i;
                int sy = y0 + j;
                if (sx < 0) sx = 0;
                if (sx >= (int)uv_sw) sx = uv_sw - 1;
                if (sy < 0) sy = 0;
                if (sy >= (int)uv_sh) sy = uv_sh - 1;
                float wx = cubic_weight(fabsf(fx - (float)sx));
                float wy = cubic_weight(fabsf(fy - (float)sy));
                result += src[src_offset + sy * uv_sw + sx * 2 + ch] * wx * wy;
            }
        }
        dst[dst_offset + dy * uv_dw + px + ch] = (uint8_t)(fminf(fmaxf(result, 0.0f), 255.0f) + 0.5f);
    }
}

void resize_nv12_bicubic(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t sw, uint32_t sh,
                         uint32_t dw, uint32_t dh,
                         cudaStream_t s) {
    resize_nv12_y_bicubic_kernel<<<grid_2d(dw, dh), block_2d(dw, dh), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh);
    resize_nv12_uv_bicubic_kernel<<<grid_2d(dw / 2, dh / 2), block_2d(dw / 2, dh / 2), 0, s>>>(
        d_src, d_dst, sw, sh, dw, dh);
}

} // namespace filters
} // namespace kagerou
