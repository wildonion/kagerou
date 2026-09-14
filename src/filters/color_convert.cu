// Kagerou SDK — NV12 <-> RGB conversion kernels.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>

namespace kagerou {
namespace filters {

// ---- BT.709 coefficients (full-range) --------------------------------------
// Forward: RGB -> YUV (full-range [0,255])
__constant__ float c_bt709_rgb2yuv[3][3] = {
    {  0.2126f,  0.7152f,  0.0722f },   // Y
    { -0.1146f, -0.3854f,  0.5000f },   // U (Cb)
    {  0.5000f, -0.4542f, -0.0458f }    // V (Cr)
};

// Inverse: YUV -> RGB (full-range [0,255])
// R = Y + 1.5748*V
// G = Y - 0.1873*U - 0.4681*V
// B = Y + 1.8556*U
__constant__ float c_bt709_yuv2rgb[3][3] = {
    { 1.0000f,  0.0000f,  1.5748f },
    { 1.0000f, -0.1873f, -0.4681f },
    { 1.0000f,  1.8556f,  0.0000f }
};

// NV12 -> RGB: each thread handles one pixel
__global__ void nv12_to_rgb_kernel(const uint8_t* __restrict__ nv12,
                                   uint8_t* __restrict__ rgb,
                                   uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Y plane
    float y_val = nv12[y * w + x] / 255.0f;

    // UV planes (half resolution, nearest-sample)
    uint32_t ux = x / 2;
    uint32_t uy = y / 2;
    uint32_t uv_offset = w * h;
    float u_val = nv12[uv_offset + uy * w + x - (x & 1)] / 255.0f - 0.5f;
    float v_val = nv12[uv_offset + uy * w + x - (x & 1) + 1] / 255.0f - 0.5f;

    // BT.709 convert
    float r = c_bt709_yuv2rgb[0][0] * y_val + c_bt709_yuv2rgb[0][2] * v_val;
    float g = c_bt709_yuv2rgb[1][0] * y_val + c_bt709_yuv2rgb[1][1] * u_val + c_bt709_yuv2rgb[1][2] * v_val;
    float b = c_bt709_yuv2rgb[2][0] * y_val + c_bt709_yuv2rgb[2][1] * u_val;

    // clamp
    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

    uint32_t idx = (y * w + x) * 3;
    rgb[idx + 0] = (uint8_t)(r * 255.0f);
    rgb[idx + 1] = (uint8_t)(g * 255.0f);
    rgb[idx + 2] = (uint8_t)(b * 255.0f);
}

// RGB -> NV12: one thread per Y sample, writes Y + subsampled UV
__global__ void rgb_to_nv12_kernel(const uint8_t* __restrict__ rgb,
                                   uint8_t* __restrict__ nv12,
                                   uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t idx = (y * w + x) * 3;
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b = rgb[idx + 2] / 255.0f;

    // Y
    float yval = c_bt709_rgb2yuv[0][0] * r + c_bt709_rgb2yuv[0][1] * g + c_bt709_rgb2yuv[0][2] * b;
    nv12[y * w + x] = (uint8_t)(yval * 255.0f);

    // UV (only even pixels write, atomicAdd for subsampling)
    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = c_bt709_rgb2yuv[1][0] * r + c_bt709_rgb2yuv[1][1] * g + c_bt709_rgb2yuv[1][2] * b;
        float v = c_bt709_rgb2yuv[2][0] * r + c_bt709_rgb2yuv[2][1] * g + c_bt709_rgb2yuv[2][2] * b;
        uint32_t uv_offset = w * h;
        nv12[uv_offset + (y / 2) * w + x]     = (uint8_t)((u + 0.5f) * 255.0f);
        nv12[uv_offset + (y / 2) * w + x + 1] = (uint8_t)((v + 0.5f) * 255.0f);
    }
}

// ---- YUV420P (planar) -> RGB -----------------------------------------------
__global__ void yuv420p_to_rgb_kernel(const uint8_t* __restrict__ y_plane,
                                      const uint8_t* __restrict__ u_plane,
                                      const uint8_t* __restrict__ v_plane,
                                      uint8_t* __restrict__ rgb,
                                      uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float y_val = y_plane[y * w + x] / 255.0f;
    uint32_t ux = x / 2;
    uint32_t uy = y / 2;
    uint32_t hw = w / 2;
    float u_val = u_plane[uy * hw + ux] / 255.0f - 0.5f;
    float v_val = v_plane[uy * hw + ux] / 255.0f - 0.5f;

    float r = c_bt709_yuv2rgb[0][0] * y_val + c_bt709_yuv2rgb[0][2] * v_val;
    float g = c_bt709_yuv2rgb[1][0] * y_val + c_bt709_yuv2rgb[1][1] * u_val + c_bt709_yuv2rgb[1][2] * v_val;
    float b = c_bt709_yuv2rgb[2][0] * y_val + c_bt709_yuv2rgb[2][1] * u_val;

    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

    uint32_t o = (y * w + x) * 3;
    rgb[o + 0] = (uint8_t)(r * 255.0f);
    rgb[o + 1] = (uint8_t)(g * 255.0f);
    rgb[o + 2] = (uint8_t)(b * 255.0f);
}

// RGB -> YUV420P
__global__ void rgb_to_yuv420p_kernel(const uint8_t* __restrict__ rgb,
                                      uint8_t* __restrict__ y_plane,
                                      uint8_t* __restrict__ u_plane,
                                      uint8_t* __restrict__ v_plane,
                                      uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t idx = (y * w + x) * 3;
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b = rgb[idx + 2] / 255.0f;

    float yval = c_bt709_rgb2yuv[0][0] * r + c_bt709_rgb2yuv[0][1] * g + c_bt709_rgb2yuv[0][2] * b;
    y_plane[y * w + x] = (uint8_t)(yval * 255.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = c_bt709_rgb2yuv[1][0] * r + c_bt709_rgb2yuv[1][1] * g + c_bt709_rgb2yuv[1][2] * b;
        float v = c_bt709_rgb2yuv[2][0] * r + c_bt709_rgb2yuv[2][1] * g + c_bt709_rgb2yuv[2][2] * b;
        uint32_t hw = w / 2;
        u_plane[(y/2)*hw + x/2] = (uint8_t)((u + 0.5f) * 255.0f);
        v_plane[(y/2)*hw + x/2] = (uint8_t)((v + 0.5f) * 255.0f);
    }
}

// ---- PQ (ST2084) -> SDR tone mapping --------------------------------------
__global__ void pq_to_sdr_kernel(const uint8_t* __restrict__ pq,
                                 uint8_t* __restrict__ rgb,
                                 uint32_t w, uint32_t h,
                                 float inv_peak_nits) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // PQ EOTF: assume 10-bit PQ stored in 3 bytes (simplified)
    // This is a simplified inverse PQ for demonstration
    uint32_t idx = (y * w + x) * 3;
    float r = pq[idx + 0] / 255.0f;
    float g = pq[idx + 1] / 255.0f;
    float b = pq[idx + 2] / 255.0f;

    // Reinhard tone mapping
    r = r * inv_peak_nits;
    g = g * inv_peak_nits;
    b = b * inv_peak_nits;
    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);

    rgb[idx + 0] = (uint8_t)(r * 255.0f);
    rgb[idx + 1] = (uint8_t)(g * 255.0f);
    rgb[idx + 2] = (uint8_t)(b * 255.0f);
}

// ---- NV12 -> RGB (limited-range BT.709) ------------------------------------
// Y: 16-235 -> 0-1, UV: 16-240 -> -0.5 to 0.5
__global__ void nv12_to_rgb_limited_kernel(const uint8_t* __restrict__ nv12,
                                           uint8_t* __restrict__ rgb,
                                           uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float y_val = fmaxf((float)nv12[y * w + x] - 16.0f, 0.0f) / 219.0f;

    uint32_t ux = x / 2;
    uint32_t uy = y / 2;
    uint32_t uv_offset = w * h;
    float u_val = fmaxf((float)nv12[uv_offset + uy * w + x - (x & 1)] - 128.0f, -128.0f) / 224.0f;
    float v_val = fmaxf((float)nv12[uv_offset + uy * w + x - (x & 1) + 1] - 128.0f, -128.0f) / 224.0f;

    float r = y_val + 1.5748f * v_val;
    float g = y_val - 0.1873f * u_val - 0.4681f * v_val;
    float b = y_val + 1.8556f * u_val;

    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

    uint32_t idx = (y * w + x) * 3;
    rgb[idx + 0] = (uint8_t)(r * 255.0f);
    rgb[idx + 1] = (uint8_t)(g * 255.0f);
    rgb[idx + 2] = (uint8_t)(b * 255.0f);
}

// ---- RGB -> NV12 (limited-range BT.709) ------------------------------------
// Y: 0-1 -> 16-235, UV: -0.5 to 0.5 -> 16-240
__global__ void rgb_to_nv12_limited_kernel(const uint8_t* __restrict__ rgb,
                                           uint8_t* __restrict__ nv12,
                                           uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t idx = (y * w + x) * 3;
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b_ch = rgb[idx + 2] / 255.0f;

    float yval = 0.2126f * r + 0.7152f * g + 0.0722f * b_ch;
    nv12[y * w + x] = (uint8_t)(yval * 219.0f + 16.0f + 0.5f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = -0.1146f * r - 0.3854f * g + 0.5000f * b_ch;
        float v =  0.5000f * r - 0.4542f * g - 0.0458f * b_ch;
        uint32_t uv_offset = w * h;
        nv12[uv_offset + (y / 2) * w + x]     = (uint8_t)(u * 224.0f + 128.0f + 0.5f);
        nv12[uv_offset + (y / 2) * w + x + 1] = (uint8_t)(v * 224.0f + 128.0f + 0.5f);
    }
}

// ---- Host wrappers ---------------------------------------------------------
void nv12_to_rgb(const uint8_t* d_nv12, uint8_t* d_rgb,
                 uint32_t w, uint32_t h, cudaStream_t s) {
    nv12_to_rgb_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_nv12, d_rgb, w, h);
}

void nv12_to_rgb_limited(const uint8_t* d_nv12, uint8_t* d_rgb,
                         uint32_t w, uint32_t h, cudaStream_t s) {
    nv12_to_rgb_limited_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_nv12, d_rgb, w, h);
}

void rgb_to_nv12(const uint8_t* d_rgb, uint8_t* d_nv12,
                 uint32_t w, uint32_t h, cudaStream_t s) {
    rgb_to_nv12_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_rgb, d_nv12, w, h);
}

void rgb_to_nv12_limited(const uint8_t* d_rgb, uint8_t* d_nv12,
                         uint32_t w, uint32_t h, cudaStream_t s) {
    rgb_to_nv12_limited_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_rgb, d_nv12, w, h);
}

// ---- RGB -> NV12 (limited-range BT.601) ------------------------------------
// Y: 0-1 -> 16-235, UV: -0.5 to 0.5 -> 16-240
__global__ void rgb_to_nv12_limited601_kernel(const uint8_t* __restrict__ rgb,
                                              uint8_t* __restrict__ nv12,
                                              uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t idx = (y * w + x) * 3;
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b_ch = rgb[idx + 2] / 255.0f;

    float yval = 0.299f * r + 0.587f * g + 0.114f * b_ch;
    nv12[y * w + x] = (uint8_t)(yval * 219.0f + 16.0f + 0.5f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = -0.16874f * r - 0.33126f * g + 0.50000f * b_ch;
        float v =  0.50000f * r - 0.41869f * g - 0.08131f * b_ch;
        uint32_t uv_offset = w * h;
        nv12[uv_offset + (y / 2) * w + x]     = (uint8_t)(u * 224.0f + 128.0f + 0.5f);
        nv12[uv_offset + (y / 2) * w + x + 1] = (uint8_t)(v * 224.0f + 128.0f + 0.5f);
    }
}

void rgb_to_nv12_limited601(const uint8_t* d_rgb, uint8_t* d_nv12,
                            uint32_t w, uint32_t h, cudaStream_t s) {
    rgb_to_nv12_limited601_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_rgb, d_nv12, w, h);
}

void yuv420p_to_rgb(const uint8_t* d_y, const uint8_t* d_u, const uint8_t* d_v,
                    uint8_t* d_rgb, uint32_t w, uint32_t h, cudaStream_t s) {
    yuv420p_to_rgb_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_y, d_u, d_v, d_rgb, w, h);
}

void rgb_to_yuv420p(const uint8_t* d_rgb, uint8_t* d_y, uint8_t* d_u, uint8_t* d_v,
                    uint32_t w, uint32_t h, cudaStream_t s) {
    rgb_to_yuv420p_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_rgb, d_y, d_u, d_v, w, h);
}

void pq_to_sdr(const uint8_t* d_pq, uint8_t* d_rgb,
                 uint32_t w, uint32_t h, float peak_nits, cudaStream_t s) {
    pq_to_sdr_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_pq, d_rgb, w, h, 1.0f / peak_nits);
}

// BGR -> RGB channel swap. Safe in-place (each thread owns its 3 bytes).
// Used for DirectShow/webcam frames, which are natively BGR.
__global__ void bgr_to_rgb_kernel(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
                                  uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t b = src[i * 3 + 0];
    uint8_t g = src[i * 3 + 1];
    uint8_t r = src[i * 3 + 2];
    dst[i * 3 + 0] = r;
    dst[i * 3 + 1] = g;
    dst[i * 3 + 2] = b;
}

void bgr_to_rgb(const uint8_t* d_bgr, uint8_t* d_rgb,
                uint32_t w, uint32_t h, cudaStream_t s) {
    uint32_t n = w * h;
    bgr_to_rgb_kernel<<<(n + 255) / 256, 256, 0, s>>>(d_bgr, d_rgb, n);
}

} // namespace filters
} // namespace kagerou
