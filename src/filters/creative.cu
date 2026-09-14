// Kagerou SDK — Creative / artistic GPU filter kernels.
// Gaussian blur, sharpen, brightness/contrast, saturation, gamma,
// vignette, film grain, directional blur, edge detect, white balance,
// lens distortion, GPU flip, crop.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <math.h>

namespace kagerou {
namespace filters {

// ============================================================================
// 1. Gaussian Blur (variable sigma, separable 2-pass)
// ============================================================================
__device__ float gaussian_weight(int x, float sigma2) {
    return expf(-(float)(x * x) / (2.0f * sigma2));
}

__global__ void gauss_h_kernel(const uint8_t* __restrict__ src,
                               uint8_t* __restrict__ dst,
                               uint32_t w, uint32_t h, int ch,
                               float sigma) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int radius = (int)(sigma * 1.5f);
    if (radius < 1) radius = 1;
    float sigma2 = sigma * sigma;
    float sum_w = 0;

    for (int c = 0; c < ch; c++) {
        float sum = 0;
        sum_w = 0;
        for (int k = -radius; k <= radius; k++) {
            int sx = (int)x + k;
            if (sx < 0) sx = 0;
            if (sx >= (int)w) sx = (int)w - 1;
            float wt = gaussian_weight(k, sigma2);
            sum += wt * src[(y * w + sx) * ch + c];
            sum_w += wt;
        }
        dst[(y * w + x) * ch + c] = (uint8_t)(sum / sum_w + 0.5f);
    }
}

__global__ void gauss_v_kernel(const uint8_t* __restrict__ src,
                               uint8_t* __restrict__ dst,
                               uint32_t w, uint32_t h, int ch,
                               float sigma) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int radius = (int)(sigma * 1.5f);
    if (radius < 1) radius = 1;
    float sigma2 = sigma * sigma;

    for (int c = 0; c < ch; c++) {
        float sum = 0;
        float sum_w = 0;
        for (int k = -radius; k <= radius; k++) {
            int sy = (int)y + k;
            if (sy < 0) sy = 0;
            if (sy >= (int)h) sy = (int)h - 1;
            float wt = gaussian_weight(k, sigma2);
            sum += wt * src[(sy * w + x) * ch + c];
            sum_w += wt;
        }
        dst[(y * w + x) * ch + c] = (uint8_t)(sum / sum_w + 0.5f);
    }
}

void gaussian_blur(const uint8_t* d_src, uint8_t* d_dst,
                   uint32_t w, uint32_t h, int channels,
                   float sigma, cudaStream_t s) {
    if (sigma < 0.5f) sigma = 0.5f;
    uint8_t* d_tmp = nullptr;
    cudaMalloc(&d_tmp, (size_t)w * h * channels);
    dim3 g = grid_2d(w, h);
    dim3 b = block_2d(w, h);
    gauss_h_kernel<<<g, b, 0, s>>>(d_src, d_tmp, w, h, channels, sigma);
    gauss_v_kernel<<<g, b, 0, s>>>(d_tmp, d_dst, w, h, channels, sigma);
    cudaFree(d_tmp);
}

// ============================================================================
// 2. Sharpen (unsharp mask)
// ============================================================================
__global__ void sharpen_kernel(const uint8_t* __restrict__ src,
                               uint8_t* __restrict__ dst,
                               uint32_t w, uint32_t h, int ch,
                               float strength) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    for (int c = 0; c < ch; c++) {
        float center = src[(y * w + x) * ch + c];
        float blur_sum = 0;
        int cnt = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int sx = (int)x + kx, sy = (int)y + ky;
                if (sx >= 0 && sx < (int)w && sy >= 0 && sy < (int)h) {
                    blur_sum += src[(sy * w + sx) * ch + c];
                    cnt++;
                }
            }
        }
        float blurred = blur_sum / cnt;
        float result = center + strength * (center - blurred);
        dst[(y * w + x) * ch + c] = (uint8_t)max(0, min(255, (int)(result + 0.5f)));
    }
}

void sharpen_rgb(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h,
                 float strength, cudaStream_t s) {
    sharpen_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, 3, strength);
}

// ============================================================================
// 3. Brightness / Contrast
// ============================================================================
__global__ void brightness_contrast_kernel(const uint8_t* __restrict__ src,
                                           uint8_t* __restrict__ dst,
                                           uint32_t w, uint32_t h, int ch,
                                           float brightness,
                                           float contrast) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = w * h * ch;
    if (i >= n) return;
    float v = (float)src[i];
    v = (v - 128.0f) * contrast + 128.0f + brightness;
    dst[i] = (uint8_t)max(0, min(255, (int)(v + 0.5f)));
}

void brightness_contrast(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t w, uint32_t h,
                         float brightness, float contrast,
                         cudaStream_t s) {
    uint32_t n = w * h * 3;
    brightness_contrast_kernel<<<(n + 255) / 256, 256, 0, s>>>(
        d_src, d_dst, w, h, 3, brightness, contrast);
}

// ============================================================================
// 4. Saturation
// ============================================================================
__global__ void saturation_kernel(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
                                  uint32_t w, uint32_t h,
                                  float factor) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t idx = (y * w + x) * 3;
    float r = src[idx], g = src[idx+1], b = src[idx+2];
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    r = lum + factor * (r - lum);
    g = lum + factor * (g - lum);
    b = lum + factor * (b - lum);
    dst[idx]   = (uint8_t)max(0, min(255, (int)(r + 0.5f)));
    dst[idx+1] = (uint8_t)max(0, min(255, (int)(g + 0.5f)));
    dst[idx+2] = (uint8_t)max(0, min(255, (int)(b + 0.5f)));
}

void saturation_rgb(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    float factor, cudaStream_t s) {
    saturation_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, factor);
}

// ============================================================================
// 5. Gamma Correction
// ============================================================================
__global__ void gamma_kernel(const uint8_t* __restrict__ src,
                             uint8_t* __restrict__ dst,
                             uint32_t w, uint32_t h, int ch,
                             float gamma) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = w * h * ch;
    if (i >= n) return;
    float inv_gamma = 1.0f / gamma;
    float v = (float)src[i] / 255.0f;
    v = powf(v, inv_gamma);
    dst[i] = (uint8_t)(v * 255.0f + 0.5f);
}

void gamma_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               float gamma, cudaStream_t s) {
    uint32_t n = w * h * 3;
    gamma_kernel<<<(n + 255) / 256, 256, 0, s>>>(
        d_src, d_dst, w, h, 3, gamma);
}

// ============================================================================
// 6. Vignette (lens darkening)
// ============================================================================
__global__ void vignette_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                uint32_t w, uint32_t h,
                                float strength) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float cx = w * 0.5f, cy = h * 0.5f;
    float max_r = sqrtf(cx * cx + cy * cy);
    float dx = (float)x - cx, dy = (float)y - cy;
    float r = sqrtf(dx * dx + dy * dy) / max_r;
    float vignette = 1.0f - strength * r * r;
    vignette = max(0.0f, vignette);

    uint32_t idx = (y * w + x) * 3;
    dst[idx]   = (uint8_t)(src[idx]   * vignette + 0.5f);
    dst[idx+1] = (uint8_t)(src[idx+1] * vignette + 0.5f);
    dst[idx+2] = (uint8_t)(src[idx+2] * vignette + 0.5f);
}

void vignette_rgb(const uint8_t* d_src, uint8_t* d_dst,
                  uint32_t w, uint32_t h,
                  float strength, cudaStream_t s) {
    vignette_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, strength);
}

// ============================================================================
// 7. Film Grain (noise overlay)
// ============================================================================
__global__ void film_grain_kernel(uint8_t* __restrict__ data,
                                  uint32_t w, uint32_t h,
                                  float amount, uint32_t seed) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Simple hash-based PRNG
    uint32_t h1 = x * 374761393u + y * 668265263u + seed;
    h1 = (h1 ^ (h1 >> 13)) * 1274126177u;
    float noise = ((float)(h1 & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;

    uint32_t idx = (y * w + x) * 3;
    float r = data[idx]   + noise * amount;
    float g = data[idx+1] + noise * amount;
    float b = data[idx+2] + noise * amount;
    data[idx]   = (uint8_t)max(0, min(255, (int)(r + 0.5f)));
    data[idx+1] = (uint8_t)max(0, min(255, (int)(g + 0.5f)));
    data[idx+2] = (uint8_t)max(0, min(255, (int)(b + 0.5f)));
}

void film_grain_rgb(uint8_t* d_data,
                    uint32_t w, uint32_t h,
                    float amount, uint32_t seed,
                    cudaStream_t s) {
    film_grain_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_data, w, h, amount, seed);
}

// ============================================================================
// 8. Directional Blur (motion blur)
// ============================================================================
__global__ void directional_blur_kernel(const uint8_t* __restrict__ src,
                                        uint8_t* __restrict__ dst,
                                        uint32_t w, uint32_t h, int ch,
                                        float angle_deg, int length) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float rad = angle_deg * 3.14159265f / 180.0f;
    float dx = cosf(rad), dy = sinf(rad);

    for (int c = 0; c < ch; c++) {
        float sum = 0;
        int cnt = 0;
        for (int k = -length / 2; k <= length / 2; k++) {
            int sx = (int)(x + dx * k + 0.5f);
            int sy = (int)(y + dy * k + 0.5f);
            if (sx >= 0 && sx < (int)w && sy >= 0 && sy < (int)h) {
                sum += src[(sy * w + sx) * ch + c];
                cnt++;
            }
        }
        dst[(y * w + x) * ch + c] = (uint8_t)(sum / cnt + 0.5f);
    }
}

void directional_blur(const uint8_t* d_src, uint8_t* d_dst,
                      uint32_t w, uint32_t h, int channels,
                      float angle_deg, int length,
                      cudaStream_t s) {
    directional_blur_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, channels, angle_deg, length);
}

// ============================================================================
// 9. Edge Detect (Sobel)
// ============================================================================
__global__ void edge_detect_kernel(const uint8_t* __restrict__ src,
                                   uint8_t* __restrict__ dst,
                                   uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Convert to grayscale inline
    auto lum = [&](int px, int py) -> float {
        if (px < 0) px = 0; if (px >= (int)w) px = w - 1;
        if (py < 0) py = 0; if (py >= (int)h) py = h - 1;
        uint32_t i = (py * w + px) * 3;
        return 0.2126f * src[i] + 0.7152f * src[i+1] + 0.0722f * src[i+2];
    };

    // Sobel
    float gx = -lum(x-1,y-1) + lum(x+1,y-1)
               -2*lum(x-1,y)   + 2*lum(x+1,y)
               -lum(x-1,y+1) + lum(x+1,y+1);
    float gy = -lum(x-1,y-1) - 2*lum(x,y-1) - lum(x+1,y-1)
               +lum(x-1,y+1) + 2*lum(x,y+1) + lum(x+1,y+1);
    float mag = sqrtf(gx * gx + gy * gy);
    uint8_t val = (uint8_t)min(255.0f, mag);
    uint32_t idx = (y * w + x) * 3;
    dst[idx] = dst[idx+1] = dst[idx+2] = val;
}

void edge_detect_rgb(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t w, uint32_t h,
                     cudaStream_t s) {
    edge_detect_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h);
}

// ============================================================================
// 10. White Balance (temperature / tint)
// ============================================================================
__global__ void white_balance_kernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     uint32_t w, uint32_t h,
                                     float temperature, float tint) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t idx = (y * w + x) * 3;
    float r = src[idx], g = src[idx+1], b = src[idx+2];
    // temperature: positive=warm (+R -B), negative=cool (-R +B)
    // tint: positive=green, negative=magenta
    r += temperature * 1.5f;
    b -= temperature * 1.5f;
    g += tint * 1.0f;
    dst[idx]   = (uint8_t)max(0, min(255, (int)(r + 0.5f)));
    dst[idx+1] = (uint8_t)max(0, min(255, (int)(g + 0.5f)));
    dst[idx+2] = (uint8_t)max(0, min(255, (int)(b + 0.5f)));
}

void white_balance_rgb(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h,
                       float temperature, float tint,
                       cudaStream_t s) {
    white_balance_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, temperature, tint);
}

// ============================================================================
// 11. Lens Distortion (barrel / pincushion)
// ============================================================================
__global__ void lens_distortion_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       uint32_t w, uint32_t h, int ch,
                                       float k1) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float cx = w * 0.5f, cy = h * 0.5f;
    float norm_x = ((float)x - cx) / cx;
    float norm_y = ((float)y - cy) / cy;
    float r2 = norm_x * norm_x + norm_y * norm_y;
    float distortion = 1.0f + k1 * r2;
    float src_x = cx + norm_x * distortion * cx;
    float src_y = cy + norm_y * distortion * cy;

    int sx = (int)(src_x + 0.5f);
    int sy = (int)(src_y + 0.5f);
    uint32_t idx = (y * w + x) * ch;
    if (sx >= 0 && sx < (int)w && sy >= 0 && sy < (int)h) {
        uint32_t sidx = (sy * w + sx) * ch;
        for (int c = 0; c < ch; c++)
            dst[idx + c] = src[sidx + c];
    } else {
        for (int c = 0; c < ch; c++)
            dst[idx + c] = 0;
    }
}

void lens_distortion(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t w, uint32_t h, int channels,
                     float k1, cudaStream_t s) {
    lens_distortion_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(
        d_src, d_dst, w, h, channels, k1);
}

// ============================================================================
// 12. Flip (horizontal / vertical on GPU)
// ============================================================================
__global__ void flip_h_kernel(const uint8_t* __restrict__ src,
                              uint8_t* __restrict__ dst,
                              uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t sx = w - 1 - x;
    uint32_t didx = (y * w + x) * 3;
    uint32_t sidx = (y * w + sx) * 3;
    dst[didx] = src[sidx];
    dst[didx+1] = src[sidx+1];
    dst[didx+2] = src[sidx+2];
}

__global__ void flip_v_kernel(const uint8_t* __restrict__ src,
                              uint8_t* __restrict__ dst,
                              uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t sy = h - 1 - y;
    uint32_t didx = (y * w + x) * 3;
    uint32_t sidx = (sy * w + x) * 3;
    dst[didx] = src[sidx];
    dst[didx+1] = src[sidx+1];
    dst[didx+2] = src[sidx+2];
}

void flip_horizontal_gpu(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t w, uint32_t h,
                         cudaStream_t s) {
    flip_h_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_src, d_dst, w, h);
}

void flip_vertical_gpu(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h,
                       cudaStream_t s) {
    flip_v_kernel<<<grid_2d(w, h), block_2d(w, h), 0, s>>>(d_src, d_dst, w, h);
}

} // namespace filters
} // namespace kagerou
