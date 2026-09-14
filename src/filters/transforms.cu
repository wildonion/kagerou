// Kagerou SDK — Geometry transforms & effects GPU kernels.
// Crop, pad, chroma key, background blur, temporal denoise, HDR tone map.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <math.h>

namespace kagerou {
namespace filters {

__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ============================================================================
// 1. Crop (extract region from RGB/NV12)
// ============================================================================
__global__ void crop_rgb_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                uint32_t src_w, uint32_t src_h,
                                uint32_t dst_w, uint32_t dst_h,
                                int crop_x, int crop_y, int channels) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int sx = (int)x + crop_x;
    int sy = (int)y + crop_y;
    if (sx < 0 || sx >= (int)src_w || sy < 0 || sy >= (int)src_h) {
        for (int c = 0; c < channels; c++)
            dst[(y * dst_w + x) * channels + c] = 0;
        return;
    }
    for (int c = 0; c < channels; c++)
        dst[(y * dst_w + x) * channels + c] = src[(sy * src_w + sx) * channels + c];
}

void crop_rgb(const uint8_t* d_src, uint8_t* d_dst,
              uint32_t src_w, uint32_t src_h,
              uint32_t dst_w, uint32_t dst_h,
              int crop_x, int crop_y, int channels,
              cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    crop_rgb_kernel<<<grid, block, 0, s>>>(d_src, d_dst, src_w, src_h,
                                           dst_w, dst_h, crop_x, crop_y, channels);
}

// NV12 crop: Y plane + UV plane
__global__ void crop_nv12_y_kernel(const uint8_t* __restrict__ src,
                                   uint8_t* __restrict__ dst,
                                   uint32_t src_w, uint32_t src_h,
                                   uint32_t dst_w, uint32_t dst_h,
                                   int crop_x, int crop_y) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    int sx = (int)x + crop_x;
    int sy = (int)y + crop_y;
    if (sx < 0 || sx >= (int)src_w || sy < 0 || sy >= (int)src_h) { dst[y * dst_w + x] = 16; return; }
    dst[y * dst_w + x] = src[sy * src_w + sx];
}

__global__ void crop_nv12_uv_kernel(const uint8_t* __restrict__ src,
                                    uint8_t* __restrict__ dst,
                                    uint32_t src_w, uint32_t src_h,
                                    uint32_t dst_w, uint32_t dst_h,
                                    int crop_x, int crop_y) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= (dst_h / 2)) return;
    int sx = (int)x + crop_x;
    int sy = (int)y + crop_y / 2;
    if (sx < 0 || sx >= (int)src_w || sy < 0 || sy >= (int)(src_h / 2)) {
        dst[y * dst_w + x * 2] = 128;
        dst[y * dst_w + x * 2 + 1] = 128;
        return;
    }
    dst[y * dst_w + x * 2]     = src[sy * src_w + sx * 2];
    dst[y * dst_w + x * 2 + 1] = src[sy * src_w + sx * 2 + 1];
}

void crop_nv12(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t src_w, uint32_t src_h,
               uint32_t dst_w, uint32_t dst_h,
               int crop_x, int crop_y,
               cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid_y((dst_w + 15) / 16, (dst_h + 15) / 16);
    crop_nv12_y_kernel<<<grid_y, block, 0, s>>>(d_src, d_dst, src_w, src_h, dst_w, dst_h, crop_x, crop_y);
    dim3 grid_uv((dst_w + 15) / 16, (dst_h / 2 + 15) / 16);
    crop_nv12_uv_kernel<<<grid_uv, block, 0, s>>>(d_src + src_w * src_h, d_dst + dst_w * dst_h,
                                                   src_w, src_h, dst_w, dst_h, crop_x, crop_y);
}

// ============================================================================
// 2. Pad (add border around RGB image)
// ============================================================================
__global__ void pad_rgb_kernel(const uint8_t* __restrict__ src,
                               uint8_t* __restrict__ dst,
                               uint32_t src_w, uint32_t src_h,
                               uint32_t dst_w, uint32_t dst_h,
                               int pad_x, int pad_y,
                               uint8_t pad_r, uint8_t pad_g, uint8_t pad_b) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int sx = (int)x - pad_x;
    int sy = (int)y - pad_y;
    if (sx < 0 || sx >= (int)src_w || sy < 0 || sy >= (int)src_h) {
        dst[(y * dst_w + x) * 3]     = pad_r;
        dst[(y * dst_w + x) * 3 + 1] = pad_g;
        dst[(y * dst_w + x) * 3 + 2] = pad_b;
        return;
    }
    dst[(y * dst_w + x) * 3]     = src[(sy * src_w + sx) * 3];
    dst[(y * dst_w + x) * 3 + 1] = src[(sy * src_w + sx) * 3 + 1];
    dst[(y * dst_w + x) * 3 + 2] = src[(sy * src_w + sx) * 3 + 2];
}

void pad_rgb(const uint8_t* d_src, uint8_t* d_dst,
             uint32_t src_w, uint32_t src_h,
             uint32_t dst_w, uint32_t dst_h,
             int pad_x, int pad_y,
             uint8_t pad_r, uint8_t pad_g, uint8_t pad_b,
             cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    pad_rgb_kernel<<<grid, block, 0, s>>>(d_src, d_dst, src_w, src_h,
                                          dst_w, dst_h, pad_x, pad_y,
                                          pad_r, pad_g, pad_b);
}

// ============================================================================
// 3. Chroma Key (green screen removal)
// ============================================================================
__global__ void chroma_key_rgb_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      uint32_t w, uint32_t h,
                                      float key_h_min, float key_h_max,
                                      float key_s_min, float key_v_min,
                                      float spill_suppress,
                                      uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
                                      float blend_edge) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    float r = src[idx]     / 255.0f;
    float g = src[idx + 1] / 255.0f;
    float b = src[idx + 2] / 255.0f;

    // RGB to HSV
    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float diff = cmax - cmin;
    float hue = 0, s = 0, v = cmax;

    if (cmax > 0.001f) s = diff / cmax;
    if (diff > 0.001f) {
        if (cmax == r)      hue = 60.0f * fmodf((g - b) / diff, 6.0f);
        else if (cmax == g) hue = 60.0f * ((b - r) / diff + 2.0f);
        else                hue = 60.0f * ((r - g) / diff + 4.0f);
        if (hue < 0) hue += 360.0f;
    }

    // Key detection with soft edge
    float key_mask = 0;
    if (hue >= key_h_min && hue <= key_h_max && s >= key_s_min && v >= key_v_min) {
        // Inside key range — compute soft edge
        float edge_low = 0, edge_high = 1.0f;

        // Distance from hue edges
        float hue_center = (key_h_min + key_h_max) * 0.5f;
        float hue_range = (key_h_max - key_h_min) * 0.5f;
        float hue_dist = fabsf(hue - hue_center);
        if (hue_dist > hue_range) hue_dist = hue_range;
        float hue_factor = 1.0f - hue_dist / hue_range;

        float sat_factor = fminf((s - key_s_min) / (1.0f - key_s_min + 0.001f), 1.0f);
        float val_factor = fminf((v - key_v_min) / (1.0f - key_v_min + 0.001f), 1.0f);

        key_mask = hue_factor * sat_factor * val_factor;
        key_mask = fminf(fmaxf(key_mask, 0.0f), 1.0f);
    }

    // Apply blend
    float alpha = 1.0f - key_mask;
    if (blend_edge > 0.001f && key_mask > 0.001f && key_mask < 0.999f) {
        // Smooth the edge
        alpha = smoothstep(0.1f, 0.9f, alpha);
    }

    // Spill suppression: reduce green channel on keyed pixels
    if (spill_suppress > 0.001f && key_mask > 0.01f) {
        float spill = fmaxf(g - fmaxf(r, b), 0.0f) * spill_suppress * key_mask;
        g -= spill;
        if (g < 0) g = 0;
    }

    dst[idx]     = (uint8_t)(alpha * r * 255.0f + (1.0f - alpha) * bg_r);
    dst[idx + 1] = (uint8_t)(alpha * g * 255.0f + (1.0f - alpha) * bg_g);
    dst[idx + 2] = (uint8_t)(alpha * b * 255.0f + (1.0f - alpha) * bg_b);
}

void chroma_key_rgb(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    float key_h_min, float key_h_max,
                    float key_s_min, float key_v_min,
                    float spill_suppress,
                    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
                    float blend_edge,
                    cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    chroma_key_rgb_kernel<<<grid, block, 0, s>>>(d_src, d_dst, w, h,
                                                  key_h_min, key_h_max,
                                                  key_s_min, key_v_min,
                                                  spill_suppress,
                                                  bg_r, bg_g, bg_b,
                                                  blend_edge);
}

// ============================================================================
// 4. Background Blur (center-weighted depth heuristic)
// ============================================================================
__global__ void bg_blur_rgb_kernel(const uint8_t* __restrict__ src,
                                   uint8_t* __restrict__ dst,
                                   uint32_t w, uint32_t h,
                                   float center_x, float center_y,
                                   float focus_radius,
                                   float blur_strength) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Distance from center (normalized 0..1)
    float dx = ((float)x - center_x) / (w * 0.5f);
    float dy = ((float)y - center_y) / (h * 0.5f);
    float dist = sqrtf(dx * dx + dy * dy);

    // Focus mask: 1.0 at center, 0.0 at edges
    float focus = 1.0f - fminf(fmaxf((dist - focus_radius) / (1.0f - focus_radius + 0.001f), 0.0f), 1.0f);
    focus = focus * focus; // smooth falloff

    // Adaptive blur radius based on distance from focus
    float blur_radius = blur_strength * (1.0f - focus);
    int radius = (int)(blur_radius * 3.0f);
    if (radius < 1) radius = 1;
    if (radius > 15) radius = 15;

    float sigma2 = blur_radius * blur_radius * 2.0f;
    if (sigma2 < 0.1f) sigma2 = 0.1f;

    // Box blur for speed (separable not needed for small kernels)
    float sum_r = 0, sum_g = 0, sum_b = 0;
    float sum_w = 0;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int sx = min(max((int)x + kx, 0), (int)w - 1);
            int sy = min(max((int)y + ky, 0), (int)h - 1);
            float d2 = (float)(kx * kx + ky * ky);
            float wt = expf(-d2 / sigma2);
            int si = (sy * w + sx) * 3;
            sum_r += wt * src[si];
            sum_g += wt * src[si + 1];
            sum_b += wt * src[si + 2];
            sum_w += wt;
        }
    }
    sum_r /= sum_w; sum_g /= sum_w; sum_b /= sum_w;

    int di = (y * w + x) * 3;
    dst[di]     = (uint8_t)(focus * src[di]     + (1.0f - focus) * sum_r);
    dst[di + 1] = (uint8_t)(focus * src[di + 1] + (1.0f - focus) * sum_g);
    dst[di + 2] = (uint8_t)(focus * src[di + 2] + (1.0f - focus) * sum_b);
}

void bg_blur_rgb(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h,
                 float center_x, float center_y,
                 float focus_radius,
                 float blur_strength,
                 cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    bg_blur_rgb_kernel<<<grid, block, 0, s>>>(d_src, d_dst, w, h,
                                               center_x, center_y,
                                               focus_radius, blur_strength);
}

// ============================================================================
// 5. Temporal Denoise (inter-frame averaging)
// ============================================================================
__global__ void temporal_avg_kernel(const uint8_t* __restrict__ prev,
                                    const uint8_t* __restrict__ curr,
                                    uint8_t* __restrict__ dst,
                                    uint32_t n,
                                    float prev_weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = prev_weight * prev[i] + (1.0f - prev_weight) * curr[i];
    dst[i] = (uint8_t)(v + 0.5f);
}

void temporal_denoise_rgb(const uint8_t* d_prev, const uint8_t* d_curr,
                          uint8_t* d_dst, uint32_t w, uint32_t h,
                          float strength,
                          cudaStream_t s) {
    uint32_t n = w * h * 3;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    temporal_avg_kernel<<<grid, block, 0, s>>>(d_prev, d_curr, d_dst, n, strength);
}

void temporal_denoise_nv12(const uint8_t* d_prev, const uint8_t* d_curr,
                           uint8_t* d_dst, uint32_t w, uint32_t h,
                           float strength,
                           cudaStream_t s) {
    uint32_t y_size = w * h;
    uint32_t uv_size = w * (h / 2);
    dim3 block(256);
    dim3 grid_y((y_size + 255) / 256);
    temporal_avg_kernel<<<grid_y, block, 0, s>>>(d_prev, d_curr, d_dst, y_size, strength);
    dim3 grid_uv((uv_size + 255) / 256);
    temporal_avg_kernel<<<grid_uv, block, 0, s>>>(d_prev + y_size, d_curr + y_size,
                                                   d_dst + y_size, uv_size, strength);
}

// ============================================================================
// 6. HDR Tone Map (Reinhard + ACES)
// ============================================================================
__global__ void hdr_reinhard_kernel(const uint8_t* __restrict__ src,
                                    uint8_t* __restrict__ dst,
                                    uint32_t w, uint32_t h,
                                    float peak_nits) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    float scale = 100.0f / peak_nits;  // normalize to SDR 100 nits
    float r = src[idx]     / 255.0f * scale;
    float g = src[idx + 1] / 255.0f * scale;
    float b = src[idx + 2] / 255.0f * scale;

    // Reinhard: L = L / (1 + L)
    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);

    dst[idx]     = (uint8_t)(fminf(r * 255.0f, 255.0f));
    dst[idx + 1] = (uint8_t)(fminf(g * 255.0f, 255.0f));
    dst[idx + 2] = (uint8_t)(fminf(b * 255.0f, 255.0f));
}

__device__ float3 aces_input(float3 x) {
    return make_float3(
        x.x * 0.59719f + x.y * 0.35458f + x.z * 0.04823f,
        x.x * 0.07600f + x.y * 0.90834f + x.z * 0.01566f,
        x.x * 0.02840f + x.y * 0.13383f + x.z * 0.83777f
    );
}

__device__ float3 aces_output(float3 x) {
    float a = x.x * (x.x + 0.0245786f) - 0.000090537f;
    float b = x.x * (0.983729f * x.x + 0.4329510f) + 0.238081f;
    float c = x.y * (x.y + 0.0245786f) - 0.000090537f;
    float d = x.y * (0.983729f * x.y + 0.4329510f) + 0.238081f;
    float e = x.z * (x.z + 0.0245786f) - 0.000090537f;
    float f = x.z * (0.983729f * x.z + 0.4329510f) + 0.238081f;
    return make_float3(a / b, c / d, e / f);
}

__global__ void hdr_aces_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                uint32_t w, uint32_t h,
                                float peak_nits) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    float scale = 100.0f / peak_nits;
    float3 in = make_float3(src[idx] / 255.0f * scale,
                             src[idx + 1] / 255.0f * scale,
                             src[idx + 2] / 255.0f * scale);

    float3 mapped = aces_output(aces_input(in));

    dst[idx]     = (uint8_t)(fminf(mapped.x * 255.0f, 255.0f));
    dst[idx + 1] = (uint8_t)(fminf(mapped.y * 255.0f, 255.0f));
    dst[idx + 2] = (uint8_t)(fminf(mapped.z * 255.0f, 255.0f));
}

void hdr_tone_map_rgb(const uint8_t* d_src, uint8_t* d_dst,
                      uint32_t w, uint32_t h,
                      int method, float peak_nits,
                      cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    if (method == 0)
        hdr_reinhard_kernel<<<grid, block, 0, s>>>(d_src, d_dst, w, h, peak_nits);
    else
        hdr_aces_kernel<<<grid, block, 0, s>>>(d_src, d_dst, w, h, peak_nits);
}

// ============================================================================
// 7. Temporal Stabilization (global motion estimation + warp)
// ============================================================================
// Simple global motion estimation using block matching on Y plane.
// Estimates translation (dx, dy) between consecutive frames.
__global__ void estimate_motion_kernel(const uint8_t* __restrict__ prev,
                                       const uint8_t* __restrict__ curr,
                                       int* __restrict__ accum_dx,
                                       int* __restrict__ accum_dy,
                                       int* __restrict__ accum_count,
                                       uint32_t w, uint32_t h,
                                       int block_size, int search_range) {
    // Each block processes one macroblock
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t mb_x = bx * block_size;
    uint32_t mb_y = by * block_size;
    if (mb_x + block_size > w || mb_y + block_size > h) return;

    extern __shared__ int sdata[];
    int* s_dx = sdata;
    int* s_dy = sdata + blockDim.x;
    int* s_cnt = sdata + blockDim.x * 2;

    int tid = threadIdx.x;
    s_dx[tid] = 0; s_dy[tid] = 0; s_cnt[tid] = 0;
    __syncthreads();

    // Simple SAD-based block matching (each thread tries one offset)
    int best_dx = 0, best_dy = 0;
    int best_sad = 999999;

    for (int sy = -search_range; sy <= search_range; sy += 2) {
        for (int sx = -search_range; sx <= search_range; sx += 2) {
            int sad = 0;
            for (int py = 0; py < block_size; py += 2) {
                for (int px = 0; px < block_size; px += 2) {
                    int cx = mb_x + px;
                    int cy = mb_y + py;
                    int px2 = cx + sx;
                    int py2 = cy + sy;
                    if (px2 < 0 || px2 >= (int)w || py2 < 0 || py2 >= (int)h) continue;
                    // RGB data: 3 bytes per pixel, use luminance for block matching
                    int ci = (cy * w + cx) * 3;
                    int pi = (py2 * w + px2) * 3;
                    int cy_curr = (int)(0.299f * curr[ci] + 0.587f * curr[ci + 1] + 0.114f * curr[ci + 2]);
                    int cy_prev = (int)(0.299f * prev[pi] + 0.587f * prev[pi + 1] + 0.114f * prev[pi + 2]);
                    sad += abs(cy_curr - cy_prev);
                }
            }
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = sx;
                best_dy = sy;
            }
        }
    }

    atomicAdd(accum_dx, best_dx);
    atomicAdd(accum_dy, best_dy);
    atomicAdd(accum_count, 1);
}

// Warp (translate) RGB image by (warp_dx, warp_dy)
__global__ void warp_translate_rgb_kernel(const uint8_t* __restrict__ src,
                                          uint8_t* __restrict__ dst,
                                          uint32_t w, uint32_t h,
                                          int warp_dx, int warp_dy) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int sx = (int)x - warp_dx;
    int sy = (int)y - warp_dy;
    int di = (y * w + x) * 3;

    if (sx < 0 || sx >= (int)w || sy < 0 || sy >= (int)h) {
        dst[di] = dst[di + 1] = dst[di + 2] = 0; // black border
        return;
    }
    int si = (sy * w + sx) * 3;
    dst[di]     = src[si];
    dst[di + 1] = src[si + 1];
    dst[di + 2] = src[si + 2];
}

void temporal_stabilize_rgb(const uint8_t* d_prev, const uint8_t* d_curr,
                            uint8_t* d_dst, uint32_t w, uint32_t h,
                            int* d_accum, int block_size, int search_range,
                            cudaStream_t s) {
    // Reset accumulators
    cudaMemsetAsync(d_accum, 0, 3 * sizeof(int), s);

    dim3 mb_block(1);
    dim3 mb_grid((w + block_size - 1) / block_size, (h + block_size - 1) / block_size);
    int smem = 3 * sizeof(int);
    estimate_motion_kernel<<<mb_grid, mb_block, smem, s>>>(d_prev, d_curr,
                                                            d_accum, d_accum + 1, d_accum + 2,
                                                            w, h, block_size, search_range);
    // Read back motion vector (CPU side needed for warp — but we keep it GPU-side)
    // For simplicity, do the warp on GPU with the averaged motion vector
    // The accum[0]/accum[2] = avg_dx, accum[1]/accum[2] = avg_dy
    // We'll do the actual warp in pipeline.cu after reading the motion vector
}

void warp_translate_rgb(const uint8_t* d_src, uint8_t* d_dst,
                        uint32_t w, uint32_t h,
                        int warp_dx, int warp_dy,
                        cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    warp_translate_rgb_kernel<<<grid, block, 0, s>>>(d_src, d_dst, w, h, warp_dx, warp_dy);
}

} // namespace filters
} // namespace kagerou
