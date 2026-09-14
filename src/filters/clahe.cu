// Kagerou SDK — CLAHE (Contrast Limited Adaptive Histogram Equalization).
// CUDA implementation: per-tile histogram + clip + CDF, bilinear interpolation.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>

namespace kagerou {
namespace filters {

static const int CLAHE_BINS = 256;

// ---- Extract one channel from interleaved RGB ------------------------------
__global__ void extract_channel_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       uint32_t n, int channel) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i * 3 + channel];
}

// ---- Write one channel back to interleaved RGB -----------------------------
__global__ void write_channel_kernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     uint32_t n, int channel) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i * 3 + channel] = src[i];
}

// ---- Pass 1: Per-tile histogram + clip + CDF (one block per tile) ----------
__global__ void tile_histogram_kernel(
    const uint8_t* __restrict__ src, int* __restrict__ tile_cdf,
    uint32_t img_w, uint32_t img_h,
    int tile_w, int tile_h, int tiles_x, int tiles_y,
    float clip_limit)
{
    extern __shared__ int s_hist[];
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    int tid = threadIdx.x;

    for (int i = tid; i < CLAHE_BINS; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    int x0 = tx * tile_w;
    int y0 = ty * tile_h;
    int x1 = min(x0 + tile_w, (int)img_w);
    int y1 = min(y0 + tile_h, (int)img_h);
    int pixels = (x1 - x0) * (y1 - y0);

    for (int i = tid; i < pixels; i += blockDim.x) {
        int px = x0 + (i % (x1 - x0));
        int py = y0 + (i / (x1 - x0));
        atomicAdd(&s_hist[src[py * img_w + px]], 1);
    }
    __syncthreads();

    // Clip
    float clip_px = (float)pixels / CLAHE_BINS * clip_limit;
    int excess = 0;
    for (int i = tid; i < CLAHE_BINS; i += blockDim.x) {
        if ((float)s_hist[i] > clip_px) {
            excess += (int)((float)s_hist[i] - clip_px);
            s_hist[i] = (int)clip_px;
        }
    }
    __syncthreads();

    int redist = excess / CLAHE_BINS;
    int rem = excess % CLAHE_BINS;
    for (int i = tid; i < CLAHE_BINS; i += blockDim.x) {
        s_hist[i] += redist + (i < rem ? 1 : 0);
    }
    __syncthreads();

    // CDF + normalize
    if (tid == 0) {
        int tile_idx = ty * tiles_x + tx;
        int sum = 0;
        for (int i = 0; i < CLAHE_BINS; i++) {
            tile_cdf[tile_idx * CLAHE_BINS + i] = sum;
            sum += s_hist[i];
        }
        if (sum > 0) {
            for (int i = 0; i < CLAHE_BINS; i++)
                tile_cdf[tile_idx * CLAHE_BINS + i] =
                    (int)((float)tile_cdf[tile_idx * CLAHE_BINS + i] * 255.0f / sum + 0.5f);
        }
    }
}

// ---- Pass 2: Bilinear interpolation between tile CDFs (one thread/pixel) ---
__global__ void interpolate_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    const int* __restrict__ tile_cdf,
    uint32_t w, uint32_t h,
    int tile_w, int tile_h, int tiles_x, int tiles_y)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float tx_f = (float)x / tile_w;
    float ty_f = (float)y / tile_h;
    int tx0 = max(0, min((int)floorf(tx_f), tiles_x - 1));
    int ty0 = max(0, min((int)floorf(ty_f), tiles_y - 1));
    int tx1 = min(tiles_x - 1, tx0 + 1);
    int ty1 = min(tiles_y - 1, ty0 + 1);
    float fx = tx_f - floorf(tx_f);
    float fy = ty_f - floorf(ty_f);

    int val = src[y * w + x];
    float c00 = (float)tile_cdf[(ty0 * tiles_x + tx0) * CLAHE_BINS + val];
    float c10 = (float)tile_cdf[(ty0 * tiles_x + tx1) * CLAHE_BINS + val];
    float c01 = (float)tile_cdf[(ty1 * tiles_x + tx0) * CLAHE_BINS + val];
    float c11 = (float)tile_cdf[(ty1 * tiles_x + tx1) * CLAHE_BINS + val];

    float result = c00 * (1-fx)*(1-fy) + c10 * fx*(1-fy) + c01 * (1-fx)*fy + c11 * fx*fy;
    dst[y * w + x] = (uint8_t)max(0, min(255, (int)(result + 0.5f)));
}

// ---- CLAHE on a single-channel grayscale plane -----------------------------
// Persistent CDF buffer to avoid per-frame cudaMalloc/cudaFree.
// Safe for multi-stream: cudaMalloc/cudaFree are serialized per-device.
static int* g_tile_cdf = nullptr;
static size_t g_tile_cdf_size = 0;

static void clahe_single_channel(const uint8_t* d_src, uint8_t* d_dst,
                                 uint32_t w, uint32_t h,
                                 float clip_limit, int tile_size,
                                 cudaStream_t s) {
    int tiles_x = (w + tile_size - 1) / tile_size;
    int tiles_y = (h + tile_size - 1) / tile_size;

    size_t needed = (size_t)tiles_x * tiles_y * CLAHE_BINS * sizeof(int);
    if (needed > g_tile_cdf_size) {
        if (g_tile_cdf) cudaFree(g_tile_cdf);
        cudaMalloc(&g_tile_cdf, needed);
        g_tile_cdf_size = needed;
    }

    dim3 grid(tiles_x, tiles_y);
    dim3 block(256);
    tile_histogram_kernel<<<grid, block, CLAHE_BINS * sizeof(int), s>>>(
        d_src, g_tile_cdf, w, h, tile_size, tile_size, tiles_x, tiles_y, clip_limit);

    dim3 g2 = grid_2d(w, h);
    dim3 b2 = block_2d(w, h);
    interpolate_kernel<<<g2, b2, 0, s>>>(
        d_src, d_dst, g_tile_cdf, w, h, tile_size, tile_size, tiles_x, tiles_y);
}

// ---- NV12 CLAHE: Y plane only, UV passthrough -----------------------------
void clahe_nv12(const uint8_t* d_src, uint8_t* d_dst,
                uint32_t w, uint32_t h,
                float clip_limit, int tile_size,
                cudaStream_t s) {
    clahe_single_channel(d_src, d_dst, w, h, clip_limit, tile_size, s);

    uint32_t uv_offset = w * h;
    uint32_t uv_size = w * (h / 2);
    cudaMemcpyAsync(d_dst + uv_offset, d_src + uv_offset,
                    uv_size, cudaMemcpyDeviceToDevice, s);
}

// ---- RGB→Luminance (single channel, BT.709) ---------------------------------
__global__ void rgb_to_lum_kernel(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t r = src[i*3], g = src[i*3+1], b = src[i*3+2];
    dst[i] = (uint8_t)(0.2126f*r + 0.7152f*g + 0.0722f*b + 0.5f);
}

// ---- Blend enhanced luminance back into RGB ----------------------------------
__global__ void blend_lum_kernel(const uint8_t* __restrict__ orig,
                                 const uint8_t* __restrict__ enhanced_lum,
                                 uint8_t* __restrict__ dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float orig_lum = 0.2126f*orig[i*3] + 0.7152f*orig[i*3+1] + 0.0722f*orig[i*3+2];
    if (orig_lum < 1.0f) orig_lum = 1.0f;
    float ratio = (float)enhanced_lum[i] / orig_lum;
    float r = orig[i*3]   * ratio;
    float g = orig[i*3+1] * ratio;
    float b = orig[i*3+2] * ratio;
    dst[i*3]   = (uint8_t)max(0, min(255, (int)(r + 0.5f)));
    dst[i*3+1] = (uint8_t)max(0, min(255, (int)(g + 0.5f)));
    dst[i*3+2] = (uint8_t)max(0, min(255, (int)(b + 0.5f)));
}

// ---- RGB CLAHE: luminance-only to avoid color fringing at tile boundaries ----
void clahe_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               float clip_limit, int tile_size,
               cudaStream_t s) {
    uint32_t n = w * h;
    dim3 b1((n + 255) / 256);
    dim3 t1(256);
    if (b1.x == 0) b1.x = 1;

    uint8_t* d_lum = nullptr;
    uint8_t* d_lum_out = nullptr;
    cudaMalloc(&d_lum, n);
    cudaMalloc(&d_lum_out, n);

    rgb_to_lum_kernel<<<b1, t1, 0, s>>>(d_src, d_lum, n);
    clahe_single_channel(d_lum, d_lum_out, w, h, clip_limit, tile_size, s);
    blend_lum_kernel<<<b1, t1, 0, s>>>(d_src, d_lum_out, d_dst, n);

    cudaFree(d_lum);
    cudaFree(d_lum_out);
}

} // namespace filters
} // namespace kagerou
