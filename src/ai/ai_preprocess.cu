// ============================================================================
// Kagerou AI — GPU Preprocessing Kernels
// ZERO CPU. All format conversion happens on CUDA cores.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace kagerou {
namespace ai {

// ============================================================================
// RGB uint8 (HWC) → NCHW float normalized [0,1] — single frame
// d_rgb:  input RGB on GPU, w*h*3 bytes
// d_nchw: output NCHW on GPU, 3*w*h floats
// frame_idx: which frame slot (0..N-1) in the batch
// ============================================================================
__global__ void rgb8_to_nchw_f32_kernel(
    const uint8_t* __restrict__ d_rgb,
    float* __restrict__ d_nchw,
    uint32_t w, uint32_t h, uint32_t frame_idx)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t plane = w * h;
    uint32_t pixel = y * w + x;
    uint32_t src_idx = pixel * 3;

    // Write to NCHW layout: [frame_idx, channel, y, x]
    uint32_t dst_base = frame_idx * 3 * plane;
    d_nchw[dst_base + 0 * plane + pixel] = d_rgb[src_idx + 0] * (1.0f / 255.0f);
    d_nchw[dst_base + 1 * plane + pixel] = d_rgb[src_idx + 1] * (1.0f / 255.0f);
    d_nchw[dst_base + 2 * plane + pixel] = d_rgb[src_idx + 2] * (1.0f / 255.0f);
}

// ============================================================================
// RGB8 → NCHW float [0, 255] — single frame (for models expecting raw pixel values)
// ============================================================================
__global__ void rgb8_to_nchw_255_kernel(
    const uint8_t* __restrict__ d_rgb,
    float* __restrict__ d_nchw,
    uint32_t w, uint32_t h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t plane = w * h;
    uint32_t pixel = y * w + x;
    uint32_t src_idx = pixel * 3;

    d_nchw[0 * plane + pixel] = (float)d_rgb[src_idx + 0];
    d_nchw[1 * plane + pixel] = (float)d_rgb[src_idx + 1];
    d_nchw[2 * plane + pixel] = (float)d_rgb[src_idx + 2];
}

// ============================================================================
// NCHW float [0,1] → RGB uint8 — single frame
// d_nchw: input NCHW on GPU, 3*w*h floats
// d_rgb:  output RGB on GPU, w*h*3 bytes
// ============================================================================
__global__ void nchw_f32_to_rgb8_kernel(
    const float* __restrict__ d_nchw,
    uint8_t* __restrict__ d_rgb,
    uint32_t w, uint32_t h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t plane = w * h;
    uint32_t pixel = y * w + x;

    float r = d_nchw[0 * plane + pixel] * 255.0f;
    float g = d_nchw[1 * plane + pixel] * 255.0f;
    float b = d_nchw[2 * plane + pixel] * 255.0f;

    d_rgb[pixel * 3 + 0] = (uint8_t)fminf(fmaxf(r + 0.5f, 0.0f), 255.0f);
    d_rgb[pixel * 3 + 1] = (uint8_t)fminf(fmaxf(g + 0.5f, 0.0f), 255.0f);
    d_rgb[pixel * 3 + 2] = (uint8_t)fminf(fmaxf(b + 0.5f, 0.0f), 255.0f);
}

// ============================================================================
// Pack 5 RGB frames into single NCHW tensor [1, 15, H, W]
// d_frames[0..4]: 5 GPU RGB pointers, each w*h*3 bytes
// d_batch: output [1, 15, H, W] float tensor on GPU
// ============================================================================
__global__ void pack_5frames_nchw_kernel(
    const uint8_t* __restrict__ f0,
    const uint8_t* __restrict__ f1,
    const uint8_t* __restrict__ f2,
    const uint8_t* __restrict__ f3,
    const uint8_t* __restrict__ f4,
    float* __restrict__ d_batch,
    uint32_t w, uint32_t h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t plane = w * h;
    uint32_t pixel = y * w + x;
    uint32_t src_idx = pixel * 3;

    const uint8_t* frames[5] = {f0, f1, f2, f3, f4};

    for (int f = 0; f < 5; f++) {
        uint32_t base = f * 3 * plane;
        d_batch[base + 0 * plane + pixel] = frames[f][src_idx + 0] * (1.0f / 255.0f);
        d_batch[base + 1 * plane + pixel] = frames[f][src_idx + 1] * (1.0f / 255.0f);
        d_batch[base + 2 * plane + pixel] = frames[f][src_idx + 2] * (1.0f / 255.0f);
    }
}

// ============================================================================
// Unpack NCHW output [1, 3, H, W] → RGB uint8
// ============================================================================
__global__ void unpack_nchw_to_rgb8_kernel(
    const float* __restrict__ d_nchw,
    uint8_t* __restrict__ d_rgb,
    uint32_t w, uint32_t h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t plane = w * h;
    uint32_t pixel = y * w + x;

    float r = d_nchw[0 * plane + pixel] * 255.0f;
    float g = d_nchw[1 * plane + pixel] * 255.0f;
    float b = d_nchw[2 * plane + pixel] * 255.0f;

    d_rgb[pixel * 3 + 0] = (uint8_t)fminf(fmaxf(r + 0.5f, 0.0f), 255.0f);
    d_rgb[pixel * 3 + 1] = (uint8_t)fminf(fmaxf(g + 0.5f, 0.0f), 255.0f);
    d_rgb[pixel * 3 + 2] = (uint8_t)fminf(fmaxf(b + 0.5f, 0.0f), 255.0f);
}

// ============================================================================
// Bilinear RGB resize kernel — shared across all AI filters
// ============================================================================
__global__ void resize_bilinear_rgb_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float fx = (float)x * (float)src_w / (float)dst_w;
    float fy = (float)y * (float)src_h / (float)dst_h;
    uint32_t x0 = min((uint32_t)fx, src_w - 1);
    uint32_t y0 = min((uint32_t)fy, src_h - 1);
    uint32_t x1 = min(x0 + 1, src_w - 1);
    uint32_t y1 = min(y0 + 1, src_h - 1);
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;

    for (int c = 0; c < 3; c++) {
        float v = (1-ax)*(1-ay)*src[(y0*src_w+x0)*3+c]
                + ax*(1-ay)*src[(y0*src_w+x1)*3+c]
                + (1-ax)*ay*src[(y1*src_w+x0)*3+c]
                + ax*ay*src[(y1*src_w+x1)*3+c];
        dst[(y*dst_w+x)*3+c] = (uint8_t)(v + 0.5f);
    }
}

// ============================================================================
// Sigmoid for single float value (host)
// ============================================================================
inline __host__ __device__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================================
// Per-frame min/max reduce for float depth buffer
// Each block reduces its chunk into d_block_min/max[blockIdx.x]
// ============================================================================
static __global__ void depth_reduce_min_max_kernel(
    const float* __restrict__ d, uint32_t n,
    float* __restrict__ d_block_min, float* __restrict__ d_block_max)
{
    extern __shared__ float sdata[];
    float* s_min = sdata;
    float* s_max = sdata + blockDim.x;

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x * 4 + tid;

    float local_min = 1e30f;
    float local_max = -1e30f;
    for (uint32_t k = 0; k < 4; k++) {
        uint32_t idx = i + k * blockDim.x;
        if (idx < n) {
            float v = d[idx];
            if (isfinite(v)) {
                local_min = fminf(local_min, v);
                local_max = fmaxf(local_max, v);
            }
        }
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_min[blockIdx.x] = s_min[0];
        d_block_max[blockIdx.x] = s_max[0];
    }
}

// Normalize depth buffer in-place: d[i] = (d[i] - dmin) / (dmax - dmin)
static __global__ void depth_normalize_kernel(
    float* __restrict__ d, uint32_t n, float dmin, float range)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = d[i];
    if (!isfinite(v)) v = dmin;
    d[i] = (range > 1e-6f) ? (v - dmin) / range : 0.0f;
}

// Host wrapper: find min/max, then normalize to [0,1]
void launch_normalize_depth(float* d_depth, uint32_t n, cudaStream_t s) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads * 4 - 1) / (threads * 4);
    if (blocks > 1024) blocks = 1024;

    float *d_bmin, *d_bmax;
    cudaMalloc(&d_bmin, blocks * sizeof(float));
    cudaMalloc(&d_bmax, blocks * sizeof(float));

    uint32_t smem = threads * 2 * sizeof(float);
    depth_reduce_min_max_kernel<<<blocks, threads, smem, s>>>(d_depth, n, d_bmin, d_bmax);

    float h_bmin[1024], h_bmax[1024];
    cudaMemcpyAsync(h_bmin, d_bmin, blocks * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_bmax, d_bmax, blocks * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    float h_min = 1e30f, h_max = -1e30f;
    for (uint32_t i = 0; i < blocks; i++) {
        if (isfinite(h_bmin[i])) h_min = fminf(h_min, h_bmin[i]);
        if (isfinite(h_bmax[i])) h_max = fmaxf(h_max, h_bmax[i]);
    }

    float range = h_max - h_min;
    uint32_t norm_blocks = (n + threads - 1) / threads;
    depth_normalize_kernel<<<norm_blocks, threads, 0, s>>>(d_depth, n, h_min, range);

    cudaFree(d_bmin);
    cudaFree(d_bmax);
}

// ============================================================================
// Float depth map [0..1] → RGB uint8 grayscale heatmap
// ============================================================================
__global__ void depth_to_rgb_kernel(
    const float* __restrict__ d_depth, uint8_t* __restrict__ d_rgb,
    uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fminf(fmaxf(d_depth[i], 0.0f), 1.0f);
    uint8_t val = (uint8_t)(v * 255.0f);
    d_rgb[i * 3 + 0] = val;
    d_rgb[i * 3 + 1] = val;
    d_rgb[i * 3 + 2] = val;
}

void launch_depth_to_rgb(const float* d_depth, uint8_t* d_rgb, uint32_t n, cudaStream_t s = 0) {
    dim3 b(256);
    dim3 g((n + 255) / 256);
    depth_to_rgb_kernel<<<g, b, 0, s>>>(d_depth, d_rgb, n);
}

// ============================================================================
// Host launch wrappers
// ============================================================================
void launch_rgb8_to_nchw(const uint8_t* d_rgb, float* d_nchw,
                         uint32_t w, uint32_t h, uint32_t frame_idx,
                         cudaStream_t s = 0) {
    dim3 b(16, 16);
    dim3 g((w+15)/16, (h+15)/16);
    rgb8_to_nchw_f32_kernel<<<g, b, 0, s>>>(d_rgb, d_nchw, w, h, frame_idx);
}

void launch_rgb8_to_nchw_255(const uint8_t* d_rgb, float* d_nchw,
                             uint32_t w, uint32_t h, cudaStream_t s = 0) {
    dim3 b(16, 16);
    dim3 g((w+15)/16, (h+15)/16);
    rgb8_to_nchw_255_kernel<<<g, b, 0, s>>>(d_rgb, d_nchw, w, h);
}

void launch_nchw_to_rgb8(const float* d_nchw, uint8_t* d_rgb,
                         uint32_t w, uint32_t h,
                         cudaStream_t s = 0) {
    dim3 b(16, 16);
    dim3 g((w+15)/16, (h+15)/16);
    nchw_f32_to_rgb8_kernel<<<g, b, 0, s>>>(d_nchw, d_rgb, w, h);
}

void launch_pack_5frames(const uint8_t* frames[5], float* d_batch,
                         uint32_t w, uint32_t h,
                         cudaStream_t s = 0) {
    dim3 b(16, 16);
    dim3 g((w+15)/16, (h+15)/16);
    pack_5frames_nchw_kernel<<<g, b, 0, s>>>(
        frames[0], frames[1], frames[2], frames[3], frames[4],
        d_batch, w, h);
}

void launch_unpack_nchw(const float* d_nchw, uint8_t* d_rgb,
                        uint32_t w, uint32_t h,
                        cudaStream_t s = 0) {
    dim3 b(16, 16);
    dim3 g((w+15)/16, (h+15)/16);
    unpack_nchw_to_rgb8_kernel<<<g, b, 0, s>>>(d_nchw, d_rgb, w, h);
}

void launch_resize_rgb(const uint8_t* src, uint8_t* dst,
                       uint32_t src_w, uint32_t src_h,
                       uint32_t dst_w, uint32_t dst_h,
                       cudaStream_t s) {
    dim3 b(16, 16);
    dim3 g((dst_w+15)/16, (dst_h+15)/16);
    resize_bilinear_rgb_kernel<<<g, b, 0, s>>>(src, dst, src_w, src_h, dst_w, dst_h);
}

// ============================================================================
// GPU Visualization Kernels — draw results onto RGB frames
// ============================================================================

// Draw filled rectangle (for bounding boxes)
__global__ void draw_rect_kernel(
    uint8_t* __restrict__ rgb, uint32_t w, uint32_t h,
    int x1, int y1, int x2, int y2,
    uint8_t r, uint8_t g, uint8_t b, int thickness)
{
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    if (px >= (uint32_t)x1 && px <= (uint32_t)x2 && py >= (uint32_t)y1 && py <= (uint32_t)y2) {
        bool edge = (px < (uint32_t)(x1 + thickness) || px > (uint32_t)(x2 - thickness) ||
                     py < (uint32_t)(y1 + thickness) || py > (uint32_t)(y2 - thickness));
        if (edge) {
            uint32_t idx = (py * w + px) * 3;
            rgb[idx + 0] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
}

// Draw circle (for landmark points)
__global__ void draw_circle_kernel(
    uint8_t* __restrict__ rgb, uint32_t w, uint32_t h,
    float cx, float cy, float radius,
    uint8_t r, uint8_t g, uint8_t b)
{
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    float dx = (float)px - cx, dy = (float)py - cy;
    if (dx * dx + dy * dy <= radius * radius) {
        uint32_t idx = (py * w + px) * 3;
        rgb[idx + 0] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
}

// Flow to RGB visualization (Middlebury color coding)
__global__ void flow_to_rgb_kernel(
    const float* __restrict__ d_flow,
    uint8_t* __restrict__ d_rgb,
    uint32_t w, uint32_t h)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    float dx = d_flow[i * 2 + 0];
    float dy = d_flow[i * 2 + 1];
    float mag = sqrtf(dx * dx + dy * dy);
    float angle = atan2f(dy, dx) / 3.14159265f; // [-1, 1]

    // HSV-like color wheel, tuned for typical webcam motion (1-8 px/frame).
    // The old gains (mag/20, mag*5) left everyday motion dark gray.
    float hue = (angle + 1.0f) * 0.5f; // [0, 1]
    float sat = fminf(mag / 6.0f, 1.0f);
    float val = fminf(mag * 30.0f, 255.0f);

    // Simple HSV to RGB (hue sectors)
    float h6 = hue * 6.0f;
    int sector = (int)h6;
    float f = h6 - sector;
    uint8_t p = 0, q = (uint8_t)(val * (1.0f - sat)), t = (uint8_t)(val * sat * f);
    uint8_t v = (uint8_t)val;
    switch (sector % 6) {
        case 0: d_rgb[i*3+0]=v; d_rgb[i*3+1]=t; d_rgb[i*3+2]=q; break;
        case 1: d_rgb[i*3+0]=t; d_rgb[i*3+1]=v; d_rgb[i*3+2]=q; break;
        case 2: d_rgb[i*3+0]=q; d_rgb[i*3+1]=v; d_rgb[i*3+2]=t; break;
        case 3: d_rgb[i*3+0]=q; d_rgb[i*3+1]=t; d_rgb[i*3+2]=v; break;
        case 4: d_rgb[i*3+0]=t; d_rgb[i*3+1]=q; d_rgb[i*3+2]=v; break;
        case 5: d_rgb[i*3+0]=v; d_rgb[i*3+1]=q; d_rgb[i*3+2]=t; break;
    }
}

void launch_draw_rect(uint8_t* rgb, uint32_t w, uint32_t h,
                      int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b, int thickness,
                      cudaStream_t s = 0) {
    dim3 b2(16, 16), g2((w+15)/16, (h+15)/16);
    draw_rect_kernel<<<g2, b2, 0, s>>>(rgb, w, h, x1, y1, x2, y2, r, g, b, thickness);
}

void launch_draw_circle(uint8_t* rgb, uint32_t w, uint32_t h,
                         float cx, float cy, float radius,
                         uint8_t r, uint8_t g, uint8_t b,
                         cudaStream_t s = 0) {
    dim3 b2(16, 16), g2((w+15)/16, (h+15)/16);
    draw_circle_kernel<<<g2, b2, 0, s>>>(rgb, w, h, cx, cy, radius, r, g, b);
}

// 3x5 bitmap font: A-Z 0-9 space '-' '.'. Each glyph = 5 rows of 3 bits
// (bit 2 = left pixel). Verified legible at scale>=2 (see repo notes).
// __constant__: host table is invisible to device code.
__constant__ uint8_t kFont3x5[40][5] = {
    {2,5,7,5,5},{6,5,6,5,6},{3,4,4,4,3},{6,5,5,5,6},{7,4,6,4,7},{7,4,6,4,4},
    {3,4,5,5,3},{5,5,7,5,5},{7,2,2,2,7},{1,1,1,5,2},{5,5,6,5,5},{4,4,4,4,7},
    {5,7,7,5,5},{6,5,5,5,5},{2,5,5,5,2},{6,5,6,4,4},{2,5,5,2,1},{6,5,6,5,5},
    {3,4,2,1,6},{7,2,2,2,2},{5,5,5,5,7},{5,5,5,5,2},{5,5,7,7,5},{5,5,2,5,5},
    {5,5,2,2,2},{7,1,2,4,7},
    {7,5,5,5,7},{2,6,2,2,7},{7,1,7,4,7},{7,1,7,1,7},{5,5,7,1,1},{7,4,7,1,7},
    {7,4,7,5,7},{7,1,1,2,2},{7,5,7,5,7},{7,5,7,1,7},
    {0,0,0,0,0},{0,0,7,0,0},{0,0,0,0,2}
};

// Device-side label text (<= 63 chars), stream-ordered via symbol copy.
__constant__ char d_text_buf[64];

__global__ void draw_text_kernel(
    uint8_t* __restrict__ rgb, uint32_t w, uint32_t h,
    int x0, int y0, int nchars, int scale,
    uint8_t fr, uint8_t fg, uint8_t fb,
    uint8_t br, uint8_t bg, uint8_t bb)
{
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    int cw = 3 * scale + 1, chh = 5 * scale;
    int lx = (int)px - x0, ly = (int)py - y0;
    if (lx < 0 || ly < 0 || lx >= nchars * cw || ly >= chh) return;
    int ci = lx / cw;
    char c = (ci < nchars) ? d_text_buf[ci] : ' ';
    if (c >= 'a' && c <= 'z') c -= 32;
    int gi;
    if (c >= 'A' && c <= 'Z') gi = c - 'A';
    else if (c >= '0' && c <= '9') gi = 26 + c - '0';
    else if (c == '-') gi = 37;
    else if (c == '.') gi = 38;
    else gi = 36; // space / unknown
    int gx = (lx - ci * cw) / scale, gy = ly / scale;
    bool dot = (gx < 3) && ((kFont3x5[gi][gy] >> (2 - gx)) & 1);
    uint32_t idx = (py * w + px) * 3;
    rgb[idx + 0] = dot ? fr : br;
    rgb[idx + 1] = dot ? fg : bg;
    rgb[idx + 2] = dot ? fb : bb;
}

void launch_draw_text(uint8_t* rgb, uint32_t w, uint32_t h,
                      int x, int y, const char* str,
                      int scale, uint8_t fr, uint8_t fg, uint8_t fb,
                      uint8_t br, uint8_t bg, uint8_t bb,
                      cudaStream_t s) {
    if (!str || !str[0] || scale < 1) return;
    size_t n = strlen(str);
    if (n > 63) n = 63;
    cudaMemcpyToSymbolAsync(d_text_buf, str, n + 1,
                            0, cudaMemcpyHostToDevice, s);
    dim3 b2(16, 16), g2((w+15)/16, (h+15)/16);
    draw_text_kernel<<<g2, b2, 0, s>>>(rgb, w, h, x, y, (int)n, scale,
                                       fr, fg, fb, br, bg, bb);
}

void launch_flow_to_rgb(const float* d_flow, uint8_t* d_rgb,
                        uint32_t w, uint32_t h,
                        cudaStream_t s = 0) {
    dim3 b2(256), g2((w*h+255)/256);
    flow_to_rgb_kernel<<<g2, b2, 0, s>>>(d_flow, d_rgb, w, h);
}

// Deinterleave NCHW planar [U_ch | V_ch] to interleaved [u0,v0, u1,v1, ...]
__global__ void deinterleave_flow_kernel(
    const float* __restrict__ planar, float* __restrict__ interleaved,
    uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    interleaved[i * 2 + 0] = planar[i];
    interleaved[i * 2 + 1] = planar[n + i];
}

void launch_deinterleave_flow(const float* d_planar, float* d_interleaved,
                              uint32_t n, cudaStream_t s = 0) {
    dim3 b(256), g((n + 255) / 256);
    deinterleave_flow_kernel<<<g, b, 0, s>>>(d_planar, d_interleaved, n);
}

// ============================================================================
// Letterbox resize: preserves aspect ratio, pads with gray (128)
// ============================================================================
__global__ void letterbox_resize_rgb_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h,
    float scale, float pad_x, float pad_y)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float src_x = ((float)x - pad_x) / scale;
    float src_y = ((float)y - pad_y) / scale;

    uint32_t dst_idx = (y * dst_w + x) * 3;

    if (src_x < 0.0f || src_y < 0.0f || src_x >= (float)src_w || src_y >= (float)src_h) {
        dst[dst_idx + 0] = 128;
        dst[dst_idx + 1] = 128;
        dst[dst_idx + 2] = 128;
        return;
    }

    uint32_t x0 = (uint32_t)src_x;
    uint32_t y0 = (uint32_t)src_y;
    uint32_t x1 = min(x0 + 1, src_w - 1);
    uint32_t y1 = min(y0 + 1, src_h - 1);
    float ax = src_x - (float)x0;
    float ay = src_y - (float)y0;

    for (int c = 0; c < 3; c++) {
        float v = (1-ax)*(1-ay)*src[(y0*src_w+x0)*3+c]
                + ax*(1-ay)*src[(y0*src_w+x1)*3+c]
                + (1-ax)*ay*src[(y1*src_w+x0)*3+c]
                + ax*ay*src[(y1*src_w+x1)*3+c];
        dst[dst_idx + c] = (uint8_t)(v + 0.5f);
    }
}

void launch_letterbox_resize_rgb(const uint8_t* src, uint8_t* dst,
                                  uint32_t src_w, uint32_t src_h,
                                  uint32_t dst_w, uint32_t dst_h,
                                  float& out_scale, float& out_pad_x, float& out_pad_y,
                                  cudaStream_t s) {
    float sx = (float)dst_w / (float)src_w;
    float sy = (float)dst_h / (float)src_h;
    out_scale = fminf(sx, sy);
    out_pad_x = ((float)dst_w - (float)src_w * out_scale) * 0.5f;
    out_pad_y = ((float)dst_h - (float)src_h * out_scale) * 0.5f;

    dim3 b(16, 16);
    dim3 g((dst_w+15)/16, (dst_h+15)/16);
    letterbox_resize_rgb_kernel<<<g, b, 0, s>>>(src, dst, src_w, src_h, dst_w, dst_h,
                                                 out_scale, out_pad_x, out_pad_y);
}

// Transform [0,1] NCHW float tensor to [-1,1] range: x = x * 2.0 - 1.0
__global__ void normalize_01_to_11_kernel(float* __restrict__ data, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    data[i] = data[i] * 2.0f - 1.0f;
}

void launch_normalize_01_to_11(float* d_data, uint32_t count, cudaStream_t s = 0) {
    dim3 b(256), g((count + 255) / 256);
    normalize_01_to_11_kernel<<<g, b, 0, s>>>(d_data, count);
}

// RGB uint8 HWC → NHWC float [0,1] (for MediaPipe Pose which expects [1,H,W,3])
__global__ void rgb8_to_nhwc_float_kernel(
    const uint8_t* __restrict__ d_rgb,
    float* __restrict__ d_out,
    uint32_t n)  // n = w * h
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_out[i * 3 + 0] = d_rgb[i * 3 + 0] * (1.0f / 255.0f);
    d_out[i * 3 + 1] = d_rgb[i * 3 + 1] * (1.0f / 255.0f);
    d_out[i * 3 + 2] = d_rgb[i * 3 + 2] * (1.0f / 255.0f);
}

void launch_rgb8_to_nhwc_float(const uint8_t* d_rgb, float* d_out,
                               uint32_t w, uint32_t h, cudaStream_t s) {
    uint32_t n = w * h;
    dim3 b(256), g((n + 255) / 256);
    rgb8_to_nhwc_float_kernel<<<g, b, 0, s>>>(d_rgb, d_out, n);
}

// GPU crop: extract rectangular region from RGB HWC source
void launch_crop_rgb(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t src_w, uint32_t src_h,
                     uint32_t crop_x, uint32_t crop_y,
                     uint32_t crop_w, uint32_t crop_h, cudaStream_t s) {
    // Copy row by row using cudaMemcpy2DAsync (GPU-native, no CPU)
    cudaMemcpy2DAsync(d_dst, crop_w * 3,
                      d_src + (crop_y * src_w + crop_x) * 3, src_w * 3,
                      crop_w * 3, crop_h,
                      cudaMemcpyDeviceToDevice, s);
}

} // namespace ai
} // namespace kagerou
