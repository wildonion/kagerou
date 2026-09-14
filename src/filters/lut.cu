// Kagerou SDK — 3D LUT color grading filter.
// Trilinear interpolation on a 17x17x17 or 33x33x33 3D LUT.

#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

namespace kagerou {
namespace filters {

// Generate built-in 3D LUT on GPU based on preset type
// LUT layout: lut[b * res * res + g * res + r] = {r_out, g_out, b_out}
// res = 17 (17*17*17 = 4913 entries, 14739 bytes)
static const int LUT_RES = 17;

__device__ __forceinline__ float3 apply_curve(float3 val, int preset) {
    float r = val.x, g = val.y, b = val.z;
    float ro, go, bo;
    switch (preset) {
        case 1: // Warm: boost reds/yellows
            ro = r * 1.1f + 0.02f;
            go = g * 1.02f + 0.01f;
            bo = b * 0.9f;
            break;
        case 2: // Cool: boost blues/cyans
            ro = r * 0.9f;
            go = g * 1.0f + 0.01f;
            bo = b * 1.1f + 0.02f;
            break;
        case 3: // Cinematic: orange-teal
            ro = r * 1.05f + (r > 0.5f ? 0.03f : -0.01f);
            go = g * 0.98f;
            bo = b * 1.08f + (b < 0.5f ? 0.04f : -0.01f);
            break;
        case 4: // Vintage: faded blacks, warm midtones
            ro = r * 0.95f + 0.05f;  // lifted blacks
            go = g * 0.92f + 0.04f;
            bo = b * 0.85f + 0.06f;
            // Desaturate slightly
            float lum = 0.299f * ro + 0.587f * go + 0.114f * bo;
            ro = ro + (lum - ro) * 0.2f;
            go = go + (lum - go) * 0.2f;
            bo = bo + (lum - bo) * 0.2f;
            break;
        case 5: // High Contrast
            ro = r < 0.5f ? r * 0.8f : r * 1.2f + 0.05f;
            go = g < 0.5f ? g * 0.8f : g * 1.2f + 0.05f;
            bo = b < 0.5f ? b * 0.8f : b * 1.2f + 0.05f;
            break;
        case 6: // Desaturate (bleach bypass)
            lum = 0.299f * r + 0.587f * g + 0.114f * b;
            ro = r + (lum - r) * 0.6f;
            go = g + (lum - g) * 0.6f;
            bo = b + (lum - b) * 0.6f;
            ro = ro * 1.1f + 0.02f;  // boost contrast
            go = go * 1.1f + 0.02f;
            bo = bo * 1.1f + 0.02f;
            break;
        default:
            ro = r; go = g; bo = b;
            break;
    }
    return make_float3(
        fminf(1.0f, fmaxf(0.0f, ro)),
        fminf(1.0f, fmaxf(0.0f, go)),
        fminf(1.0f, fmaxf(0.0f, bo))
    );
}

// Generate LUT on GPU
__global__ void generate_lut_kernel(uint8_t* __restrict__ lut, int preset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = LUT_RES * LUT_RES * LUT_RES;
    if (idx >= total) return;

    int b_idx = idx / (LUT_RES * LUT_RES);
    int g_idx = (idx / LUT_RES) % LUT_RES;
    int r_idx = idx % LUT_RES;

    float r = (float)r_idx / (LUT_RES - 1);
    float g = (float)g_idx / (LUT_RES - 1);
    float b = (float)b_idx / (LUT_RES - 1);

    float3 result = apply_curve(make_float3(r, g, b), preset);

    lut[idx * 3 + 0] = (uint8_t)(result.x * 255.0f + 0.5f);
    lut[idx * 3 + 1] = (uint8_t)(result.y * 255.0f + 0.5f);
    lut[idx * 3 + 2] = (uint8_t)(result.z * 255.0f + 0.5f);
}

// Trilinear LUT interpolation kernel (one thread per pixel)
__global__ void lut3d_apply_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ lut,
    uint32_t w, uint32_t h,
    float strength)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    float r = (float)src[idx + 0] / 255.0f;
    float g = (float)src[idx + 1] / 255.0f;
    float b = (float)src[idx + 2] / 255.0f;

    // Map to LUT coordinates
    float lr = r * (LUT_RES - 1);
    float lg = g * (LUT_RES - 1);
    float lb = b * (LUT_RES - 1);

    int r0 = (int)lr, r1 = min(r0 + 1, LUT_RES - 1);
    int g0 = (int)lg, g1 = min(g0 + 1, LUT_RES - 1);
    int b0 = (int)lb, b1 = min(b0 + 1, LUT_RES - 1);
    float fr = lr - r0, fg = lg - g0, fb = lb - b0;

    // Trilinear interpolation
    float3 result = make_float3(0, 0, 0);
    for (int ch = 0; ch < 3; ++ch) {
        float c000 = (float)lut[((b0 * LUT_RES + g0) * LUT_RES + r0) * 3 + ch];
        float c100 = (float)lut[((b0 * LUT_RES + g0) * LUT_RES + r1) * 3 + ch];
        float c010 = (float)lut[((b0 * LUT_RES + g1) * LUT_RES + r0) * 3 + ch];
        float c110 = (float)lut[((b0 * LUT_RES + g1) * LUT_RES + r1) * 3 + ch];
        float c001 = (float)lut[((b1 * LUT_RES + g0) * LUT_RES + r0) * 3 + ch];
        float c101 = (float)lut[((b1 * LUT_RES + g0) * LUT_RES + r1) * 3 + ch];
        float c011 = (float)lut[((b1 * LUT_RES + g1) * LUT_RES + r0) * 3 + ch];
        float c111 = (float)lut[((b1 * LUT_RES + g1) * LUT_RES + r1) * 3 + ch];

        float c00 = c000 + fr * (c100 - c000);
        float c10 = c010 + fr * (c110 - c010);
        float c01 = c001 + fr * (c101 - c001);
        float c11 = c011 + fr * (c111 - c011);

        float c0 = c00 + fg * (c10 - c00);
        float c1 = c01 + fg * (c11 - c01);
        float val = c0 + fb * (c1 - c0);

        // Strength blend: output = lerp(original, lut_result, strength)
        float orig = (float)src[idx + ch];
        val = orig + strength * (val - orig);
        ((uint8_t*)dst)[idx + ch] = (uint8_t)max(0.0f, min(255.0f, val + 0.5f));
    }
}

// ---- Host API: Generate LUT + Apply ---------------------------------------
void lut3d_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               const uint8_t* d_lut, int lut_res,
               float strength,
               cudaStream_t s) {
    // lut_res is currently always LUT_RES (17) — parameter kept for future flexibility
    dim3 b = block_2d(w, h);
    dim3 g = grid_2d(w, h);
    lut3d_apply_kernel<<<g, b, 0, s>>>(d_src, d_dst, d_lut, w, h, strength);
}

// Generate a built-in LUT on GPU. Caller must cudaFree the returned pointer.
uint8_t* generate_builtin_lut(int preset, cudaStream_t s) {
    uint8_t* d_lut = nullptr;
    cudaMalloc(&d_lut, LUT_RES * LUT_RES * LUT_RES * 3);
    int total = LUT_RES * LUT_RES * LUT_RES;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    generate_lut_kernel<<<grid, block, 0, s>>>(d_lut, preset);
    cudaStreamSynchronize(s);
    return d_lut;
}

} // namespace filters
} // namespace kagerou
