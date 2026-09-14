#include "kagerou/filters_common.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace kagerou {
namespace filters {

// Y plane compositor: bilinear sample from source, write to output position
__global__ void composite_nv12_y_kernel(
    uint8_t* __restrict__ dst,
    uint32_t out_w, uint32_t out_h,
    const uint8_t* __restrict__ src,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_x, uint32_t dst_y,
    uint32_t tile_w, uint32_t tile_h)
{
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= tile_w || dy >= tile_h) return;

    // Bilinear sample from source
    float fx = (float)dx * (float)src_w / (float)tile_w;
    float fy = (float)dy * (float)src_h / (float)tile_h;

    uint32_t x0 = (uint32_t)fx;
    uint32_t y0 = (uint32_t)fy;
    uint32_t x1 = (x0 + 1 < src_w) ? x0 + 1 : x0;
    uint32_t y1 = (y0 + 1 < src_h) ? y0 + 1 : y0;
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float val = src[y0 * src_w + x0] * (1-wx) * (1-wy)
              + src[y0 * src_w + x1] * wx * (1-wy)
              + src[y1 * src_w + x0] * (1-wx) * wy
              + src[y1 * src_w + x1] * wx * wy;

    uint32_t ox = dst_x + dx;
    uint32_t oy = dst_y + dy;
    if (ox < out_w && oy < out_h)
        dst[oy * out_w + ox] = (uint8_t)(val + 0.5f);
}

// UV plane compositor
__global__ void composite_nv12_uv_kernel(
    uint8_t* __restrict__ dst,
    uint32_t out_w, uint32_t out_h,
    const uint8_t* __restrict__ src,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_x, uint32_t dst_y,
    uint32_t tile_w, uint32_t tile_h)
{
    uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t uv_tile_w = tile_w;
    uint32_t uv_tile_h = tile_h / 2;
    if (dx >= uv_tile_w / 2 || dy >= uv_tile_h) return;

    // UV is interleaved at half resolution
    uint32_t px = dx * 2;
    uint32_t py = dy;

    float fx = (float)px * (float)src_w / (float)uv_tile_w;
    float fy = (float)py * (float)(src_h / 2) / (float)uv_tile_h;

    uint32_t x0 = (uint32_t)fx;
    uint32_t y0 = (uint32_t)fy;
    uint32_t x0e = x0 & ~1u;
    uint32_t x1e = ((x0e + 2) < src_w) ? x0e + 2 : x0e;
    uint32_t y1 = (y0 + 1 < src_h / 2) ? y0 + 1 : y0;
    float wx = fx - (float)x0e;
    float wy = fy - (float)y0;

    uint32_t src_uv_offset = src_w * src_h;
    for (int ch = 0; ch < 2; ++ch) {
        float v00 = src[src_uv_offset + y0 * src_w + x0e + ch];
        float v10 = src[src_uv_offset + y0 * src_w + x1e + ch];
        float v01 = src[src_uv_offset + y1 * src_w + x0e + ch];
        float v11 = src[src_uv_offset + y1 * src_w + x1e + ch];

        float val = v00 * (1-wx) * (1-wy)
                  + v10 * wx * (1-wy)
                  + v01 * (1-wx) * wy
                  + v11 * wx * wy;

        uint32_t dst_uv_offset = out_w * out_h;
        uint32_t ox = dst_x + px + ch;
        uint32_t oy = dst_y / 2 + py;
        if (ox < out_w && oy < out_h / 2)
            dst[dst_uv_offset + oy * out_w + ox] = (uint8_t)(val + 0.5f);
    }
}

// Host function: composite N tiles into output
// tiles: array of tile descriptors (up to MAX_TILES)
// out_w/out_h: output dimensions (cols*tile_w, rows*tile_h)
void composite_nv12(const CompositeTile* tiles, int count,
                    uint8_t* d_out, uint32_t out_w, uint32_t out_h,
                    cudaStream_t s)
{
    // Zero output
    cudaMemsetAsync(d_out, 0, out_w * out_h * 3 / 2, s);

    for (int i = 0; i < count; ++i) {
        if (!tiles[i].active || !tiles[i].d_src) continue;

        // Y plane
        composite_nv12_y_kernel<<<grid_2d(tiles[i].tile_w, tiles[i].tile_h),
                                  block_2d(tiles[i].tile_w, tiles[i].tile_h), 0, s>>>(
            d_out, out_w, out_h,
            tiles[i].d_src, tiles[i].src_w, tiles[i].src_h,
            tiles[i].dst_x, tiles[i].dst_y,
            tiles[i].tile_w, tiles[i].tile_h);

        // UV plane
        composite_nv12_uv_kernel<<<grid_2d(tiles[i].tile_w / 2, tiles[i].tile_h / 2),
                                   block_2d(tiles[i].tile_w / 2, tiles[i].tile_h / 2), 0, s>>>(
            d_out, out_w, out_h,
            tiles[i].d_src, tiles[i].src_w, tiles[i].src_h,
            tiles[i].dst_x, tiles[i].dst_y,
            tiles[i].tile_w, tiles[i].tile_h);
    }
}

// ---- Helper: compute grid layout for N participants -------------------------
// Returns cols x rows that best fits N tiles in a grid
void compute_composite_grid(int n_participants, uint32_t out_w, uint32_t out_h,
                            int& cols, int& rows, uint32_t& tile_w, uint32_t& tile_h) {
    if (n_participants <= 1) { cols = 1; rows = 1; }
    else if (n_participants <= 2) { cols = 2; rows = 1; }
    else if (n_participants <= 4) { cols = 2; rows = 2; }
    else if (n_participants <= 6) { cols = 3; rows = 2; }
    else if (n_participants <= 9) { cols = 3; rows = 3; }
    else { cols = 4; rows = 4; }

    tile_w = out_w / cols;
    tile_h = out_h / rows;
}

} // namespace filters
} // namespace kagerou
