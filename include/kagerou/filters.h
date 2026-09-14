#pragma once
// Kagerou SDK — CUDA filter declarations.

#include "common.h"
#include "config.h"

namespace kagerou {
namespace filters {

// ---- Color Conversion (RGB <-> NV12) --------------------------------------
// NV12 -> RGB (W x H) — full-range BT.709
void nv12_to_rgb(const uint8_t* d_nv12, uint8_t* d_rgb,
                 uint32_t w, uint32_t h, cudaStream_t s = 0);

// NV12 -> RGB (W x H) — limited-range BT.709 (NVDEC output)
void nv12_to_rgb_limited(const uint8_t* d_nv12, uint8_t* d_rgb,
                         uint32_t w, uint32_t h, cudaStream_t s = 0);

// RGB -> NV12 (W x H) — full-range BT.709
void rgb_to_nv12(const uint8_t* d_rgb, uint8_t* d_nv12,
                 uint32_t w, uint32_t h, cudaStream_t s = 0);

// RGB -> NV12 (W x H) — limited-range BT.709
void rgb_to_nv12_limited(const uint8_t* d_rgb, uint8_t* d_nv12,
                         uint32_t w, uint32_t h, cudaStream_t s = 0);

// RGB -> NV12 (W x H) — limited-range BT.601 (SD content: players assume 601
// below 720p; feeding 709 there shifts reds/blues and crushes contrast)
void rgb_to_nv12_limited601(const uint8_t* d_rgb, uint8_t* d_nv12,
                            uint32_t w, uint32_t h, cudaStream_t s = 0);

// BGR -> RGB channel swap (webcam/DirectShow native order). In-place safe.
void bgr_to_rgb(const uint8_t* d_bgr, uint8_t* d_rgb,
                uint32_t w, uint32_t h, cudaStream_t s = 0);

// ---- Color Conversion (YUV420P <-> RGB) -----------------------------------
void yuv420p_to_rgb(const uint8_t* d_y, const uint8_t* d_u, const uint8_t* d_v,
                    uint8_t* d_rgb, uint32_t w, uint32_t h, cudaStream_t s = 0);

void rgb_to_yuv420p(const uint8_t* d_rgb, uint8_t* d_y, uint8_t* d_u, uint8_t* d_v,
                    uint32_t w, uint32_t h, cudaStream_t s = 0);

// ---- HDR Tone Mapping -----------------------------------------------------
// PQ (SMPTE ST2084) -> SDR with Reinhard
void pq_to_sdr(const uint8_t* d_pq, uint8_t* d_rgb,
               uint32_t w, uint32_t h, float peak_nits = 1000.0f,
               cudaStream_t s = 0);

// ---- Scale / Resize -------------------------------------------------------
// Bilinear resize
void resize_bilinear(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t sw, uint32_t sh,
                     uint32_t dw, uint32_t dh,
                     int channels = 3, cudaStream_t s = 0);

// Bicubic resize
void resize_bicubic(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t sw, uint32_t sh,
                    uint32_t dw, uint32_t dh,
                    int channels = 3, cudaStream_t s = 0);

// NV12 bilinear resize (Y + UV planes separately)
void resize_nv12_bilinear(const uint8_t* d_src, uint8_t* d_dst,
                          uint32_t sw, uint32_t sh,
                          uint32_t dw, uint32_t dh,
                          cudaStream_t s = 0);

// NV12 bicubic resize (Y + UV planes separately)
void resize_nv12_bicubic(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t sw, uint32_t sh,
                         uint32_t dw, uint32_t dh,
                         cudaStream_t s = 0);

// ---- Denoise (Bilateral Filter) -------------------------------------------
// RGB bilateral denoise
void denoise_bilateral(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h, int channels = 3,
                       float sigma_spatial = 15.0f,
                       float sigma_color  = 25.0f,
                       int kernel_size = 5,
                       cudaStream_t s = 0);

// NV12 bilateral denoise (Y plane filtered, UV plane light blur)
void denoise_nv12_bilateral(const uint8_t* d_src, uint8_t* d_dst,
                            uint32_t w, uint32_t h,
                            float sigma_spatial = 15.0f,
                            float sigma_color  = 25.0f,
                            int kernel_size = 5,
                            cudaStream_t s = 0);

// ---- Super Resolution (Bicubic + Sharpen) ---------------------------------
// Simple 2x upscale with bicubic + unsharp mask sharpening
void super_res_2x(const uint8_t* d_src, uint8_t* d_dst,
                  uint32_t sw, uint32_t sh,
                  int channels = 3,
                  float sharpen_strength = 0.5f,
                  cudaStream_t s = 0);

// ---- Frame Interpolation --------------------------------------------------
// Blend two frames: output = (1-alpha)*frame_a + alpha*frame_b
void frame_blend(const uint8_t* d_frame_a, const uint8_t* d_frame_b,
                 uint8_t* d_out, uint32_t w, uint32_t h,
                 int channels, float alpha,
                 cudaStream_t s = 0);

// ---- CLAHE (Contrast Limited Adaptive Histogram Equalization) -------------
// NV12: applies CLAHE to Y plane only, light blur on UV
void clahe_nv12(const uint8_t* d_src, uint8_t* d_dst,
                uint32_t w, uint32_t h,
                float clip_limit, int tile_size,
                cudaStream_t s = 0);

// RGB: applies CLAHE to each channel independently
void clahe_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               float clip_limit, int tile_size,
               cudaStream_t s = 0);

// ---- 3D LUT Color Grading ------------------------------------------------
// Apply 3D LUT (17x17x17) to RGB image via trilinear interpolation
// d_lut: GPU buffer, 17*17*17*3 bytes (R,G,B output for each LUT entry)
void lut3d_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               const uint8_t* d_lut, int lut_res,
               float strength,
               cudaStream_t s = 0);

// ---- Gaussian Blur (variable sigma, separable 2-pass) ---------------------
void gaussian_blur(const uint8_t* d_src, uint8_t* d_dst,
                   uint32_t w, uint32_t h, int channels = 3,
                   float sigma = 2.0f,
                   cudaStream_t s = 0);

// ---- Sharpen (unsharp mask) -----------------------------------------------
void sharpen_rgb(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h,
                 float strength = 1.0f,
                 cudaStream_t s = 0);

// ---- Brightness / Contrast ------------------------------------------------
// brightness: -255..255 offset, contrast: 0.1..3.0 multiplier
void brightness_contrast(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t w, uint32_t h,
                         float brightness = 0.0f,
                         float contrast = 1.0f,
                         cudaStream_t s = 0);

// ---- Saturation -----------------------------------------------------------
// factor: 0.0 = grayscale, 1.0 = original, >1.0 = oversaturated
void saturation_rgb(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    float factor = 1.0f,
                    cudaStream_t s = 0);

// ---- Gamma Correction -----------------------------------------------------
// gamma: <1.0 brighten, 1.0 = identity, >1.0 darken
void gamma_rgb(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t w, uint32_t h,
               float gamma = 1.0f,
               cudaStream_t s = 0);

// ---- Vignette (lens darkening) --------------------------------------------
// strength: 0.0 = none, 1.0 = heavy
void vignette_rgb(const uint8_t* d_src, uint8_t* d_dst,
                  uint32_t w, uint32_t h,
                  float strength = 0.5f,
                  cudaStream_t s = 0);

// ---- Film Grain -----------------------------------------------------------
// amount: 0..100 (grain intensity), seed: random seed
void film_grain_rgb(uint8_t* d_data,
                    uint32_t w, uint32_t h,
                    float amount = 20.0f, uint32_t seed = 42,
                    cudaStream_t s = 0);

// ---- Directional Blur (motion blur) ---------------------------------------
void directional_blur(const uint8_t* d_src, uint8_t* d_dst,
                      uint32_t w, uint32_t h, int channels = 3,
                      float angle_deg = 0.0f, int length = 10,
                      cudaStream_t s = 0);

// ---- Edge Detect (Sobel) --------------------------------------------------
// Output is grayscale edge magnitude
void edge_detect_rgb(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t w, uint32_t h,
                     cudaStream_t s = 0);

// ---- White Balance (temperature / tint) -----------------------------------
void white_balance_rgb(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h,
                       float temperature = 0.0f,
                       float tint = 0.0f,
                       cudaStream_t s = 0);

// ---- Lens Distortion (barrel / pincushion) --------------------------------
// k1: negative = barrel, positive = pincushion
void lens_distortion(const uint8_t* d_src, uint8_t* d_dst,
                     uint32_t w, uint32_t h, int channels = 3,
                     float k1 = -0.3f,
                     cudaStream_t s = 0);

// ---- GPU Flip (horizontal / vertical) -------------------------------------
void flip_horizontal_gpu(const uint8_t* d_src, uint8_t* d_dst,
                         uint32_t w, uint32_t h,
                         cudaStream_t s = 0);

void flip_vertical_gpu(const uint8_t* d_src, uint8_t* d_dst,
                       uint32_t w, uint32_t h,
                       cudaStream_t s = 0);

// ---- GPU Compositor (multi-stream layout) -----------------------------------
// Composites N NV12 source frames into a single output at specified positions.
// Each source is bilinear-scaled to its tile size, then blitted into position.
struct CompositeTile {
    const uint8_t* d_src;   // device pointer to source NV12 frame
    uint32_t src_w, src_h;  // source resolution
    uint32_t dst_x, dst_y;  // position in output (top-left corner)
    uint32_t tile_w, tile_h;// tile size in output
    bool     active;        // false = black tile
};

// Composite N tiles into a single NV12 output
void composite_nv12(const CompositeTile* tiles, int count,
                    uint8_t* d_out, uint32_t out_w, uint32_t out_h,
                    cudaStream_t s = 0);

// Compute grid layout for N participants (cols, rows, tile size)
void compute_composite_grid(int n_participants, uint32_t out_w, uint32_t out_h,
                            int& cols, int& rows, uint32_t& tile_w, uint32_t& tile_h);

// ---- Crop (extract region) -------------------------------------------------
void crop_rgb(const uint8_t* d_src, uint8_t* d_dst,
              uint32_t src_w, uint32_t src_h,
              uint32_t dst_w, uint32_t dst_h,
              int crop_x, int crop_y, int channels = 3,
              cudaStream_t s = 0);

void crop_nv12(const uint8_t* d_src, uint8_t* d_dst,
               uint32_t src_w, uint32_t src_h,
               uint32_t dst_w, uint32_t dst_h,
               int crop_x, int crop_y,
               cudaStream_t s = 0);

// ---- Pad (add border) -----------------------------------------------------
void pad_rgb(const uint8_t* d_src, uint8_t* d_dst,
             uint32_t src_w, uint32_t src_h,
             uint32_t dst_w, uint32_t dst_h,
             int pad_x, int pad_y,
             uint8_t pad_r = 0, uint8_t pad_g = 0, uint8_t pad_b = 0,
             cudaStream_t s = 0);

// ---- Chroma Key (green screen removal) ------------------------------------
// key_h_min/max: hue range to key (green ~60-160 degrees)
// key_s_min: minimum saturation to be considered key color
// key_v_min: minimum brightness to be considered key color
// spill_suppress: reduce green spill on foreground (0=off, 1=full)
// bg_r/g/b: replacement background color
void chroma_key_rgb(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    float key_h_min = 60.0f, float key_h_max = 160.0f,
                    float key_s_min = 0.3f, float key_v_min = 0.2f,
                    float spill_suppress = 0.5f,
                    uint8_t bg_r = 0, uint8_t bg_g = 0, uint8_t bg_b = 0,
                    float blend_edge = 1.0f,
                    cudaStream_t s = 0);

// ---- Background Blur (center-weighted focus) ------------------------------
// focus_radius: normalized radius (0..1) of the in-focus area
// blur_strength: max blur radius in pixels for out-of-focus areas
void bg_blur_rgb(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h,
                 float center_x, float center_y,
                 float focus_radius = 0.3f,
                 float blur_strength = 8.0f,
                 cudaStream_t s = 0);

// ---- Temporal Denoise (inter-frame averaging) ----------------------------
// d_prev: previous frame, d_curr: current frame, d_dst: output
// strength: 0.0 = no effect (all current), 1.0 = all previous
void temporal_denoise_rgb(const uint8_t* d_prev, const uint8_t* d_curr,
                          uint8_t* d_dst, uint32_t w, uint32_t h,
                          float strength = 0.25f,
                          cudaStream_t s = 0);

void temporal_denoise_nv12(const uint8_t* d_prev, const uint8_t* d_curr,
                           uint8_t* d_dst, uint32_t w, uint32_t h,
                           float strength = 0.25f,
                           cudaStream_t s = 0);

// ---- HDR Tone Map ---------------------------------------------------------
// method: 0 = Reinhard, 1 = ACES
void hdr_tone_map_rgb(const uint8_t* d_src, uint8_t* d_dst,
                      uint32_t w, uint32_t h,
                      int method = 0, float peak_nits = 1000.0f,
                      cudaStream_t s = 0);

// ---- Temporal Stabilization -----------------------------------------------
// Estimate global motion between two frames and warp to stabilize
// d_accum: 3 ints on GPU [sum_dx, sum_dy, count] for motion estimation
void temporal_stabilize_rgb(const uint8_t* d_prev, const uint8_t* d_curr,
                            uint8_t* d_dst, uint32_t w, uint32_t h,
                            int* d_accum, int block_size = 16, int search_range = 8,
                            cudaStream_t s = 0);

void warp_translate_rgb(const uint8_t* d_src, uint8_t* d_dst,
                        uint32_t w, uint32_t h,
                        int warp_dx, int warp_dy,
                        cudaStream_t s = 0);

} // namespace filters
} // namespace kagerou
