#pragma once
// ============================================================================
// Kagerou AI Filters — Public API
// All AI filters run on GPU via ONNX Runtime CUDA execution provider
// ============================================================================

#include <cstdint>
#include <cuda_runtime.h>

namespace kagerou {
namespace ai {

// ============================================================================
// AI Denoise — FastDVDNet temporal noise reduction
// Input:  5 consecutive RGB frames (d_prev4, d_prev3, d_prev2, d_prev1, d_curr)
// Output: denoised RGB frame (d_dst)
// Each frame: w*h*3 bytes (RGB uint8)
// ============================================================================
bool ai_denoise_init(const char* model_path, int device_id = 0);
bool ai_denoise(const uint8_t* d_prev4, const uint8_t* d_prev3,
                const uint8_t* d_prev2, const uint8_t* d_prev1,
                const uint8_t* d_curr, uint8_t* d_dst,
                uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Background Removal — RMBG-2.0 segmentation
// Input:  RGB frame (d_src)
// Output: alpha mask (d_alpha, w*h bytes, 0=bg 255=fg)
// Also available: composited output with blurred background
// ============================================================================
bool ai_bg_removal_init(const char* model_path, int device_id = 0);
bool ai_bg_removal(const uint8_t* d_src, uint8_t* d_alpha,
                   uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Object Detection — YOLOv8n (Ultralytics, 640x640, 80 COCO classes)
// Input:  RGB frame; boxes returned in input pixel coordinates.
// Export stores sigmoided class scores, so no sigmoid is applied here.
// ============================================================================
struct DetectBox {
    float x1, y1, x2, y2;
    float score;
    int cls; // 0..79 (COCO)
};

bool ai_detect_init(const char* model_path, int device_id = 0);
// Returns number of objects (<= max_boxes).
int ai_detect(const uint8_t* d_src, uint32_t w, uint32_t h,
              DetectBox* boxes_out, int max_boxes, cudaStream_t stream = 0);
// COCO class name (lowercase, e.g. "person") + per-class overlay color.
const char* ai_detect_class_name(int cls);
void ai_detect_color(int cls, uint8_t* rgb_out3);

// GPU text label (3x5 bitmap font): draws bg-filled string at (x,y).
void launch_draw_text(uint8_t* rgb, uint32_t w, uint32_t h,
                      int x, int y, const char* str,
                      int scale, uint8_t fr, uint8_t fg, uint8_t fb,
                      uint8_t br, uint8_t bg, uint8_t bb,
                      cudaStream_t s = 0);

// ============================================================================
// AI Face Detection — YuNet (OpenCV Zoo, 640x640)
// Input:  RGB frame; boxes returned in input pixel coordinates.
// ============================================================================
struct FaceBox {
    float x1, y1, x2, y2;
    float score;
};

bool ai_face_init(const char* model_path, int device_id = 0);
// Returns number of faces (<= max_boxes).
int ai_face(const uint8_t* d_src, uint32_t w, uint32_t h,
            FaceBox* boxes_out, int max_boxes, cudaStream_t stream = 0);

// ============================================================================
// AI Hand Tracking — BlazePalm detect + hand landmark (21 joints)
// joints[i] in input pixel coordinates. Returns number of hands.
// ============================================================================
struct HandJoints {
    float x[21], y[21];
    float score;
};

bool ai_hands_init(const char* palm_path, const char* landmark_path, int device_id = 0);
int ai_hands(const uint8_t* d_src, uint32_t w, uint32_t h,
             HandJoints* hands_out, int max_hands, cudaStream_t stream = 0);

// ============================================================================
// AI Gaze — YuNet face box + MediaPipe iris per eye.
// Returns eyes found (<= max_eyes). (cx,cy) = iris center in frame pixels,
// (dx,dy) = gaze direction (unit-ish, iris offset from eye center).
// ============================================================================
struct GazeEye {
    float cx, cy, dx, dy;
    float er;    // eye radius px (for redirect mask)
    float score;
};

bool ai_gaze_init(const char* iris_path, int device_id = 0);
int ai_gaze(const uint8_t* d_src, uint32_t w, uint32_t h,
            GazeEye* eyes_out, int max_eyes, cudaStream_t stream = 0);

// Eye-contact redirect: shifts eyeball pixels toward the lens inside a
// feathered ellipse per eye. d_src/d_dst must be DISTINCT buffers.
// strength 0..1 (cap ~0.6 — beyond looks demonic).
bool ai_eye_correct(const uint8_t* d_src, uint8_t* d_dst,
                    uint32_t w, uint32_t h,
                    const GazeEye* eyes, int neyes, float strength,
                    cudaStream_t stream = 0);

// ============================================================================
// AI Low-Light — Zero-DCE (single frame enhance)
// ============================================================================
bool ai_lowlight_init(const char* model_path, int device_id = 0);
bool ai_lowlight(const uint8_t* d_src, uint8_t* d_dst,
                 uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Style — AnimeGANv3 Hayao (256px stylize + upscale)
// ============================================================================
bool ai_anime_init(const char* model_path, int device_id = 0);
bool ai_anime(const uint8_t* d_src, uint8_t* d_dst,
              uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Matting — RVM MobileNetV3, stateless (bg blur composite)
// ============================================================================
bool ai_matting_init(const char* model_path, int device_id = 0, bool use_trt = true);
bool ai_matting(const uint8_t* d_src, uint8_t* d_dst,
                uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Pose Estimation — MediaPipe Pose (33 3D landmarks)
// ============================================================================
struct PoseLandmark {
    float x, y, z;
    float visibility;
};

bool ai_pose_init(const char* model_path, int device_id = 0);
bool ai_pose(const uint8_t* d_src, uint32_t w, uint32_t h,
             PoseLandmark* landmarks_out, cudaStream_t stream = 0);

// ============================================================================
// AI Depth Estimation — Depth Anything V2
// Input:  RGB frame
// Output: depth map (d_depth, w*h floats, relative depth)
// ============================================================================
bool ai_depth_init(const char* model_path, int device_id = 0);
bool ai_depth(const uint8_t* d_src, float* d_depth,
              uint32_t w, uint32_t h, cudaStream_t stream = 0);

// ============================================================================
// AI Optical Flow — RAFT small
// Input:  two consecutive RGB frames
// Output: flow field (d_flow, w*h*2 floats: dx,dy per pixel)
// ============================================================================
bool ai_flow_init(const char* model_path, int device_id = 0);
bool ai_flow(const uint8_t* d_frame1, const uint8_t* d_frame2,
             float* d_flow, uint32_t w, uint32_t h, cudaStream_t stream = 0);

} // namespace ai
} // namespace kagerou
