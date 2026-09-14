#pragma once
// Kagerou SDK — pipeline configuration structs.

#include "common.h"
#include <string>

namespace kagerou {

// ---- codec types -----------------------------------------------------------
enum class VideoCodec {
    kH264 = 0,   // AVC
    kH265,       // HEVC
    kVP9,
    kAV1
};

enum class AudioCodec {
    kAAC = 0,
    kMP3,
    kOpus,
    kFLAC,
    kNone
};

enum class RateControl {
    kCBR = 0,    // constant bitrate
    kVBR,        // variable bitrate
    kCQP,        // constant QP
    kCRF         // constant rate factor
};

enum class ColorRange {
    kLimited = 0,  // BT.709 limited range: Y 16-235, UV 16-240 (NVDEC default)
    kFull          // BT.709 full range: Y 0-255, UV 0-255
};

// ---- decoder config --------------------------------------------------------
struct DecoderConfig {
    VideoCodec codec = VideoCodec::kH264;
    uint32_t   max_width  = 3840;
    uint32_t   max_height = 2160;
};

// ---- encoder config --------------------------------------------------------
struct EncoderConfig {
    VideoCodec   codec       = VideoCodec::kH264;
    uint32_t     width       = 1920;
    uint32_t     height      = 1080;
    uint32_t     fps         = 30;
    uint32_t     bitrate_kbps = 5000;
    RateControl  rc          = RateControl::kCBR;
    uint32_t     qp          = 20;          // for CQP/CRF
    uint32_t     gop_size    = 30;          // keyframe interval
    bool         b_frames    = true;        // allow B-frames
    PixelFormat  pixel_fmt   = PixelFormat::kNV12;  // NVENC prefers NV12
};

// ---- filter configs --------------------------------------------------------
struct DenoiseConfig {
    bool    enabled = false;
    float   sigma_spatial = 15.0f;   // spatial sigma (bilateral)
    float   sigma_color  = 25.0f;   // color sigma (bilateral)
    int     kernel_size  = 5;        // must be odd: 3, 5, 7
};

struct SuperResConfig {
    bool    enabled = false;
    float   scale_factor = 2.0f;    // Always 2x (kernel hardcoded)
    float   sharpen_strength = 0.5f;
};

struct FrameInterpConfig {
    bool    enabled = false;
    uint32_t target_fps = 60;        // interpolate to this FPS
};

struct ColorConvertConfig {
    bool    enabled = false;
    PixelFormat target_fmt = PixelFormat::kRGB;
    // BT.709 vs BT.2020 selection
    bool    hdr_input  = false;
    bool    hdr_output = false;
    float   hdr_peak_nits = 1000.0f; // for tone mapping
};

struct ScaleConfig {
    bool    enabled = false;
    uint32_t target_width  = 1920;
    uint32_t target_height = 1080;
    // 0=bilinear, 1=bicubic, 2=nearest
    int     interpolation  = 0;
};

struct CLAHEConfig {
    bool    enabled = false;
    float   clip_limit = 3.0f;     // contrast limit (1.0-10.0, higher=more contrast)
    int     tile_size  = 8;        // tile grid size (NxN tiles)
};

enum class LUTPreset {
    kNone = 0,
    kWarm,           // warm tone: boost reds/yellows
    kCool,           // cool tone: boost blues/cyans
    kCinematic,      // orange-teal split tone
    kVintage,        // faded blacks, warm midtones
    kHighContrast,   // aggressive contrast boost
    kDesaturate,     // partial desaturation (bleach bypass)
};

struct LUTConfig {
    bool       enabled = false;
    LUTPreset  preset  = LUTPreset::kNone;
    float      strength = 1.0f;    // 0.0=no effect, 1.0=full LUT
    // Built-in 17x17x17 LUT generated on GPU from preset
    static const int LUT_RES = 17;
};

struct BlurConfig {
    bool  enabled = false;
    float sigma = 2.0f;
};

struct SharpenConfig {
    bool  enabled = false;
    float strength = 1.0f;
};

struct BrightContrastConfig {
    bool  enabled = false;
    float brightness = 0.0f;    // -255..255
    float contrast   = 1.0f;    // 0.1..3.0
};

struct SaturationConfig {
    bool  enabled = false;
    float factor = 1.4f;        // 0=gray, 1=normal, >1=saturated
};

struct GammaConfig {
    bool  enabled = false;
    float gamma = 0.8f;         // <1 brighten, >1 darken
};

struct VignetteConfig {
    bool  enabled = false;
    float strength = 0.6f;      // 0=none, 1=heavy
};

struct FilmGrainConfig {
    bool    enabled = false;
    float   amount = 25.0f;     // 0..100
    uint32_t seed = 42;
};

struct EdgeDetectConfig {
    bool enabled = false;
};

struct WhiteBalanceConfig {
    bool  enabled = false;
    float temperature = 15.0f;  // +warm, -cool
    float tint = 5.0f;          // +green, -magenta
};

struct LensDistortionConfig {
    bool  enabled = false;
    float k1 = -0.3f;           // negative=barrel, positive=pincushion
};

struct FlipConfig {
    bool  enabled = false;
    int   mode = 1;             // 1=horizontal, 2=vertical
};

struct DirectionalBlurConfig {
    bool  enabled = false;
    float angle = 0.0f;         // degrees
    int   length = 10;          // blur kernel length
};

struct CropConfig {
    bool    enabled = false;
    int     x = 0;
    int     y = 0;
    uint32_t width = 0;         // 0 = auto (no crop)
    uint32_t height = 0;
};

struct PadConfig {
    bool    enabled = false;
    int     top = 0, bottom = 0, left = 0, right = 0;
    uint8_t pad_r = 0, pad_g = 0, pad_b = 0;
};

struct ChromaKeyConfig {
    bool    enabled = false;
    float   hue_min = 60.0f;    // green screen hue range
    float   hue_max = 160.0f;
    float   sat_min = 0.3f;
    float   val_min = 0.2f;
    float   spill_suppress = 0.5f;
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
};

struct TemporalDenoiseConfig {
    bool    enabled = false;
    float   strength = 0.25f;   // 0=off, 1=max
};

struct TemporalStabConfig {
    bool    enabled = false;
    int     block_size = 16;
    int     search_range = 8;
    float   smooth_factor = 0.8f; // exponential smoothing of motion vectors
};

struct BackgroundBlurConfig {
    bool    enabled = false;
    float   focus_radius = 0.3f;
    float   blur_strength = 8.0f;
};

struct HDRConfig {
    bool    enabled = false;
    int     method = 0;         // 0=Reinhard, 1=ACES
    float   peak_nits = 100.0f; // SDR default; set higher for true HDR sources
};

// ---- pipeline config -------------------------------------------------------
struct PipelineConfig {
    DecoderConfig  decoder;
    EncoderConfig  encoder;
    DenoiseConfig  denoise;
    SuperResConfig super_res;
    FrameInterpConfig frame_interp;
    ColorConvertConfig color_convert;
    ScaleConfig    scale;
    CLAHEConfig    clahe;
    LUTConfig      lut;
    BlurConfig     blur;
    SharpenConfig  sharpen;
    BrightContrastConfig bright_contrast;
    SaturationConfig saturation;
    GammaConfig    gamma;
    VignetteConfig vignette;
    FilmGrainConfig film_grain;
    EdgeDetectConfig edge_detect;
    WhiteBalanceConfig white_balance;
    LensDistortionConfig lens_distortion;
    FlipConfig     flip;
    DirectionalBlurConfig dir_blur;
    CropConfig     crop;
    PadConfig      pad;
    ChromaKeyConfig chroma_key;
    TemporalDenoiseConfig temporal_denoise;
    TemporalStabConfig temporal_stab;
    BackgroundBlurConfig bg_blur;
    HDRConfig      hdr;

    // AI filters (ONNX Runtime GPU)
    bool           ai_denoise_enabled   = false;
    bool           ai_bg_removal_enabled = false;
    bool           ai_depth_enabled     = false;
    bool           ai_flow_enabled      = false;
    bool           ai_pose_enabled      = false;
    bool           ai_matting_enabled   = false;
    bool           ai_face_enabled      = false;
    bool           ai_hands_enabled     = false;
    bool           ai_gaze_enabled      = false;
    bool           ai_lowlight_enabled  = false;
    bool           ai_anime_enabled     = false;
    bool           ai_detect_enabled    = false;

    bool           ai_autoframe_enabled = false;
    uint8_t        ai_bg_color[3]       = {0, 0, 0}; // solid bg color for bg removal

    // model paths (empty = use defaults from models/ directory)
    std::string    ai_denoise_model;
    std::string    ai_depth_model;
    std::string    ai_flow_model;
    std::string    ai_pose_model;
    std::string    ai_matting_model;
    std::string    ai_face_model;
    std::string    ai_palm_model;
    std::string    ai_handlm_model;
    std::string    ai_iris_model;
    std::string    ai_lowlight_model;
    std::string    ai_anime_model;
    std::string    ai_detect_model;

    // global
    int            device_id = 0;       // GPU to use
    bool           verbose   = false;
    uint32_t       warmup_frames = 10;  // warmup GPU before benchmark
    ColorRange    color_range = ColorRange::kLimited;  // NVDEC outputs limited range
};

} // namespace kagerou
