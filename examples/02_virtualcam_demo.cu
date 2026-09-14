// Kagerou Virtual Camera -- Full AI + Non-AI Pipeline
// Capture from real camera -> GPU filters -> Virtual Camera output
// Build: build.bat virtualcam sdk minimp4 onnx
// Run:   bin\kagerou_virtualcam.exe

#if !defined(KAGEROU_USE_NVDEC_SDK)
#error "Need SDK: build.bat all sdk"
#endif

#include <windows.h>
#include <commdlg.h>
#include <dshow.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <mmreg.h>
#include <ksmedia.h>
#include <functiondiscoverykeys_devpkey.h>
#include <propvarutil.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>

#include "kagerou/common.h"
#include "kagerou/filters.h"
#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/gesture.h"
#include "kagerou/pip.h"
#include "../screen_cap/screen_shm.h"
#if !defined(KAGEROU_NO_VCODEC_SDK)
#include <cuda.h> // driver API: retain the primary context to share with NVENC
#endif
#include "kagerou/ai/ort_wrapper.h"
#include "../virtualcam/kagerou_vcam_shm.h"

#include "../src/filters/color_convert.cu"
#include "../src/filters/denoise.cu"
#include "../src/filters/clahe.cu"
#include "../src/filters/scale.cu"
#include "../src/filters/super_res.cu"
#include "../src/filters/frame_interp.cu"
#include "../src/filters/transforms.cu"
#include "../src/filters/lut.cu"
#include "../src/filters/creative.cu"
#include "../src/ai/ort_wrapper.cu"
#include "../src/ai/ai_preprocess.cu"
#include "../src/ai/ai_denoise.cu"
#include "../src/ai/ai_depth.cu"
#include "../src/ai/ai_flow.cu"
#include "../src/ai/ai_pose_estimation.cu"
#include "../src/ai/ai_lowlight.cu"
#include "../src/ai/ai_face.cu"
#include "../src/ai/ai_yolo.cu"
#include "../src/ai/ai_anime.cu"
#include "../src/ai/ai_matting.cu"
#include "../src/ai/ai_hands.cu"
#include "../src/ai/ai_gaze.cu"
#include "../src/encoder.cu"
#include "kagerou/fileio.h"

static std::string get_exe_dir() {
    char path[MAX_PATH] = {};
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string s(path);
    size_t pos = s.find_last_of("\\/");
    return (pos != std::string::npos) ? s.substr(0, pos) : ".";
}

static std::string resolve_model(const char* rel) {
    std::string p = get_exe_dir() + "\\..\\" + rel;
    for (auto& c : p) if (c == '/') c = '\\';
    char full[MAX_PATH] = {};
    DWORD n = GetFullPathNameA(p.c_str(), MAX_PATH, full, nullptr);
    if (n > 0 && n < MAX_PATH) return std::string(full);
    return p;
}

static bool file_exists(const char* p) {
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

using namespace kagerou;

// ============================================================================
// State
// ============================================================================
static uint32_t g_cap_w = 640, g_cap_h = 480;
// Max program-feed frame (screen/tutorial up to 1080p for readable text).
// All fixed scratch buffers are sized for this so any source fits.
// NOTE: players assume BT.601 below 720p and BT.709 at/above it, so the
// record path picks the NV12 matrix by take height (see rec_write_frame).
static const uint32_t FEED_MAX_W = 1920, FEED_MAX_H = 1080;
static const size_t FEED_MAX_RGB = (size_t)FEED_MAX_W * FEED_MAX_H * 3;
static std::atomic<bool> g_cam_alive{false};
static std::atomic<bool> g_cam_black{false}; // lens shutter/cover: frames are ~zero
// ---- tutorial source: screen capture (fit in FEED_MAX box) + camera PiP.
// 0 camera, 1 screen, 2 tutorial.
static int g_src_mode = 0;
static std::thread g_screen_thread;
static std::atomic<bool> g_screen_run{false};
static std::mutex g_screen_mtx;
static uint8_t* g_screen_buf = nullptr; // host RGB, <=640x480 box
static uint32_t g_screen_w = 0, g_screen_h = 0;
static std::atomic<uint64_t> g_screen_seq{0};
// Capture diagnostics (read by take.log): last blit result + mean brightness.
static std::atomic<int> g_screen_blt{0};   // 1 = BitBlt ok, 2 = StretchBlt ok, <0 = failed
static std::atomic<unsigned long> g_screen_blterr{0};
static std::atomic<float> g_screen_mean{0};
static std::atomic<float> g_screen_fps{0}; // capture rate EMA (which side lags?)
static std::atomic<uint64_t> g_cam_seq{0}; // bumps per fresh camera frame
static std::atomic<bool> g_running{true};
static uint8_t* g_cam_buf = nullptr;
static uint8_t* g_display_out = nullptr;
static std::mutex g_frame_mutex;

// Preview-only OSD border (never touches the take): 0 none, 1 amber hold,
// 2 green confirm. Defined early: the GPU thread runs before the RECORD tab
// section below but consumes this after every display memcpy.
static std::atomic<int> g_rec_osd{0};
static std::chrono::steady_clock::time_point g_rec_osd_t0 = std::chrono::steady_clock::now();
static void rec_osd_border(uint8_t* bgr, uint32_t w, uint32_t h, int mode) {
    uint8_t b = 0, g = 0, r = 0;
    if (mode == 1) { b = 80; g = 160; r = 255; }       // amber: hold counting
    else if (mode == 2) { b = 80; g = 220; r = 80; }   // green: action fired
    else return;
    const uint32_t T = 6;
    auto px = [&](uint32_t x, uint32_t y) {
        size_t o = ((size_t)y * w + x) * 3;
        bgr[o] = b; bgr[o+1] = g; bgr[o+2] = r;
    };
    for (uint32_t y = 0; y < T && y < h; y++)
        for (uint32_t x = 0; x < w; x++) px(x, y);
    for (uint32_t y = (h > T ? h - T : 0); y < h; y++)
        for (uint32_t x = 0; x < w; x++) px(x, y);
    for (uint32_t y = T; y + T < h; y++) {
        for (uint32_t x = 0; x < T && x < w; x++) px(x, y);
        for (uint32_t x = (w > T ? w - T : 0); x < w; x++) px(x, y);
    }
}

static uint8_t* g_d_rgb = nullptr;
static uint8_t* g_d_tmp = nullptr;
static uint8_t* g_d_ai_s0 = nullptr; // small-frame staging for heavy AI
static uint8_t* g_d_ai_s1 = nullptr;
static uint8_t* g_d_camdet = nullptr; // camera RGB for tutorial tracking
static uint32_t g_det_iw = 640, g_det_ih = 480; // detector input dims (SR-aware overlay mapping)
static uint32_t g_denoise_last_w = 0, g_denoise_last_h = 0;
static uint8_t* g_d_sr_out = nullptr;
static uint8_t* g_d_lut = nullptr;
static int g_lut_preset = 0;
static cudaStream_t g_stream = 0;

static HWND g_hwnd = nullptr;
static HFONT g_font_ui = 0;
static HFONT g_font_st = 0;
static HFONT g_font_tab = 0;
static HFONT g_font_emoji = 0;

static bool g_denoise = false, g_clahe = false, g_sr = false;
static bool g_lut_on = false;
static bool g_blur = false, g_sharp = false;
static bool g_bright = false, g_sat = false, g_gamma = false;
static bool g_vignette = false, g_grain = false, g_edge = false;
static bool g_wb = false, g_flip = false, g_lens = false;
static int g_flip_mode = 0;
// second-wave filters (full CLI parity)
static bool g_interp = false, g_bgblur = false;
static bool g_tempden = false, g_tempstab = false;
static bool g_chroma = false;
static bool g_dirblur = false, g_hdr = false, g_cropzoom = false;
static int g_dirblur_idx = 0; // 1: 45deg/20px, 2: 135deg/12px
static int g_hdr_method = 0;  // 1 Reinhard, 2 ACES
static int g_zoom_idx = 0;    // 1:1.25x, 2:1.5x, 3:2x center zoom
static uint8_t* g_d_temporal_prev = nullptr;
static uint8_t* g_d_stage = nullptr;
static uint8_t* g_d_zoom = nullptr;
static int* g_d_motion_accum = nullptr;
static bool g_temporal_has_prev = false;
static int g_stab_dx = 0, g_stab_dy = 0;

static bool g_ai_denoise = false;
static bool g_ai_depth = false;
static bool g_ai_flow = false;
static bool g_ai_pose = false;
static bool g_ai_matting = false;
static bool g_ai_face = false;
static bool g_ai_hands = false;
static bool g_ai_gaze = false;
static bool g_ai_lowlight = false;
static bool g_ai_anime = false;
static bool g_ai_detect = false;
static bool g_autoframe = false;
static int g_af_idx = 0; // 0 off, 1 wide 3.0x, 2 med 2.5x, 3 tight 2.0x
// auto-frame smoothed window (frame px) + own face cache
static float g_af_x = 0, g_af_y = 0, g_af_w = 0, g_af_h = 0;
static int g_af_fw = 0, g_af_fh = 0;
static bool g_af_init = false;
static kagerou::ai::FaceBox g_af_boxes[4] = {};
// Tutorial-mode auto-frame: crop the camera to the face (camera coords, from
// the last tutorial face inference) before the PiP composite, so the corner
// box follows you instead of showing the whole frame.
static float g_afp_x = 0, g_afp_y = 0, g_afp_w = 0, g_afp_h = 0;
static int g_afp_fw = 0, g_afp_fh = 0;
static bool g_afp_init = false;
static uint8_t* g_afp_cam = nullptr;
static size_t g_afp_cam_sz = 0;

static std::atomic<bool> g_ai_denoise_ready{false};
static std::atomic<bool> g_ai_depth_ready{false};
static std::atomic<bool> g_ai_flow_ready{false};
static std::atomic<bool> g_ai_pose_ready{false};
static std::atomic<bool> g_ai_matting_ready{false};
static std::atomic<bool> g_ai_face_ready{false};
static std::atomic<bool> g_ai_hands_ready{false};
static std::atomic<bool> g_ai_gaze_ready{false};
static std::atomic<bool> g_ai_lowlight_ready{false};
static std::atomic<bool> g_ai_anime_ready{false};
static std::atomic<bool> g_ai_detect_ready{false};

// cached detections (drawn every frame, refreshed at inference cadence)
static kagerou::ai::FaceBox g_face_boxes[4] = {};
static kagerou::ai::DetectBox g_det_boxes[16] = {};
static kagerou::ai::HandJoints g_hand_joints[2] = {};
static kagerou::ai::GazeEye g_gaze_eyes[2] = {};
// True until background init + warmup finishes. The GPU thread skips AI
// while set, so a toggle during warmup can't contend on ORT sessions.
static std::atomic<bool> g_ai_warming{true};

static float* g_d_ai_depth_buf = nullptr;
static uint8_t* g_d_ai_vis = nullptr;
static uint8_t* g_d_ai_matt = nullptr;
static bool g_matt_cached = false;
static uint8_t* g_d_flow_prev = nullptr;
static uint8_t* g_d_flow_curr = nullptr;
static float* g_d_flow_buf = nullptr;
static bool g_flow_has_prev = false;
static kagerou::ai::PoseLandmark g_pose_landmarks[33];

static const int DENOISE_RING_SIZE = 5;
static uint8_t* g_d_denoise_ring[DENOISE_RING_SIZE] = {};
static int g_denoise_ring_idx = 0;
static int g_denoise_ring_count = 0;

static VirtualCamWriter g_vcam;
static bool g_vcam_active = false;
static uint8_t* g_d_nv12 = nullptr;
static uint8_t* g_h_nv12 = nullptr;

static double g_fps = 0;
static int g_frame_count = 0;
static int g_disp_w = 640, g_disp_h = 480;

static COLORREF CLR_BG = RGB(14, 14, 20);
static COLORREF CLR_PANEL = RGB(22, 22, 32);
static COLORREF CLR_PANEL2 = RGB(28, 28, 40);
static COLORREF CLR_LINE = RGB(48, 48, 66);
static COLORREF CLR_TEXT = RGB(225, 225, 238);
static COLORREF CLR_DIM = RGB(130, 130, 152);
static COLORREF CLR_ACCENT = RGB(0, 180, 255);
static COLORREF CLR_AI = RGB(255, 122, 61);
static COLORREF CLR_OK = RGB(46, 204, 113);
static COLORREF CLR_BAD = RGB(231, 76, 60);

// ============================================================================
// Sidebar button model -- one shared layout for paint + hit testing
// ============================================================================
static const int SIDE_W = 252;
static const int STATUS_H = 28;
static const int TAB_H = 34;
static const int BTN_H = 36;
static const int BTN_GAP = 6;
static const int SEC_GAP = 10;
static const int SEC_TITLE_H = 22;

struct BtnDef {
    const char* name;    // short name
    const char* key;     // shortcut hint
    bool* state;
    int section;         // 0 enhance, 1 color, 2 fx, 3 ai
    int special;         // 0 toggle, 1 lut, 2 flip, 3 dirblur, 4 hdr, 5 zoom
};
static BtnDef g_btns[] = {
    {"Denoise",  "D", &g_denoise, 0, 0}, {"CLAHE",   "C", &g_clahe, 0, 0},
    {"SuperRes", "S", &g_sr,      0, 0}, {"Blur",    "B", &g_blur,  0, 0},
    {"Sharp",    "N", &g_sharp,   0, 0}, {"Interp",  "I", &g_interp,0, 0},
    {"TempDenoise","U", &g_tempden,0, 0}, {"TempStab","Y", &g_tempstab,0, 0},
    {"LUT",      "L", &g_lut_on,  1, 1}, {"Bright/Contrast", "1", &g_bright, 1, 0},
    {"Saturation","2", &g_sat,    1, 0}, {"Gamma",   "3", &g_gamma, 1, 0},
    {"White Bal.","W", &g_wb,     1, 0}, {"HDR",     "H", &g_hdr,   1, 4},
    {"Vignette", "V", &g_vignette,2, 0}, {"Film Grain","G", &g_grain, 2, 0},
    {"Edge",     "E", &g_edge,    2, 0}, {"Flip",    "F", &g_flip,   2, 2},
    {"Lens",     "Q", &g_lens,    2, 0}, {"DirBlur", "O", &g_dirblur,2, 3},
    {"BgBlur",   "M", &g_bgblur,  2, 0}, {"Chroma",  "K", &g_chroma, 2, 0},
    {"CropZoom", "X", &g_cropzoom,2, 5},
    {"AI Denoise","7", &g_ai_denoise, 3, 0}, {"AI Depth","8", &g_ai_depth, 3, 0},
    {"AI Flow",  "9", &g_ai_flow, 3, 0}, {"AI Pose", "0", &g_ai_pose, 3, 0},
    {"AI Matting","A", &g_ai_matting, 3, 0}, {"AI Face","P", &g_ai_face, 3, 0},
    {"AI Hands","J", &g_ai_hands, 3, 0}, {"AI Gaze","Z", &g_ai_gaze, 3, 0},
    {"AI LowLight","5", &g_ai_lowlight, 3, 0}, {"AI Anime","6", &g_ai_anime, 3, 0},
    {"AI Detect","F10", &g_ai_detect, 3, 0},
    {"AutoFrame","F7", &g_autoframe, 3, 6},
};
static const int NBTN = sizeof(g_btns)/sizeof(g_btns[0]);
static const char* SEC_NAMES[] = {"ENHANCE", "COLOR", "EFFECTS", "AI MODELS"};

struct BtnRect { int x, y, w, h; };
static int g_hover = -1;
static int g_pressed = -1;
static int g_scroll = 0;
static int g_content_h = 0;

static const int SIDE_TOP = 70; // buttons live below the header pills

// Computes sidebar button + section-header rects for the current client
// size. The list is taller than the window, so it scrolls with the wheel.
static int layout_buttons(int client_h, BtnRect* out, BtnRect* sec_out) {
    int y = TAB_H + SIDE_TOP + 8 - g_scroll;
    int x = 12, w = SIDE_W - 24;
    int last_sec = -1;
    for (int s = 0; s < 4; s++) sec_out[s] = {0,0,0,0};
    for (int i = 0; i < NBTN; i++) {
        if (g_btns[i].section != last_sec) {
            last_sec = g_btns[i].section;
            y += (i == 0) ? 0 : SEC_GAP;
            sec_out[last_sec] = {x, y, w, SEC_TITLE_H};
            y += SEC_TITLE_H;
        }
        out[i] = {x, y, w, BTN_H};
        y += BTN_H + BTN_GAP;
    }
    g_content_h = y + g_scroll;
    return NBTN;
}

static bool rect_visible(BtnRect r, int client_h) {
    if (r.w <= 0) return false;
    int bottom = client_h - STATUS_H - 8;
    return (r.y + r.h >= TAB_H + SIDE_TOP && r.y <= bottom);
}

static int max_scroll(int client_h) {
    int visible = (client_h - STATUS_H - 8) - (TAB_H + SIDE_TOP + 8);
    int m = g_content_h - (TAB_H + SIDE_TOP + 8) - visible;
    return m > 0 ? m : 0;
}

// AI button state for the UI: model file loaded + warmup finished.
// 0 = failed/missing (never usable), 1 = warming (wait), 2 = ready.
static int ai_button_state(const BtnDef& b) {
    bool r = false;
    if (b.state == &g_ai_denoise) r = g_ai_denoise_ready.load();
    else if (b.state == &g_ai_depth) r = g_ai_depth_ready.load();
    else if (b.state == &g_ai_flow) r = g_ai_flow_ready.load();
    else if (b.state == &g_ai_pose) r = g_ai_pose_ready.load();
    else if (b.state == &g_ai_matting) r = g_ai_matting_ready.load();
    else if (b.state == &g_ai_face) r = g_ai_face_ready.load();
    else if (b.state == &g_ai_hands) r = g_ai_hands_ready.load();
    else if (b.state == &g_ai_gaze) r = g_ai_gaze_ready.load();
    else if (b.state == &g_ai_lowlight) r = g_ai_lowlight_ready.load();
    else if (b.state == &g_ai_anime) r = g_ai_anime_ready.load();
    else if (b.state == &g_ai_detect) r = g_ai_detect_ready.load();
    else if (b.state == &g_autoframe) r = g_ai_face_ready.load();
    else return 2;
    if (!r) return 0;
    return g_ai_warming.load() ? 1 : 2;
}

// Forward declarations
static void gen_test_pattern(uint8_t* rgb, uint32_t w, uint32_t h, int f);
// GAME tab (defined in 03_game_mode.inl, included before draw_tabs)
static void game_tick_gpu();
static void game_draw_canvas();
static void render_game_panel(HDC hdc, int W, int H);
static int game_hittest(int mx, int my, int W, int H);
static void game_fire(int id);
static void game_kill();
static void game_on_key(WPARAM w);
static bool game_is_on();
// Windows DIBs (StretchDIBits) expect BGR byte order, but our buffers are
// RGB. Swap R/B in place for display only (GPU/vcam paths stay RGB).
static void swap_rb_inplace(uint8_t* rgb, size_t pixels) {
    for (size_t i = 0; i < pixels; i++) {
        uint8_t t = rgb[i * 3];
        rgb[i * 3] = rgb[i * 3 + 2];
        rgb[i * 3 + 2] = t;
    }
}
static void process_frame_full(uint8_t* d_rgb, uint32_t w, uint32_t h,
                                uint8_t* d_out, uint32_t& out_w, uint32_t& out_h);
// Live-cut recorder hook (defined in the RECORD tab section below).
// fresh = a new camera frame arrived since the last call (dupes skipped).
static void rec_write_frame(const uint8_t* d_rgb, uint32_t w, uint32_t h, bool fresh);
// Gesture director tick (defined in the RECORD tab section below).
static void rec_gesture_tick();
// Overlay pass: draws enabled detector annotations for preview/vcam AFTER
// the take taps the clean frame. In tutorial mode results map into the PiP.
static void draw_ai_overlays();

namespace kagerou { namespace ai {
// visualization helpers (defined in src/ai/ai_preprocess.cu)
void launch_normalize_depth(float* d_depth, uint32_t n, cudaStream_t s);
void launch_depth_to_rgb(const float* d_depth, uint8_t* d_rgb, uint32_t n, cudaStream_t s);
void launch_flow_to_rgb(const float* d_flow, uint8_t* d_rgb, uint32_t w, uint32_t h, cudaStream_t s);
void launch_draw_circle(uint8_t* rgb, uint32_t w, uint32_t h, float cx, float cy, float radius,
                        uint8_t r, uint8_t g, uint8_t b, cudaStream_t s);
void launch_draw_rect(uint8_t* rgb, uint32_t w, uint32_t h, int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b, int thickness, cudaStream_t s);
} }

// Enumerate real capture devices via DirectShow. Skips our own virtual
// camera (capturing it would feed our output back into our input).
static std::vector<std::string> enum_cameras() {
    std::vector<std::string> out;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool uninit = SUCCEEDED(hr);
    ICreateDevEnum* dev = nullptr;
    hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                          IID_ICreateDevEnum, (void**)&dev);
    if (SUCCEEDED(hr) && dev) {
        IEnumMoniker* en = nullptr;
        if (dev->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &en, 0) == S_OK && en) {
            IMoniker* mon = nullptr;
            while (en->Next(1, &mon, nullptr) == S_OK) {
                IPropertyBag* bag = nullptr;
                if (SUCCEEDED(mon->BindToStorage(nullptr, nullptr, IID_IPropertyBag, (void**)&bag)) && bag) {
                    VARIANT v; VariantInit(&v);
                    if (SUCCEEDED(bag->Read(L"FriendlyName", &v, nullptr)) && v.vt == VT_BSTR) {
                        char name[256] = {};
                        WideCharToMultiByte(CP_ACP, 0, v.bstrVal, -1, name, sizeof(name), nullptr, nullptr);
                        // never capture ourselves
                        if (!strstr(name, "Kagerou") && name[0])
                            out.push_back(name);
                    }
                    VariantClear(&v);
                    bag->Release();
                }
                mon->Release();
            }
            en->Release();
        }
        dev->Release();
    }
    if (uninit) CoUninitialize();
    return out;
}

// ============================================================================
// Screen capture thread: helper process (fast DXGI) with GDI fallback.
// DIBs are BGR: swapped to RGB to match the camera path.
// ============================================================================

// Retired: in-process DXGI duplication cannot work in this process
// (phantom NVIDIA output). Kept compiling but unused; capture lives in
// screen_cap.exe (helper) + GDI fallback. See screen_helper_* below.
struct DxgiCap {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    IDXGIOutputDuplication* dup = nullptr;
    ID3D11Texture2D* staging = nullptr;
    uint32_t sw = 0, sh = 0, tw = 0, th = 0;
};
static void dxgi_close(DxgiCap& c) {
    if (c.dup) { c.dup->Release(); c.dup = nullptr; }
    if (c.staging) { c.staging->Release(); c.staging = nullptr; }
    if (c.ctx) { c.ctx->Release(); c.ctx = nullptr; }
    if (c.dev) { c.dev->Release(); c.dev = nullptr; }
    c.sw = c.sh = 0;
}
static std::atomic<int> g_dxgi_fail{0};
static std::atomic<long> g_dxgi_hr{0};

static bool dxgi_open(DxgiCap& c) {
    dxgi_close(c);
    IDXGIFactory1* fac = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac)) || !fac) {
        g_dxgi_fail = 1; return false;
    }
    // NOTE: DXGI adapter/output topology is NOT stable per process here.
    // A bare probe sees Intel-first; this process (CUDA-active) sees
    // NVIDIA-first with a phantom panel output that refuses duplication.
    // Root cause is the process GPU preference (likely High-performance):
    // outputs attach to the preferred adapter. So ask explicitly for the
    // minimum-power (iGPU) adapter first, then fall back to the legacy walk.
    bool ok = false;
    bool any_primary = false;
    static const D3D_FEATURE_LEVEL fls[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    IDXGIFactory6* fac6 = nullptr;
    if (SUCCEEDED(fac->QueryInterface(__uuidof(IDXGIFactory6), (void**)&fac6)) && fac6) {
        for (UINT oi = 0; !ok; oi++) {
            IDXGIAdapter1* ad = nullptr;
            if (fac6->EnumAdapterByGpuPreference(oi, DXGI_GPU_PREFERENCE_MINIMUM_POWER,
                                                 __uuidof(IDXGIAdapter1), (void**)&ad) != S_OK || !ad)
                break;
            // (output walk below is shared; mark and reuse via goto-free flow)
            for (UINT ox = 0; !ok; ox++) {
                IDXGIOutput* out = nullptr;
                if (ad->EnumOutputs(ox, &out) != S_OK || !out) break;
                DXGI_OUTPUT_DESC od = {};
                out->GetDesc(&od);
                bool primary = (od.DesktopCoordinates.left == 0 && od.DesktopCoordinates.top == 0);
                uint32_t sw = (uint32_t)(od.DesktopCoordinates.right - od.DesktopCoordinates.left);
                uint32_t sh = (uint32_t)(od.DesktopCoordinates.bottom - od.DesktopCoordinates.top);
                if (primary) any_primary = true;
                ID3D11Device* dev = nullptr;
                ID3D11DeviceContext* ctx = nullptr;
                IDXGIOutputDuplication* dup = nullptr;
                D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_1_0_CORE;
                HRESULT dhr = E_FAIL;
                if (primary)
                    dhr = D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                            fls, 4, D3D11_SDK_VERSION, &dev, &got, &ctx);
                if (primary && SUCCEEDED(dhr) && dev) {
                    IDXGIOutput1* out1 = nullptr;
                    HRESULT qhr = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
                    if (SUCCEEDED(qhr) && out1) {
                        HRESULT uhr = out1->DuplicateOutput(dev, &dup);
                        if (SUCCEEDED(uhr) && dup) {
                            c.dev = dev; c.ctx = ctx; c.dup = dup;
                            c.staging = nullptr; c.sw = sw; c.sh = sh;
                            c.tw = 0; c.th = 0;
                            dev = nullptr; ctx = nullptr; dup = nullptr;
                            ok = true; g_dxgi_fail = 0;
                        } else { g_dxgi_fail = 15; g_dxgi_hr = uhr; }
                        if (out1) out1->Release();
                    } else { g_dxgi_fail = 14; g_dxgi_hr = qhr; }
                } else if (primary) { g_dxgi_fail = 13; g_dxgi_hr = dhr; }
                if (dup) dup->Release();
                if (ctx) ctx->Release();
                if (dev) dev->Release();
                out->Release();
            }
            ad->Release();
        }
        fac6->Release();
        if (ok) { fac->Release(); return true; }
    }
    for (UINT ai = 0; !ok; ai++) {
        IDXGIAdapter1* ad = nullptr;
        if (fac->EnumAdapters1(ai, &ad) != S_OK) break;
        for (UINT oi = 0; !ok; oi++) {
            IDXGIOutput* out = nullptr;
            if (ad->EnumOutputs(oi, &out) != S_OK) break;
            DXGI_OUTPUT_DESC od = {};
            out->GetDesc(&od);
            bool primary = (od.DesktopCoordinates.left == 0 && od.DesktopCoordinates.top == 0);
            uint32_t sw = (uint32_t)(od.DesktopCoordinates.right - od.DesktopCoordinates.left);
            uint32_t sh = (uint32_t)(od.DesktopCoordinates.bottom - od.DesktopCoordinates.top);
            if (primary) any_primary = true;
            ID3D11Device* dev = nullptr;
            ID3D11DeviceContext* ctx = nullptr;
            IDXGIOutputDuplication* dup = nullptr;
            ID3D11Texture2D* staging = nullptr;
            D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_1_0_CORE;
            HRESULT dhr = E_FAIL;
            // NOTE: no FEED-size gate here — large desktops are downscaled
            // from native frames in dxgi_frame (see staging logic there).
            if (primary)
                dhr = D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                        fls, 4, D3D11_SDK_VERSION, &dev, &got, &ctx);
            if (primary && SUCCEEDED(dhr) && dev) {
                IDXGIOutput1* out1 = nullptr;
                HRESULT qhr = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
                if (SUCCEEDED(qhr) && out1) {
                    HRESULT uhr = out1->DuplicateOutput(dev, &dup);
                    if (SUCCEEDED(uhr) && dup) {
                        // Staging is created lazily from the first real frame:
                        // DesktopCoordinates are DPI-virtualized and often do
                        // NOT match the physical framebuffer (e.g. 1440x810
                        // coords vs 2880x1620 pixels at 200% scaling).
                        c.dev = dev; c.ctx = ctx; c.dup = dup;
                        c.staging = nullptr; c.sw = sw; c.sh = sh;
                        c.tw = 0; c.th = 0;
                        dev = nullptr; ctx = nullptr;
                        dup = nullptr; staging = nullptr;
                        ok = true;
                        g_dxgi_fail = 0;
                    } else { g_dxgi_fail = 5; g_dxgi_hr = uhr; }
                    if (out1) out1->Release();
                } else { g_dxgi_fail = 4; g_dxgi_hr = qhr; }
            } else if (primary) {
                g_dxgi_fail = 3; g_dxgi_hr = dhr;
            }
            if (staging) staging->Release();
            if (dup) dup->Release();
            if (ctx) ctx->Release();
            if (dev) dev->Release();
            out->Release();
        }
        ad->Release();
    }
    fac->Release();
    if (!ok && !any_primary && g_dxgi_fail == 0) g_dxgi_fail = 2;
    return ok;
}

// Copies the newest frame as RGB24 into rgb (c.tw x c.th, FEED box fit of
// the native framebuffer, bilinear BGRA->RGB). Static screen yields
// timeouts: caller republishes its last buffer then.
static int dxgi_frame(DxgiCap& c, uint8_t* rgb) {
    if (!c.dup) return 0;
    IDXGIResource* res = nullptr;
    DXGI_OUTDUPL_FRAME_INFO fi = {};
    HRESULT hr = c.dup->AcquireNextFrame(30, &fi, &res);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) return 2;
    if (FAILED(hr) || !res) { g_dxgi_fail = 7; g_dxgi_hr = hr; return 0; }
    int rc = 0;
    if (fi.LastPresentTime.QuadPart != 0) {
        ID3D11Texture2D* src = nullptr;
        if (SUCCEEDED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) && src) {
            D3D11_TEXTURE2D_DESC sd = {};
            src->GetDesc(&sd);
            if (sd.Format == DXGI_FORMAT_B8G8R8A8_UNORM &&
                sd.Width >= 160 && sd.Height >= 90 &&
                sd.Width <= 4096 && sd.Height <= 2304) {
                // (Re)create staging + target size from the REAL framebuffer.
                if (!c.staging || sd.Width != c.sw || sd.Height != c.sh) {
                    if (c.staging) { c.staging->Release(); c.staging = nullptr; }
                    D3D11_TEXTURE2D_DESC td = {};
                    td.Width = sd.Width; td.Height = sd.Height;
                    td.MipLevels = 1; td.ArraySize = 1;
                    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                    td.SampleDesc.Count = 1;
                    td.Usage = D3D11_USAGE_STAGING;
                    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                    HRESULT thr = c.dev->CreateTexture2D(&td, nullptr, &c.staging);
                    if (SUCCEEDED(thr) && c.staging) {
                        c.sw = sd.Width; c.sh = sd.Height;
                        double s = (double)FEED_MAX_W / sd.Width <
                                   (double)FEED_MAX_H / sd.Height
                                   ? (double)FEED_MAX_W / sd.Width
                                   : (double)FEED_MAX_H / sd.Height;
                        if (s > 1.0) s = 1.0;
                        c.tw = ((uint32_t)(sd.Width * s)) & ~1u;
                        c.th = ((uint32_t)(sd.Height * s)) & ~1u;
                        if (c.tw < 160) c.tw = 160;
                        if (c.th < 90) c.th = 90;
                    } else { g_dxgi_fail = 6; g_dxgi_hr = thr; }
                }
                if (c.staging && c.tw && c.th) {
                    g_dxgi_fail = 0;
                    c.ctx->CopyResource(c.staging, src);
                    D3D11_MAPPED_SUBRESOURCE mp = {};
                    if (SUCCEEDED(c.ctx->Map(c.staging, 0, D3D11_MAP_READ, 0, &mp)) && mp.pData) {
                        const uint8_t* s = (const uint8_t*)mp.pData;
                        float fx = (float)c.sw / c.tw, fy = (float)c.sh / c.th;
                        for (uint32_t y = 0; y < c.th; y++) {
                            uint32_t sy = (uint32_t)(y * fy);
                            if (sy >= c.sh) sy = c.sh - 1;
                            const uint8_t* row = s + (size_t)sy * mp.RowPitch;
                            uint8_t* d = rgb + (size_t)y * c.tw * 3;
                            for (uint32_t x = 0; x < c.tw; x++) {
                                uint32_t sx = (uint32_t)(x * fx);
                                if (sx >= c.sw) sx = c.sw - 1;
                                d[x*3+0] = row[sx*4+2];
                                d[x*3+1] = row[sx*4+1];
                                d[x*3+2] = row[sx*4+0];
                            }
                        }
                        rc = 1;
                    }
                    c.ctx->Unmap(c.staging, 0);
                }
            } else { g_dxgi_fail = 8; g_dxgi_hr = (long)sd.Format; }
        }
        if (src) src->Release();
    } else rc = 2; // present-but-empty: treat as static
    res->Release();
    c.dup->ReleaseFrame();
    return rc;
}

// ---- screen capture helper (separate bare process) --------------------------
// In-process DXGI duplication fails here (phantom NVIDIA output refuses it
// while a bare probe works). The helper links no CUDA/ORT/NVENC, so it
// enumerates Intel-first and duplicates fine. SHM seqlock feed; GDI below
// stays as fallback.
static PROCESS_INFORMATION g_scr_pi = {};
static bool g_scr_proc = false;
static HANDLE g_scr_map = nullptr;
static ScreenShm* g_scr_shm = nullptr;
static HANDLE g_scr_stop = nullptr;
static char g_scr_mapname[128] = "";
static char g_scr_evname[128] = "";

static void scr_names_init() {
    if (!g_scr_mapname[0]) {
        char tag[32];
        snprintf(tag, sizeof(tag), "%lu", GetCurrentProcessId());
        kscr_names(tag, g_scr_mapname, sizeof(g_scr_mapname),
                   g_scr_evname, sizeof(g_scr_evname));
    }
}
static uint8_t* g_screen_nat = nullptr; // native-size staging (helper feed)
static size_t g_screen_nat_sz = 0;
static int g_scr_fails = 0;
static std::chrono::steady_clock::time_point g_scr_start_t{};

static void screen_helper_stop() {
    if (g_scr_stop) SetEvent(g_scr_stop);
    if (g_scr_proc && g_scr_pi.hProcess) {
        if (WaitForSingleObject(g_scr_pi.hProcess, 2000) == WAIT_TIMEOUT)
            TerminateProcess(g_scr_pi.hProcess, 0);
        CloseHandle(g_scr_pi.hProcess);
        CloseHandle(g_scr_pi.hThread);
        g_scr_pi = {};
    }
    g_scr_proc = false;
    if (g_scr_shm) { UnmapViewOfFile(g_scr_shm); g_scr_shm = nullptr; }
    if (g_scr_map) { CloseHandle(g_scr_map); g_scr_map = nullptr; }
    if (g_scr_stop) { CloseHandle(g_scr_stop); g_scr_stop = nullptr; }
    g_scr_fails = 0;
}

static bool screen_helper_start() {
    screen_helper_stop();
    scr_names_init();
    std::string exe = get_exe_dir() + "\\screen_cap.exe";
    if (GetFileAttributesA(exe.c_str()) == INVALID_FILE_ATTRIBUTES) return false;
    g_scr_stop = CreateEventA(NULL, TRUE, FALSE, g_scr_evname);
    if (!g_scr_stop) return false;
    ResetEvent(g_scr_stop);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "\"%s\" %lu", exe.c_str(), GetCurrentProcessId());
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        NULL, NULL, &si, &pi)) {
        screen_helper_stop();
        return false;
    }
    g_scr_pi = pi;
    g_scr_proc = true;
    g_scr_start_t = std::chrono::steady_clock::now();
    for (int i = 0; i < 100 && !g_scr_map; i++) {
        g_scr_map = OpenFileMappingA(FILE_MAP_READ, FALSE, g_scr_mapname);
        if (!g_scr_map) Sleep(20);
    }
    if (!g_scr_map) { screen_helper_stop(); return false; }
    g_scr_shm = (ScreenShm*)MapViewOfFile(g_scr_map, FILE_MAP_READ, 0, 0,
                                          sizeof(ScreenShm) + KSCR_MAX_BYTES);
    if (!g_scr_shm) { screen_helper_stop(); return false; }
    return true;
}

// Seqlock-read the latest helper frame, downscale (bilinear) into the FEED
// box and publish. Returns true on a fresh publish.
static bool screen_helper_poll() {
    if (!g_scr_proc || !g_scr_shm) return false;
    if (g_scr_shm->magic != KSCR_MAGIC || g_scr_shm->version != KSCR_VERSION)
        return false;
    if (g_scr_shm->state == 2) return false; // helper failed
    LONG s1 = g_scr_shm->seq;
    if ((s1 & 1) || g_scr_shm->state != 1) return false;
    // Helper already publishes take-size RGB (FEED box fit).
    uint32_t tw = g_scr_shm->w, th = g_scr_shm->h;
    if (!tw || !th || tw > FEED_MAX_W || th > FEED_MAX_H) return false;
    size_t n = (size_t)tw * th * 3;
    if (!g_screen_buf) {
        g_screen_buf = (uint8_t*)malloc(FEED_MAX_RGB);
        if (!g_screen_buf) return false;
    }
    memcpy(g_screen_buf, (const uint8_t*)(g_scr_shm + 1), n);
    MemoryBarrier();
    g_screen_w = tw; g_screen_h = th;
    long acc = 0; int cnt = 0;
    for (size_t i = 0; i < (size_t)tw * th * 3; i += 997) {
        acc += g_screen_buf[i]; cnt++;
    }
    g_screen_mean = cnt ? (float)acc / cnt : 0.0f;
    g_screen_seq++;
    g_screen_blt = 4; // helper
    auto now = std::chrono::steady_clock::now();
    static auto prev = now;
    double dt = std::chrono::duration<double>(now - prev).count();
    prev = now;
    if (dt > 0.001 && dt < 2.0) {
        float inst = (float)(1.0 / dt);
        float ema = g_screen_fps.load();
        g_screen_fps = ema <= 0 ? inst : ema * 0.9f + inst * 0.1f;
    }
    return true;
}

static void screen_thread_func() {
    auto help_last_try = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    // Persistent DC + DIB (GDI fallback; recreated only on resolution change).
    HDC hs = NULL, hm = NULL;
    HBITMAP hb = NULL, hstock = NULL;
    void* bits = nullptr;
    uint32_t cur_tw = 0, cur_th = 0;
    while (g_screen_run.load()) {
        // Idle in camera mode: no capture at all (battery/GPU friendly).
        if (g_src_mode == 0) {
            screen_helper_stop();
            Sleep(500);
            continue;
        }
        int sw = GetSystemMetrics(SM_CXSCREEN);
        int sh = GetSystemMetrics(SM_CYSCREEN);
        if (sw < 160 || sh < 120) { Sleep(200); continue; }
        // Helper fast path (throttled start attempts).
        {
            auto now = std::chrono::steady_clock::now();
            if (!g_scr_proc &&
                std::chrono::duration<double>(now - help_last_try).count() > 5.0) {
                help_last_try = now;
                screen_helper_start();
            }
        }
        if (g_scr_proc) {
            DWORD wr = WaitForSingleObject(g_scr_pi.hProcess, 0);
            if (wr == WAIT_OBJECT_0) {
                screen_helper_stop(); // helper died
            } else if (screen_helper_poll()) {
                g_scr_fails = 0; // success resets the circuit breaker
                Sleep(5);
                continue;
            } else {
                // Startup grace (process spawn + DXGI open + first frame
                // take ~1s) plus success-reset above: only truly persistent
                // failure trips the breaker into GDI fallback.
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - g_scr_start_t).count() < 4.0) {
                    Sleep(20);
                    continue;
                }
                if (++g_scr_fails > 120) {
                    screen_helper_stop(); // persistent failure -> GDI below
                } else {
                    Sleep(5);
                    continue;
                }
            }
        }
        double s = (double)FEED_MAX_W / sw < (double)FEED_MAX_H / sh
                     ? (double)FEED_MAX_W / sw : (double)FEED_MAX_H / sh;
        if (s > 1.0) s = 1.0;
        uint32_t tw = ((uint32_t)(sw * s)) & ~1u;
        uint32_t th = ((uint32_t)(sh * s)) & ~1u;
        if (tw < 160 || th < 90) { Sleep(200); continue; }
        if (!hs) hs = GetDC(NULL);
        if (!hs) { Sleep(200); continue; }
        if (!hm) {
            hm = CreateCompatibleDC(hs);
            if (hm) hstock = (HBITMAP)GetCurrentObject(hm, OBJ_BITMAP);
        }
        if (!hm) { ReleaseDC(NULL, hs); hs = NULL; Sleep(200); continue; }
        if (!hb || tw != cur_tw || th != cur_th) {
            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = tw;
            bmi.bmiHeader.biHeight = -(int)th;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 24;
            bmi.bmiHeader.biCompression = BI_RGB;
            void* nbits = nullptr;
            HBITMAP nhb = CreateDIBSection(hm, &bmi, DIB_RGB_COLORS, &nbits, NULL, 0);
            if (!nhb || !nbits) {
                if (nhb) DeleteObject(nhb);
                g_screen_blt = -3;
                Sleep(100);
                continue;
            }
            SelectObject(hm, nhb); // deselects hb (or stock on first pass)
            if (hb) DeleteObject(hb); // our old DIB only; stock untouched
            hb = nhb; bits = nbits; cur_tw = tw; cur_th = th;
        }
        BOOL ok = FALSE;
        if (tw == (uint32_t)sw && th == (uint32_t)sh) {
            ok = BitBlt(hm, 0, 0, tw, th, hs, 0, 0, SRCCOPY);
            g_screen_blt = ok ? 1 : -1;
        } else {
            ok = StretchBlt(hm, 0, 0, tw, th, hs, 0, 0, sw, sh, SRCCOPY);
            g_screen_blt = ok ? 2 : -2;
        }
        if (!ok) g_screen_blterr = GetLastError();
            {
                std::lock_guard<std::mutex> lock(g_screen_mtx);
                if (!g_screen_buf)
                    g_screen_buf = (uint8_t*)malloc(FEED_MAX_RGB);
                if (g_screen_buf) {
                    memcpy(g_screen_buf, bits, (size_t)tw * th * 3);
                    swap_rb_inplace(g_screen_buf, (size_t)tw * th); // BGR -> RGB
                    long acc = 0; int cnt = 0;
                    for (size_t i = 0; i < (size_t)tw * th * 3; i += 997) {
                        acc += g_screen_buf[i]; cnt++;
                    }
                    g_screen_mean = cnt ? (float)acc / cnt : 0.0f;
                    g_screen_w = tw; g_screen_h = th;
                    g_screen_seq++;
                    auto now = std::chrono::steady_clock::now();
                    static auto prev = now;
                    double dt = std::chrono::duration<double>(now - prev).count();
                    prev = now;
                    if (dt > 0.001 && dt < 2.0) {
                        float inst = (float)(1.0 / dt);
                        float ema = g_screen_fps.load();
                        g_screen_fps = ema <= 0 ? inst : ema * 0.9f + inst * 0.1f;
                    }
                }
            }
        Sleep(5);
    }
    if (hm) {
        if (hstock) SelectObject(hm, hstock);
        if (hb) DeleteObject(hb);
        DeleteDC(hm);
    }
    if (hs) ReleaseDC(NULL, hs);
    screen_helper_stop();
    if (g_screen_nat) { free(g_screen_nat); g_screen_nat = nullptr; g_screen_nat_sz = 0; }
}

static bool screen_ensure() {
    if (g_screen_thread.joinable()) return true;
    g_screen_run = true;
    try { g_screen_thread = std::thread(screen_thread_func); }
    catch (...) { g_screen_run = false; return false; }
    return true;
}

// Camera PiP: SDK implementation (kagerou/pip.h, unit-tested).
// Caller must hold g_frame_mutex (shares g_cam_buf).
static void composite_pip(uint8_t* screen, uint32_t sw, uint32_t sh) {
    if (!g_cam_alive.load() || !g_cam_buf) return;
    uint32_t cw = g_cap_w, ch = g_cap_h;
    uint8_t* cam = g_cam_buf;
    // Tutorial auto-frame: crop the camera to the face (boxes from the last
    // tutorial face inference, in camera coords) so the PiP follows you.
    // Reuses the camera-mode smoothing constants; separate state so the two
    // modes never clobber each other.
    if (g_autoframe && g_af_idx > 0 && g_ai_face_ready && g_cap_w && g_cap_h) {
        int bi = -1; float bs = 0.4f;
        for (int i = 0; i < 4; i++)
            if (g_face_boxes[i].score > bs) { bs = g_face_boxes[i].score; bi = i; }
        if (bi >= 0) {
            float af_expand[] = {0, 2.5f, 2.0f, 1.5f};
            float fw = (g_face_boxes[bi].x2 - g_face_boxes[bi].x1) * (float)g_cap_w;
            float fh = (g_face_boxes[bi].y2 - g_face_boxes[bi].y1) * (float)g_cap_h;
            float tw = fw * af_expand[g_af_idx], th = fh * af_expand[g_af_idx];
            float tcx = (g_face_boxes[bi].x1 + g_face_boxes[bi].x2) * 0.5f * (float)g_cap_w;
            float tcy = (g_face_boxes[bi].y1 + g_face_boxes[bi].y2) * 0.5f * (float)g_cap_h - 0.1f * th;
            float tx = tcx - tw * 0.5f, ty = tcy - th * 0.5f;
            if (!g_afp_init || g_afp_fw != (int)g_cap_w || g_afp_fh != (int)g_cap_h) {
                g_afp_x = tx; g_afp_y = ty; g_afp_w = tw; g_afp_h = th;
                g_afp_fw = g_cap_w; g_afp_fh = g_cap_h; g_afp_init = true;
            } else {
                float dx = tx - g_afp_x, dy = ty - g_afp_y;
                float dw = tw - g_afp_w, dh = th - g_afp_h;
                if (fabsf(dx) < 5) dx = 0; if (fabsf(dy) < 5) dy = 0;
                if (fabsf(dw) < 4) dw = 0; if (fabsf(dh) < 4) dh = 0;
                g_afp_x += dx * 0.12f; g_afp_y += dy * 0.12f;
                g_afp_w += dw * 0.12f; g_afp_h += dh * 0.12f;
            }
            if (g_afp_w > (float)g_cap_w) g_afp_w = g_cap_w;
            if (g_afp_h > (float)g_cap_h) g_afp_h = g_cap_h;
            if (g_afp_x < 0) g_afp_x = 0; if (g_afp_y < 0) g_afp_y = 0;
            if (g_afp_x + g_afp_w > (float)g_cap_w) g_afp_x = g_cap_w - g_afp_w;
            if (g_afp_y + g_afp_h > (float)g_cap_h) g_afp_y = g_cap_h - g_afp_h;
            uint32_t crop_w = ((uint32_t)g_afp_w) & ~1u, crop_h = ((uint32_t)g_afp_h) & ~1u;
            int cx = (int)g_afp_x & ~1, cy = (int)g_afp_y & ~1;
            if (crop_w >= 32 && crop_h >= 32) {
                size_t need = (size_t)crop_w * crop_h * 3;
                if (!g_afp_cam || g_afp_cam_sz < need) {
                    if (g_afp_cam) free(g_afp_cam);
                    g_afp_cam = (uint8_t*)malloc(need); g_afp_cam_sz = need;
                }
                if (g_afp_cam) {
                    for (uint32_t row = 0; row < crop_h; row++)
                        memcpy(g_afp_cam + (size_t)row * crop_w * 3,
                               g_cam_buf + ((size_t)(cy + (int)row) * g_cap_w + cx) * 3,
                               crop_w * 3);
                    cam = g_afp_cam; cw = crop_w; ch = crop_h;
                }
            }
        }
    }
    kagerou::pip::composite(cam, cw, ch, screen, sw, sh);
}

// ============================================================================
// Camera capture thread
// ============================================================================
static void cam_thread_func() {
    char cmd[1024];
    size_t cand = 0;
    std::vector<std::string> cams;
    while (g_running) {
        if (cand == 0) {
            // Re-enumerate each cycle: hot-plugged cameras appear by themselves.
            cams = enum_cameras();
            if (cams.empty()) {
                printf("[CAM] no cameras found, retrying...\n"); fflush(stdout);
                Sleep(1000);
                continue;
            }
        }
        const std::string& name = cams[cand % cams.size()];
        sprintf(cmd, "ffmpeg -hide_banner -loglevel error "
                "-f dshow -video_size %ux%u -framerate 30 "
                "-i video=\"%s\" "
                "-f rawvideo -pix_fmt rgb24 pipe:1",
                g_cap_w, g_cap_h, name.c_str());
        FILE* pipe = _popen(cmd, "rb");
        if (!pipe) { Sleep(1000); continue; }
        // Probe: try to read one full frame with a timeout-ish check.
        // If the device name is wrong ffmpeg exits immediately -> fread fails.
        size_t sz = (size_t)g_cap_w * g_cap_h * 3;
        uint8_t* buf = (uint8_t*)malloc(sz);
        size_t got0 = fread(buf, 1, sz, pipe);
        if (got0 < sz) {
            free(buf);
            _pclose(pipe);
            printf("[CAM] \"%s\" not available, trying next...\n", name.c_str());
            fflush(stdout);
            cand++;
            if (cand % cams.size() == 0) { cand = 0; Sleep(500); }
            continue;
        }
        printf("[CAM] Capturing from \"%s\"\n", name.c_str());
        fflush(stdout);
        g_cam_alive = true;
        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            memcpy(g_cam_buf, buf, sz);
        }
        bool ok = true;
        while (g_running && ok) {
            size_t got = 0;
            while (got < sz) {
                size_t r = fread(buf + got, 1, sz - got, pipe);
                if (r == 0) { ok = false; break; }
                got += r;
            }
            if (!ok) break;
            {
                std::lock_guard<std::mutex> lock(g_frame_mutex);
                memcpy(g_cam_buf, buf, sz);
            }
            g_cam_seq++;
        }
        free(buf);
        _pclose(pipe);
        g_cam_alive = false;
        // Move on and re-enumerate from scratch next cycle.
        cand = 0;
        Sleep(500);
    }
}

// ============================================================================
// GPU processing thread (runs independently of UI)
// ============================================================================
// Recorder state needed by the frame loop below (burn overlays when taking).
// Full recorder block lives further down; these two are hoisted so the
// tap-order decision sees them. 0 idle, 1 recording, 2 finalizing, 3 done, 4 failed.
static std::atomic<int> g_rec_state{0};
static std::atomic<bool> g_rec_paused{false};
static void gpu_thread_func() {
    auto t0 = std::chrono::high_resolution_clock::now();
    int fc = 0;

    while (g_running) {
        size_t cap_sz = (size_t)g_cap_w * g_cap_h * 3;
        uint8_t* local_buf = (uint8_t*)malloc(FEED_MAX_RGB); // fits any feed
        bool has_frame = false;
        uint32_t fw = g_cap_w, fh = g_cap_h; // feed dims this iteration
        uint64_t feed_seq = 0;

        // Tutorial source: screen (mode 1) or screen + camera PiP (mode 2).
        if (g_src_mode != 0) {
            screen_ensure();
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            std::lock_guard<std::mutex> slock(g_screen_mtx);
            if (g_screen_w && g_screen_h && g_screen_buf) {
                fw = g_screen_w; fh = g_screen_h;
                memcpy(local_buf, g_screen_buf, (size_t)fw * fh * 3);
                feed_seq = g_screen_seq.load();
                has_frame = true;
                if (g_src_mode == 2)
                    composite_pip(local_buf, fw, fh); // frame lock held
            }
        }
        if (!has_frame) {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            if (g_cam_alive && g_cam_buf) {
                memcpy(local_buf, g_cam_buf, cap_sz);
                feed_seq = g_cam_seq.load();
                has_frame = true;
            }
        }

        size_t sz = (size_t)fw * fh * 3;
        if (!has_frame) {
            gen_test_pattern(local_buf, fw, fh, g_frame_count);
        }
        // Black-frame watch: a shuttered/covered camera reads "alive" but
        // delivers zeros (records as solid green). Hysteresis avoids flicker.
        // Screen feeds skip it (dark IDEs are normal).
        {
            static int black_n = 0;
            if (!has_frame || g_src_mode != 0) { black_n = 0; g_cam_black = false; }
            else {
                long acc = 0; int cnt = 0;
                for (size_t i = 0; i < sz; i += 997) { acc += local_buf[i]; cnt++; }
                double mean = cnt ? (double)acc / cnt : 255.0;
                if (mean < 3.0) { if (++black_n > 30) g_cam_black = true; }
                else { black_n = 0; if (mean > 8.0) g_cam_black = false; }
            }
        }
        g_frame_count++;

        // Tutorial tracking: face+hands infer on the CAMERA feed (results in
        // camera coords for PiP mapping + gestures). The program feed flows
        // to the take untouched; overlays draw after the take tap.
        bool ai_live = !g_ai_warming.load();
        if (g_src_mode != 0 && g_cam_alive.load() && g_cam_buf && g_d_camdet) {
            bool need_face = ai_live && g_ai_face_ready &&
                (g_ai_face || (g_autoframe && g_af_idx > 0 && g_src_mode == 2));
            bool need_hands = ai_live && g_ai_hands && g_ai_hands_ready;
            bool need_pose = ai_live && g_ai_pose && g_ai_pose_ready;
            bool need_gaze = ai_live && g_ai_gaze && g_ai_gaze_ready;
            bool need_detect = ai_live && g_ai_detect && g_ai_detect_ready;
            if (need_face || need_hands || need_pose || need_gaze || need_detect) {
                g_det_iw = g_cap_w; g_det_ih = g_cap_h;
                {
                    std::lock_guard<std::mutex> lock(g_frame_mutex);
                    cudaMemcpyAsync(g_d_camdet, g_cam_buf, cap_sz,
                                    cudaMemcpyHostToDevice, g_stream);
                }
                if (need_face && (g_frame_count % 2) == 0)
                    ai::ai_face(g_d_camdet, g_cap_w, g_cap_h, g_face_boxes, 4, g_stream);
                if (need_hands && (g_frame_count % 3) == 0)
                    ai::ai_hands(g_d_camdet, g_cap_w, g_cap_h, g_hand_joints, 2, g_stream);
                if (need_pose)
                    ai::ai_pose(g_d_camdet, g_cap_w, g_cap_h, g_pose_landmarks, g_stream);
                if (need_gaze && (g_frame_count % 3) == 0)
                    ai::ai_gaze(g_d_camdet, g_cap_w, g_cap_h, g_gaze_eyes, 2, g_stream);
                if (need_detect && (g_frame_count % 2) == 0)
                    ai::ai_detect(g_d_camdet, g_cap_w, g_cap_h, g_det_boxes, 16, g_stream);
            }
        }

        bool any_filter = g_denoise || g_clahe || g_sr || g_lut_on ||
                          g_blur || g_sharp || g_bright || g_sat || g_gamma ||
                          g_vignette || g_grain || g_edge || g_wb || g_flip || g_lens ||
                          g_interp || g_bgblur || g_tempden || g_tempstab || g_chroma ||
                          g_dirblur || g_hdr || g_cropzoom ||
                          g_ai_denoise || g_ai_depth || g_ai_flow || g_ai_pose ||
                          g_ai_matting || g_ai_face || g_ai_hands || g_ai_gaze ||
                          g_ai_lowlight || g_ai_anime || g_ai_detect || g_autoframe;

        if (!any_filter) {
            g_disp_w = fw; g_disp_h = fh;
            cudaMemcpyAsync(g_d_rgb, local_buf, sz, cudaMemcpyHostToDevice, g_stream);
        } else {
            cudaMemcpyAsync(g_d_rgb, local_buf, sz, cudaMemcpyHostToDevice, g_stream);
            uint32_t ww, wh;
            process_frame_full(g_d_rgb, fw, fh, g_d_tmp, ww, wh);
            g_disp_w = ww; g_disp_h = wh;
        }

        // When recording, tracking overlays are drawn BEFORE the take tap
        // so face/gaze/pose/hands are burned into the saved video
        // (same-stream launches on g_stream, so ordering is exact).
        // Otherwise the take tap stays clean and overlays go to
        // preview + vcam only.
        static uint64_t s_last_seq = 0;
        bool fresh = (feed_seq != s_last_seq);
        s_last_seq = feed_seq;
        bool rec_active = (g_rec_state.load() == 1) && !g_rec_paused;
        if (rec_active) draw_ai_overlays();
        rec_write_frame(g_d_rgb, g_disp_w, g_disp_h, fresh);
        if (!rec_active) draw_ai_overlays();
        game_tick_gpu(); // GAME tab: gestures -> injector/trainer (no-op unless on)

        // NOTE: the ffmpeg pipe delivers RGB (rgb24). The preview swap
        // below must be atomic with the download: a UI paint landing
        // between them flashes an unswapped (wrong-color) frame.
        // Display download: RGB->BGR swap runs on the GPU into g_d_tmp
        // (a 6MB CPU swap loop here costs ~6ms/frame at 1080p). g_d_tmp is
        // process scratch, free once the chain above has consumed it; the
        // vcam block reuses it only after this download completes.
        {
            size_t wsz = (size_t)g_disp_w * g_disp_h * 3;
            filters::bgr_to_rgb(g_d_rgb, g_d_tmp, g_disp_w, g_disp_h, g_stream);
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            cudaMemcpyAsync(g_display_out, g_d_tmp, wsz, cudaMemcpyDeviceToHost, g_stream);
            cudaStreamSynchronize(g_stream);
            int osd = g_rec_osd.load();
            if (osd == 2 && std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - g_rec_osd_t0).count() > 0.5)
                { osd = 0; g_rec_osd = 0; }
            if (osd) rec_osd_border(g_display_out, g_disp_w, g_disp_h, osd);
        }

        // Virtual-cam output is LOCKED to capture size (640x480). Changing
        // dimensions mid-stream breaks DirectShow downstream negotiation
        // (Chrome shows "camera unavailable"). Downscale when SR is on.
        if (g_vcam_active && g_d_nv12 && g_h_nv12) {
            auto now = std::chrono::high_resolution_clock::now();
            uint64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
            const uint8_t* d_vcam_src = g_d_rgb;
            if (g_disp_w != g_cap_w || g_disp_h != g_cap_h) {
                filters::resize_bilinear(g_d_rgb, g_d_tmp, g_disp_w, g_disp_h,
                                         g_cap_w, g_cap_h, 3, g_stream);
                d_vcam_src = g_d_tmp;
            }
            // Limited-range BT.601 (players assume 601 below 720p; 709 here
            // shifts reds/blues and crushes contrast).
            filters::rgb_to_nv12_limited601(d_vcam_src, g_d_nv12, g_cap_w, g_cap_h, g_stream);
            uint32_t nv12_sz = g_cap_w * g_cap_h * 3 / 2;
            cudaMemcpyAsync(g_h_nv12, g_d_nv12, nv12_sz, cudaMemcpyDeviceToHost, g_stream);
            cudaStreamSynchronize(g_stream);
            g_vcam.WriteFrame(g_h_nv12, g_cap_w, g_cap_h, ts);
        }

        free(local_buf);

        fc++;
        auto now = std::chrono::high_resolution_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        if (el >= 1.0) { g_fps = fc / el; fc = 0; t0 = now; }

        Sleep(1);
    }
}

// ============================================================================
// Test pattern
// ============================================================================
static void gen_test_pattern(uint8_t* rgb, uint32_t w, uint32_t h, int f) {
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            int bar = ((x + f * 2) * 7 / w) % 7;
            uint8_t c[] = {40,40,50, 60,60,75, 50,50,65, 45,45,58, 55,55,72, 65,65,82, 35,35,45};
            uint8_t* p = &c[bar * 3];
            rgb[(y*w+x)*3]=p[0]; rgb[(y*w+x)*3+1]=p[1]; rgb[(y*w+x)*3+2]=p[2];
        }
}

// ============================================================================
// AI model init
// ============================================================================
static void init_ai_models() {
    std::string denoise_path = resolve_model("models/denoise/fastdvdnet.onnx");
    std::string depth_path = resolve_model("models/depth/depth_anything_v2.onnx");
    std::string flow_path = resolve_model("models/flow/raft_small.onnx");
    std::string pose_path = resolve_model("models/pose/pose_landmark.onnx");
    // NOTE: static-shape copy (TRT EP rejects the fully-symbolic original).
    std::string matting_path = resolve_model("models/matting/rvm_mobilenetv3_static.onnx");
    std::string face_path = resolve_model("models/face/yunet_2023mar.onnx");
    std::string palm_path = resolve_model("models/hands/palm_detection_lite.onnx");
    std::string handlm_path = resolve_model("models/hands/hand_landmark_lite.onnx");
    std::string iris_path = resolve_model("models/gaze/iris_landmark.onnx");
    std::string lowlight_path = resolve_model("models/lowlight/zero_dce.onnx");
    std::string anime_path = resolve_model("models/style/animeganv3_hayao.onnx");
    std::string detect_path = resolve_model("models/detect/yolov8n.onnx");
    printf("[AI] Loading from exe dir: %s\n", get_exe_dir().c_str());
    fflush(stdout);
    g_ai_denoise_ready = file_exists(denoise_path.c_str()) ? ai::ai_denoise_init(denoise_path.c_str()) : false;
    if (!file_exists(denoise_path.c_str())) printf("[AI] denoise model file missing, skipping\n");
    g_ai_depth_ready = file_exists(depth_path.c_str()) ? ai::ai_depth_init(depth_path.c_str()) : false;
    if (!file_exists(depth_path.c_str())) printf("[AI] depth model file missing, skipping\n");
    g_ai_flow_ready = file_exists(flow_path.c_str()) ? ai::ai_flow_init(flow_path.c_str()) : false;
    if (!file_exists(flow_path.c_str())) printf("[AI] flow model file missing, skipping\n");
    g_ai_pose_ready = file_exists(pose_path.c_str()) ? ai::ai_pose_init(pose_path.c_str()) : false;
    if (!file_exists(pose_path.c_str())) printf("[AI] pose model file missing, skipping\n");
    g_ai_matting_ready = file_exists(matting_path.c_str()) ? ai::ai_matting_init(matting_path.c_str()) : false;
    if (!file_exists(matting_path.c_str())) printf("[AI] matting model file missing, skipping\n");
    g_ai_face_ready = file_exists(face_path.c_str()) ? ai::ai_face_init(face_path.c_str()) : false;
    if (!file_exists(face_path.c_str())) printf("[AI] face model file missing, skipping\n");
    g_ai_hands_ready = (file_exists(palm_path.c_str()) && file_exists(handlm_path.c_str()))
        ? ai::ai_hands_init(palm_path.c_str(), handlm_path.c_str()) : false;
    if (!file_exists(palm_path.c_str()) || !file_exists(handlm_path.c_str()))
        printf("[AI] hands model files missing, skipping\n");
    g_ai_gaze_ready = file_exists(iris_path.c_str()) ? ai::ai_gaze_init(iris_path.c_str()) : false;
    if (!file_exists(iris_path.c_str())) printf("[AI] gaze model file missing, skipping\n");
    g_ai_lowlight_ready = file_exists(lowlight_path.c_str()) ? ai::ai_lowlight_init(lowlight_path.c_str()) : false;
    if (!file_exists(lowlight_path.c_str())) printf("[AI] lowlight model file missing, skipping\n");
    g_ai_anime_ready = file_exists(anime_path.c_str()) ? ai::ai_anime_init(anime_path.c_str()) : false;
    if (!file_exists(anime_path.c_str())) printf("[AI] anime model file missing, skipping\n");
    g_ai_detect_ready = file_exists(detect_path.c_str()) ? ai::ai_detect_init(detect_path.c_str()) : false;
    if (!file_exists(detect_path.c_str())) printf("[AI] detect model file missing, skipping\n");
    printf("[AI] denoise=%s depth=%s flow=%s pose=%s\n",
        g_ai_denoise_ready?"OK":"FAIL", g_ai_depth_ready?"OK":"FAIL",
        g_ai_flow_ready?"OK":"FAIL", g_ai_pose_ready?"OK":"FAIL");
    printf("[AI] matting=%s face=%s hands=%s gaze=%s lowlight=%s anime=%s detect=%s\n",
        g_ai_matting_ready?"OK":"FAIL", g_ai_face_ready?"OK":"FAIL",
        g_ai_hands_ready?"OK":"FAIL", g_ai_gaze_ready?"OK":"FAIL",
        g_ai_lowlight_ready?"OK":"FAIL", g_ai_anime_ready?"OK":"FAIL",
        g_ai_detect_ready?"OK":"FAIL");
    size_t frame_sz = FEED_MAX_RGB;
    size_t feed_px = (size_t)FEED_MAX_W * FEED_MAX_H;
    cudaMalloc(&g_d_ai_vis, frame_sz);
    cudaMalloc(&g_d_ai_matt, frame_sz);
    if (g_ai_depth_ready) cudaMalloc(&g_d_ai_depth_buf, feed_px * sizeof(float));
    if (g_ai_flow_ready) {
        cudaMalloc(&g_d_flow_prev, frame_sz);
        cudaMalloc(&g_d_flow_curr, frame_sz);
        cudaMalloc(&g_d_flow_buf, feed_px * 2 * sizeof(float));
    }
    if (g_ai_denoise_ready) {
        for (int i = 0; i < DENOISE_RING_SIZE; i++)
            cudaMalloc(&g_d_denoise_ring[i], frame_sz);
    }

    // ---- Warmup: TensorRT builds its engines lazily on the first Run(),
    // which takes minutes per model and would freeze the frame loop on
    // first toggle. Do it once here in the background thread; engines are
    // cached to trt_engines/ so later runs start fast.
    printf("[AI] Warming up (one-time engine build, then cached).\n");
    printf("[AI] First run only: the GPU will be busy, preview may stutter until done.\n");
    fflush(stdout);
    uint8_t *d_win = nullptr, *d_wout = nullptr;
    cudaMalloc(&d_win, FEED_MAX_RGB);
    cudaMalloc(&d_wout, FEED_MAX_RGB);
    cudaMemset(d_win, 0, frame_sz);
    if (g_ai_denoise_ready) {
        printf("[AI] Warmup 1/11: denoise (this one is big, be patient)...\n"); fflush(stdout);
        for (int i = 0; i < DENOISE_RING_SIZE; i++)
            cudaMemcpy(g_d_denoise_ring[i], d_win, frame_sz, cudaMemcpyDeviceToDevice);
        ai::ai_denoise(g_d_denoise_ring[0], g_d_denoise_ring[1], g_d_denoise_ring[2],
            g_d_denoise_ring[3], g_d_denoise_ring[4], d_wout, g_cap_w, g_cap_h, 0);
        printf("[AI] Warmup 1/11 done: denoise.\n"); fflush(stdout);
        Sleep(1500); // let the frame loop breathe between engine builds
    } else printf("[AI] Warmup 1/11 skipped: denoise model not loaded.\n");
    if (g_ai_depth_ready) {
        printf("[AI] Warmup 2/11: depth...\n"); fflush(stdout);
        ai::ai_depth(d_win, g_d_ai_depth_buf, g_cap_w, g_cap_h, 0);
        printf("[AI] Warmup 2/11 done: depth.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 2/11 skipped: depth model not loaded.\n");
    if (g_ai_flow_ready) {
        printf("[AI] Warmup 3/11: flow...\n"); fflush(stdout);
        cudaMemcpy(g_d_flow_prev, d_win, frame_sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_d_flow_curr, d_win, frame_sz, cudaMemcpyDeviceToDevice);
        ai::ai_flow(g_d_flow_prev, g_d_flow_curr, g_d_flow_buf, g_cap_w, g_cap_h, 0);
        g_flow_has_prev = true;
        printf("[AI] Warmup 3/11 done: flow.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 3/11 skipped: flow model not loaded.\n");
    if (g_ai_pose_ready) {
        printf("[AI] Warmup 4/11: pose...\n"); fflush(stdout);
        ai::ai_pose(d_win, g_cap_w, g_cap_h, g_pose_landmarks, 0);
        printf("[AI] Warmup 4/11 done: pose.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 4/11 skipped: pose model not loaded.\n");
    if (g_ai_matting_ready) {
        printf("[AI] Warmup 5/11: matting...\n"); fflush(stdout);
        ai::ai_matting(d_win, d_wout, g_cap_w, g_cap_h, 0);
        printf("[AI] Warmup 5/11 done: matting.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 5/11 skipped: matting model not loaded.\n");
    if (g_ai_face_ready) {
        printf("[AI] Warmup 6/11: face...\n"); fflush(stdout);
        kagerou::ai::FaceBox wb[2];
        ai::ai_face(d_win, g_cap_w, g_cap_h, wb, 2, 0);
        printf("[AI] Warmup 6/11 done: face.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 6/11 skipped: face model not loaded.\n");
    if (g_ai_hands_ready) {
        printf("[AI] Warmup 7/11: hands...\n"); fflush(stdout);
        kagerou::ai::HandJoints wh[1];
        ai::ai_hands(d_win, g_cap_w, g_cap_h, wh, 1, 0);
        printf("[AI] Warmup 7/11 done: hands.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 7/11 skipped: hands models not loaded.\n");
    if (g_ai_gaze_ready) {
        printf("[AI] Warmup 8/11: gaze...\n"); fflush(stdout);
        kagerou::ai::GazeEye we[1];
        ai::ai_gaze(d_win, g_cap_w, g_cap_h, we, 1, 0);
        printf("[AI] Warmup 8/11 done: gaze.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 8/11 skipped: gaze model not loaded.\n");
    if (g_ai_lowlight_ready) {
        printf("[AI] Warmup 9/11: lowlight...\n"); fflush(stdout);
        ai::ai_lowlight(d_win, d_wout, g_cap_w, g_cap_h, 0);
        printf("[AI] Warmup 9/11 done: lowlight.\n"); fflush(stdout);
        Sleep(1500);
    } else printf("[AI] Warmup 9/11 skipped: lowlight model not loaded.\n");
    if (g_ai_anime_ready) {
        printf("[AI] Warmup 10/11: anime...\n"); fflush(stdout);
        ai::ai_anime(d_win, d_wout, g_cap_w, g_cap_h, 0);
        printf("[AI] Warmup 10/11 done: anime.\n"); fflush(stdout);
    } else printf("[AI] Warmup 10/11 skipped: anime model not loaded.\n");
    if (g_ai_detect_ready) {
        printf("[AI] Warmup 11/11: detect...\n"); fflush(stdout);
        kagerou::ai::DetectBox wd[2];
        ai::ai_detect(d_win, g_cap_w, g_cap_h, wd, 2, 0);
        printf("[AI] Warmup 11/11 done: detect.\n"); fflush(stdout);
    } else printf("[AI] Warmup 11/11 skipped: detect model not loaded.\n");
    if (d_win) cudaFree(d_win);
    if (d_wout) cudaFree(d_wout);
    g_ai_warming = false;
    printf("[AI] Ready -- toggles respond instantly from here.\n"); fflush(stdout);
}

// ============================================================================
// GPU filter chain
// ============================================================================
static void process_frame_full(uint8_t* d_rgb, uint32_t w, uint32_t h,
                                uint8_t* d_out, uint32_t& out_w, uint32_t& out_h) {
    out_w = w; out_h = h;

    // ---- AI first, at capture size: side buffers are capture-sized and
    // inference is cheaper here. Each model also draws a visible overlay
    // (previously depth/flow wrote side buffers nobody ever displayed).
    // Heavy models infer at reduced rate; the cached overlay blends every
    // frame so the preview stays smooth.
    bool ai_live = !g_ai_warming.load();
    // Heavy generative models run at reduced res on big frames: full-res
    // Zero-DCE/FastDVDnet/AnimeGAN at 1080p costs 100ms+/frame and freezes
    // the loop (fine on camera, stuck on screen takes). Small-frame path
    // keeps identical per-frame semantics, just fewer pixels.
    uint32_t ai_sw = w, ai_sh = h;
    bool ai_small = false;
    if ((uint64_t)w * h > (uint64_t)640 * 480 &&
        (g_ai_denoise || g_ai_lowlight || g_ai_anime)) {
        double s = 640.0 / w < 480.0 / h ? 640.0 / w : 480.0 / h;
        ai_sw = ((uint32_t)(w * s)) & ~1u;
        ai_sh = ((uint32_t)(h * s)) & ~1u;
        if (ai_sw >= 160 && ai_sh >= 90) ai_small = true;
    }
    g_det_iw = w; g_det_ih = h; // overlay maps detector->program coords
    // AI denoise is a camera-sensor model with a blocking sync; on a 1080p
    // screen/tutorial feed it stalls the loop, and a digital screen capture
    // has no sensor noise to remove. Camera mode only.
    if (ai_live && g_ai_denoise && g_ai_denoise_ready && g_src_mode == 0) {
        if (ai_small && (g_denoise_last_w != w || g_denoise_last_h != h)) {
            g_denoise_ring_count = 0; // dims changed: refill temporal ring
            g_denoise_last_w = w; g_denoise_last_h = h;
        }
        if (ai_small) {
            filters::resize_bilinear(d_rgb, g_d_ai_s0, w, h, ai_sw, ai_sh, 3, g_stream);
            cudaMemcpyAsync(g_d_denoise_ring[g_denoise_ring_idx], g_d_ai_s0,
                            (size_t)ai_sw * ai_sh * 3, cudaMemcpyDeviceToDevice, g_stream);
            g_denoise_ring_idx = (g_denoise_ring_idx + 1) % DENOISE_RING_SIZE;
            if (g_denoise_ring_count < DENOISE_RING_SIZE) g_denoise_ring_count++;
            if (g_denoise_ring_count >= DENOISE_RING_SIZE) {
                int idx = g_denoise_ring_idx;
                ai::ai_denoise(g_d_denoise_ring[(idx+0)%5], g_d_denoise_ring[(idx+1)%5],
                    g_d_denoise_ring[(idx+2)%5], g_d_denoise_ring[(idx+3)%5],
                    g_d_denoise_ring[(idx+4)%5], g_d_ai_s1, ai_sw, ai_sh, g_stream);
                filters::resize_bilinear(g_d_ai_s1, d_rgb, ai_sw, ai_sh, w, h, 3, g_stream);
            }
        } else {
            cudaMemcpyAsync(g_d_denoise_ring[g_denoise_ring_idx], d_rgb, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
            g_denoise_ring_idx = (g_denoise_ring_idx + 1) % DENOISE_RING_SIZE;
            if (g_denoise_ring_count < DENOISE_RING_SIZE) g_denoise_ring_count++;
            if (g_denoise_ring_count >= DENOISE_RING_SIZE) {
                int idx = g_denoise_ring_idx;
                ai::ai_denoise(g_d_denoise_ring[(idx+0)%5], g_d_denoise_ring[(idx+1)%5],
                    g_d_denoise_ring[(idx+2)%5], g_d_denoise_ring[(idx+3)%5],
                    g_d_denoise_ring[(idx+4)%5], d_rgb, w, h, g_stream);
            }
        }
    }
    if (ai_live && g_ai_depth && g_ai_depth_ready && g_d_ai_vis) {
        if ((g_frame_count % 3) == 0)
            ai::ai_depth(d_rgb, g_d_ai_depth_buf, w, h, g_stream);
        ai::launch_normalize_depth(g_d_ai_depth_buf, w*h, g_stream);
        ai::launch_depth_to_rgb(g_d_ai_depth_buf, g_d_ai_vis, w*h, g_stream);
        filters::frame_blend(d_rgb, g_d_ai_vis, d_rgb, w, h, 3, 0.55f, g_stream);
    }
    // Snapshot the clean frame BEFORE any overlay blends below, so flow
    // compares raw frames (not its own past visualizations -> ghosting).
    if (ai_live && g_ai_flow && g_ai_flow_ready && g_d_flow_curr)
        cudaMemcpyAsync(g_d_flow_curr, d_rgb, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
    if (ai_live && g_ai_flow && g_ai_flow_ready && g_d_ai_vis) {
        if (g_flow_has_prev && ((g_frame_count % 2) == 0))
            ai::ai_flow(g_d_flow_prev, g_d_flow_curr, g_d_flow_buf, w, h, g_stream);
        if (g_flow_has_prev) {
            ai::launch_flow_to_rgb(g_d_flow_buf, g_d_ai_vis, w, h, g_stream);
            filters::frame_blend(d_rgb, g_d_ai_vis, d_rgb, w, h, 3, 0.6f, g_stream);
        }
        // rotate: current becomes previous (pointer swap, no copy)
        { uint8_t* t = g_d_flow_prev; g_d_flow_prev = g_d_flow_curr; g_d_flow_curr = t; }
        g_flow_has_prev = true;
    }
    // Pose/gaze infer here only in camera mode; in tutorial mode the gpu
    // loop runs them on the camera feed instead. Draws: draw_ai_overlays.
    if (ai_live && g_ai_pose && g_ai_pose_ready && g_src_mode == 0) {
        ai::ai_pose(d_rgb, w, h, g_pose_landmarks, g_stream);
    }
    // Full-replace effects. Matting is the heaviest (two full-res convs
    // fall back to CUDA): infer every 2nd frame into cache, replay cached
    // composite on skip frames. Lowlight/anime are cheap: every frame.
    if (ai_live && g_ai_matting && g_ai_matting_ready && g_d_ai_matt) {
        if ((g_frame_count % 2) == 0) {
            if (ai::ai_matting(d_rgb, g_d_ai_matt, w, h, g_stream)) {
                g_matt_cached = true;
                cudaMemcpyAsync(d_rgb, g_d_ai_matt, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
            }
        } else if (g_matt_cached) {
            cudaMemcpyAsync(d_rgb, g_d_ai_matt, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
    }
    if (ai_live && g_ai_lowlight && g_ai_lowlight_ready) {
        if (ai_small) {
            filters::resize_bilinear(d_rgb, g_d_ai_s0, w, h, ai_sw, ai_sh, 3, g_stream);
            if (ai::ai_lowlight(g_d_ai_s0, g_d_ai_s1, ai_sw, ai_sh, g_stream))
                filters::resize_bilinear(g_d_ai_s1, d_rgb, ai_sw, ai_sh, w, h, 3, g_stream);
        } else {
            uint8_t* d_tmp = d_out;
            if (ai::ai_lowlight(d_rgb, d_tmp, w, h, g_stream))
                cudaMemcpyAsync(d_rgb, d_tmp, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
    }
    if (ai_live && g_ai_anime && g_ai_anime_ready) {
        if (ai_small) {
            filters::resize_bilinear(d_rgb, g_d_ai_s0, w, h, ai_sw, ai_sh, 3, g_stream);
            if (ai::ai_anime(g_d_ai_s0, g_d_ai_s1, ai_sw, ai_sh, g_stream))
                filters::resize_bilinear(g_d_ai_s1, d_rgb, ai_sw, ai_sh, w, h, 3, g_stream);
        } else {
            uint8_t* d_tmp = d_out;
            if (ai::ai_anime(d_rgb, d_tmp, w, h, g_stream))
                cudaMemcpyAsync(d_rgb, d_tmp, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
    }
    // Face/hands infer here only in camera mode. In tutorial mode the gpu
    // loop runs them on the CAMERA feed instead (results in camera coords,
    // mapped into the PiP by draw_ai_overlays). Draws live in that pass.
    if (ai_live && g_ai_face && g_ai_face_ready && g_src_mode == 0) {
        if ((g_frame_count % 2) == 0)
            ai::ai_face(d_rgb, w, h, g_face_boxes, 4, g_stream);
    }
    if (ai_live && g_ai_hands && g_ai_hands_ready && g_src_mode == 0) {
        if ((g_frame_count % 3) == 0)
            ai::ai_hands(d_rgb, w, h, g_hand_joints, 2, g_stream);
    }
    if (ai_live && g_ai_gaze && g_ai_gaze_ready && g_src_mode == 0) {
        if ((g_frame_count % 3) == 0)
            ai::ai_gaze(d_rgb, w, h, g_gaze_eyes, 2, g_stream);
    }
    if (ai_live && g_ai_detect && g_ai_detect_ready && g_src_mode == 0) {
        if ((g_frame_count % 2) == 0)
            ai::ai_detect(d_rgb, w, h, g_det_boxes, 16, g_stream);
    }

    // ---- temporal group (consecutive clean frames at capture size) ----
    bool need_prev = g_interp || g_tempden || g_tempstab;
    if (need_prev && g_d_stage && g_d_temporal_prev)
        cudaMemcpyAsync(g_d_stage, d_rgb, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
    if (need_prev && g_temporal_has_prev && g_d_stage && g_d_temporal_prev) {
        if (g_interp) {
            filters::frame_blend(g_d_temporal_prev, d_rgb, g_d_tmp, w, h, 3, 0.5f, g_stream);
            cudaMemcpyAsync(d_rgb, g_d_tmp, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
        if (g_tempden) {
            filters::temporal_denoise_rgb(g_d_temporal_prev, d_rgb, g_d_tmp, w, h, 0.25f, g_stream);
            cudaMemcpyAsync(d_rgb, g_d_tmp, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
        if (g_tempstab && g_d_motion_accum) {
            // stabilize only estimates (block match -> motion vector).
            // Read it back, smooth it, and warp -- otherwise d_dst is
            // never written and the picture freezes on stale data.
            filters::temporal_stabilize_rgb(g_d_temporal_prev, d_rgb, g_d_tmp, w, h,
                                            g_d_motion_accum, 16, 8, g_stream);
            int h_acc[3] = {};
            cudaMemcpy(h_acc, g_d_motion_accum, 3 * sizeof(int), cudaMemcpyDeviceToHost);
            int raw_dx = h_acc[2] > 0 ? h_acc[0] / h_acc[2] : 0;
            int raw_dy = h_acc[2] > 0 ? h_acc[1] / h_acc[2] : 0;
            g_stab_dx = (int)(0.85f * g_stab_dx + 0.15f * raw_dx);
            g_stab_dy = (int)(0.85f * g_stab_dy + 0.15f * raw_dy);
            if (g_stab_dx != 0 || g_stab_dy != 0)
                filters::warp_translate_rgb(d_rgb, g_d_tmp, w, h,
                                            g_stab_dx, g_stab_dy, g_stream);
            else
                cudaMemcpyAsync(g_d_tmp, d_rgb, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
            cudaMemcpyAsync(d_rgb, g_d_tmp, w*h*3, cudaMemcpyDeviceToDevice, g_stream);
        }
    }
    if (need_prev && g_d_stage && g_d_temporal_prev) {
        uint8_t* t = g_d_temporal_prev; g_d_temporal_prev = g_d_stage; g_d_stage = t;
        g_temporal_has_prev = true;
    }

    uint8_t* d_src = d_rgb;
    uint8_t* d_dst = d_out;

    // Super-res is camera-only: 2x on a 960x540 feed would overflow g_d_sr_out.
    if (g_sr && g_src_mode == 0) {
        out_w = w * 2; out_h = h * 2;
        filters::super_res_2x(d_src, g_d_sr_out, w, h, 3, 0.5f, g_stream);
        d_src = g_d_sr_out; d_dst = d_rgb;
    }

    auto sw = [&]() { uint8_t* t = d_src; d_src = d_dst; d_dst = t; };
    if (g_denoise) { filters::denoise_bilateral(d_src, d_dst, out_w, out_h, 3, 15.f, 25.f, 5, g_stream); sw(); }
    if (g_clahe) { filters::clahe_rgb(d_src, d_dst, out_w, out_h, 3.f, 64, g_stream); sw(); }
    if (g_lut_on && g_d_lut) { filters::lut3d_rgb(d_src, d_dst, out_w, out_h, g_d_lut, 17, 1.f, g_stream); sw(); }
    if (g_bright) { filters::brightness_contrast(d_src, d_dst, out_w, out_h, 20.f, 1.2f, g_stream); sw(); }
    if (g_sat) { filters::saturation_rgb(d_src, d_dst, out_w, out_h, 1.4f, g_stream); sw(); }
    if (g_gamma) { filters::gamma_rgb(d_src, d_dst, out_w, out_h, 0.8f, g_stream); sw(); }
    if (g_wb) { filters::white_balance_rgb(d_src, d_dst, out_w, out_h, 15.f, 5.f, g_stream); sw(); }
    if (g_blur) { filters::gaussian_blur(d_src, d_dst, out_w, out_h, 3, 2.f, g_stream); sw(); }
    if (g_sharp) { filters::sharpen_rgb(d_src, d_dst, out_w, out_h, 1.5f, g_stream); sw(); }
    if (g_vignette) { filters::vignette_rgb(d_src, d_dst, out_w, out_h, 0.6f, g_stream); sw(); }
    if (g_lens) { filters::lens_distortion(d_src, d_dst, out_w, out_h, 3, -0.3f, g_stream); sw(); }
    if (g_edge) { filters::edge_detect_rgb(d_src, d_dst, out_w, out_h, g_stream); sw(); }
    if (g_grain) { filters::film_grain_rgb(d_src, out_w, out_h, 25.f, g_frame_count, g_stream); }
    if (g_flip) {
        if (g_flip_mode==1) { filters::flip_horizontal_gpu(d_src, d_dst, out_w, out_h, g_stream); sw(); }
        else if (g_flip_mode==2) { filters::flip_vertical_gpu(d_src, d_dst, out_w, out_h, g_stream); sw(); }
    }
    if (g_dirblur) {
        float angs[] = {45.f, 135.f}; int lens[] = {20, 12};
        int k = g_dirblur_idx - 1; if (k < 0) k = 0; if (k > 1) k = 1;
        filters::directional_blur(d_src, d_dst, out_w, out_h, 3, angs[k], lens[k], g_stream); sw();
    }
    if (g_bgblur) { filters::bg_blur_rgb(d_src, d_dst, out_w, out_h, out_w*0.5f, out_h*0.5f, 0.3f, 8.0f, g_stream); sw(); }
    if (g_hdr) { filters::hdr_tone_map_rgb(d_src, d_dst, out_w, out_h, g_hdr_method-1, 100.0f, g_stream); sw(); }
    if (g_chroma) { filters::chroma_key_rgb(d_src, d_dst, out_w, out_h, 60.f, 160.f, 0.3f, 0.2f, 0.5f, 0, 0, 0, 1.0f, g_stream); sw(); }
    if (g_cropzoom && g_d_zoom) {
        float zf = g_zoom_idx==1 ? 1.25f : g_zoom_idx==2 ? 1.5f : 2.0f;
        uint32_t cw = (uint32_t)(out_w / zf) & ~1u, ch = (uint32_t)(out_h / zf) & ~1u;
        if (cw < 16) cw = 16; if (ch < 16) ch = 16;
        int cx = ((int)out_w - (int)cw) / 2, cy = ((int)out_h - (int)ch) / 2;
        filters::crop_rgb(d_src, g_d_zoom, out_w, out_h, cw, ch, cx, cy, 3, g_stream);
        filters::resize_bilinear(g_d_zoom, d_dst, cw, ch, out_w, out_h, 3, g_stream); sw();
    }
    // Auto-frame: face-tracked crop window with smoothing + deadband.
    // Reuses the zoom scratch (cropzoom already finished above if on).
    // AutoFrame follows faces in the program feed (camera mode only: in
    // tutorial mode face boxes live in camera coords, not program coords).
    if (g_autoframe && g_af_idx > 0 && ai_live && g_ai_face_ready && g_d_zoom &&
        g_src_mode == 0) {
        if ((g_frame_count % 4) == 0) {
            int n = ai::ai_face(d_src, out_w, out_h, g_af_boxes, 4, g_stream);
            if (n <= 0) {
                for (int i = 0; i < 4; i++) g_af_boxes[i].score = 0;
            }
        }
        int bi = -1; float bs = 0.4f; // measured max ~0.5 on real faces
        for (int i = 0; i < 4; i++)
            if (g_af_boxes[i].score > bs) { bs = g_af_boxes[i].score; bi = i; }
        if (bi >= 0) {
            float af_expand[] = {0, 2.5f, 2.0f, 1.5f};
            float fw = g_af_boxes[bi].x2 - g_af_boxes[bi].x1;
            float fh = g_af_boxes[bi].y2 - g_af_boxes[bi].y1;
            float tw = fw * af_expand[g_af_idx], th = fh * af_expand[g_af_idx];
            if (tw > (float)out_w) tw = (float)out_w;
            if (th > (float)out_h) th = (float)out_h;
            float tcx = (g_af_boxes[bi].x1 + g_af_boxes[bi].x2) * 0.5f;
            float tcy = (g_af_boxes[bi].y1 + g_af_boxes[bi].y2) * 0.5f - 0.1f * th;
            float tx = tcx - tw * 0.5f, ty = tcy - th * 0.5f;
            if (!g_af_init || g_af_fw != (int)out_w || g_af_fh != (int)out_h) {
                g_af_x = tx; g_af_y = ty; g_af_w = tw; g_af_h = th;
                g_af_fw = out_w; g_af_fh = out_h; g_af_init = true;
            } else {
                float dx = tx - g_af_x, dy = ty - g_af_y;
                float dw = tw - g_af_w, dh = th - g_af_h;
                if (fabsf(dx) < 5) dx = 0; if (fabsf(dy) < 5) dy = 0;
                if (fabsf(dw) < 4) dw = 0; if (fabsf(dh) < 4) dh = 0;
                g_af_x += dx * 0.12f; g_af_y += dy * 0.12f;
                g_af_w += dw * 0.12f; g_af_h += dh * 0.12f;
            }
            if (g_af_w > out_w) g_af_w = out_w;
            if (g_af_h > out_h) g_af_h = out_h;
            if (g_af_x < 0) g_af_x = 0; if (g_af_y < 0) g_af_y = 0;
            if (g_af_x + g_af_w > out_w) g_af_x = out_w - g_af_w;
            if (g_af_y + g_af_h > out_h) g_af_y = out_h - g_af_h;
            uint32_t cw = ((uint32_t)g_af_w) & ~1u, ch = ((uint32_t)g_af_h) & ~1u;
            if (cw >= 16 && ch >= 16) {
                filters::crop_rgb(d_src, g_d_zoom, out_w, out_h, cw, ch,
                                  (int)g_af_x, (int)g_af_y, 3, g_stream);
                filters::resize_bilinear(g_d_zoom, d_dst, cw, ch, out_w, out_h, 3, g_stream);
                sw();
            }
        }
    }

    // Caller (display + virtual-cam paths) always reads the result from
    // d_rgb. With an odd number of swaps the result sits in d_out instead,
    // so copy it back. Without this, enabling 1/3/5... filters shows stale
    // frames (looks frozen) and the vcam converts the wrong buffer.
    if (d_src != d_rgb) {
        cudaMemcpyAsync(d_rgb, d_src, (size_t)out_w * out_h * 3,
                        cudaMemcpyDeviceToDevice, g_stream);
    }
}

// ============================================================================
// Transcode tab -- drives bin\kagerou.exe as a child process, streams its log.
// Live preview keeps running behind it, so you can stream AND transcode.
// ============================================================================
static int g_tab = 0; // 0 = live camera, 1 = file transcode, 2 = live-cut record
static double g_selftest_sec = 0; // --selftest N: headless record/cut/stop self-check

// transcode options (independent from the live pipeline flags)
static bool tc_denoise=false, tc_clahe=false, tc_sr=false, tc_interp=false;
static bool tc_blur=false, tc_sharp=false;
static bool tc_bright=false, tc_sat=false, tc_gamma=false;
static bool tc_vignette=false, tc_grain=false, tc_edge=false;
static bool tc_wb=false, tc_flip=false, tc_lens=false;
static bool tc_bgblur=false, tc_tempden=false, tc_tempstab=false, tc_chroma=false;
static bool tc_ai_denoise=false, tc_ai_depth=false, tc_ai_flow=false, tc_ai_pose=false;
static bool tc_ai_matting=false, tc_ai_face=false, tc_ai_hands=false;
static bool tc_ai_gaze=false, tc_ai_lowlight=false, tc_ai_anime=false;
static bool tc_ai_detect=false;
static bool tc_autoframe=false;
static int tc_lut_preset=0, tc_flip_mode=0; // 0=off
static int tc_codec=0;    // 0=h264 1=h265
static int tc_fps_idx=0;  // 0:30 1:60
static int tc_br_idx=0;   // 0:auto 1:5M 2:10M 3:20M
static int tc_scale_idx=0;   // 0 off, 1:0.5x, 2:1.5x, 3:2x (needs probed dims)
static int tc_dirblur_idx=0; // 0 off, 1:45/20, 2:135/12
static int tc_hdr=0;         // 0 off, 1 Reinhard, 2 ACES
static int tc_peak_idx=0;    // 0:100 1:200 2:400 nits (with HDR)
static int tc_crop_idx=0;    // 0 off, 1:center 75%, 2:center 50% (needs probed dims)
static int tc_vid_w=0, tc_vid_h=0; // ffprobe result, 0 = unknown
static char tc_input[MAX_PATH] = "";
static std::atomic<int> g_tc_state{0}; // 0 idle, 1 running, 2 done, 3 failed
static std::atomic<int> g_tc_frames{0};
static std::atomic<int> g_tc_total{0};
static std::atomic<DWORD> g_tc_exit{0};
static std::atomic<bool> g_tc_cancel{false};
static HANDLE g_tc_proc = nullptr;
static std::thread g_tc_thread;
static std::mutex g_tc_log_mtx;
static std::vector<std::string> g_tc_log;
static const int TC_LOG_MAX = 200;
static RECT tc_rc_browse, tc_rc_go, tc_rc_codec, tc_rc_fps, tc_rc_br;
static RECT tc_rc_chips[37];
static RECT tc_rc_logbox = {0,0,0,0};
static int tc_log_y0 = 0;
static int g_tc_log_pos = -1; // -1 = follow tail, else first visible line
static int tc_log_vis = 10;

static void tc_log(const std::string& line) {
    std::lock_guard<std::mutex> l(g_tc_log_mtx);
    if ((int)g_tc_log.size() >= TC_LOG_MAX) g_tc_log.erase(g_tc_log.begin());
    g_tc_log.push_back(line);
}

static const char* TC_LUTS[] = {"warm","cool","cinema","vintage","contrast","desat"};
static const char* TC_LUTS_DISP[] = {"Warm","Cool","Cinema","Vintage","Contrast","Desat"};

static std::string tc_outdir() {
    std::string in(tc_input);
    std::string base = in;
    size_t p = base.find_last_of("\\/");
    if (p != std::string::npos) base = base.substr(p+1);
    size_t d = base.find_last_of('.');
    if (d != std::string::npos) base = base.substr(0, d);
    if (base.empty()) base = "output";
    return get_exe_dir() + "\\output\\" + base;
}

static bool tc_ai_ok(bool flag, const std::atomic<bool>& ready) {
    return flag && ready.load() && !g_ai_warming.load();
}

static std::string tc_cmdline() {
    std::string cmd = "\"" + get_exe_dir() + "\\kagerou.exe\" \"" + tc_input + "\"";
    cmd += " -o \"" + tc_outdir() + "\"";
    if (tc_denoise) cmd += " --denoise";
    if (tc_clahe) cmd += " --clahe";
    if (tc_sr) cmd += " --super-res";
    if (tc_interp) cmd += " --frame-interp";
    if (tc_lut_preset > 0) { cmd += " --lut "; cmd += TC_LUTS[tc_lut_preset-1]; }
    if (tc_blur) cmd += " --blur 2.0";
    if (tc_sharp) cmd += " --sharpen 1.5";
    if (tc_bright) cmd += " --brightness 20 --contrast 1.2";
    if (tc_sat) cmd += " --saturation 1.4";
    if (tc_gamma) cmd += " --gamma 0.8";
    if (tc_vignette) cmd += " --vignette 0.6";
    if (tc_grain) cmd += " --grain 25";
    if (tc_edge) cmd += " --edge-detect";
    if (tc_wb) cmd += " --white-balance";
    if (tc_lens) cmd += " --lens-distort -0.3";
    if (tc_flip_mode == 1) cmd += " --flip h";
    else if (tc_flip_mode == 2) cmd += " --flip v";
    if (tc_bgblur) cmd += " --bg-blur 8";
    if (tc_tempden) cmd += " --temporal-denoise 0.25";
    if (tc_tempstab) cmd += " --temporal-stab";
    if (tc_chroma) cmd += " --chroma-key";
    if (tc_dirblur_idx == 1) cmd += " --dir-blur 45 20";
    else if (tc_dirblur_idx == 2) cmd += " --dir-blur 135 12";
    if (tc_hdr == 1) cmd += " --hdr 0";
    else if (tc_hdr == 2) cmd += " --hdr 1";
    if (tc_hdr > 0) {
        int pn[] = {100, 200, 400};
        cmd += " --peak-nits " + std::to_string(pn[tc_peak_idx]);
    }
    if (tc_scale_idx > 0 && tc_vid_w > 0 && tc_vid_h > 0) {
        float f[] = {0, 0.5f, 1.5f, 2.0f};
        int W = ((int)(tc_vid_w * f[tc_scale_idx])) & ~1;
        int H = ((int)(tc_vid_h * f[tc_scale_idx])) & ~1;
        if (W < 64) W = 64; if (H < 64) H = 64;
        cmd += " --scale " + std::to_string(W) + "x" + std::to_string(H);
    }
    if (tc_crop_idx > 0 && tc_vid_w > 0 && tc_vid_h > 0) {
        float f = (tc_crop_idx == 1) ? 0.75f : 0.5f;
        int W = ((int)(tc_vid_w * f)) & ~1, H = ((int)(tc_vid_h * f)) & ~1;
        int X = (tc_vid_w - W) / 2, Y = (tc_vid_h - H) / 2;
        cmd += " --crop " + std::to_string(W) + "x" + std::to_string(H) +
               "+" + std::to_string(X) + "+" + std::to_string(Y);
    }
    // Absolute model paths: the child runs with CWD=bin/, where the
    // default relative "models/..." paths don't resolve (this was the
    // "AI depth failed on frame" spam -- init failed, every frame failed).
    if (tc_ai_ok(tc_ai_denoise, g_ai_denoise_ready))
        cmd += " --ai-denoise \"" + resolve_model("models/denoise/fastdvdnet.onnx") + "\"";
    if (tc_ai_ok(tc_ai_depth, g_ai_depth_ready))
        cmd += " --ai-depth \"" + resolve_model("models/depth/depth_anything_v2.onnx") + "\"";
    if (tc_ai_ok(tc_ai_flow, g_ai_flow_ready))
        cmd += " --ai-flow \"" + resolve_model("models/flow/raft_small.onnx") + "\"";
    if (tc_ai_ok(tc_ai_pose, g_ai_pose_ready))
        cmd += " --ai-pose \"" + resolve_model("models/pose/pose_landmark.onnx") + "\"";
    if (tc_ai_ok(tc_ai_matting, g_ai_matting_ready))
        cmd += " --ai-matting \"" + resolve_model("models/matting/rvm_mobilenetv3_static.onnx") + "\"";
    if (tc_ai_ok(tc_ai_face, g_ai_face_ready))
        cmd += " --ai-face \"" + resolve_model("models/face/yunet_2023mar.onnx") + "\"";
    if (tc_ai_ok(tc_ai_hands, g_ai_hands_ready))
        cmd += " --ai-hands";
    if (tc_ai_ok(tc_ai_gaze, g_ai_gaze_ready))
        cmd += " --ai-gaze";
    if (tc_ai_ok(tc_ai_lowlight, g_ai_lowlight_ready))
        cmd += " --ai-lowlight \"" + resolve_model("models/lowlight/zero_dce.onnx") + "\"";
    if (tc_ai_ok(tc_ai_anime, g_ai_anime_ready))
        cmd += " --ai-anime \"" + resolve_model("models/style/animeganv3_hayao.onnx") + "\"";
    if (tc_ai_ok(tc_ai_detect, g_ai_detect_ready))
        cmd += " --ai-detect \"" + resolve_model("models/detect/yolov8n.onnx") + "\"";
    if (tc_autoframe && !g_ai_warming.load() && g_ai_face_ready.load())
        cmd += " --ai-autoframe";
    if (tc_codec == 1) cmd += " --codec h265";
    cmd += (tc_fps_idx == 1) ? " --fps 60" : " --fps 30";
    int brs[] = {0, 5000, 10000, 20000};
    if (brs[tc_br_idx]) cmd += " --bitrate " + std::to_string(brs[tc_br_idx]);
    return cmd;
}

static void tc_worker(std::string cmd, std::string workdir, std::string outdir) {
    SECURITY_ATTRIBUTES sa = {sizeof(sa), nullptr, TRUE};
    HANDLE hRead = nullptr, hWrite = nullptr;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
        tc_log("error: cannot create log pipe."); g_tc_state = 3; return;
    }
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si = {}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWrite; si.hStdError = hWrite;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi = {};
    // CreateDirectoryA is single-level -- ensure the parent exists first.
    CreateDirectoryA((workdir + "\\output").c_str(), nullptr);
    CreateDirectoryA(outdir.c_str(), nullptr);
    BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, workdir.c_str(), &si, &pi);
    CloseHandle(hWrite);
    if (!ok) {
        tc_log("error: cannot start kagerou.exe -- run: build.bat all sdk minimp4 onnx");
        CloseHandle(hRead); g_tc_state = 3; return;
    }
    g_tc_proc = pi.hProcess;
    char buf[4096]; DWORD n = 0;
    std::string cur; cur.reserve(512);
    while (g_running) {
        BOOL r = ReadFile(hRead, buf, sizeof(buf) - 1, &n, nullptr);
        if (!r || n == 0) break;
        for (DWORD i = 0; i < n; i++) {
            char c = buf[i];
            if (c == '\n' || c == '\r') {
                if (!cur.empty()) {
                    int f = 0, tot = 0;
                    if (sscanf(cur.c_str(), "  processed %d frames", &f) == 1) {
                        g_tc_frames = f; // progress tick: counted, not stored
                    } else {
                        if (sscanf(cur.c_str(), "[input] %*d total NALUs, %d VCL", &tot) == 1)
                            g_tc_total = tot;
                        tc_log(cur);
                    }
                    cur.clear();
                }
            } else if (c != 0) cur += c;
        }
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    g_tc_exit = code;
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess); CloseHandle(hRead);
    g_tc_proc = nullptr;
    if (g_tc_cancel) { tc_log("cancelled by user."); g_tc_state = 3; }
    else if (code == 0) { tc_log("done. Output in: " + outdir); g_tc_state = 2; }
    else { char m[64]; sprintf(m, "failed (exit %lu). See log above.", code); tc_log(m); g_tc_state = 3; }
}

static void tc_start() {
    if (g_tc_state == 1) return;
    if (!tc_input[0]) { tc_log("select an input file first (Browse...)."); return; }
    if (!file_exists(tc_input)) { tc_log("input file not found."); return; }
    if (!file_exists((get_exe_dir() + "\\kagerou.exe").c_str())) {
        tc_log("kagerou.exe missing -- run: build.bat all sdk minimp4 onnx"); return;
    }
    if (g_tc_thread.joinable()) g_tc_thread.join();
    { std::lock_guard<std::mutex> l(g_tc_log_mtx); g_tc_log.clear(); }
    g_tc_log_pos = -1;
    g_tc_frames = 0; g_tc_total = 0; g_tc_cancel = false;
    g_tc_state = 1;
    tc_log(std::string("input: ") + tc_input);
    std::string cmd = tc_cmdline();
    tc_log(cmd);
    g_tc_thread = std::thread(tc_worker, cmd, get_exe_dir(), tc_outdir());
}

static void tc_stop() {
    if (g_tc_state != 1) return;
    g_tc_cancel = true;
    if (g_tc_proc) TerminateProcess(g_tc_proc, 1);
}

static void tc_probe_dims() {
    tc_vid_w = tc_vid_h = 0;
    if (!tc_input[0]) return;
    char cmd[1024];
    sprintf(cmd, "ffprobe -hide_banner -loglevel error -select_streams v:0 "
            "-show_entries stream=width,height -of csv=p=0 \"%s\"", tc_input);
    FILE* p = _popen(cmd, "r");
    if (!p) return;
    int w = 0, h = 0;
    if (fscanf(p, "%d,%d", &w, &h) == 2 && w > 0 && h > 0) {
        tc_vid_w = w; tc_vid_h = h;
        char m[96]; sprintf(m, "video: %dx%d", w, h);
        tc_log(m);
    }
    _pclose(p);
}

static void tc_browse() {
    if (g_tc_state == 1) return;
    char path[MAX_PATH] = "";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = path;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "Video\0*.mp4;*.mkv;*.avi;*.mov;*.webm;*.h264;*.h265\0All files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn)) {
        strncpy(tc_input, path, MAX_PATH - 1);
        tc_input[MAX_PATH-1] = 0;
        tc_log(std::string("selected: ") + tc_input);
        tc_probe_dims();
    }
}

// AI usability for transcode chips (mirrors sidebar logic)
// 0 usable, 1 warming, 2 missing
static int tc_ai_usable(int i) {
    bool ok = false;
    if (i == 25) ok = g_ai_denoise_ready.load();
    else if (i == 26) ok = g_ai_depth_ready.load();
    else if (i == 27) ok = g_ai_flow_ready.load();
    else if (i == 28) ok = g_ai_pose_ready.load();
    else if (i == 29) ok = g_ai_matting_ready.load();
    else if (i == 30) ok = g_ai_face_ready.load();
    else if (i == 31) ok = g_ai_hands_ready.load();
    else if (i == 32) ok = g_ai_gaze_ready.load();
    else if (i == 33) ok = g_ai_lowlight_ready.load();
    else if (i == 34) ok = g_ai_anime_ready.load();
    else if (i == 35) ok = g_ai_detect_ready.load();
    else if (i == 36) ok = g_ai_face_ready.load();
    if (!ok) return 2;
    return g_ai_warming.load() ? 1 : 0;
}

// AI usability for transcode chips (mirrors sidebar logic)
static int tc_ai_state(bool flag, const std::atomic<bool>& ready) {
    (void)flag;
    if (!ready.load()) return 0;
    return g_ai_warming.load() ? 1 : 2;
}

// ============================================================================
// UI
// ============================================================================
static void draw_rounded_rect(HDC hdc, int x, int y, int w, int h, int r, HBRUSH br, COLORREF border) {
    HPEN pen = CreatePen(PS_SOLID, 1, border);
    HPEN op = (HPEN)SelectObject(hdc, pen);
    HBRUSH ob = (HBRUSH)SelectObject(hdc, br);
    RoundRect(hdc, x, y, x+w, y+h, r*2, r*2);
    SelectObject(hdc, op); SelectObject(hdc, ob); DeleteObject(pen);
}

static void draw_pill(HDC hdc, int x, int y, int w, int h, const char* txt, COLORREF bg, COLORREF fg, HFONT f) {
    HBRUSH bb = CreateSolidBrush(bg);
    draw_rounded_rect(hdc, x, y, w, h, h/2, bb, bg);
    DeleteObject(bb);
    SelectObject(hdc, f);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, fg);
    RECT r = {x, y, x+w, y+h};
    DrawTextA(hdc, txt, -1, &r, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
}

static int g_hover_tab = -1;

// ---- transcode panel hit ids ----
enum { TID_BROWSE = 1, TID_GO = 2, TID_CODEC = 3, TID_FPS = 4, TID_BR = 5, TID_CHIP = 10 };
struct TcTog { const char* name; bool* flag; };
static TcTog tc_togs[] = {
    {"Denoise",&tc_denoise},{"CLAHE",&tc_clahe},{"SuperRes",&tc_sr},{"Frame2x",&tc_interp},
    {"Blur",&tc_blur},{"Sharp",&tc_sharp},{"B/C",&tc_bright},{"Saturate",&tc_sat},
    {"Gamma",&tc_gamma},{"Vignette",&tc_vignette},{"Grain",&tc_grain},{"Edge",&tc_edge},
    {"WhiteBal",&tc_wb},{"Lens",&tc_lens},
    {"BgBlur",&tc_bgblur},{"TempDen",&tc_tempden},{"TempStab",&tc_tempstab},{"Chroma",&tc_chroma},
};
static const int NTC_TOG = sizeof(tc_togs)/sizeof(tc_togs[0]); // 18
// chip index: 0..17 toggles, 18 LUT, 19 Flip, 20 scale, 21 dirblur,
// 22 hdr, 23 crop, 24 peaknits, 25..35 AI toggles, 36 autoframe
static const int NTC_CHIP = 37;

static void tc_chip_label(int i, char* out, size_t n) {
    if (i < NTC_TOG) { strncpy(out, tc_togs[i].name, n-1); out[n-1] = 0; return; }
    if (i == 18) {
        if (tc_lut_preset == 0) snprintf(out, n, "LUT");
        else snprintf(out, n, "LUT: %s", TC_LUTS_DISP[tc_lut_preset-1]);
        return;
    }
    if (i == 19) {
        if (tc_flip_mode == 0) snprintf(out, n, "Flip");
        else snprintf(out, n, tc_flip_mode == 2 ? "Flip: Vert" : "Flip: Horz");
        return;
    }
    if (i == 20) {
        if (tc_scale_idx == 0 || tc_vid_w <= 0) snprintf(out, n, "Scale");
        else { const char* sn[] = {"", "0.5x", "1.5x", "2x"}; snprintf(out, n, "Scale %s", sn[tc_scale_idx]); }
        return;
    }
    if (i == 21) {
        if (tc_dirblur_idx == 0) snprintf(out, n, "DirBlur");
        else snprintf(out, n, tc_dirblur_idx == 2 ? "DB 135/12" : "DB 45/20");
        return;
    }
    if (i == 22) {
        if (tc_hdr == 0) snprintf(out, n, "HDR");
        else snprintf(out, n, tc_hdr == 2 ? "HDR: ACES" : "HDR: Reinhard");
        return;
    }
    if (i == 23) {
        if (tc_crop_idx == 0 || tc_vid_w <= 0) snprintf(out, n, "Crop");
        else snprintf(out, n, tc_crop_idx == 2 ? "Crop 50%" : "Crop 75%");
        return;
    }
    if (i == 24) {
        const char* pn[] = {"100", "200", "400"};
        snprintf(out, n, "Peak %s", pn[tc_peak_idx]);
        return;
    }
    if (i == 36) { strncpy(out, "AutoFrame", n-1); out[n-1] = 0; return; }
    const char* nm[] = {"AI Denoise", "AI Depth", "AI Flow", "AI Pose",
                        "AI Matting", "AI Face", "AI Hands", "AI Gaze",
                        "AI LowLight", "AI Anime", "AI Detect"};
    strncpy(out, nm[i-25], n-1); out[n-1] = 0;
}

static bool tc_chip_on(int i) {
    if (i < NTC_TOG) return *tc_togs[i].flag;
    if (i == 18) return tc_lut_preset > 0;
    if (i == 19) return tc_flip_mode > 0;
    if (i == 20) return tc_scale_idx > 0;
    if (i == 21) return tc_dirblur_idx > 0;
    if (i == 22) return tc_hdr > 0;
    if (i == 23) return tc_crop_idx > 0;
    if (i == 24) return true; // peak-nits always "set", applies with HDR
    if (i == 25) return tc_ai_denoise;
    if (i == 26) return tc_ai_depth;
    if (i == 27) return tc_ai_flow;
    if (i == 28) return tc_ai_pose;
    if (i == 29) return tc_ai_matting;
    if (i == 30) return tc_ai_face;
    if (i == 31) return tc_ai_hands;
    if (i == 32) return tc_ai_gaze;
    if (i == 33) return tc_ai_lowlight;
    if (i == 34) return tc_ai_anime;
    if (i == 35) return tc_ai_detect;
    return tc_autoframe;
}

// 0 usable, 1 warming, 2 missing
static int tc_chip_state(int i) {
    if (i < 25) {
        // scale/crop need probed video dims
        if ((i == 20 || i == 23) && tc_vid_w <= 0) return 2;
        return 0;
    }
    const std::atomic<bool>* rd = &g_ai_denoise_ready;
    if (i == 26) rd = &g_ai_depth_ready;
    else if (i == 27) rd = &g_ai_flow_ready;
    else if (i == 28) rd = &g_ai_pose_ready;
    else if (i == 29) rd = &g_ai_matting_ready;
    else if (i == 30) rd = &g_ai_face_ready;
    else if (i == 31) rd = &g_ai_hands_ready;
    else if (i == 32) rd = &g_ai_gaze_ready;
    else if (i == 33) rd = &g_ai_lowlight_ready;
    else if (i == 34) rd = &g_ai_anime_ready;
    else if (i == 35) rd = &g_ai_detect_ready;
    else if (i == 36) rd = &g_ai_face_ready;
    else return 2;
    if (!rd->load()) return 2;
    return g_ai_warming.load() ? 1 : 0;
}

static void tc_layout(int W, int H) {
    int px = SIDE_W + 14, pw = W - SIDE_W - 28;
    if (pw < 200) pw = 200;
    tc_rc_browse = {W - 134, TAB_H + 66, W - 14, TAB_H + 96};
    int cols = 5;
    int cw = (pw - 4 * 8) / cols;
    int gx = px, gy = TAB_H + 208;
    for (int i = 0; i < NTC_CHIP; i++) {
        int cx = gx + (i % cols) * (cw + 8);
        int cy = gy + (i / cols) * 38;
        tc_rc_chips[i] = {cx, cy, cx + cw, cy + 30};
    }
    int rows = (NTC_CHIP + cols - 1) / cols;
    int fy = gy + rows * 38 + 30;
    int fw = (pw - 2 * 8) / 3;
    tc_rc_codec = {gx, fy, gx + fw, fy + 30};
    tc_rc_fps = {gx + fw + 8, fy, gx + 2 * fw + 8, fy + 30};
    tc_rc_br = {gx + 2 * fw + 16, fy, gx + pw, fy + 30};
    int goy = fy + 44;
    tc_rc_go = {gx, goy, gx + pw, goy + 42};
    tc_log_y0 = goy + 42 + 34;
    (void)H;
}

static void tc_draw_chip(HDC hdc, RECT rc, const char* label, bool on, bool ai, int locked, bool hov) {
    COLORREF bg = on ? (ai ? CLR_AI : RGB(0, 130, 190)) : CLR_PANEL2;
    if (locked) bg = RGB(24, 24, 32);
    else if (!on && hov) bg = RGB(42, 42, 58);
    HBRUSH bb = CreateSolidBrush(bg);
    draw_rounded_rect(hdc, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top,
                      6, bb, on ? RGB(255,255,255) : (hov && !locked ? CLR_ACCENT : CLR_LINE));
    DeleteObject(bb);
    if (locked) SetTextColor(hdc, RGB(90, 90, 110));
    else SetTextColor(hdc, on ? RGB(255,255,255) : CLR_TEXT);
    SelectObject(hdc, g_font_ui);
    RECT lr = {rc.left + 10, rc.top, rc.right - 44, rc.bottom};
    DrawTextA(hdc, label, -1, &lr, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
    SelectObject(hdc, g_font_st);
    if (locked && ai) {
        if (locked == 1) draw_pill(hdc, rc.right - 38, rc.top + 7, 28, 16, "...", RGB(52,52,68), CLR_AI, g_font_st);
        else draw_pill(hdc, rc.right - 38, rc.top + 7, 28, 16, "N/A", RGB(52,52,68), CLR_DIM, g_font_st);
    } else if (on) draw_pill(hdc, rc.right - 38, rc.top + 7, 28, 16, "ON", RGB(255,255,255), ai ? CLR_AI : RGB(0,130,190), g_font_st);
    else draw_pill(hdc, rc.right - 38, rc.top + 7, 28, 16, "OFF", RGB(52,52,68), CLR_DIM, g_font_st);
}

static void render_transcode_panel(HDC hdc, int W, int H) {
    tc_layout(W, H);
    int px = SIDE_W + 14, pw = W - SIDE_W - 28;
    bool running = (g_tc_state == 1);
    SelectObject(hdc, g_font_st);
    SetBkMode(hdc, TRANSPARENT);

    // ---- input row ----
    SetTextColor(hdc, CLR_DIM);
    RECT lr = {px, TAB_H + 44, px + pw, TAB_H + 62};
    DrawTextA(hdc, "INPUT VIDEO FILE", -1, &lr, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    HBRUSH fb = CreateSolidBrush(CLR_PANEL2);
    draw_rounded_rect(hdc, px, TAB_H + 66, W - 140 - px, 30, 6, fb, CLR_LINE);
    DeleteObject(fb);
    SelectObject(hdc, g_font_ui);
    SetTextColor(hdc, tc_input[0] ? CLR_TEXT : CLR_DIM);
    RECT fr2 = {px + 10, TAB_H + 66, W - 146, TAB_H + 96};
    DrawTextA(hdc, tc_input[0] ? tc_input : "no file selected", -1, &fr2,
              DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_PATH_ELLIPSIS);
    tc_draw_chip(hdc, tc_rc_browse, "Browse...", false, false, 0, g_hover == 1001);

    // ---- output row ----
    SelectObject(hdc, g_font_st);
    SetTextColor(hdc, CLR_DIM);
    RECT or2 = {px, TAB_H + 102, px + pw, TAB_H + 120};
    std::string od = tc_input[0] ? ("OUT:  " + tc_outdir()) : "OUT:  (select input first)";
    DrawTextA(hdc, od.c_str(), -1, &or2, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_PATH_ELLIPSIS);

    // ---- filter chips ----
    SetTextColor(hdc, CLR_DIM);
    RECT fl = {px, TAB_H + 128, px + pw, TAB_H + 148};
    DrawTextA(hdc, "FILTERS  (live preview keeps streaming behind this tab)", -1, &fl, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    char nm[48];
    for (int i = 0; i < NTC_CHIP; i++) {
        tc_chip_label(i, nm, sizeof(nm));
        bool ai = (i >= 25);
        int ust = 0; // 0 usable, 1 warming, 2 missing
        if (ai) {
            ust = tc_ai_usable(i);
        } else ust = tc_chip_state(i);
        bool on = tc_chip_on(i) && ust == 0;
        tc_draw_chip(hdc, tc_rc_chips[i], nm, on, ai, ust, g_hover == 1010 + i);
    }

    // ---- format row ----
    SetTextColor(hdc, CLR_DIM);
    RECT fl2 = {px, tc_rc_codec.top - 24, px + pw, tc_rc_codec.top - 6};
    DrawTextA(hdc, "FORMAT", -1, &fl2, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    char cb[32]; sprintf(cb, "Codec: %s", tc_codec ? "H265" : "H264");
    tc_draw_chip(hdc, tc_rc_codec, cb, false, false, 0, g_hover == 1003);
    char fb2[32]; sprintf(fb2, "FPS: %d", tc_fps_idx ? 60 : 30);
    tc_draw_chip(hdc, tc_rc_fps, fb2, false, false, 0, g_hover == 1004);
    const char* brs[] = {"BR: Auto", "BR: 5M", "BR: 10M", "BR: 20M"};
    tc_draw_chip(hdc, tc_rc_br, brs[tc_br_idx], false, false, 0, g_hover == 1005);

    // ---- go / stop ----
    {
        COLORREF bg = running ? RGB(150, 40, 40) : RGB(0, 140, 90);
        if (!running && g_hover == 1002) bg = RGB(0, 170, 110);
        HBRUSH bb = CreateSolidBrush(bg);
        int gw = tc_rc_go.right - tc_rc_go.left;
        draw_rounded_rect(hdc, tc_rc_go.left, tc_rc_go.top, gw,
                          tc_rc_go.bottom - tc_rc_go.top, 8, bb,
                          running ? RGB(255,255,255) : RGB(0,200,130));
        DeleteObject(bb);
        SelectObject(hdc, g_font_ui);
        SetTextColor(hdc, RGB(255,255,255));
        const char* t = running ? "STOP" : "TRANSCODE";
        DrawTextA(hdc, t, -1, (LPRECT)&tc_rc_go, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
    }

    // ---- progress ----
    int fr = g_tc_frames.load(), tot = g_tc_total.load();
    char ps[128];
    if (running && tot > 0) sprintf(ps, "%d / %d frames  (%d%%)", fr, tot, fr * 100 / tot);
    else if (running) sprintf(ps, "%d frames...", fr);
    else if (g_tc_state == 2) sprintf(ps, "done. Output in: %s", tc_input[0] ? tc_outdir().c_str() : "");
    else if (g_tc_state == 3) sprintf(ps, "failed. See log.");
    else sprintf(ps, "idle. Pick a file, choose filters, hit TRANSCODE.  (T)");
    SelectObject(hdc, g_font_st);
    SetTextColor(hdc, running ? CLR_ACCENT : CLR_DIM);
    RECT pr = {px, tc_rc_go.bottom + 6, px + pw, tc_rc_go.bottom + 24};
    DrawTextA(hdc, ps, -1, &pr, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
    if ((running && tot > 0) || g_tc_state == 2) {
        double p = (g_tc_state == 2) ? 1.0 : (tot > 0 ? (double)fr / tot : 0);
        if (p > 1) p = 1;
        int bw = (int)((W - SIDE_W - 28) * p);
        HBRUSH pb = CreateSolidBrush(g_tc_state == 2 ? CLR_OK : CLR_ACCENT);
        RECT bar = {px, tc_rc_go.bottom + 26, px + bw, tc_rc_go.bottom + 32};
        FillRect(hdc, &bar, pb);
        DeleteObject(pb);
    }

    // ---- log box (wheel-scrollable, follows tail by default) ----
    int ly0 = tc_log_y0;
    int ly1 = H - STATUS_H - 10;
    tc_rc_logbox = (ly1 > ly0 + 40) ? RECT{px, ly0, W - 14, ly1} : RECT{0,0,0,0};
    if (ly1 > ly0 + 40) {
        HPEN pen = CreatePen(PS_SOLID, 1, CLR_LINE);
        HPEN op = (HPEN)SelectObject(hdc, pen);
        HBRUSH ob = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, px, ly0, W - 14, ly1);
        SelectObject(hdc, op); SelectObject(hdc, ob); DeleteObject(pen);
        SelectObject(hdc, g_font_st);
        SetTextColor(hdc, CLR_TEXT);
        // Clip text to the box so scrolled lines never paint outside it.
        IntersectClipRect(hdc, px + 1, ly0 + 1, W - 15, ly1 - 1);
        std::lock_guard<std::mutex> l(g_tc_log_mtx);
        int lh = 18, n = (ly1 - ly0 - 8) / lh;
        if (n < 1) n = 1;
        tc_log_vis = n;
        int total = (int)g_tc_log.size();
        int start = 0;
        if (g_tc_log_pos < 0 || g_tc_log_pos > total - n) {
            // follow tail
            start = total > n ? total - n : 0;
            g_tc_log_pos = -1;
        } else start = g_tc_log_pos;
        for (int i = start; i < total; i++) {
            RECT tlr = {px + 8, ly0 + 4 + (i - start) * lh, W - 20, ly0 + 4 + (i - start + 1) * lh};
            const std::string& s = g_tc_log[i];
            COLORREF c = CLR_TEXT;
            if (s.find("error") != std::string::npos || s.find("failed") != std::string::npos) c = CLR_BAD;
            else if (s.find("done.") == 0) c = CLR_OK;
            else if (s.find("Filter:") != std::string::npos) c = CLR_AI;
            SetTextColor(hdc, c);
            DrawTextA(hdc, s.c_str(), -1, &tlr, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
        }
        SelectClipRgn(hdc, NULL); // release the log-box clip
    }
}

static int tc_hittest(int mx, int my, int W, int H) {
    if (mx < SIDE_W) return 0;
    tc_layout(W, H);
    if (mx >= tc_rc_browse.left && mx < tc_rc_browse.right && my >= tc_rc_browse.top && my < tc_rc_browse.bottom) return TID_BROWSE;
    if (mx >= tc_rc_go.left && mx < tc_rc_go.right && my >= tc_rc_go.top && my < tc_rc_go.bottom) return TID_GO;
    if (mx >= tc_rc_codec.left && mx < tc_rc_codec.right && my >= tc_rc_codec.top && my < tc_rc_codec.bottom) return TID_CODEC;
    if (mx >= tc_rc_fps.left && mx < tc_rc_fps.right && my >= tc_rc_fps.top && my < tc_rc_fps.bottom) return TID_FPS;
    if (mx >= tc_rc_br.left && mx < tc_rc_br.right && my >= tc_rc_br.top && my < tc_rc_br.bottom) return TID_BR;
    for (int i = 0; i < NTC_CHIP; i++) {
        RECT r = tc_rc_chips[i];
        if (mx >= r.left && mx < r.right && my >= r.top && my < r.bottom) return TID_CHIP + i;
    }
    return 0;
}

static void tc_fire(int id) {
    bool running = (g_tc_state == 1);
    if (id == TID_BROWSE) { tc_browse(); return; }
    if (id == TID_GO) { if (running) tc_stop(); else tc_start(); return; }
    if (running) return; // options locked while running
    if (id == TID_CODEC) { tc_codec = 1 - tc_codec; return; }
    if (id == TID_FPS) { tc_fps_idx = 1 - tc_fps_idx; return; }
    if (id == TID_BR) { tc_br_idx = (tc_br_idx + 1) % 4; return; }
    int i = id - TID_CHIP;
    if (i < 0 || i >= NTC_CHIP) return;
    if (i < NTC_TOG) { *tc_togs[i].flag = !*tc_togs[i].flag; return; }
    if (i == 18) { tc_lut_preset = (tc_lut_preset + 1) % 7; return; }
    if (i == 19) { tc_flip_mode = (tc_flip_mode + 1) % 3; return; }
    if (i == 20) { if (tc_vid_w > 0) tc_scale_idx = (tc_scale_idx + 1) % 4; return; }
    if (i == 21) { tc_dirblur_idx = (tc_dirblur_idx + 1) % 3; return; }
    if (i == 22) { tc_hdr = (tc_hdr + 1) % 3; return; }
    if (i == 23) { if (tc_vid_w > 0) tc_crop_idx = (tc_crop_idx + 1) % 3; return; }
    if (i == 24) { tc_peak_idx = (tc_peak_idx + 1) % 3; return; }
    const std::atomic<bool>* rd = &g_ai_denoise_ready;
    bool* fl = &tc_ai_denoise;
    if (i == 26) { rd = &g_ai_depth_ready; fl = &tc_ai_depth; }
    else if (i == 27) { rd = &g_ai_flow_ready; fl = &tc_ai_flow; }
    else if (i == 28) { rd = &g_ai_pose_ready; fl = &tc_ai_pose; }
    else if (i == 29) { rd = &g_ai_matting_ready; fl = &tc_ai_matting; }
    else if (i == 30) { rd = &g_ai_face_ready; fl = &tc_ai_face; }
    else if (i == 31) { rd = &g_ai_hands_ready; fl = &tc_ai_hands; }
    else if (i == 32) { rd = &g_ai_gaze_ready; fl = &tc_ai_gaze; }
    else if (i == 33) { rd = &g_ai_lowlight_ready; fl = &tc_ai_lowlight; }
    else if (i == 34) { rd = &g_ai_anime_ready; fl = &tc_ai_anime; }
    else if (i == 35) { rd = &g_ai_detect_ready; fl = &tc_ai_detect; }
    else if (i == 36) {
        if (g_ai_warming.load() || !g_ai_face_ready.load()) return;
        tc_autoframe = !tc_autoframe; return;
    }
    if (!rd->load() || g_ai_warming.load()) return;
    *fl = !*fl;
}

// ============================================================================
// Voice commentary: WASAPI microphone capture -> 16-bit WAV, muxed with the
// take at finalize (ffmpeg aac). mmdevapi only, no new deps.
// ============================================================================
static std::atomic<bool> g_mic_on{true};   // default: record voice
static std::atomic<float> g_mic_gain{1.0f}; // 1x / 2x / 4x boost for quiet mics
// Mic selection: 0 = OFF, 1 = system default, 2+ = enumerated device.
struct MicDevInfo { std::wstring id; std::string name; };
static std::vector<MicDevInfo> g_mic_devs;
static int g_mic_sel = 1;
static std::wstring g_au_dev; // device id for the running take (empty = default)

static std::string narrow(const wchar_t* ws) {
    // ANSI codepage: the UI draws with DrawTextA, so UTF-8 bytes here would
    // show as mojibake for non-ASCII device names.
    char b[128] = {};
    WideCharToMultiByte(CP_ACP, 0, ws, -1, b, sizeof(b) - 1, nullptr, nullptr);
    return std::string(b);
}

static void mic_enum() {
    g_mic_devs.clear();
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool uninit = SUCCEEDED(hr);
    IMMDeviceEnumerator* de = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), (void**)&de);
    if (SUCCEEDED(hr) && de) {
        IMMDeviceCollection* coll = nullptr;
        if (SUCCEEDED(de->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &coll)) && coll) {
            UINT n = 0;
            coll->GetCount(&n);
            for (UINT i = 0; i < n && i < 8; i++) {
                IMMDevice* d = nullptr;
                if (FAILED(coll->Item(i, &d)) || !d) continue;
                LPWSTR wid = nullptr;
                std::wstring id = SUCCEEDED(d->GetId(&wid)) && wid ? wid : L"";
                if (wid) CoTaskMemFree(wid);
                IPropertyStore* ps = nullptr;
                std::string nm = "mic";
                if (SUCCEEDED(d->OpenPropertyStore(STGM_READ, &ps)) && ps) {
                    PROPVARIANT pv; PropVariantInit(&pv);
                    if (SUCCEEDED(ps->GetValue(PKEY_Device_FriendlyName, &pv)) &&
                        pv.vt == VT_LPWSTR && pv.pwszVal)
                        nm = narrow(pv.pwszVal);
                    PropVariantClear(&pv);
                    ps->Release();
                }
                if (!id.empty()) g_mic_devs.push_back({id, nm});
                d->Release();
            }
            coll->Release();
        }
        de->Release();
    }
    if (uninit) CoUninitialize();
    if (g_mic_sel > (int)g_mic_devs.size() + 1) g_mic_sel = 1;
}

static std::string mic_label() {
    if (g_mic_sel <= 0) return "MIC OFF";
    if (g_mic_sel == 1 || g_mic_sel - 2 >= (int)g_mic_devs.size()) return "MIC DEF";
    std::string n = g_mic_devs[g_mic_sel - 2].name;
    if (n.size() > 12) n = n.substr(0, 12);
    return "MIC " + n;
}

// Dropdown list (meeting-app style): rows 0=OFF, 1=DEF, 2+i=device i.
static bool g_mic_list_open = false;
static int mic_row_count() { return (int)g_mic_devs.size() + 2; }
static std::string mic_row_label(int i) {
    if (i <= 0) return "OFF (no voice)";
    if (i == 1) return "System default";
    std::string n = g_mic_devs[i - 2].name;
    if (n.size() > 34) n = n.substr(0, 34);
    return n;
}
static std::atomic<float> g_mic_level{0};  // peak 0..1 for the meter
static std::atomic<bool> g_au_cap{false};
static std::thread g_au_thread;
static std::string g_au_wav;
static std::string g_au_msg;
static std::atomic<bool> g_au_silent{false}; // >3s of digital silence
static uint32_t g_au_rate = 48000, g_au_ch = 2;
static uint64_t g_au_data_bytes = 0;
// Wall-clock (steady_clock) time of the FIRST audio sample written to the
// take wav. aselect's `t` is 0-based (audio-file start), so cut ranges
// (stored as absolute steady-clock seconds) must be shifted by this to land
// in the audio timeline. 0 = no audio captured yet.
static double g_au_epoch = 0;

// Recorder-state accessors (defined in the RECORD section below; the audio
// block runs before it in this TU).
static bool rec_audio_open();
static double rec_now();

static void au_write_header(FILE* f, uint32_t rate, uint32_t ch, uint32_t data) {
    uint8_t h[44] = {};
    memcpy(h, "RIFF", 4);
    *(uint32_t*)(h + 4) = 36 + data;
    memcpy(h + 8, "WAVEfmt ", 8);
    *(uint32_t*)(h + 16) = 16;
    *(uint16_t*)(h + 20) = 1;
    *(uint16_t*)(h + 22) = (uint16_t)ch;
    *(uint32_t*)(h + 24) = rate;
    *(uint32_t*)(h + 28) = rate * ch * 2;
    *(uint16_t*)(h + 32) = (uint16_t)(ch * 2);
    *(uint16_t*)(h + 34) = 16;
    memcpy(h + 36, "data", 4);
    *(uint32_t*)(h + 40) = data;
    fwrite(h, 1, 44, f);
}

static void au_thread_func(std::string path) {
    FILE* f = nullptr;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool uninit = SUCCEEDED(hr);
    IMMDeviceEnumerator* de = nullptr;
    IMMDevice* dev = nullptr;
    IAudioClient* cli = nullptr;
    IAudioCaptureClient* cap = nullptr;
    WAVEFORMATEX* fmt = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), (void**)&de);
    if (FAILED(hr) || !de) { g_au_msg = "no audio devices"; goto done; }
    if (!g_au_dev.empty()) {
        hr = de->GetDevice(g_au_dev.c_str(), &dev);
        if (FAILED(hr) || !dev) { g_au_msg = "mic device gone"; goto done; }
    } else {
        hr = de->GetDefaultAudioEndpoint(eCapture, eConsole, &dev);
        if (FAILED(hr) || !dev) { g_au_msg = "no microphone found"; goto done; }
    }
    hr = dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&cli);
    if (FAILED(hr) || !cli) { g_au_msg = "mic open failed"; goto done; }
    hr = cli->GetMixFormat(&fmt);
    if (FAILED(hr) || !fmt) { g_au_msg = "mic format failed"; goto done; }
    {
        bool isfloat = (fmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT);
        if (!isfloat && fmt->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
            auto* ex = (WAVEFORMATEXTENSIBLE*)fmt;
            isfloat = (ex->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) &&
                      fmt->wBitsPerSample == 32;
        }
        if (!isfloat) { g_au_msg = "mic format unsupported"; goto done; }
        g_au_rate = fmt->nSamplesPerSec;
        g_au_ch = fmt->nChannels;
        if (g_au_ch < 1 || g_au_ch > 8 || g_au_rate < 8000 || g_au_rate > 192000) {
            g_au_msg = "mic format out of range"; goto done;
        }
    }
    hr = cli->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, fmt, nullptr);
    if (FAILED(hr)) { g_au_msg = "mic init failed"; goto done; }
    hr = cli->GetService(__uuidof(IAudioCaptureClient), (void**)&cap);
    if (FAILED(hr) || !cap) { g_au_msg = "mic capture failed"; goto done; }
    f = fopen(path.c_str(), "wb");
    if (!f) { g_au_msg = "wav create failed"; goto done; }
    au_write_header(f, g_au_rate, g_au_ch, 0);
    hr = cli->Start();
    if (FAILED(hr)) { g_au_msg = "mic start failed"; goto done; }
    g_au_msg.clear();
    g_au_silent = false;
    int silent_iters = 0;
    while (g_au_cap.load()) {
        Sleep(10);
        // Gate: paused spans and CUT gaps have no video frames, so no audio
        // is written for them either — the take stays lip-synced by itself.
        // CUT ranges already on disk get snipped at finalize (see below).
        float iter_peak = 0;
        bool iter_wrote = false;
        UINT32 packets = 0;
        if (FAILED(cap->GetNextPacketSize(&packets))) break;
        while (packets) {
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            if (FAILED(cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
            float peak = 0;
            if (frames > 0) {
                std::vector<int16_t> tmp;
                tmp.resize((size_t)frames * g_au_ch);
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                    memset(tmp.data(), 0, tmp.size() * 2);
                } else {
                    float* fs = (float*)data;
                    float gain = g_mic_gain.load();
                    for (size_t i = 0; i < tmp.size(); i++) {
                        float v = fs[i] * gain;
                        if (v > 1.0f) v = 1.0f;
                        else if (v < -1.0f) v = -1.0f;
                        tmp[i] = (int16_t)(v * 32767.0f);
                        float a = v < 0 ? -v : v;
                        if (a > peak) peak = a;
                    }
                }
                // Re-check per packet (not per 10ms batch): a CUT/pause that
                // lands mid-batch — or after thread starvation drains a
                // backlog — must not leak post-cut packets into the wav, or
                // every later word shifts late and lip-sync drifts per cut.
                if (rec_audio_open()) {
                    if (g_au_epoch <= 0) g_au_epoch = rec_now(); // first sample = wav t0
                    fwrite(tmp.data(), 2, tmp.size(), f);
                    g_au_data_bytes += (uint64_t)tmp.size() * 2;
                    iter_wrote = true;
                } else {
                    peak = 0; // gated span: meter flat, nothing written
                }
            }
            if (peak > iter_peak) iter_peak = peak;
            cap->ReleaseBuffer(frames);
            if (FAILED(cap->GetNextPacketSize(&packets))) break;
        }
        if (iter_wrote) {
            g_mic_level = iter_peak;
            // Digital silence for >3s: muted mic, privacy block, or dead device.
            if (iter_peak > 0.002f) { silent_iters = 0; g_au_silent = false; }
            else if (++silent_iters > 300) g_au_silent = true;
        } else {
            g_mic_level = 0; // gated span: no warning either
        }
    }
    cli->Stop();
done:
    if (f) { // patch sizes, close
        fseek(f, 0, SEEK_SET);
        au_write_header(f, g_au_rate, g_au_ch, (uint32_t)g_au_data_bytes);
        fclose(f);
    } else {
        remove(path.c_str());
    }
    if (cap) cap->Release();
    if (cli) cli->Release();
    if (dev) dev->Release();
    if (de) de->Release();
    if (fmt) CoTaskMemFree(fmt);
    if (uninit) CoUninitialize();
}

static void au_stop() {
    g_au_cap = false;
    if (g_au_thread.joinable()) g_au_thread.join();
}

// Starts mic capture for a take. Non-blocking; failure just means video-only.
static void au_start(const std::string& path) {
    au_stop();
    if (g_mic_sel >= 2 && g_mic_sel - 2 < (int)g_mic_devs.size())
        g_au_dev = g_mic_devs[g_mic_sel - 2].id;
    else
        g_au_dev.clear();
    g_au_wav = path;
    g_au_data_bytes = 0;
    g_au_epoch = 0;
    g_mic_level = 0;
    g_au_msg.clear();
    g_au_cap = true;
    g_au_thread = std::thread(au_thread_func, path);
}

// ============================================================================
// RECORD tab -- live NVENC recording of the filtered camera feed ("live cut").
// Step 1: flat take recording to MP4. Segments/CUT/markers/gestures follow.
// The recorder taps the SAME processed frame as the virtual cam (g_d_rgb),
// resized to capture size, so every right-panel filter applies to the take.
// ============================================================================
static std::atomic<int> g_rec_frames{0};
static std::atomic<int> g_rec_dropped{0};
static std::string g_rec_msg = "idle";
static std::mutex g_rec_mtx;
static kagerou::Encoder g_rec_enc;
static bool g_rec_enc_ok = false;
// NVENC session/context affinity: init, encode, flush and destroy must all
// run on the SAME thread (the GPU thread). UI/worker threads only post
// requests; the gpu loop services them inside rec_write_frame.
static std::atomic<int> g_rec_enc_req{0}; // 0 none, 1 init, 2 flush+destroy
static kagerou::EncoderConfig g_rec_init_cfg;
#if !defined(KAGEROU_NO_VCODEC_SDK)
static CUcontext g_rec_init_ctx = nullptr; // shared primary context (no private ctx)
#endif
static std::atomic<bool> g_rec_init_done{false};
static std::atomic<bool> g_rec_eos_done{false};
static std::vector<std::vector<uint8_t>> g_rec_eos_tail;
static uint8_t* g_rec_rgb = nullptr;   // cap-size RGB staging (alloc'd in main)
static uint8_t* g_rec_nv12 = nullptr;  // cap-size NV12 staging (alloc'd in main)
static FILE* g_rec_file = nullptr;
static std::string g_rec_h264, g_rec_mp4, g_rec_dir;
static std::chrono::steady_clock::time_point g_rec_t0;
static std::thread g_rec_thread;
static int g_rec_codec = 0;   // 0 h264, 1 h265
static int g_rec_br_idx = 3;  // 0:5M 1:8M 2:12M 3:20M 4:40M (default 20M HQ)
static bool g_rec_mic = false;      // mic was on for this take
static std::string g_rec_wav;       // take wav path (voice)
static uint32_t g_rec_w = 640, g_rec_h = 480; // take dims (follow source)
static RECT rc_rec_go, rc_rec_codec, rc_rec_br, rc_rec_gain;
static RECT rc_rec_cut, rc_rec_prev, rc_rec_pause, rc_rec_mark, rc_rec_mic;
static RECT rc_rec_src[3]; // CAM / SCREEN / TUTORIAL
static const uint32_t REC_BR[] = {5000, 8000, 12000, 20000, 40000};

// ---- segment state: every seg file starts with an IDR (self-contained),
// so kept segs concatenate losslessly and CUT segs just get deleted.
static const int REC_SEG_TARGET = 120; // rotate at first IDR past this many frames
static const int REC_SEG_MIN = 5;      // shorter than this at close = trash
struct RecSeg { int idx, start, end; bool kept; };
// A cut removes video frames [start,end) and audio wall-time [t0,t1].
// t0 = cut press, t1 = resume (next IDR); both steady-clock seconds.
// Audio recorded in (t0,t1] is gated off live; [t(start),t0] is snipped
// at finalize so voice stays lip-synced after every cut.
struct RecCut { int start, end; double t0, t1; bool live; };
struct RecMark { int frame; std::string label; };
static std::vector<double> g_rec_ft; // wall time per submitted frame
static std::vector<RecSeg> g_rec_segs;
static std::vector<RecCut> g_rec_cuts;
static std::vector<RecMark> g_rec_marks;
static int g_rec_seg_idx = 0;
static int g_rec_seg_bs = 0;       // bitstream frames in the open seg
static int g_rec_seg_start = 0;    // submitted-frame index of open seg start
static int g_rec_submitted = 0;    // total submitted frames (take timeline)
static std::atomic<bool> g_rec_discarding{false}; // after CUT: drop until next IDR
static int g_rec_mark_n = 0;
static std::string g_rec_basedir;  // <dir>\<base> holds seg files

// Annex-B scan: true if this access unit contains an IDR picture.
static bool rec_is_idr(const uint8_t* bs, size_t n, bool h265) {
    for (size_t i = 0; i + 4 < n; i++) {
        if (bs[i] == 0 && bs[i+1] == 0 && bs[i+2] == 0 && bs[i+3] == 1) {
            uint8_t nal = bs[i+4];
            if (!h265) { if ((nal & 0x1F) == 5) return true; }
            else { uint8_t t = (nal >> 1) & 0x3F; if (t == 19 || t == 20) return true; }
            i += 4;
        }
    }
    return false;
}
static std::string rec_seg_path(int idx) {
    char s[64]; snprintf(s, sizeof(s), "seg_%03d.h264", idx);
    return g_rec_basedir + "\\" + s;
}

// NVENC emits SPS/PPS (+VPS) only with the very first IDR. Cache them from
// seg 0 so every later segment (rotation / post-CUT resume) can be prefixed
// and stays independently decodable. Without this, players fail with
// "non-existing PPS referenced" at every segment joint.
static std::vector<uint8_t> g_rec_psps;
static void rec_collect_psps(const uint8_t* bs, size_t n, bool h265) {
    if (!bs || n < 8 || !g_rec_psps.empty()) return;
    size_t i = 0;
    while (i + 4 < n) {
        size_t sc = 0;
        if (bs[i] == 0 && bs[i+1] == 0 && bs[i+2] == 0 && bs[i+3] == 1) sc = 4;
        else if (bs[i] == 0 && bs[i+1] == 0 && bs[i+2] == 1) sc = 3;
        else { i++; continue; }
        size_t end = n;
        for (size_t j = i + sc + 1; j + 2 < n; j++) {
            if (bs[j] == 0 && bs[j+1] == 0 &&
                (bs[j+2] == 1 || (j + 3 < n && bs[j+2] == 0 && bs[j+3] == 1))) {
                end = j; break;
            }
        }
        uint8_t nal = bs[i + sc];
        bool keep, vcl;
        if (!h265) {
            uint8_t t = nal & 0x1F;
            keep = (t == 7 || t == 8); vcl = (t <= 5);
        } else {
            uint8_t t = (nal >> 1) & 0x3F;
            keep = (t >= 32 && t <= 34); vcl = (t < 32);
        }
        if (keep) g_rec_psps.insert(g_rec_psps.end(), bs + i, bs + end);
        if (vcl && !g_rec_psps.empty()) break; // param sets precede slices
        i = (end >= n) ? n : end;
    }
}

static double rec_elapsed() {
    if (g_rec_state.load() != 1 && g_rec_state.load() != 2) return 0.0;
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - g_rec_t0).count();
}
// GB free on the recording volume; <0 = unknown.
static double rec_disk_free_gb() {
    std::string dir = g_rec_dir.empty() ? get_exe_dir() : g_rec_dir;
    ULARGE_INTEGER free = {};
    if (!GetDiskFreeSpaceExA(dir.c_str(), &free, nullptr, nullptr)) return -1.0;
    return (double)free.QuadPart / (1024.0 * 1024.0 * 1024.0);
}
static void rec_set_msg(const std::string& m) {
    std::lock_guard<std::mutex> l(g_rec_mtx); g_rec_msg = m;
}
static std::string rec_get_msg() {
    std::lock_guard<std::mutex> l(g_rec_mtx); return g_rec_msg;
}

static bool rec_start() {
    if (g_rec_state.load() == 1 || g_rec_state.load() == 2) return false;
    if (g_rec_enc_req.load() != 0) return false; // encoder busy
#if defined(KAGEROU_NO_VCODEC_SDK)
    rec_set_msg("NVENC missing: rebuild with 'sdk'");
    g_rec_state = 4;
    return false;
#else
    std::string dir = g_rec_dir.empty() ? (get_exe_dir() + "\\recordings") : g_rec_dir;
    kagerou::fileio::create_directory(dir.c_str());
    time_t tt = time(nullptr); struct tm lt; localtime_s(&lt, &tt);
    char base[64]; strftime(base, sizeof(base), "take_%Y%m%d_%H%M%S", &lt);
    g_rec_basedir = dir + "\\" + base;
    kagerou::fileio::create_directory(g_rec_basedir.c_str());
    g_rec_h264 = ""; // flat path unused in segment mode (kept for messages)
    g_rec_mp4  = dir + "\\" + base + ".mp4";
    g_rec_dir  = dir;
    g_rec_segs.clear(); g_rec_cuts.clear(); g_rec_marks.clear();
    g_rec_ft.clear(); g_rec_ft.reserve(10800);
    g_rec_psps.clear();
    g_rec_seg_idx = 0; g_rec_seg_bs = 0; g_rec_seg_start = 0;
    g_rec_submitted = 0; g_rec_discarding = false; g_rec_paused = false;
    g_rec_mark_n = 0;
    kagerou::EncoderConfig ec;
    ec.codec = g_rec_codec ? kagerou::VideoCodec::kH265 : kagerou::VideoCodec::kH264;
    { // take dims follow the live source (camera cap-size, screen/tutorial feed)
        std::lock_guard<std::mutex> slock(g_screen_mtx);
        if (g_src_mode != 0 && g_screen_w && g_screen_h) {
            g_rec_w = g_screen_w; g_rec_h = g_screen_h;
        } else { g_rec_w = g_cap_w; g_rec_h = g_cap_h; }
    }
    ec.width = g_rec_w; ec.height = g_rec_h; ec.fps = 30;
    ec.bitrate_kbps = REC_BR[g_rec_br_idx];
    // Screen content: CBR at the selected bitrate (exact sizes, quality on
    // edges/text). Camera: CQP for consistent per-frame quality.
    ec.rc = (g_src_mode != 0) ? kagerou::RateControl::kCBR
                              : kagerou::RateControl::kCQP;
    ec.gop_size = 60; ec.b_frames = false;
    ec.qp = 20;
    // Encoder init runs on the GPU thread (context affinity): post it and
    // let the take start; frames flow once init completes (~100ms).
    // The session shares the CUDA primary context: a private cuCtxCreate
    // would hijack the calling thread's current context and break every
    // runtime-API call on it (green takes / abort crashes).
    g_rec_eos_tail.clear();
    g_rec_init_cfg = ec;
#if !defined(KAGEROU_NO_VCODEC_SDK)
    g_rec_init_ctx = nullptr;
    {
        CUdevice dev = 0;
        CUcontext primary = nullptr;
        if (cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&dev, 0) == CUDA_SUCCESS &&
            cuDevicePrimaryCtxRetain(&primary, dev) == CUDA_SUCCESS && primary)
            g_rec_init_ctx = primary;
        // NOTE: nullptr ctx falls back to a private context (legacy path).
    }
#endif
    g_rec_init_done = false;
    g_rec_enc_req = 1;
    g_rec_file = fopen(rec_seg_path(0).c_str(), "wb");
    if (!g_rec_file) {
        rec_set_msg("cannot create output file"); g_rec_state = 4; return false;
    }
    g_rec_frames = 0; g_rec_dropped = 0;
    g_rec_osd = 0;
    g_mic_list_open = false;
    g_rec_mic = g_mic_on.load();
    if (g_rec_mic) {
        g_rec_wav = g_rec_basedir + "\\take.wav";
        au_start(g_rec_wav);
    } else g_rec_wav.clear();
    g_rec_t0 = std::chrono::steady_clock::now();
    // Per-take diagnostics: filter state + input state, so a bad take
    // explains itself. Never deleted (sits next to edit.csv).
    {
        FILE* lf = fopen((g_rec_basedir + "\\take.log").c_str(), "w");
        if (lf) {
            fprintf(lf, "take %s\nbuild %s %s\ncodec=%s br=%ukbps cap=%ux%u take=%ux%u src=%d mic=%s gain=%.0f\n",
                    base, __DATE__, __TIME__,
                    g_rec_codec ? "h265" : "h264", REC_BR[g_rec_br_idx],
                    g_cap_w, g_cap_h, g_rec_w, g_rec_h, g_src_mode,
                    mic_label().c_str(), g_mic_gain.load());
            if (g_src_mode != 0)
                fprintf(lf, "screen blt=%d err=%lu mean=%.1f %ux%u cap_fps=%.0f helper=%d\n",
                        g_screen_blt.load(), g_screen_blterr.load(),
                        g_screen_mean.load(), g_screen_w, g_screen_h,
                        g_screen_fps.load(), g_scr_proc ? 1 : 0);
            fprintf(lf, "cam_alive=%d cam_black=%d\n",
                    g_cam_alive.load() ? 1 : 0, g_cam_black.load() ? 1 : 0);
            fprintf(lf, "filters on:");
            if (g_denoise) fprintf(lf, " denoise");
            if (g_clahe) fprintf(lf, " clahe");
            if (g_sr) fprintf(lf, " superres");
            if (g_lut_on) fprintf(lf, " lut%d", g_lut_preset);
            if (g_blur) fprintf(lf, " blur");
            if (g_sharp) fprintf(lf, " sharp");
            if (g_bright) fprintf(lf, " bright");
            if (g_sat) fprintf(lf, " sat");
            if (g_gamma) fprintf(lf, " gamma");
            if (g_vignette) fprintf(lf, " vignette");
            if (g_grain) fprintf(lf, " grain");
            if (g_edge) fprintf(lf, " edge");
            if (g_wb) fprintf(lf, " wb");
            if (g_flip) fprintf(lf, " flip%d", g_flip_mode);
            if (g_lens) fprintf(lf, " lens");
            if (g_interp) fprintf(lf, " interp");
            if (g_bgblur) fprintf(lf, " bgblur");
            if (g_tempden) fprintf(lf, " tempden");
            if (g_tempstab) fprintf(lf, " tempstab");
            if (g_chroma) fprintf(lf, " chroma");
            if (g_dirblur) fprintf(lf, " dirblur%d", g_dirblur_idx);
            if (g_hdr) fprintf(lf, " hdr%d", g_hdr_method);
            if (g_cropzoom) fprintf(lf, " zoom%d", g_zoom_idx);
            if (g_ai_denoise) fprintf(lf, " ai-denoise");
            if (g_ai_depth) fprintf(lf, " ai-depth");
            if (g_ai_flow) fprintf(lf, " ai-flow");
            if (g_ai_pose) fprintf(lf, " ai-pose");
            if (g_ai_matting) fprintf(lf, " ai-matting");
            if (g_ai_face) fprintf(lf, " ai-face");
            if (g_ai_detect) fprintf(lf, " ai-detect");
            if (g_ai_hands) fprintf(lf, " ai-hands");
            if (g_ai_gaze) fprintf(lf, " ai-gaze");
            if (g_ai_lowlight) fprintf(lf, " ai-lowlight");
            if (g_ai_anime) fprintf(lf, " ai-anime");
            if (g_autoframe) fprintf(lf, " autoframe%d", g_af_idx);
            fprintf(lf, "\n");
            fclose(lf);
        }
    }
    rec_set_msg("recording");
    g_rec_state = 1;
    printf("[REC] started: %s\n", g_rec_basedir.c_str()); fflush(stdout);
    return true;
#endif
}

// Called from the GPU thread with the processed frame (g_d_rgb, ww x wh RGB).
// Wall-clock paced (~30fps CFR): the loop runs unpaced, so without this the
// take would contain duplicate frames (fast GPU) or play fast (slow GPU).
// Finalize muxes at the MEASURED fps, so playback is realtime either way.
static void rec_write_frame(const uint8_t* d_rgb, uint32_t w, uint32_t h, bool fresh) {
    (void)fresh;
    // Service encoder requests here: this runs on the GPU thread, the only
    // thread that may touch the NVENC session (see g_rec_enc_req).
    int rq = g_rec_enc_req.exchange(0);
    if (rq == 1) {
        if (g_rec_enc_ok) { g_rec_enc.destroy(); g_rec_enc_ok = false; }
#if defined(KAGEROU_NO_VCODEC_SDK)
        g_rec_enc_ok = (g_rec_enc.init(g_rec_init_cfg) == kagerou::Error::kOk);
#else
        g_rec_enc_ok = (g_rec_enc.init(g_rec_init_cfg, g_rec_init_ctx) == kagerou::Error::kOk);
#endif
        // Grab SPS/PPS once, straight from the session: prefixes every
        // segment file (rotation / post-CUT resume) so each decodes alone.
        // Bitstream scanning stays as fallback below.
        g_rec_psps.clear();
        if (g_rec_enc_ok)
            g_rec_enc.get_sequence_header(g_rec_psps);
        g_rec_init_done = true;
        if (!g_rec_enc_ok) { rec_set_msg("encoder init failed"); g_rec_state = 4; }
    } else if (rq == 2) {
        if (g_rec_enc_ok) g_rec_enc.flush(g_rec_eos_tail);
        if (g_rec_enc_ok) { g_rec_enc.destroy(); g_rec_enc_ok = false; }
        g_rec_eos_done = true;
    }
    if (g_rec_state.load() != 1 || !g_rec_enc_ok || !g_rec_file) return;
    if (!g_rec_rgb || !g_rec_nv12 || g_rec_paused) return;
    static auto s_last_t = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - s_last_t).count() < 1.0 / 30.0)
        return;
    s_last_t = now;
    const uint8_t* src = d_rgb;
    if (w != g_rec_w || h != g_rec_h) {
        filters::resize_bilinear(d_rgb, g_rec_rgb, w, h,
                                 g_rec_w, g_rec_h, 3, g_stream);
        src = g_rec_rgb;
    }
    // Limited-range matrix by take height (players assume 601 below 720p,
    // 709 at/above; the wrong one shifts reds/blues and crushes contrast).
    if (g_rec_h >= 720)
        filters::rgb_to_nv12_limited(src, g_rec_nv12, g_rec_w, g_rec_h, g_stream);
    else
        filters::rgb_to_nv12_limited601(src, g_rec_nv12, g_rec_w, g_rec_h, g_stream);
    // Synchronous launch errors are otherwise silent (void wrapper): surface
    // them into the take log (also clears sticky state so one bad launch
    // can't poison the rest of the take).
    {
        static bool logged = false;
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess && !logged) {
            logged = true;
            FILE* lf = fopen((g_rec_basedir + "\\take.log").c_str(), "a");
            if (lf) {
                fprintf(lf, "CONVERT launch err: %s\n", cudaGetErrorString(lerr));
                fclose(lf);
            }
        }
    }
    // Never abort the app on a transient GPU error mid-take: drop frames,
    // and fail the take (not the process) if the stream stays broken.
    static int s_sync_fails = 0;
    if (cudaStreamSynchronize(g_stream) != cudaSuccess) {
        g_rec_dropped++;
        if (++s_sync_fails > 30) {
            rec_set_msg("GPU error, take stopped"); g_rec_state = 4;
        }
        return;
    }
    s_sync_fails = 0;
    kagerou::Frame f;
    f.d_data = g_rec_nv12;
    f.width = g_rec_w; f.height = g_rec_h;
    f.stride = g_rec_w; f.fmt = kagerou::PixelFormat::kNV12;
    f.plane_count = 2;
    f.plane_size[0] = g_rec_w * g_rec_h;
    f.plane_size[1] = g_rec_w * g_rec_h / 2;
    std::vector<uint8_t> bs;
    if (    g_rec_enc.encode(f, bs) != kagerou::Error::kOk) { g_rec_dropped++; return; }
    g_rec_submitted++;
    g_rec_frames++;
    g_rec_ft.push_back(std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    if (bs.empty()) return;
    bool h265 = (g_rec_codec == 1);
    if (g_rec_seg_idx == 0 && g_rec_psps.empty())
        rec_collect_psps(bs.data(), bs.size(), h265);
    bool idr = rec_is_idr(bs.data(), bs.size(), h265);
    if (g_rec_discarding) {
        if (!idr) return; // CUT gap: drop P-frames until the next IDR
        g_rec_discarding = false;
        if (!g_rec_cuts.empty() && g_rec_cuts.back().end < 0) {
            g_rec_cuts.back().end = g_rec_submitted;
            g_rec_cuts.back().t1 = rec_now();
        }
        printf("[REC] resumed at IDR (frame %d)\n", g_rec_submitted); fflush(stdout);
    }
    // Rotate at the first IDR past the target (failsafe at 2x target).
    if ((g_rec_seg_bs >= REC_SEG_TARGET && idr) || g_rec_seg_bs >= 2 * REC_SEG_TARGET) {
        fclose(g_rec_file);
        g_rec_segs.push_back({g_rec_seg_idx, g_rec_seg_start, g_rec_submitted, true});
        g_rec_seg_idx++;
        g_rec_seg_start = g_rec_submitted;
        g_rec_seg_bs = 0;
        g_rec_file = fopen(rec_seg_path(g_rec_seg_idx).c_str(), "wb");
        if (!g_rec_file) { rec_set_msg("segment file failed"); g_rec_state = 4; return; }
    }
    // First payload of a non-zero segment is an IDR by construction but
    // carries no param sets: prefix the cached SPS/PPS (+VPS) so the file
    // decodes standalone (rotation joints + post-CUT resumes).
    if (g_rec_seg_bs == 0 && g_rec_seg_idx > 0 && !g_rec_psps.empty())
        fwrite(g_rec_psps.data(), 1, g_rec_psps.size(), g_rec_file);
    fwrite(bs.data(), 1, bs.size(), g_rec_file);
    g_rec_seg_bs++;
    if (g_rec_seg_bs == 1) {
        // First payload of this segment: log source vs encoder-input
        // brightness at frame CENTER (corners are often black).
        // Green take + srcmean>0 + ymean=0 => conversion never landed.
        // Green take + srcmean=0 => upstream (capture/process) is black.
        uint8_t probe[64] = {};
        size_t coff = ((size_t)g_rec_h / 2 * g_rec_w + g_rec_w / 2);
        double ymean = -1.0, srcmean = -1.0;
        if (cudaMemcpy(probe, g_rec_nv12 + coff, sizeof(probe), cudaMemcpyDeviceToHost)
            == cudaSuccess) {
            long acc = 0;
            for (int i = 0; i < 64; i++) acc += probe[i];
            ymean = acc / 64.0;
        }
        size_t so = coff * 3;
        if (w == g_rec_w && h == g_rec_h &&
            cudaMemcpy(probe, src + so, sizeof(probe), cudaMemcpyDeviceToHost)
            == cudaSuccess) {
            long acc = 0;
            for (int i = 0; i < 64; i++) acc += probe[i];
            srcmean = acc / 64.0;
        }
        FILE* lf = fopen((g_rec_basedir + "\\take.log").c_str(), "a");
        if (lf) {
            fprintf(lf, "seg %d srcmean=%.1f ymean=%.1f disp=%ux%u submitted=%d\n",
                    g_rec_seg_idx, srcmean, ymean, w, h, g_rec_submitted);
            fclose(lf);
        }
    }
}

// Drop the in-progress segment (the sneeze): its file is deleted and writing
// resumes at the next IDR, so the cut range is [seg_start, resume).
static double rec_now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Audio gate: voice records only while video frames are being written
// (recording, not paused, not in a CUT gap). Keeps takes lip-synced.
static bool rec_audio_open() {
    return g_rec_state.load() == 1 &&
           !g_rec_paused.load() && !g_rec_discarding.load();
}

static void rec_cut() {
    if (g_rec_state.load() != 1 || g_rec_paused) return;
    if (g_rec_file) { fclose(g_rec_file); g_rec_file = nullptr; }
    remove(rec_seg_path(g_rec_seg_idx).c_str());
    g_rec_cuts.push_back({g_rec_seg_start, -1, rec_now(), -1, true});
    printf("[REC] CUT seg %d (frames %d..)\n", g_rec_seg_idx, g_rec_seg_start);
    fflush(stdout);
    g_rec_seg_idx++;
    g_rec_seg_start = g_rec_submitted;
    g_rec_seg_bs = 0;
    g_rec_discarding = true;
    g_rec_file = fopen(rec_seg_path(g_rec_seg_idx).c_str(), "wb");
    if (!g_rec_file) { rec_set_msg("segment file failed"); g_rec_state = 4; }
}

// Drop the last KEPT segment (mistake is one segment back).
static void rec_cut_prev() {
    if (g_rec_state.load() != 1 || g_rec_paused) return;
    for (int i = (int)g_rec_segs.size() - 1; i >= 0; i--) {
        if (g_rec_segs[i].kept) {
            g_rec_segs[i].kept = false;
            remove(rec_seg_path(g_rec_segs[i].idx).c_str());
            double a = (g_rec_segs[i].start >= 0 && g_rec_segs[i].start < (int)g_rec_ft.size())
                       ? g_rec_ft[g_rec_segs[i].start] : -1.0;
            double b = (g_rec_segs[i].end >= 0 && g_rec_segs[i].end < (int)g_rec_ft.size())
                       ? g_rec_ft[g_rec_segs[i].end] : rec_now();
            g_rec_cuts.push_back({g_rec_segs[i].start, g_rec_segs[i].end, a, b, false});
            printf("[REC] CUT PREV seg %d (frames %d-%d)\n",
                   g_rec_segs[i].idx, g_rec_segs[i].start, g_rec_segs[i].end);
            fflush(stdout);
            return;
        }
    }
}

static void rec_pause_toggle() {
    if (g_rec_state.load() != 1) return;
    g_rec_paused = !g_rec_paused;
    rec_set_msg(g_rec_paused ? "paused" : "recording");
    printf("[REC] %s\n", g_rec_paused ? "paused" : "resumed"); fflush(stdout);
}

static void rec_mark() {
    if (g_rec_state.load() != 1 || g_rec_paused) return;
    char lb[32]; snprintf(lb, sizeof(lb), "mark %d", ++g_rec_mark_n);
    g_rec_marks.push_back({g_rec_submitted, lb});
    printf("[REC] %s at frame %d\n", lb, g_rec_submitted); fflush(stdout);
}

// ============================================================================
// Gesture director: heuristic classifier on the 21 hand landmarks
// (MediaPipe order: 0 wrist, thumb 1-4, index 5-8, middle 9-12,
// ring 13-16, pinky 17-20). No new model -- reuses the Hands AI output.
//   open palm (hold 0.5s) = CUT | thumbs-up = MARK
//   peace = PAUSE/RESUME | fist (hold 2s) = STOP
// ============================================================================
static void rec_stop(); // defined below (finalize section)
// Gesture vocabulary lives in the SDK (kagerou/ai/gesture.h); short aliases.
typedef kagerou::ai::HandGesture RecGest;
static const RecGest RG_NONE = kagerou::ai::HandGesture::kNone;
static const RecGest RG_PALM = kagerou::ai::HandGesture::kPalm;
static const RecGest RG_FIST = kagerou::ai::HandGesture::kFist;
static const RecGest RG_THUMB = kagerou::ai::HandGesture::kThumb;
static const RecGest RG_PEACE = kagerou::ai::HandGesture::kPeace;
static std::string g_rec_gest_txt = "gestures idle";
static std::mutex g_rec_gest_mtx;
static void rec_gest_set(const std::string& t) {
    std::lock_guard<std::mutex> l(g_rec_gest_mtx); g_rec_gest_txt = t;
}
static std::string rec_gest_get() {
    std::lock_guard<std::mutex> l(g_rec_gest_mtx); return g_rec_gest_txt;
}

static RecGest rec_classify(const kagerou::ai::HandJoints& j) {
    return kagerou::ai::classify_gesture(j);
}

// PiP rect shared with the compositor (single source of truth in pip.h).
static void pip_rect(uint32_t sw, uint32_t sh, int* px, int* py, int* pw, int* ph) {
    kagerou::pip::PiPBox b = {};
    if (!kagerou::pip::box(sw, sh, &b)) { *pw = 0; return; }
    *px = b.x; *py = b.y; *pw = b.w; *ph = b.h;
}

// Overlay pass: detector annotations for preview/vcam, drawn AFTER the take
// taps its clean frame. Camera mode: native coords. Tutorial mode: face +
// hands results live in CAMERA coords (inferred on the camera feed) and map
// into the PiP box, so tracking follows YOU, not the recorded screen.
static void draw_ai_overlays() {
    bool ai_live = !g_ai_warming.load();
    if (!ai_live) { rec_gesture_tick(); return; }
    uint32_t w = g_disp_w, h = g_disp_h;
    if (!w || !h || !g_d_rgb) { rec_gesture_tick(); return; }
    bool tuto = (g_src_mode != 0);
    float sx = 1.0f, sy = 1.0f;
    int ox = 0, oy = 0;
    // Overlay span for NORMALIZED coords (pose): program dims in camera
    // mode, PiP dims in tutorial mode.
    float spanx = (float)w, spany = (float)h;
    if (tuto) {
        // Tutorial detectors run on the camera feed (cap dims).
        int px, py, pw, ph;
        pip_rect(w, h, &px, &py, &pw, &ph);
        if (pw <= 0) { rec_gesture_tick(); return; }
        sx = (float)pw / g_cap_w; sy = (float)ph / g_cap_h;
        ox = px; oy = py;
        spanx = (float)pw; spany = (float)ph;
    } else if (g_det_iw && g_det_ih) {
        // Camera mode: detectors run at program dims, except under SR.
        sx = (float)w / g_det_iw; sy = (float)h / g_det_ih;
    }
    if (g_ai_pose && g_ai_pose_ready) {
        float r = (float)(w > h ? w : h) / 160.0f * (sx + sy) * 0.5f;
        if (r < 2.0f) r = 2.0f;
        for (int i = 0; i < 33; i++) {
            const auto& lm = g_pose_landmarks[i];
            if (lm.visibility > 0.5f && lm.x >= 0.f && lm.x <= 1.f && lm.y >= 0.f && lm.y <= 1.f)
                ai::launch_draw_circle(g_d_rgb, w, h, ox + lm.x * spanx, oy + lm.y * spany,
                                       r, 255, 40, 40, g_stream);
        }
    }
    if (g_ai_face && g_ai_face_ready) {
        for (int i = 0; i < 4; i++) {
            const auto& fb = g_face_boxes[i];
            if (fb.score <= 0.5f) continue;
            ai::launch_draw_rect(g_d_rgb, w, h,
                (int)(ox + fb.x1 * sx), (int)(oy + fb.y1 * sy),
                (int)(ox + fb.x2 * sx), (int)(oy + fb.y2 * sy),
                0, 255, 0, 2, g_stream);
        }
    }
    if (g_ai_hands && g_ai_hands_ready) {
        float r = (float)(w > h ? w : h) / 200.0f * (sx + sy) * 0.5f;
        if (r < 2.0f) r = 2.0f;
        for (int k = 0; k < 2; k++) {
            if (g_hand_joints[k].score <= 0.5f) continue;
            for (int j = 0; j < 21; j++)
                ai::launch_draw_circle(g_d_rgb, w, h,
                    ox + g_hand_joints[k].x[j] * sx,
                    oy + g_hand_joints[k].y[j] * sy,
                    r, 0, 255, 255, g_stream);
        }
    }
    if (g_ai_gaze && g_ai_gaze_ready) {
        float gr = tuto ? 4.0f * (sx + sy) * 0.5f : 4.0f;
        if (gr < 1.5f) gr = 1.5f;
        for (int k = 0; k < 2; k++) {
            if (g_gaze_eyes[k].score < 0.4f) continue;
            ai::launch_draw_circle(g_d_rgb, w, h,
                ox + g_gaze_eyes[k].cx * sx, oy + g_gaze_eyes[k].cy * sy,
                gr, 0, 255, 0, g_stream);
            ai::launch_draw_circle(g_d_rgb, w, h,
                ox + (g_gaze_eyes[k].cx + g_gaze_eyes[k].dx * 24.0f) * sx,
                oy + (g_gaze_eyes[k].cy + g_gaze_eyes[k].dy * 24.0f) * sy,
                gr * 0.5f, 255, 255, 0, g_stream);
        }
    }
    if (g_ai_detect && g_ai_detect_ready) {
        // Detector coords: program dims in camera mode, cap dims in tutorial.
        float dsx = tuto ? (float)sx : (g_det_iw ? (float)w / g_det_iw : 1.0f);
        float dsy = tuto ? (float)sy : (g_det_ih ? (float)h / g_det_ih : 1.0f);
        for (int k = 0; k < 16; k++) {
            const auto& db = g_det_boxes[k];
            if (db.score <= 0.35f) continue;
            uint8_t dc[3];
            ai::ai_detect_color(db.cls, dc);
            int bx1 = (int)(ox + db.x1 * dsx), by1 = (int)(oy + db.y1 * dsy);
            int bx2 = (int)(ox + db.x2 * dsx), by2 = (int)(oy + db.y2 * dsy);
            ai::launch_draw_rect(g_d_rgb, w, h, bx1, by1, bx2, by2,
                                 dc[0], dc[1], dc[2], 2, g_stream);
            int ty = by1 - 12;
            if (ty < 0) ty = by2 + 2;
            if (ty < 0) ty = 0;
            ai::launch_draw_text(g_d_rgb, w, h, bx1 < 0 ? 0 : bx1, ty,
                                 ai::ai_detect_class_name(db.cls),
                                 2, 255, 255, 255, dc[0], dc[1], dc[2], g_stream);
        }
    }
    game_draw_canvas(); // GAME tab canvas (crosshair/targets/HUD)
    rec_gesture_tick();
}

static void rec_gesture_tick() {
    // best visible hand
    int best = -1;
    for (int k = 0; k < 2; k++)
        if (g_hand_joints[k].score > 0.5f &&
            (best < 0 || g_hand_joints[k].score > g_hand_joints[best].score))
            best = k;
    RecGest g = (best >= 0) ? rec_classify(g_hand_joints[best]) : RG_NONE;

    static RecGest s_last = RG_NONE;
    static auto s_since = std::chrono::steady_clock::now();
    static auto s_cool_until = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (g != s_last) { s_last = g; s_since = now; }
    double stable = std::chrono::duration<double>(now - s_since).count();
    bool cooling = (now < s_cool_until);
    bool recording = (g_rec_state.load() == 1);

    const char* nm[] = {"--", "PALM", "FIST", "THUMB", "PEACE"};
    char txt[96];
    if (!recording)
        snprintf(txt, sizeof(txt), "gestures idle (start a take)");
    else if (g == RG_NONE)
        snprintf(txt, sizeof(txt), "gestures ready: palm=cut thumb=mark");
    else
        snprintf(txt, sizeof(txt), "saw %s %.1fs%s", nm[(int)g], stable,
                 cooling ? " (cooldown)" : "");
    rec_gest_set(txt);

    // Preview OSD: amber while a hold gesture stabilizes, green on fire.
    if (recording && !cooling && g != RG_NONE && stable > 0.05)
        g_rec_osd = 1;
    else if (g_rec_osd.load() == 1)
        g_rec_osd = 0;

    if (!recording || cooling) return;
    bool fire = false;
    if (g == RG_PALM && stable >= 0.5) { rec_cut(); fire = true; }
    else if (g == RG_THUMB && stable >= 0.3) { rec_mark(); fire = true; }
    else if (g == RG_PEACE && stable >= 0.3) { rec_pause_toggle(); fire = true; }
    else if (g == RG_FIST && stable >= 2.0) { rec_stop(); fire = true; }
    if (fire) {
        s_cool_until = now + std::chrono::seconds(2);
        s_since = now; // require a fresh hold for the next trigger
        g_rec_osd = 2; g_rec_osd_t0 = now;
    }
}

static void rec_finalize() {
    // Measured fps over the take: keeps playback realtime even if the GPU
    // couldn't sustain 30 during heavy background work (e.g. TRT warmup).
    double el = rec_elapsed();
    uint32_t fps = 30;
    if (el > 0.5) {
        fps = (uint32_t)((double)g_rec_frames.load() / el + 0.5);
        if (fps < 5) fps = 5;
        if (fps > 60) fps = 60;
    }
    // Flush must run on the GPU thread (context affinity): request it and
    // wait for the gpu loop to drain + destroy the session.
    g_rec_eos_tail.clear();
    g_rec_eos_done = false;
    g_rec_enc_req = 2;
    for (int i = 0; i < 500 && !g_rec_eos_done.load(); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::vector<std::vector<uint8_t>> tail = std::move(g_rec_eos_tail);
    // Stopped mid-CUT-gap: the drained tail belongs to the cut region.
    bool drop_tail = g_rec_discarding;
    if (g_rec_file) {
        if (!drop_tail)
            for (auto& f : tail) if (!f.empty()) { fwrite(f.data(), 1, f.size(), g_rec_file); g_rec_seg_bs++; }
        fclose(g_rec_file); g_rec_file = nullptr;
    }
    if (!g_rec_cuts.empty() && g_rec_cuts.back().end < 0) {
        g_rec_cuts.back().end = g_rec_submitted;
        g_rec_cuts.back().t1 = rec_now();
    }
    // Register or trash the trailing segment.
    if (g_rec_seg_bs >= REC_SEG_MIN)
        g_rec_segs.push_back({g_rec_seg_idx, g_rec_seg_start, g_rec_submitted, true});
    else
        remove(rec_seg_path(g_rec_seg_idx).c_str());

    int kept = 0;
    for (auto& s : g_rec_segs) if (s.kept) kept++;
    if (kept == 0) {
        rec_set_msg("everything cut, discarded"); g_rec_state = 4;
    } else {
        // Concat kept segs (all start with IDR) -> MP4, no re-encode.
        // Absolute forward-slash paths: the concat demuxer resolves
        // relative entries against the process CWD, not the list file.
        std::string list = g_rec_basedir + "\\list.txt";
        FILE* lf = fopen(list.c_str(), "w");
        for (auto& s : g_rec_segs) {
            if (!s.kept) continue;
            std::string ap = rec_seg_path(s.idx);
            for (auto& c : ap) if (c == '\\') c = '/';
            fprintf(lf, "file '%s'\n", ap.c_str());
        }
        fclose(lf);
        std::string flog = g_rec_basedir + "\\concat.log";
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -hide_banner -nostdin -y -r %u -f concat -safe 0 -i \"%s\" -c copy -movflags +faststart \"%s\" 2>\"%s\"",
            fps, list.c_str(), g_rec_mp4.c_str(), flog.c_str());
        printf("[REC] concat %d segs @%ufps...\n", kept, fps); fflush(stdout);
        bool ok = (system(cmd) == 0);
        // EDL: markers + cuts as CSV (frame + seconds), NLE-friendly.
        char edl[512];
        snprintf(edl, sizeof(edl), "%s\\edit.csv", g_rec_basedir.c_str());
        FILE* ef = fopen(edl, "w");
        if (ef) {
            fprintf(ef, "type,frame,time_s,label\n");
            for (auto& m : g_rec_marks)
                fprintf(ef, "mark,%d,%.2f,%s\n", m.frame, m.frame / 30.0, m.label.c_str());
            for (auto& c : g_rec_cuts)
                fprintf(ef, "cut,%d,%.2f,cut %d-%d\n", c.start, c.start / 30.0, c.start, c.end);
            fclose(ef);
        }
        if (ok) {
            // Success: segs served their purpose, keep list+csv+log next to the mp4.
            for (auto& s : g_rec_segs) remove(rec_seg_path(s.idx).c_str());
            remove(list.c_str());
            // Voice track: mux the mic wav under the video (same take).
            bool voiced = false;
            std::string voicemsg;
            if (!g_rec_wav.empty()) {
                FILE* wf = fopen(g_rec_wav.c_str(), "rb");
                long wsize = 0;
                if (wf) { fseek(wf, 0, SEEK_END); wsize = ftell(wf); fclose(wf); }
                double wsec = (g_au_rate && g_au_ch)
                    ? (wsize - 44) / (double)(g_au_rate * g_au_ch * 2) : 0.0;
                if (wsec > 1.0) {
                    std::string av = g_rec_basedir + "\\take_av.mp4";
                    std::string alog = g_rec_basedir + "\\av.log";
                    // Normalize quiet voice to broadcast level (single-pass
                    // loudnorm = static gain + limiter, no pumping). Skip when
                    // the mic captured only digital silence (would boost hiss).
                    double vmean = -100.0;
                    {
                        std::string vlog = g_rec_basedir + "\\vol.log";
                        char vcmd[1024];
                        snprintf(vcmd, sizeof(vcmd),
                            "ffmpeg -hide_banner -nostdin -i \"%s\" -af volumedetect -f null - 2>\"%s\"",
                            g_rec_wav.c_str(), vlog.c_str());
                        system(vcmd);
                        FILE* vf = fopen(vlog.c_str(), "r");
                        if (vf) {
                            char ln[256];
                            while (fgets(ln, sizeof(ln), vf)) {
                                const char* p = strstr(ln, "mean_volume:");
                                if (p) { vmean = atof(p + 12); break; }
                            }
                            fclose(vf);
                        }
                        remove(vlog.c_str());
                    }
                    // Deterministic voice lift: static gain to -20dB mean +
                    // limiter (single-pass loudnorm under-adapts on short
                    // takes and left takes quiet).
                    char lift[96] = "";
                    double applied = 0.0;
                    if (vmean > -50.0) {
                        double g = -20.0 - vmean;
                        if (g < 0.0) g = 0.0;
                        if (g > 30.0) g = 30.0;
                        applied = g;
                        snprintf(lift, sizeof(lift),
                            "volume=%.1fdB,alimiter=limit=0.9", g);
                    }
                    // CUT sync: snip the audio ranges whose video was cut
                    // (pause spans + CUT gaps are already gated silent live).
                    // live cuts store press time in t0; cut-prev stores [a,b].
                    std::string snip;
                    for (auto& c : g_rec_cuts) {
                        double A = -1, B = -1;
                        if (c.live) {
                            if (c.start >= 0 && c.start < (int)g_rec_ft.size() &&
                                c.t0 > 0) {
                                A = g_rec_ft[c.start]; B = c.t0;
                            }
                        } else if (c.t0 >= 0 && c.t1 > c.t0) {
                            A = c.t0; B = c.t1;
                        }
                        if (A >= 0 && B > A + 0.05) {
                            // A/B are absolute steady-clock seconds; aselect's
                            // `t` is 0-based (audio-file start). Shift into the
                            // audio timeline using the first-sample wall time,
                            // or the range matches nothing and no voice is cut
                            // -> the take drifts out of lip-sync after every cut.
                            if (g_au_epoch > 0) { A -= g_au_epoch; B -= g_au_epoch; }
                            if (A < 0) A = 0;
                            if (!snip.empty()) snip += "+";
                            char r[96];
                            snprintf(r, sizeof(r), "between(t\\,%.3f\\,%.3f)", A, B);
                            snip += r;
                        }
                    }
                    char aff[4096] = "";
                    {
                        std::string chain;
                        if (!snip.empty())
                            chain += "aselect='not(" + snip + ")',asetpts=N/SR/TB,aresample=async=1";
                        if (lift[0]) {
                            if (!chain.empty()) chain += ",";
                            chain += lift;
                        }
                        if (!chain.empty())
                            snprintf(aff, sizeof(aff), " -af \"%s\"", chain.c_str());
                    }
                    char acmd[8192];
                    snprintf(acmd, sizeof(acmd),
                        "ffmpeg -hide_banner -nostdin -y -i \"%s\" -i \"%s\" -c:v copy -c:a aac -b:a 128k%s -movflags +faststart -shortest \"%s\" 2>\"%s\"",
                        g_rec_mp4.c_str(), g_rec_wav.c_str(), aff, av.c_str(), alog.c_str());
                    if (system(acmd) == 0) {
                        remove(g_rec_mp4.c_str());
                        remove(g_rec_wav.c_str());
                        remove(alog.c_str());
                        MoveFileA(av.c_str(), g_rec_mp4.c_str());
                        voiced = true;
                        FILE* tlf = fopen((g_rec_basedir + "\\take.log").c_str(), "a");
                        if (tlf) {
                            fprintf(tlf, "voice mean=%.1f lift=%.1f snip=%d\n",
                                    vmean, applied, snip.empty() ? 0 : 1);
                            fclose(tlf);
                        }
                    } else voicemsg = " (voice mux failed, .wav kept)";
                } else {
                    remove(g_rec_wav.c_str());
                    voicemsg = " (no voice: mic silent?)";
                }
            }
            char msg[300];
            snprintf(msg, sizeof(msg), "saved: %s (%d segs, %d cuts, %d marks%s%s)",
                     g_rec_mp4.c_str(), kept, (int)g_rec_cuts.size(), (int)g_rec_marks.size(),
                     voiced ? " + voice" : "", voiced ? "" : voicemsg.c_str());
            rec_set_msg(msg); g_rec_state = 3;
        } else {
            // FAILURE: keep everything (segs + list + ffmpeg log) so the
            // take is recoverable by hand. Do NOT delete the footage.
            char msg[300];
            snprintf(msg, sizeof(msg), "concat failed, segs kept in %s (see concat.log)",
                     g_rec_basedir.c_str());
            rec_set_msg(msg); g_rec_state = 4;
        }
    }
    printf("[REC] %s\n", rec_get_msg().c_str()); fflush(stdout);
}

static void rec_stop() {
    if (g_rec_state.load() != 1) return;
    g_rec_paused = false;
    g_rec_osd = 0;
    au_stop(); // end voice with video (both started together)
    g_rec_state = 2; rec_set_msg("finalizing...");
    if (g_rec_thread.joinable()) g_rec_thread.join();
    g_rec_thread = std::thread(rec_finalize);
}

static RECT rc_rec_view; // self-view preview
static int rec_ctl_y = 0;       // controls base (below preview)
// Larger record controls (readability).
static const int REC_SRC_H = 28;
static const int REC_FMT_H = 32;
static const int REC_GO_H = 46;
static const int REC_CUT_H = 34;
static const int REC_MIC_H = 30;
// Fixed control strip below the preview: gap 8 + src + gap 8 +
// label/out/warn block (58) + codec + gap 6 + GO + gap 6 + CUT + gap 6 + mic.
// Preview takes all remaining vertical space (no 340px cap).
static const int REC_CTL_H = 8 + 28+8 + 58 + 32+6 + 46+6 + 34+6 + 30;
static void rec_layout(int W, int H) {
    int px = SIDE_W + 14, pw = W - SIDE_W - 28;
    if (pw < 200) pw = 200;
    int avail = H - TAB_H - STATUS_H - 14 - REC_CTL_H;
    if (avail < 160) avail = 160;
    double fa = (g_disp_w > 0 && g_disp_h > 0) ? (double)g_disp_w / (double)g_disp_h
                                               : (double)g_cap_w / (double)g_cap_h;
    int ph = avail, pwid = (int)(ph * fa);
    if (pwid > pw) { pwid = pw; ph = (int)(pwid / fa); }
    if (ph < 100) ph = 100;
    int vy = TAB_H + 6;
    rc_rec_view = {px + (pw - pwid) / 2, vy, px + (pw - pwid) / 2 + pwid, vy + ph};
    int sy2 = vy + ph + 8, sw2 = (pw - 2 * 8) / 3;
    rc_rec_src[0] = {px, sy2, px + sw2, sy2 + REC_SRC_H};
    rc_rec_src[1] = {px + sw2 + 8, sy2, px + 2 * sw2 + 8, sy2 + REC_SRC_H};
    rc_rec_src[2] = {px + 2 * sw2 + 16, sy2, px + pw, sy2 + REC_SRC_H};
    int y = sy2 + REC_SRC_H + 8; // controls base
    rec_ctl_y = y;
    int fw = (pw - 2 * 8) / 3, fy = y + 58;
    rc_rec_codec = {px, fy, px + fw, fy + REC_FMT_H};
    rc_rec_br = {px + fw + 8, fy, px + 2 * fw + 8, fy + REC_FMT_H};
    rc_rec_gain = {px + 2 * fw + 16, fy, px + pw, fy + REC_FMT_H};
    rc_rec_go = {px, fy + REC_FMT_H + 6, px + pw, fy + REC_FMT_H + 6 + REC_GO_H};
    int cy = fy + REC_FMT_H + 6 + REC_GO_H + 6;
    rc_rec_cut = {px, cy, px + pw, cy + REC_CUT_H};
    int ry = cy + REC_CUT_H + 6, sw = (pw - 3 * 8) / 4;
    rc_rec_mic = {px, ry, px + sw, ry + REC_MIC_H};
    rc_rec_prev = {px + sw + 8, ry, px + 2 * sw + 8, ry + REC_MIC_H};
    rc_rec_pause = {px + 2 * sw + 16, ry, px + 3 * sw + 16, ry + REC_MIC_H};
    rc_rec_mark = {px + 3 * sw + 24, ry, px + pw, ry + REC_MIC_H};
}

// ids: 1 = rec/stop, 2 = codec, 3 = bitrate, 4 = cut, 5 = cut prev,
// 6 = pause/resume, 7 = mark, 8 = mic list, 9/10/11 = cam/screen/tutorial,
// 12 = mic gain, 20+i = mic list row i
static int rec_hittest(int mx, int my, int W, int H) {
    rec_layout(W, H);
    if (g_mic_list_open) {
        int px = SIDE_W + 14, pw = W - SIDE_W - 28;
        int ry0 = rc_rec_mic.bottom + 4;
        for (int i = 0; i < mic_row_count() && i < 10; i++) {
            int ry = ry0 + i * 24;
            if (ry + 22 >= H - STATUS_H - 8) break;
            if (mx >= px && mx < px + pw && my >= ry && my < ry + 22)
                return 20 + i;
        }
    }
    const RECT* rcs[] = {nullptr, &rc_rec_go, &rc_rec_codec, &rc_rec_br,
                         &rc_rec_cut, &rc_rec_prev, &rc_rec_pause, &rc_rec_mark,
                         &rc_rec_mic,
                         &rc_rec_src[0], &rc_rec_src[1], &rc_rec_src[2],
                         &rc_rec_gain};
    for (int id = 1; id <= 12; id++) {
        const RECT* r = rcs[id];
        if (mx >= r->left && mx < r->right && my >= r->top && my < r->bottom)
            return id;
    }
    return -1;
}

static void rec_fire(int id) {
    if (id == 1) {
        if (g_rec_state.load() == 1) rec_stop(); else rec_start();
    }
    else if (id == 2) { if (g_rec_state.load() != 1) g_rec_codec = 1 - g_rec_codec; }
    else if (id == 3) { if (g_rec_state.load() != 1) g_rec_br_idx = (g_rec_br_idx + 1) % 5; }
    else if (id == 4) rec_cut();
    else if (id == 5) rec_cut_prev();
    else if (id == 6) rec_pause_toggle();
    else if (id == 7) rec_mark();
    else if (id == 8) {
        if (g_rec_state.load() != 1) {
            if (!g_mic_list_open) mic_enum(); // refresh on open
            g_mic_list_open = !g_mic_list_open;
        }
    }
    else if (id >= 20 && id < 30) {
        int i = id - 20;
        if (i >= 0 && i < mic_row_count()) {
            g_mic_sel = i;
            if (g_mic_sel > (int)g_mic_devs.size() + 1) g_mic_sel = 1;
            g_mic_on = (g_mic_sel != 0);
        }
        g_mic_list_open = false;
    }
    else if (id >= 9 && id <= 11) { screen_ensure(); g_src_mode = id - 9; }
    else if (id == 12) {
        if (g_rec_state.load() != 1) {
            float g = g_mic_gain.load();
            g_mic_gain = (g < 1.5f) ? 2.0f : (g < 3.0f ? 4.0f : 1.0f);
        }
    }
}

static void render_record_panel(HDC hdc, int W, int H) {
    rec_layout(W, H);
    int px = SIDE_W + 14, pw = W - SIDE_W - 28;
    int st = g_rec_state.load();
    bool recording = (st == 1);
    SelectObject(hdc, g_font_st);
    SetBkMode(hdc, TRANSPARENT);

    // ---- self-view (same live frame as the virtual cam) ----
    if (g_display_out && g_disp_w > 0 && g_disp_h > 0) {
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = g_disp_w;
        bmi.bmiHeader.biHeight = -(int)g_disp_h;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;
        SetStretchBltMode(hdc, HALFTONE);
        { std::lock_guard<std::mutex> lock(g_frame_mutex);
          StretchDIBits(hdc, rc_rec_view.left, rc_rec_view.top,
              rc_rec_view.right - rc_rec_view.left,
              rc_rec_view.bottom - rc_rec_view.top,
              0, 0, g_disp_w, g_disp_h,
              g_display_out, &bmi, DIB_RGB_COLORS, SRCCOPY); }
        HPEN pen = CreatePen(PS_SOLID, 1, g_rec_state.load() == 1 ? RGB(255, 60, 60) : CLR_LINE);
        HPEN op = (HPEN)SelectObject(hdc, pen);
        HBRUSH ob = (HBRUSH)GetStockObject(NULL_BRUSH);
        HBRUSH oo = (HBRUSH)SelectObject(hdc, ob);
        Rectangle(hdc, rc_rec_view.left, rc_rec_view.top,
                  rc_rec_view.right + 1, rc_rec_view.bottom + 1);
        SelectObject(hdc, oo); SelectObject(hdc, op); DeleteObject(pen);
        if (!g_cam_alive) {
            SetTextColor(hdc, RGB(255,255,255));
            SelectObject(hdc, g_font_ui);
            RECT nr = {rc_rec_view.left, rc_rec_view.top + 8,
                       rc_rec_view.right, rc_rec_view.top + 32};
            DrawTextA(hdc, "NO SIGNAL -- test pattern", -1, &nr,
                      DT_CENTER|DT_VCENTER|DT_SINGLELINE);
            SelectObject(hdc, g_font_st);
        }
    }

    // source selector: CAM / SCREEN / TUTORIAL (screen + camera PiP)
    {
        const char* sn[] = {"CAM", "SCREEN", "TUTORIAL"};
        for (int i = 0; i < 3; i++) {
            bool sel = (g_src_mode == i);
            bool hov = (g_hover == 2009 + i);
            COLORREF bg = sel ? RGB(0, 130, 190) : (hov ? RGB(42, 42, 58) : CLR_PANEL2);
            HBRUSH bb = CreateSolidBrush(bg);
            draw_rounded_rect(hdc, rc_rec_src[i].left, rc_rec_src[i].top,
                              rc_rec_src[i].right - rc_rec_src[i].left,
                              rc_rec_src[i].bottom - rc_rec_src[i].top,
                              6, bb, sel ? RGB(255,255,255) : CLR_LINE);
            DeleteObject(bb);
            SetTextColor(hdc, sel ? RGB(255,255,255) : CLR_TEXT);
            SelectObject(hdc, g_font_st);
            RECT sr = {rc_rec_src[i].left, rc_rec_src[i].top,
                       rc_rec_src[i].right, rc_rec_src[i].bottom};
            DrawTextA(hdc, sn[i], -1, &sr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        }
    }

    int y = rec_ctl_y;
    SetTextColor(hdc, CLR_DIM);
    RECT lr = {px, y, px + pw, y + 16};
    DrawTextA(hdc, "LIVE CUT RECORDER  (vcam mirrors this feed)", -1, &lr,
              DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    // output path
    SelectObject(hdc, g_font_ui);
    SetTextColor(hdc, CLR_TEXT);
    std::string od = "OUT:  " + (g_rec_dir.empty() ? (get_exe_dir() + "\\recordings") : g_rec_dir);
    RECT or2 = {px, y + 18, px + pw, y + 38};
    DrawTextA(hdc, od.c_str(), -1, &or2, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_PATH_ELLIPSIS);
    if (!g_cam_alive) {
        SetTextColor(hdc, CLR_BAD);
        RECT nr = {px, y + 40, px + pw, y + 56};
        DrawTextA(hdc, "NO CAMERA — recording test pattern", -1, &nr,
                  DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    } else if (g_cam_black) {
        SetTextColor(hdc, RGB(255, 90, 90));
        RECT nr = {px, y + 40, px + pw, y + 56};
        DrawTextA(hdc, "CAMERA BLACK — check shutter/cover (take will be green)", -1, &nr,
                  DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    }

    // codec / bitrate / mic-gain chips
    char cb[32]; sprintf(cb, "Codec: %s", g_rec_codec ? "H265" : "H264");
    bool locked_fmt = recording;
    COLORREF bgc = locked_fmt ? RGB(24,24,32) : CLR_PANEL2;
    HBRUSH bb;
    struct { RECT* r; const char* t; int hov; } fmt[] = {
        {&rc_rec_codec, cb, 2002},
        {&rc_rec_br, nullptr, 2003},
        {&rc_rec_gain, nullptr, 2012},
    };
    const char* brs[] = {"BR: 5M", "BR: 8M", "BR: 12M", "BR: 20M", "BR: 40M"};
    char gnt[16]; snprintf(gnt, sizeof(gnt), "GAIN %.0fx", g_mic_gain.load());
    const char* ftext[] = {cb, brs[g_rec_br_idx], gnt};
    SelectObject(hdc, g_font_ui);
    for (int i = 0; i < 3; i++) {
        bool hov = (!locked_fmt && g_hover == fmt[i].hov);
        COLORREF b2c = hov ? RGB(42, 42, 58) : bgc;
        bb = CreateSolidBrush(b2c);
        draw_rounded_rect(hdc, fmt[i].r->left, fmt[i].r->top,
                          fmt[i].r->right - fmt[i].r->left,
                          fmt[i].r->bottom - fmt[i].r->top,
                          6, bb, hov ? CLR_ACCENT : CLR_LINE);
        DeleteObject(bb);
        SetTextColor(hdc, locked_fmt ? RGB(90,90,110) : CLR_TEXT);
        RECT cr = {fmt[i].r->left + 10, fmt[i].r->top, fmt[i].r->right - 10, fmt[i].r->bottom};
        DrawTextA(hdc, ftext[i], -1, &cr, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    }

    // big REC / STOP button
    COLORREF bg = recording ? RGB(150, 40, 40) : RGB(0, 140, 90);
    if (!recording && g_hover == 2001) bg = RGB(0, 170, 110);
    bb = CreateSolidBrush(bg);
    draw_rounded_rect(hdc, rc_rec_go.left, rc_rec_go.top, pw,
                      rc_rec_go.bottom - rc_rec_go.top,
                      8, bb, RGB(255,255,255));
    DeleteObject(bb);
    SetTextColor(hdc, RGB(255,255,255));
    SelectObject(hdc, g_font_ui);
    char gob[64];
    if (recording) {
        double e = rec_elapsed();
        sprintf(gob, "STOP   %02d:%02d", (int)e / 60, (int)e % 60);
    } else if (st == 2) sprintf(gob, "FINALIZING...");
    else sprintf(gob, "START RECORDING  (F9)");
    RECT gr = {rc_rec_go.left, rc_rec_go.top, rc_rec_go.right, rc_rec_go.bottom};
    DrawTextA(hdc, gob, -1, &gr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);

    // CUT button (full width, amber)
    {
        bool can = recording && !g_rec_paused;
        COLORREF bg = can ? RGB(160, 110, 20) : RGB(24, 24, 32);
        if (can && g_hover == 2004) bg = RGB(190, 130, 25);
        HBRUSH b2 = CreateSolidBrush(bg);
        draw_rounded_rect(hdc, rc_rec_cut.left, rc_rec_cut.top, pw,
                          rc_rec_cut.bottom - rc_rec_cut.top,
                          8, b2, can ? RGB(255,255,255) : CLR_LINE);
        DeleteObject(b2);
        SetTextColor(hdc, can ? RGB(255,255,255) : RGB(90,90,110));
        SelectObject(hdc, g_font_ui);
        RECT cr2 = {rc_rec_cut.left, rc_rec_cut.top, rc_rec_cut.right, rc_rec_cut.bottom};
        DrawTextA(hdc, "CUT LAST ~4s  (F8)", -1, &cr2, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
    }
    // mic / prev / pause / mark row (+ live mic meter under it)
    bool show_micmsg = !g_au_msg.empty() && !recording;
    {
        bool can = recording && !g_rec_paused;
        std::string mics = mic_label();
        const char* mict = mics.c_str();
        struct { RECT* r; const char* t; int id; bool en; } btns[] = {
            {&rc_rec_mic, mict, 2008, !recording},
            {&rc_rec_prev, "CUT PREV", 2005, can},
            {&rc_rec_pause, g_rec_paused ? "RESUME" : "PAUSE", 2006, recording},
            {&rc_rec_mark, "MARK (4)", 2007, can},
        };
        for (auto& b : btns) {
            COLORREF bg = b.en ? CLR_PANEL2 : RGB(24, 24, 32);
            if (b.en && g_hover == b.id) bg = RGB(42, 42, 58);
            HBRUSH b2 = CreateSolidBrush(bg);
            draw_rounded_rect(hdc, b.r->left, b.r->top,
                              b.r->right - b.r->left,
                              b.r->bottom - b.r->top, 6, b2,
                              (b.en && g_hover == b.id) ? CLR_ACCENT : CLR_LINE);
            DeleteObject(b2);
            SetTextColor(hdc, b.en ? CLR_TEXT : RGB(90, 90, 110));
            SelectObject(hdc, g_font_st);
            RECT lr2 = {b.r->left, b.r->top, b.r->right, b.r->bottom};
            DrawTextA(hdc, b.t, -1, &lr2, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        }
        // mic level meter under the row (smoothed peak)
        {
            static float shown = 0;
            float target = g_mic_level.load();
            if (target > shown) shown = target;
            else shown += (target - shown) * 0.35f;
            if (shown < 0) shown = 0; if (shown > 1) shown = 1;
            int my = rc_rec_mic.bottom + 3;
            HBRUSH tb = CreateSolidBrush(RGB(24, 24, 32));
            draw_rounded_rect(hdc, px, my, pw, 5, 2, tb, RGB(24, 24, 32));
            DeleteObject(tb);
            if (shown > 0.01f) {
                int mw = (int)(pw * shown);
                if (mw < 4) mw = 4;
                COLORREF mc = shown > 0.92f ? RGB(220, 60, 60) : RGB(60, 200, 110);
                HBRUSH mb = CreateSolidBrush(mc);
                draw_rounded_rect(hdc, px, my, mw, 5, 2, mb, mc);
                DeleteObject(mb);
            }
        }
        if (show_micmsg) {
            SetTextColor(hdc, CLR_BAD);
            RECT ar = {px, rc_rec_mic.bottom + 10, px + pw, rc_rec_mic.bottom + 28};
            std::string am = "mic: " + g_au_msg;
            DrawTextA(hdc, am.c_str(), -1, &ar, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
        }
    }

    // stats line
    SelectObject(hdc, g_font_st);
    char sl[256];
    if (recording) {
        double free_gb = rec_disk_free_gb();
        char df[32] = "";
        if (free_gb >= 0) snprintf(df, sizeof(df), "   %.1f GB free", free_gb);
        const char* sname[3] = {"CAM", "SCR", "TUT"};
        const char* micwarn = (g_rec_mic && g_au_silent.load()) ? "   MIC SILENT?" : "";
        const char* warmwarn = g_ai_warming.load() ? "   AI WARMING (may stutter)" : "";
        char scr[48] = "";
        if (g_src_mode != 0)
            snprintf(scr, sizeof(scr), "   scr=%.0f/%dfps",
                     g_screen_mean.load(), (int)(g_screen_fps.load() + 0.5f));
        sprintf(sl, "%s REC%s  %d frames   %d segs   %d cuts   %d marks   %d dropped%s%s%s%s",
                sname[g_src_mode < 0 || g_src_mode > 2 ? 0 : g_src_mode],
                g_rec_paused ? " (PAUSED)" : "", g_rec_frames.load(),
                (int)g_rec_segs.size(), (int)g_rec_cuts.size(),
                (int)g_rec_marks.size(), g_rec_dropped.load(), df, micwarn, scr, warmwarn);
    }
    else
        sprintf(sl, "%s", rec_get_msg().c_str());
    SetTextColor(hdc, recording ? RGB(255, 90, 90) :
                    (st == 3 ? CLR_OK : (st == 4 ? CLR_BAD : CLR_DIM)));
    int sy = rc_rec_mark.bottom + (show_micmsg ? 30 : 8);
    RECT sr2 = {px, sy, px + pw, sy + 20};
    DrawTextA(hdc, sl, -1, &sr2, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_PATH_ELLIPSIS);

    // timeline: last few segs + open cut/marks
    {
        int ty = sy + 24;
        SetTextColor(hdc, CLR_DIM);
        RECT tr = {px, ty, px + pw, ty + 18};
        DrawTextA(hdc, "TIMELINE (this take)", -1, &tr, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
        ty += 20;
        int n = (int)g_rec_segs.size();
        int from = n > 6 ? n - 6 : 0;
        for (int i = from; i < n && ty + 18 < H - STATUS_H - 8; i++) {
            const RecSeg& s = g_rec_segs[i];
            char lb[128];
            snprintf(lb, sizeof(lb), "seg %d   %.1f-%.1fs   %s",
                     s.idx, s.start / 30.0, s.end / 30.0,
                     s.kept ? "KEPT" : "CUT");
            SetTextColor(hdc, s.kept ? CLR_OK : CLR_BAD);
            RECT lr3 = {px + 8, ty, px + pw, ty + 18};
            DrawTextA(hdc, lb, -1, &lr3, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            ty += 18;
        }
        if (g_rec_state.load() == 1 && ty + 18 < H - STATUS_H - 8) {
            char lb[128];
            if (g_rec_discarding)
                snprintf(lb, sizeof(lb), ">> cutting... resumes at next IDR");
            else
                snprintf(lb, sizeof(lb), ">> rec seg %d from %.1fs%s",
                         g_rec_seg_idx, g_rec_seg_start / 30.0,
                         g_rec_paused ? " (PAUSED)" : "");
            SetTextColor(hdc, g_rec_discarding ? RGB(255, 200, 80) : CLR_ACCENT);
            RECT lr3 = {px + 8, ty, px + pw, ty + 18};
            DrawTextA(hdc, lb, -1, &lr3, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            ty += 18;
        }
        for (int i = (int)g_rec_marks.size() - 1; i >= 0 && ty + 18 < H - STATUS_H - 8; i--) {
            char lb[128];
            snprintf(lb, sizeof(lb), "MARK %s @ %.1fs",
                     g_rec_marks[i].label.c_str(), g_rec_marks[i].frame / 30.0);
            SetTextColor(hdc, RGB(255, 220, 120));
            RECT lr3 = {px + 8, ty, px + pw, ty + 18};
            DrawTextA(hdc, lb, -1, &lr3, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            ty += 18;
            if (i == (int)g_rec_marks.size() - 3) break; // last 2 marks only
        }
        // Gesture director: live status + emoji legend (what each hand does).
        if (ty + 62 < H - STATUS_H - 8) {
            if (!g_ai_hands) {
                SetTextColor(hdc, CLR_DIM);
                SelectObject(hdc, g_font_st);
                RECT hr = {px, ty + 4, px + pw, ty + 24};
                DrawTextA(hdc, "Gestures OFF -- press J (Hands AI) to direct.",
                          -1, &hr, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            } else {
                SetTextColor(hdc, CLR_AI);
                std::string gt = rec_gest_get();
                RECT hr = {px, ty + 4, px + pw, ty + 22};
                DrawTextA(hdc, gt.c_str(), -1, &hr,
                          DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            }
            // legend: emoji (emoji font) + action (ui font), 2x2 grid
            struct { const wchar_t* e; const char* t; } lg[] = {
                {L"\xD83D\xDD90", "hold = CUT"},
                {L"\xD83D\xDC4D", "= MARK"},
                {L"\x270C", "= PAUSE"},
                {L"\xD83D\xDC4A", "hold = STOP"},
            };
            int lx = px + 8, ly = ty + 24;
            int colw = (pw - 8) / 2;
            for (int i = 0; i < 4; i++) {
                int cx = lx + (i % 2) * colw, cy = ly + (i / 2) * 20;
                if (cy + 18 >= H - STATUS_H - 8) break;
                SelectObject(hdc, g_font_emoji);
                SetTextColor(hdc, RGB(255, 255, 255));
                RECT er = {cx, cy, cx + 22, cy + 18};
                DrawTextW(hdc, lg[i].e, -1, &er, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
                SelectObject(hdc, g_font_st);
                SetTextColor(hdc, CLR_DIM);
                RECT tr2 = {cx + 24, cy, cx + colw, cy + 18};
                DrawTextA(hdc, lg[i].t, -1, &tr2, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
            }
        }

    // Mic dropdown overlay (meeting-app style list, drawn last = on top).
    if (g_mic_list_open) {
        int n = mic_row_count();
        if (n > 10) n = 10;
        int ry0 = rc_rec_mic.bottom + 4;
        int rh = 0;
        for (int i = 0; i < n; i++) {
            int ry = ry0 + i * 24;
            if (ry + 22 >= H - STATUS_H - 8) break;
            rh = ry + 22 - ry0;
        }
        if (rh > 0) {
            HBRUSH lb = CreateSolidBrush(RGB(18, 18, 28));
            draw_rounded_rect(hdc, px, ry0 - 3, pw, rh + 6, 6, lb, CLR_ACCENT);
            DeleteObject(lb);
            SelectObject(hdc, g_font_st);
            for (int i = 0; i < n; i++) {
                int ry = ry0 + i * 24;
                if (ry + 22 >= H - STATUS_H - 8) break;
                bool sel = (g_mic_sel == i);
                bool hov = (g_hover == 2020 + i);
                if (sel || hov) {
                    HBRUSH rb = CreateSolidBrush(sel ? RGB(0, 110, 160) : RGB(42, 42, 58));
                    draw_rounded_rect(hdc, px + 3, ry, pw - 6, 22, 4, rb, CLR_LINE);
                    DeleteObject(rb);
                }
                SetTextColor(hdc, sel ? RGB(255,255,255) : CLR_TEXT);
                std::string t = (sel ? "> " : "   ") + mic_row_label(i);
                RECT lr = {px + 10, ry, px + pw - 10, ry + 22};
                DrawTextA(hdc, t.c_str(), -1, &lr, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
            }
        }
    }
    }
}

// ---- GPU pipeline hint: header button + info dialog ----
static int g_hover_gpu = 0;
static RECT gpu_hint_rect(int W) {
    RECT r = {W - 14 - 150, 5, W - 14, TAB_H - 5};
    return r;
}
static bool hit_gpu_hint(int mx, int my, int W) {
    RECT r = gpu_hint_rect(W);
    return mx >= r.left && mx < r.right && my >= r.top && my < r.bottom;
}
static void show_gpu_dialog(HWND h) {
    MessageBoxA(h,
        "LIVE tab -- camera / screen capture, no decoder, no encoder.\n"
        "Frame uploads to the GPU once, then every filter + AI model\n"
        "runs on-device (TensorRT FP16). Raw frames go to the virtual\n"
        "camera via shared memory. NVDEC and NVENC are NOT used here.\n"
        "\n"
        "RECORD tab -- taps the same live GPU frame into NVENC.\n"
        "Segments encode on the NVENC chip straight from device memory.\n"
        "AI overlays burn in when enabled; voice muxes in after stop.\n"
        "NVDEC is NOT used (live feed, nothing to decode).\n"
        "\n"
        "TRANSCODE tab -- file in, file out, fully GPU for H.264/H.265.\n"
        "NVDEC decodes to GPU memory, filters + AI run on-device,\n"
        "NVENC encodes back. AV1/VP9/etc. fall back to CPU decode\n"
        "in ffmpeg, then upload once -- the rest stays on GPU.\n"
        "\n"
        "AI on all tabs: TensorRT EP first (engines built once at\n"
        "warmup, cached in trt_engines/), CUDA EP fallback per model.\n"
        "Frames never leave the GPU; only tiny result tensors\n"
        "(boxes, scores, landmarks) download for CPU decode + NMS.",
        "How Kagerou runs on GPU", MB_OK | MB_ICONINFORMATION);
}

#include "03_game_mode.inl"

// Tab bar geometry (shared by draw + hit-test): 4 tabs with gaps.
// Short labels: the selected tab's full name shows in the title area.
static const int TAB_GAP = 5;
static int tab_tw() { return (SIDE_W - 24 - 3 * TAB_GAP) / 4; }
static int tab_tx(int t) { return 12 + t * (tab_tw() + TAB_GAP); }

static void draw_tabs(HDC hdc, int W) {
    const char* names[4] = {"LIVE", "TRANS", "REC", "GAME"};
    int tw = tab_tw();
    SelectObject(hdc, g_font_tab ? g_font_tab : g_font_st);
    SetBkMode(hdc, TRANSPARENT);
    for (int t = 0; t < 4; t++) {
        int x = tab_tx(t);
        bool sel = (g_tab == t);
        bool hov = (t == g_hover_tab);
        COLORREF bg = sel ? RGB(0, 130, 190) : (hov ? RGB(42,42,58) : CLR_PANEL2);
        HBRUSH bb = CreateSolidBrush(bg);
        draw_rounded_rect(hdc, x, 5, tw, TAB_H - 10, 5, bb, sel ? RGB(255,255,255) : CLR_LINE);
        DeleteObject(bb);
        SetTextColor(hdc, sel ? RGB(255,255,255) : CLR_DIM);
        RECT r = {x, 5, x + tw, TAB_H - 5};
        DrawTextA(hdc, names[t], -1, &r, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
    }
    // title in main area
    SelectObject(hdc, g_font_ui);
    SetTextColor(hdc, CLR_TEXT);
    RECT tr2 = {SIDE_W + 14, 0, W - 14, TAB_H};
    const char* ttl = (g_tab == 0) ? "Live Virtual Camera" :
                      (g_tab == 1) ? "File Transcoder" :
                      (g_tab == 2) ? "Live Cut Recorder" : "Gesture Game Control";
    DrawTextA(hdc, ttl, -1, &tr2, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    // GPU pipeline hint button (top-right, all tabs)
    {
        RECT gr = gpu_hint_rect(W);
        bool hov = (g_hover_gpu != 0);
        HBRUSH bb = CreateSolidBrush(hov ? RGB(20, 90, 60) : RGB(16, 60, 44));
        draw_rounded_rect(hdc, gr.left, gr.top, gr.right - gr.left,
                          gr.bottom - gr.top, 5, bb, RGB(0, 200, 130));
        DeleteObject(bb);
        SelectObject(hdc, g_font_st);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(140, 255, 200));
        DrawTextA(hdc, "GPU: HOW IT RUNS", -1, &gr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
    }
}

// Paints into an offscreen DC; called from WM_PAINT. Flicker-free.
static void render_frame(HDC hdc, int W, int H) {
    int vid_x = SIDE_W, vid_w = W - SIDE_W;
    int vid_y = TAB_H, vid_h = H - TAB_H - STATUS_H;

    HBRUSH br_bg = CreateSolidBrush(CLR_BG);
    HBRUSH br_pn = CreateSolidBrush(CLR_PANEL);
    RECT rc = {0, 0, W, H};
    FillRect(hdc, &rc, br_bg);
    RECT side = {0, 0, SIDE_W, H};
    FillRect(hdc, &side, br_pn);
    DeleteObject(br_bg); DeleteObject(br_pn);

    HPEN pen = CreatePen(PS_SOLID, 1, CLR_LINE);
    HPEN op = (HPEN)SelectObject(hdc, pen);
    MoveToEx(hdc, SIDE_W, 0, NULL); LineTo(hdc, SIDE_W, H);
    MoveToEx(hdc, 0, TAB_H, NULL); LineTo(hdc, W, TAB_H);
    MoveToEx(hdc, 0, H-STATUS_H, NULL); LineTo(hdc, W, H-STATUS_H);
    SelectObject(hdc, op); DeleteObject(pen);

    draw_tabs(hdc, W);

    // ---- sidebar header ----
    SelectObject(hdc, g_font_ui);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, CLR_TEXT);
    RECT tr = {14, TAB_H + 8, SIDE_W-14, TAB_H + 30};
    DrawTextA(hdc, "Kagerou", -1, &tr, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    SelectObject(hdc, g_font_st);
    SetTextColor(hdc, CLR_DIM);
    RECT sr2 = {14, TAB_H + 30, SIDE_W-14, TAB_H + 46};
    DrawTextA(hdc, "VIRTUAL CAMERA", -1, &sr2, DT_LEFT|DT_VCENTER|DT_SINGLELINE);

    // status pills under header
    char fps_txt[32]; sprintf(fps_txt, "%.0f FPS", g_fps);
    draw_pill(hdc, 14, TAB_H + 50, 76, 20, fps_txt, RGB(35,35,50), CLR_TEXT, g_font_st);
    if (g_cam_alive) draw_pill(hdc, 94, TAB_H + 50, 66, 20, "CAM", RGB(20,70,45), CLR_OK, g_font_st);
    else draw_pill(hdc, 94, TAB_H + 50, 66, 20, "NO CAM", RGB(70,30,30), CLR_BAD, g_font_st);
    if (g_vcam_active) draw_pill(hdc, 164, TAB_H + 50, 74, 20, "VCAM ON", RGB(20,70,45), CLR_OK, g_font_st);
    else draw_pill(hdc, 164, TAB_H + 50, 74, 20, "VCAM OFF", RGB(70,30,30), CLR_BAD, g_font_st);

    if (g_tab == 3) {
        render_game_panel(hdc, W, H);
    } else {
    // ---- sidebar buttons ----
    BtnRect rects[40];
    BtnRect secs[4];
    layout_buttons(H, rects, secs);
    SelectObject(hdc, g_font_st);
    for (int s = 0; s < 4; s++) {
        if (!rect_visible(secs[s], H)) continue;
        SetTextColor(hdc, CLR_DIM);
        RECT sr3 = {14, secs[s].y, SIDE_W-14, secs[s].y + secs[s].h};
        const char* title = SEC_NAMES[s];
        char ait[64];
        if (s == 3 && g_ai_warming.load()) {
            snprintf(ait, sizeof(ait), "%s  ...warming up", SEC_NAMES[s]);
            title = ait;
            SetTextColor(hdc, CLR_AI);
        }
        DrawTextA(hdc, title, -1, &sr3, DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    }
    // scroll hint / position bar
    int ms = max_scroll(H);
    if (ms > 0) {
        int track_y0 = TAB_H + SIDE_TOP + 4, track_y1 = H - STATUS_H - 12;
        int th = (track_y1 - track_y0) * (track_y1 - track_y0) / g_content_h;
        if (th < 24) th = 24;
        int ty = track_y0 + (track_y1 - track_y0 - th) * g_scroll / ms;
        HBRUSH sb2 = CreateSolidBrush(RGB(60, 60, 80));
        draw_rounded_rect(hdc, SIDE_W - 8, ty, 4, th, 2, sb2, RGB(60, 60, 80));
        DeleteObject(sb2);
    }
    for (int i = 0; i < NBTN; i++) {
        if (!rect_visible(rects[i], H)) continue;
        BtnDef& b = g_btns[i];
        int aist = (b.section == 3) ? ai_button_state(b) : 2;
        bool locked = (aist < 2);
        bool on = *b.state && !locked;
        bool ai = (b.section == 3);
        COLORREF on_bg = ai ? CLR_AI : RGB(0, 130, 190);
        COLORREF bg = on ? on_bg : CLR_PANEL2;
        if (locked) bg = RGB(24, 24, 32); // disabled: flat dark, no hover
        else if (!on && (i == g_hover || i == g_pressed))
            bg = RGB(42, 42, 58); // hover feedback
        COLORREF bd = on ? RGB(255,255,255) : (i == g_hover ? CLR_ACCENT : CLR_LINE);
        HBRUSH bb = CreateSolidBrush(bg);
        draw_rounded_rect(hdc, rects[i].x, rects[i].y, rects[i].w, rects[i].h, 6, bb, bd);
        DeleteObject(bb);

        if (locked) SetTextColor(hdc, RGB(90, 90, 110));
        else SetTextColor(hdc, on ? RGB(255,255,255) : CLR_TEXT);
        SelectObject(hdc, g_font_ui);
        RECT lr = {rects[i].x + 12, rects[i].y, rects[i].x + rects[i].w - 64, rects[i].y + rects[i].h};
        const char* nm = b.name;
        char disp[64];
        if (b.special == 1 && on) {
            const char* ln[]={"","Warm","Cool","Cinema","Vintage","Contrast","Desat"};
            snprintf(disp, sizeof(disp), "LUT: %s", ln[g_lut_preset]);
            nm = disp;
        } else if (b.special == 2 && on) {
            nm = (g_flip_mode == 2) ? "Flip: Vert" : "Flip: Horz";
        } else if (b.special == 3 && on) {
            nm = (g_dirblur_idx == 2) ? "DirBlur: 135" : "DirBlur: 45";
        } else if (b.special == 4 && on) {
            nm = (g_hdr_method == 2) ? "HDR: ACES" : "HDR: Reinhard";
        } else if (b.special == 5 && on) {
            const char* zn[]={"","1.25x","1.5x","2x"};
            snprintf(disp, sizeof(disp), "Zoom: %s", zn[g_zoom_idx]);
            nm = disp;
        } else if (b.special == 6 && on) {
            const char* an[]={"","Wide","Med","Tight"};
            snprintf(disp, sizeof(disp), "Auto: %s", an[g_af_idx]);
            nm = disp;
        }
        DrawTextA(hdc, nm, -1, &lr, DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
        SelectObject(hdc, g_font_st);
        char kh[8]; sprintf(kh, "%s", b.key);
        RECT kr = {rects[i].x + rects[i].w - 62, rects[i].y, rects[i].x + rects[i].w - 40, rects[i].y + rects[i].h};
        SetTextColor(hdc, on ? RGB(255,255,255) : CLR_DIM);
        DrawTextA(hdc, kh, -1, &kr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        // ON/OFF pill (locked AI buttons show status instead)
        if (locked && aist == 1) draw_pill(hdc, rects[i].x + rects[i].w - 38, rects[i].y + 8, 28, 16, "...", RGB(52,52,68), CLR_AI, g_font_st);
        else if (locked) draw_pill(hdc, rects[i].x + rects[i].w - 38, rects[i].y + 8, 28, 16, "N/A", RGB(52,52,68), CLR_DIM, g_font_st);
        else if (on) draw_pill(hdc, rects[i].x + rects[i].w - 38, rects[i].y + 8, 28, 16, "ON", RGB(255,255,255), ai ? CLR_AI : RGB(0,130,190), g_font_st);
        else draw_pill(hdc, rects[i].x + rects[i].w - 38, rects[i].y + 8, 28, 16, "OFF", RGB(52,52,68), CLR_DIM, g_font_st);
    }
    } // end sidebar (game tab renders its own panel above)

    // ---- main area: live video, transcode panel, or record panel ----
    if (g_tab == 1) {
        render_transcode_panel(hdc, W, H);
    } else
    if (g_tab == 2) {
        render_record_panel(hdc, W, H);
    } else
    if (g_display_out && vid_w > 0 && vid_h > 0) {
        double va = (double)vid_w / (double)vid_h;
        double fa = (double)g_disp_w / (double)g_disp_h;
        int dw = vid_w, dh = vid_h, dx = vid_x, dy = vid_y;
        if (fa > va) { dh = (int)(vid_w / fa); dy = vid_y + (vid_h - dh) / 2; }
        else { dw = (int)(vid_h * fa); dx = vid_x + (vid_w - dw) / 2; }
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = g_disp_w;
        bmi.bmiHeader.biHeight = -(int)g_disp_h;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;
        SetStretchBltMode(hdc, HALFTONE);
        { std::lock_guard<std::mutex> lock(g_frame_mutex);
          StretchDIBits(hdc, dx, dy, dw, dh, 0, 0, g_disp_w, g_disp_h,
              g_display_out, &bmi, DIB_RGB_COLORS, SRCCOPY); }
        if (!g_cam_alive) {
            SetTextColor(hdc, RGB(255,255,255));
            SelectObject(hdc, g_font_ui);
            const char* t = "NO SIGNAL -- showing test pattern";
            RECT nr = {dx, dy + 12, dx + dw, dy + 36};
            DrawTextA(hdc, t, -1, &nr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        } else if (g_cam_black) {
            SetTextColor(hdc, RGB(255, 90, 90));
            SelectObject(hdc, g_font_ui);
            const char* t = "CAMERA BLACK -- check lens shutter / cover";
            RECT nr = {dx, dy + 12, dx + dw, dy + 36};
            DrawTextA(hdc, t, -1, &nr, DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        }
    }

    // ---- status bar ----
    SelectObject(hdc, g_font_st);
    SetTextColor(hdc, CLR_DIM);
    char st[128];
    sprintf(st, "  %.0f FPS    %ux%u", g_fps, g_cap_w, g_cap_h);
    RECT sb = {SIDE_W, H-STATUS_H, W, H};
    DrawTextA(hdc, st, -1, &sb, DT_LEFT|DT_VCENTER|DT_SINGLELINE);

    char ac[128];
    if (g_tab == 3) {
        if (game_is_on()) { sprintf(ac, "game live"); SetTextColor(hdc, RGB(0, 220, 130)); }
        else { sprintf(ac, "game idle"); SetTextColor(hdc, CLR_DIM); }
    } else
    if (g_tab == 2) {
        int st = g_rec_state.load();
        if (st == 1) {
            double e = rec_elapsed();
            sprintf(ac, "REC %02d:%02d  %d frames", (int)e / 60, (int)e % 60,
                    g_rec_frames.load());
            SetTextColor(hdc, RGB(255, 90, 90));
        }
        else if (st == 2) { sprintf(ac, "finalizing..."); SetTextColor(hdc, CLR_ACCENT); }
        else if (st == 3) { sprintf(ac, "take saved"); SetTextColor(hdc, CLR_OK); }
        else if (st == 4) { sprintf(ac, "record failed"); SetTextColor(hdc, CLR_BAD); }
        else { sprintf(ac, "record idle"); SetTextColor(hdc, CLR_DIM); }
    } else
    if (g_tab == 1) {
        if (g_tc_state == 1) {
            int fr = g_tc_frames.load(), tot = g_tc_total.load();
            if (tot > 0) sprintf(ac, "transcoding %d/%d (%d%%)", fr, tot, fr * 100 / tot);
            else sprintf(ac, "transcoding %d frames...", fr);
            SetTextColor(hdc, CLR_ACCENT);
        } else if (g_tc_state == 2) { sprintf(ac, "transcode done"); SetTextColor(hdc, CLR_OK); }
        else if (g_tc_state == 3) { sprintf(ac, "transcode failed"); SetTextColor(hdc, CLR_BAD); }
        else { sprintf(ac, "transcode idle"); SetTextColor(hdc, CLR_DIM); }
    } else {
        int active = 0;
        for (int i = 0; i < NBTN; i++) if (*g_btns[i].state) active++;
        if (active == 0) sprintf(ac, "passthrough");
        else sprintf(ac, "%d filter%s active", active, active > 1 ? "s" : "");
        SetTextColor(hdc, active ? CLR_AI : CLR_DIM);
    }
    RECT ar = {SIDE_W, H-STATUS_H, W - 10, H};
    DrawTextA(hdc, ac, -1, &ar, DT_RIGHT|DT_VCENTER|DT_SINGLELINE);
}

// ============================================================================
// Input handling
// ============================================================================
static void toggle_lut() {
    g_lut_preset = (g_lut_preset+1) % 7;
    g_lut_on = (g_lut_preset > 0);
    if (g_d_lut) { cudaFree(g_d_lut); g_d_lut = nullptr; }
    if (g_lut_on) g_d_lut = filters::generate_builtin_lut(g_lut_preset, g_stream);
}
static void toggle_flip() { g_flip_mode = (g_flip_mode+1)%3; g_flip = (g_flip_mode>0); }
static void cycle_dirblur() { g_dirblur_idx = (g_dirblur_idx+1)%3; g_dirblur = (g_dirblur_idx>0); }
static void cycle_hdr() { g_hdr_method = (g_hdr_method+1)%3; g_hdr = (g_hdr_method>0); }
static void cycle_zoom() { g_zoom_idx = (g_zoom_idx+1)%4; g_cropzoom = (g_zoom_idx>0); }
static void reset_all() {
    g_denoise = g_clahe = g_sr = g_lut_on = g_blur = g_sharp = false;
    g_bright = g_sat = g_gamma = g_vignette = g_grain = g_edge = false;
    g_wb = g_flip = g_lens = false;
    g_interp = g_bgblur = g_tempden = g_tempstab = g_chroma = false;
    g_dirblur = g_hdr = g_cropzoom = false;
    g_stab_dx = g_stab_dy = 0;
    g_temporal_has_prev = false;
    g_ai_denoise = g_ai_depth = g_ai_flow = g_ai_pose = false;
    g_ai_matting = g_ai_face = g_ai_hands = g_ai_gaze = false;
    g_ai_lowlight = g_ai_anime = g_ai_detect = false;
    g_matt_cached = false;
    g_autoframe = false; g_af_idx = 0; g_af_init = false; g_afp_init = false;
    g_flip_mode = 0; g_lut_preset = 0;
    g_dirblur_idx = 0; g_hdr_method = 0; g_zoom_idx = 0;
    if (g_d_lut) { cudaFree(g_d_lut); g_d_lut = nullptr; }
}

static void toggle_ai(bool& flag, const std::atomic<bool>& ready) {
    if (g_ai_warming.load() || !ready.load()) return; // warming / no model: ignore
    flag = !flag;
}

static void fire_button(int i) {
    if (i < 0 || i >= NBTN) return;
    BtnDef& b = g_btns[i];
    if (b.section == 3 && ai_button_state(b) < 2) return; // warming / no model: ignore
    if (b.special == 1) toggle_lut();
    else if (b.special == 2) toggle_flip();
    else if (b.special == 3) cycle_dirblur();
    else if (b.special == 4) cycle_hdr();
    else if (b.special == 5) cycle_zoom();
    else if (b.special == 6) {
        g_af_idx = (g_af_idx + 1) % 4;
        g_autoframe = (g_af_idx > 0);
        if (!g_autoframe) { g_af_init = false; g_afp_init = false; }
    }
    else *b.state = !*b.state;
}

static int hit_button(int mx, int my) {
    if (!g_hwnd) return -1;
    RECT rc; GetClientRect(g_hwnd, &rc);
    BtnRect rects[40];
    BtnRect secs[4];
    layout_buttons(rc.bottom, rects, secs);
    for (int i = 0; i < NBTN; i++) {
        if (!rect_visible(rects[i], rc.bottom)) continue;
        if (mx >= rects[i].x && mx < rects[i].x + rects[i].w &&
            my >= rects[i].y && my < rects[i].y + rects[i].h)
            return i;
    }
    return -1;
}

static int hit_tab(int mx, int my) {
    if (my < 5 || my >= TAB_H - 5) return -1;
    int tw = tab_tw();
    for (int t = 0; t < 4; t++) {
        int x = tab_tx(t);
        if (mx >= x && mx < x + tw) return t;
    }
    return -1;
}

LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_DESTROY:
        game_kill(); // release any injected mouse/keyboard holds
        if (g_tc_state == 1) tc_stop();
        if (g_tc_thread.joinable()) g_tc_thread.join();
        if (g_rec_state == 1) rec_stop();
        if (g_rec_thread.joinable()) g_rec_thread.join();
        au_stop();
        g_running = false; PostQuitMessage(0); return 0;
    case WM_ERASEBKGND: return 1;
    case WM_GETMINMAXINFO: {
        MINMAXINFO* mi = (MINMAXINFO*)l;
        mi->ptMinTrackSize.x = 900; mi->ptMinTrackSize.y = 600;
        return 0;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc_s = BeginPaint(h, &ps);
        RECT rc; GetClientRect(h, &rc);
        int W = rc.right, H = rc.bottom;
        HDC hdc = CreateCompatibleDC(hdc_s);
        HBITMAP hbm = CreateCompatibleBitmap(hdc_s, W > 0 ? W : 1, H > 0 ? H : 1);
        HBITMAP obm = (HBITMAP)SelectObject(hdc, hbm);
        render_frame(hdc, W, H);
        BitBlt(hdc_s, ps.rcPaint.left, ps.rcPaint.top,
               ps.rcPaint.right - ps.rcPaint.left, ps.rcPaint.bottom - ps.rcPaint.top,
               hdc, ps.rcPaint.left, ps.rcPaint.top, SRCCOPY);
        SelectObject(hdc, obm);
        DeleteObject(hbm); DeleteDC(hdc);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_TIMER:
        if (g_tc_state != 1 && g_tc_thread.joinable()) g_tc_thread.join();
        if (g_rec_state != 1 && g_rec_state != 2 && g_rec_thread.joinable())
            g_rec_thread.join();
        InvalidateRect(h, NULL, FALSE);
        // Headless injector self-check: scripted mouse/keys -> quit.
        if (g_gametest) {
            static DWORD gt0 = 0;
            if (!gt0) gt0 = GetTickCount();
            double gel = (GetTickCount() - gt0) / 1000.0;
            game_selftest_tick(gel);
            if (gel > 8.0) DestroyWindow(h);
        }
        // Headless self-test: record -> cut -> stop -> quit, no focus needed.
        if (g_selftest_sec > 0) {
            static DWORD st0 = 0;
            static bool st_go = false, st_cut = false;
            if (!st0) st0 = GetTickCount();
            double el = (GetTickCount() - st0) / 1000.0;
            if (el >= 2.0 && !st_go) { st_go = true; rec_start(); }
            if (el >= 4.0 && !st_cut) { st_cut = true; rec_cut(); }
            if (el >= g_selftest_sec && g_rec_state.load() == 1) rec_stop();
            int st = g_rec_state.load();
            if (((st == 3 || st == 4) && el > g_selftest_sec + 2.0) ||
                el > g_selftest_sec + 90.0)
                DestroyWindow(h);
        }
        return 0;
    case WM_SIZE:
        g_scroll = 0;
        InvalidateRect(h, NULL, FALSE);
        return 0;
    case WM_MOUSEWHEEL: {
        POINT pt = {LOWORD(l), HIWORD(l)};
        ScreenToClient(h, &pt);
        // wheel over the transcode log box scrolls the log, not the sidebar
        if (g_tab == 1 && tc_rc_logbox.right > tc_rc_logbox.left &&
            pt.x >= tc_rc_logbox.left && pt.x < tc_rc_logbox.right &&
            pt.y >= tc_rc_logbox.top && pt.y < tc_rc_logbox.bottom) {
            int d = GET_WHEEL_DELTA_WPARAM(w) / WHEEL_DELTA;
            std::lock_guard<std::mutex> lk(g_tc_log_mtx);
            int total = (int)g_tc_log.size();
            int maxstart = total > tc_log_vis ? total - tc_log_vis : 0;
            int pos = (g_tc_log_pos < 0) ? maxstart : g_tc_log_pos;
            pos -= d * 3;
            if (pos >= maxstart) { g_tc_log_pos = -1; } // snapped to bottom: follow
            else {
                if (pos < 0) pos = 0;
                g_tc_log_pos = pos;
            }
            InvalidateRect(h, NULL, FALSE);
            return 0;
        }
        RECT rc; GetClientRect(h, &rc);
        BtnRect rtmp[40]; BtnRect stmp[4];
        layout_buttons(rc.bottom, rtmp, stmp); // refresh g_content_h
        int d = GET_WHEEL_DELTA_WPARAM(w) / WHEEL_DELTA;
        g_scroll -= d * 36;
        int ms = max_scroll(rc.bottom);
        if (g_scroll < 0) g_scroll = 0;
        if (g_scroll > ms) g_scroll = ms;
        InvalidateRect(h, NULL, FALSE);
        return 0;
    }
    case WM_MOUSEMOVE: {
        int t = hit_tab(LOWORD(l), HIWORD(l));
        if (t != g_hover_tab) { g_hover_tab = t; InvalidateRect(h, NULL, FALSE); }
        if (t >= 0) return 0;
        RECT rc; GetClientRect(h, &rc);
        int gh = hit_gpu_hint(LOWORD(l), HIWORD(l), rc.right) ? 1 : 0;
        if (gh != g_hover_gpu) { g_hover_gpu = gh; InvalidateRect(h, NULL, FALSE); }
        if (gh) return 0;
        int nh = -1;
        if (g_tab == 1) {
            int id = tc_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) nh = 1000 + id;
        } else if (g_tab == 2) {
            int id = rec_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) nh = 2000 + id;
            else { int i = hit_button(LOWORD(l), HIWORD(l)); if (i >= 0) nh = i; }
        } else if (g_tab == 3) {
            int id = game_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) nh = 3000 + id;
        } else {
            int i = hit_button(LOWORD(l), HIWORD(l));
            if (i >= 0) nh = i;
        }
        if (nh != g_hover) { g_hover = nh; InvalidateRect(h, NULL, FALSE); }
        return 0;
    }
    case WM_LBUTTONDOWN: {
        int t = hit_tab(LOWORD(l), HIWORD(l));
        if (t >= 0) { if (t != g_tab) { g_tab = t; InvalidateRect(h, NULL, FALSE); } g_mic_list_open = false; return 0; }
        RECT rc; GetClientRect(h, &rc);
        if (hit_gpu_hint(LOWORD(l), HIWORD(l), rc.right)) { show_gpu_dialog(h); return 0; }
        int np = -1;
        if (g_tab == 1) {
            int id = tc_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) np = 1000 + id;
        } else if (g_tab == 2) {
            int id = rec_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) np = 2000 + id;
            else { int i = hit_button(LOWORD(l), HIWORD(l)); if (i >= 0) np = i; }
        } else if (g_tab == 3) {
            int id = game_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0) np = 3000 + id;
        } else {
            int i = hit_button(LOWORD(l), HIWORD(l));
            if (i >= 0) np = i;
        }
        g_pressed = np;
        if (np >= 0) InvalidateRect(h, NULL, FALSE);
        SetCapture(h);
        return 0;
    }
    case WM_LBUTTONUP: {
        RECT rc; GetClientRect(h, &rc);
        if (g_pressed >= 3000) {
            int id = game_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0 && 3000 + id == g_pressed) game_fire(id);
            g_pressed = -1;
            ReleaseCapture();
            InvalidateRect(h, NULL, FALSE);
            return 0;
        } else if (g_pressed >= 2000) {
            int id = rec_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0 && 2000 + id == g_pressed) rec_fire(id);
            g_pressed = -1;
            ReleaseCapture();
            InvalidateRect(h, NULL, FALSE);
            return 0;
        } else if (g_pressed >= 1000) {
            int id = tc_hittest(LOWORD(l), HIWORD(l), rc.right, rc.bottom);
            if (id > 0 && 1000 + id == g_pressed) tc_fire(id);
        } else if (g_tab == 0 || g_tab == 2) {
            int i = hit_button(LOWORD(l), HIWORD(l));
            if (g_pressed >= 0 && i == g_pressed) fire_button(i);
        }
        g_pressed = -1;
        ReleaseCapture();
        InvalidateRect(h, NULL, FALSE);
        return 0;
    }
    case WM_KEYDOWN:
        switch (w) {
        case VK_ESCAPE: DestroyWindow(h); break;
        case 'D': g_denoise = !g_denoise; break;
        case 'C': g_clahe = !g_clahe; break;
        case 'S': g_sr = !g_sr; break;
        case 'L': toggle_lut(); break;
        case 'B': g_blur = !g_blur; break;
        case 'N': g_sharp = !g_sharp; break;
        case '1': g_bright = !g_bright; break;
        case '2': g_sat = !g_sat; break;
        case '3': g_gamma = !g_gamma; break;
        case 'V': g_vignette = !g_vignette; break;
        case 'G': g_grain = !g_grain; break;
        case 'E': g_edge = !g_edge; break;
        case 'W': g_wb = !g_wb; break;
        case 'F': toggle_flip(); break;
        case 'Q': g_lens = !g_lens; break;
        case 'I': g_interp = !g_interp; break;
        case 'O': cycle_dirblur(); break;
        case 'M': g_bgblur = !g_bgblur; break;
        case 'U': g_tempden = !g_tempden; break;
        case 'Y': g_tempstab = !g_tempstab; break;
        case 'H': cycle_hdr(); break;
        case 'K': g_chroma = !g_chroma; break;
        case 'X': cycle_zoom(); break;
        case 'R': reset_all(); break;
        case '7': toggle_ai(g_ai_denoise, g_ai_denoise_ready); break;
        case '8': toggle_ai(g_ai_depth, g_ai_depth_ready); break;
        case '9': toggle_ai(g_ai_flow, g_ai_flow_ready); break;
        case '0': toggle_ai(g_ai_pose, g_ai_pose_ready); break;
        case 'A': toggle_ai(g_ai_matting, g_ai_matting_ready); break;
        case 'P': toggle_ai(g_ai_face, g_ai_face_ready); break;
        case 'J': toggle_ai(g_ai_hands, g_ai_hands_ready); break;
        case 'Z': toggle_ai(g_ai_gaze, g_ai_gaze_ready); break;
        case '5': toggle_ai(g_ai_lowlight, g_ai_lowlight_ready); break;
        case '6': toggle_ai(g_ai_anime, g_ai_anime_ready); break;
        case VK_F10: toggle_ai(g_ai_detect, g_ai_detect_ready); break;
        case VK_F7:
            g_af_idx = (g_af_idx + 1) % 4;
            g_autoframe = (g_af_idx > 0);
            if (!g_autoframe) { g_af_init = false; g_afp_init = false; }
            break;
        case 'T': if (g_tab == 1) { if (g_tc_state == 1) tc_stop(); else tc_start(); } break;
        case VK_F9: if (g_rec_state == 1) rec_stop(); else rec_start(); break;
        case VK_F5: screen_ensure(); g_src_mode = (g_src_mode + 1) % 3; break;
        case VK_F4: { // TEMP DIAG: full adapter/output topology in-app
            FILE* df = fopen("C:\\Users\\teknotek2025\\Desktop\\wildonion\\hanzo\\src\\HardLab\\Kagerou\\bin\\dx_dbg.log", "a");
            IDXGIFactory1* fac = nullptr;
            HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac);
            if (df) fprintf(df, "F4 factory=0x%lX\n", hr);
            if (SUCCEEDED(hr) && fac) {
                for (UINT fai = 0; fai < 4; fai++) {
                    IDXGIAdapter1* adx = nullptr;
                    if (fac->EnumAdapters1(fai, &adx) != S_OK || !adx) break;
                    DXGI_ADAPTER_DESC1 ddx = {};
                    adx->GetDesc1(&ddx);
                    if (df) fprintf(df, "F4 ad%u ven=%04X %S\n", fai, ddx.VendorId, ddx.Description);
                    for (UINT foi = 0; foi < 4; foi++) {
                        IDXGIOutput* ox = nullptr;
                        if (adx->EnumOutputs(foi, &ox) != S_OK || !ox) break;
                        DXGI_OUTPUT_DESC odx = {};
                        ox->GetDesc(&odx);
                        if (df) fprintf(df, "F4   out%u %dx%d @(%d,%d) att=%d\n", foi,
                            odx.DesktopCoordinates.right - odx.DesktopCoordinates.left,
                            odx.DesktopCoordinates.bottom - odx.DesktopCoordinates.top,
                            odx.DesktopCoordinates.left, odx.DesktopCoordinates.top,
                            (int)odx.AttachedToDesktop);
                        ox->Release();
                    }
                    adx->Release();
                }
                IDXGIAdapter1* ad = nullptr;
                if (fac->EnumAdapters1(0, &ad) == S_OK && ad) {
                    IDXGIOutput* out = nullptr;
                    if (ad->EnumOutputs(0, &out) == S_OK && out) {
                        static const D3D_FEATURE_LEVEL fls[] = {
                            D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
                            D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
                        ID3D11Device* dev = nullptr;
                        ID3D11DeviceContext* ctx = nullptr;
                        D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_1_0_CORE;
                        hr = D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                               fls, 4, D3D11_SDK_VERSION, &dev, &got, &ctx);
                        if (df) fprintf(df, "F4 device=0x%lX fl=0x%X\n", hr, got);
                        if (SUCCEEDED(hr) && dev) {
                            IDXGIOutput1* out1 = nullptr;
                            hr = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
                            if (df) fprintf(df, "F4 out1=0x%lX\n", hr);
                            if (SUCCEEDED(hr) && out1) {
                                IDXGIOutputDuplication* dup = nullptr;
                                hr = out1->DuplicateOutput(dev, &dup);
                                if (df) fprintf(df, "F4 dupe=0x%lX\n", hr);
                                if (dup) { dup->Release(); }
                                out1->Release();
                            }
                            ctx->Release();
                            dev->Release();
                        }
                        out->Release();
                    }
                    ad->Release();
                }
                fac->Release();
            }
            if (df) fclose(df);
            break;
        }
        case VK_F8: rec_cut(); break;
        case '4': rec_mark(); break;
        case VK_F6: g_tab = (g_tab + 1) % 4; break;
        case VK_F12: game_on_key(w); break;
        }
        InvalidateRect(h, NULL, FALSE);
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest") && i + 1 < argc)
            g_selftest_sec = atof(argv[++i]);
        if (!strcmp(argv[i], "--gametest"))
            g_gametest = true;
    }
    printf("=== Kagerou Virtual Camera ===\n");
    printf("Non-AI: D=denoise C=CLAHE S=super-res L=LUT B=blur N=sharp\n");
    printf("        1=B/C 2=sat 3=gamma V=vignette G=grain E=edge W=WB F=flip Q=lens\n");
    printf("AI:     7=AI-denoise 8=depth 9=flow 0=pose\n");
    printf("        A=matting P=face J=hands Z=gaze 5=lowlight 6=anime\n");
    printf("REC:    F9=start/stop recording  F8=cut last ~4s  4=mark (RECORD tab)\n");
    printf("        F5=cycle source CAM/SCREEN/TUTORIAL\n");
    printf("        voice commentary records from the default mic (MIC chip toggles)\n");
    printf("        gestures (Hands AI on): palm=cut thumb=mark peace=pause fist(2s)=stop\n");
    printf("        R=reset all  ESC=quit\n\n"); fflush(stdout);

    SetProcessDPIAware();
    cudaSetDevice(0);
    cudaStreamCreate(&g_stream);

    size_t sz = (size_t)g_cap_w * g_cap_h * 3;
    size_t sr_sz = (size_t)g_cap_w * 2 * g_cap_h * 2 * 3; // camera SR only
    g_cam_buf = (uint8_t*)malloc(sz);
    g_display_out = (uint8_t*)malloc(FEED_MAX_RGB);
    memset(g_display_out, 30, FEED_MAX_RGB);
    cudaMalloc(&g_d_rgb, FEED_MAX_RGB);
    cudaMalloc(&g_d_tmp, FEED_MAX_RGB);
    cudaMalloc(&g_d_ai_s0, (size_t)640 * 480 * 3);
    cudaMalloc(&g_d_ai_s1, (size_t)640 * 480 * 3);
    cudaMalloc(&g_d_camdet, sz);
    cudaMalloc(&g_d_sr_out, sr_sz);
    cudaMalloc(&g_d_temporal_prev, FEED_MAX_RGB);
    cudaMalloc(&g_d_stage, FEED_MAX_RGB);
    cudaMalloc(&g_d_zoom, FEED_MAX_RGB);
    cudaMalloc(&g_d_motion_accum, 3 * sizeof(int));
    cudaMemset(g_d_motion_accum, 0, 3 * sizeof(int));

    if (g_vcam.Open()) { g_vcam_active = true; printf("[VCAM] Shared memory OK\n"); }
    else printf("[VCAM] Not available\n");

    // NV12 staging sized for the max feed frame.
    size_t nv12_max = FEED_MAX_RGB / 2;
    cudaMalloc(&g_d_nv12, nv12_max);
    g_h_nv12 = (uint8_t*)malloc(nv12_max);

    // Recorder staging at feed-max size (RGB + NV12).
    cudaMalloc(&g_rec_rgb, FEED_MAX_RGB);
    cudaMalloc(&g_rec_nv12, FEED_MAX_RGB / 2);

    // Create the window FIRST so it appears instantly. AI models (esp.
    // depth/TRT) can take minutes -- load them in the background.
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "KagerouVCam";
    RegisterClassExA(&wc);

    g_hwnd = CreateWindowExA(0, "KagerouVCam", "Kagerou Virtual Camera",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        1120, 720, NULL, NULL, wc.hInstance, NULL);

    g_font_ui = CreateFontA(18,0,0,0,FW_MEDIUM,0,0,0,DEFAULT_CHARSET,0,0,0,0,"Segoe UI");
    g_font_emoji = CreateFontA(18,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,0,0,"Segoe UI Emoji");
    g_font_st = CreateFontA(15,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,0,0,"Consolas");
    g_font_tab = CreateFontA(14,0,0,0,FW_MEDIUM,0,0,0,DEFAULT_CHARSET,0,0,0,0,"Segoe UI");

    ShowWindow(g_hwnd, SW_SHOW);
    UpdateWindow(g_hwnd);

    SetTimer(g_hwnd, 1, 33, nullptr);

    mic_enum(); // cache mic list for the MIC chip

    printf("[WIN] OK\n[LOOP] Running...\n\n"); fflush(stdout);

    std::thread cam(cam_thread_func);
    std::thread gpu(gpu_thread_func);
    std::thread ai_init([]() {
        printf("[AI] Loading models in background...\n"); fflush(stdout);
        init_ai_models();
        printf("[AI] Background load done.\n"); fflush(stdout);
    });

    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    g_running = false;
    if (g_tc_state == 1) tc_stop();
    if (g_rec_state == 1) rec_stop();
    if (cam.joinable()) cam.join();
    if (gpu.joinable()) gpu.join();
    if (ai_init.joinable()) ai_init.join();
    if (g_tc_thread.joinable()) g_tc_thread.join();
    if (g_rec_thread.joinable()) g_rec_thread.join();
    // NOTE: no encoder destroy here -- it must run on the GPU thread (done
    // via the flush handshake above); process teardown reclaims the rest.
    if (g_rec_file) { fclose(g_rec_file); g_rec_file = nullptr; }

    KillTimer(g_hwnd, 1);
    g_vcam.Close();
    cudaStreamDestroy(g_stream);
    if (g_d_rgb) cudaFree(g_d_rgb);
    if (g_d_tmp) cudaFree(g_d_tmp);
    if (g_d_ai_s0) cudaFree(g_d_ai_s0);
    if (g_d_ai_s1) cudaFree(g_d_ai_s1);
    if (g_d_camdet) cudaFree(g_d_camdet);
    if (g_d_sr_out) cudaFree(g_d_sr_out);
    if (g_d_temporal_prev) cudaFree(g_d_temporal_prev);
    if (g_d_stage) cudaFree(g_d_stage);
    if (g_d_zoom) cudaFree(g_d_zoom);
    if (g_d_motion_accum) cudaFree(g_d_motion_accum);
    if (g_d_lut) cudaFree(g_d_lut);
    if (g_d_nv12) cudaFree(g_d_nv12);
    if (g_h_nv12) free(g_h_nv12);
    if (g_rec_rgb) cudaFree(g_rec_rgb);
    if (g_rec_nv12) cudaFree(g_rec_nv12);
    if (g_d_ai_depth_buf) cudaFree(g_d_ai_depth_buf);
    if (g_d_ai_vis) cudaFree(g_d_ai_vis);
    if (g_d_ai_matt) cudaFree(g_d_ai_matt);
    if (g_d_flow_prev) cudaFree(g_d_flow_prev);
    if (g_d_flow_curr) cudaFree(g_d_flow_curr);
    if (g_d_flow_buf) cudaFree(g_d_flow_buf);
    for (int i = 0; i < DENOISE_RING_SIZE; i++)
        if (g_d_denoise_ring[i]) cudaFree(g_d_denoise_ring[i]);
    if (g_afp_cam) { free(g_afp_cam); g_afp_cam = nullptr; g_afp_cam_sz = 0; }
    g_screen_run = false;
    if (g_screen_thread.joinable()) g_screen_thread.join();
    screen_helper_stop();
    if (g_screen_buf) free(g_screen_buf);
    if (g_screen_nat) free(g_screen_nat);
    if (g_cam_buf) free(g_cam_buf);
    if (g_display_out) free(g_display_out);
    if (g_font_ui) DeleteObject(g_font_ui);
    if (g_font_emoji) DeleteObject(g_font_emoji);
    if (g_font_st) DeleteObject(g_font_st);
    if (g_font_tab) DeleteObject(g_font_tab);
    return 0;
}
