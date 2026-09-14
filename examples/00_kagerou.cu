// Kagerou SDK -- Unified CLI
// Single executable: decode -> GPU filters -> encode -> output
//
// Usage:
//   kagerou.exe <input> [options]
//   kagerou.exe benchmark
//   kagerou.exe test
//
// Transcode options:
//   -o <path>          Output folder (default: output/)
//   --denoise          Enable bilateral denoise filter
//   --scale WxH        Enable scale filter (e.g. --scale 1280x720)
//   --super-res        Enable 2x super resolution filter
//   --frame-interp     Enable frame interpolation (2x FPS)
//   --clahe            Enable CLAHE contrast enhancement
//   --lut <preset>     Enable LUT color grading (warm|cool|cinematic|vintage|contrast|desat)
//   --all              Enable all filters
//   --fps <n>          Output FPS (default: 30)
//   --bitrate <kbps>   Output bitrate in kbps (default: auto ~0.07 bpp)
//   --codec <h264|h265> Output codec (default: h264)
//   -v                 Verbose output
//   -h                 Show help
//
// Batch mode:
//   kagerou.exe batch <input>   Transcode at multiple quality configs

#include "../src/pipeline.cu"
#include "../include/kagerou/fileio.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>

static void print_usage() {
    printf("Kagerou GPU Video Transcoding SDK\n\n");
    printf("Usage:\n");
    printf("  kagerou.exe <input> [options]   Transcode video\n");
    printf("  kagerou.exe benchmark           Run filter benchmarks\n");
    printf("  kagerou.exe test                Run unit tests\n");
    printf("  kagerou.exe batch <input>       Batch transcode (multiple quality configs)\n");
    printf("\nGeneral options:\n");
    printf("  -o <path>              Output folder (default: output/)\n");
    printf("  --fps <n>              Output FPS (default: 30)\n");
    printf("  --bitrate <kbps>       Output bitrate (default: auto ~0.07 bpp)\n");
    printf("  --codec <h264|h265>    Output codec (default: h264)\n");
    printf("  -v                     Verbose output\n");
    printf("  -h                     Show help\n");
    printf("\nGPU Filters (all run on GPU, zero-copy):\n");
    printf("  --denoise              Bilateral denoise -- reduces noise, smooths textures\n");
    printf("                           eg: kagerou.exe in.mp4 --denoise\n");
    printf("  --scale WxH            Resize to target resolution\n");
    printf("                           eg: kagerou.exe in.mp4 --scale 1280x720\n");
    printf("  --super-res            2x super resolution (sharpened bicubic upscale)\n");
    printf("                           eg: kagerou.exe in.mp4 --super-res\n");
    printf("  --frame-interp         Frame interpolation -- doubles FPS via blending\n");
    printf("                           eg: kagerou.exe in.mp4 --frame-interp --fps 60\n");
    printf("  --clahe                Adaptive histogram equalization (contrast boost)\n");
    printf("                           eg: kagerou.exe in.mp4 --clahe\n");
    printf("  --lut <preset>         LUT color grading (warm|cool|cinematic|vintage|contrast|desat)\n");
    printf("                           eg: kagerou.exe in.mp4 --lut cinematic\n");
    printf("  --blur [sigma]         Gaussian blur (sigma=0.5..20, default: 2.0)\n");
    printf("                           eg: kagerou.exe in.mp4 --blur 4.0\n");
    printf("  --sharpen [str]        Unsharp mask sharpen (str=0.1..5.0, default: 1.0)\n");
    printf("                           eg: kagerou.exe in.mp4 --sharpen 1.5\n");
    printf("  --brightness <n>       Brightness offset (-255..255, default: 0)\n");
    printf("                           eg: kagerou.exe in.mp4 --brightness 30\n");
    printf("  --contrast <n>         Contrast multiplier (0.1..3.0, default: 1.0)\n");
    printf("                           eg: kagerou.exe in.mp4 --contrast 1.4\n");
    printf("  --saturation [n]       Color saturation (0=gray, 1=normal, default: 1.4)\n");
    printf("                           eg: kagerou.exe in.mp4 --saturation 1.6\n");
    printf("  --gamma [n]            Gamma (<1=brighten, >1=darken, default: 0.8)\n");
    printf("                           eg: kagerou.exe in.mp4 --gamma 0.7\n");
    printf("  --vignette [str]       Lens darkening (0=none, 1=heavy, default: 0.6)\n");
    printf("                           eg: kagerou.exe in.mp4 --vignette 0.4\n");
    printf("  --grain [amt]          Film grain noise (0..100, default: 25)\n");
    printf("                           eg: kagerou.exe in.mp4 --grain 30\n");
    printf("  --edge-detect          Sobel edge detection ( outputs edge magnitude )\n");
    printf("                           eg: kagerou.exe in.mp4 --edge-detect\n");
    printf("  --white-balance        Warm/cool color temperature shift\n");
    printf("                           eg: kagerou.exe in.mp4 --white-balance\n");
    printf("  --lens-distort [k]     Barrel/pincushion distortion (k<0=barrel, default: -0.3)\n");
    printf("                           eg: kagerou.exe in.mp4 --lens-distort -0.5\n");
    printf("  --flip <h|v>           Flip horizontal or vertical\n");
    printf("                           eg: kagerou.exe in.mp4 --flip h\n");
    printf("  --dir-blur <ang> <len> Directional/motion blur (angle in deg, length in px)\n");
    printf("                           eg: kagerou.exe in.mp4 --dir-blur 45 20\n");
    printf("  --crop WxH+X+Y         Crop region from frame\n");
    printf("                           eg: kagerou.exe in.mp4 --crop 640x480+100+50\n");
    printf("  --chroma-key           Green screen removal (green bg -> black bg)\n");
    printf("                           eg: kagerou.exe in.mp4 --chroma-key\n");
    printf("  --chroma-sat <n>       Chroma key: min saturation (default: 0.3, lower=catches more)\n");
    printf("                           eg: kagerou.exe in.mp4 --chroma-key --chroma-sat 0.1\n");
    printf("  --chroma-hue <min:max> Chroma key: hue range in degrees (default: 60:160)\n");
    printf("                           eg: kagerou.exe in.mp4 --chroma-key --chroma-hue 50:170\n");
    printf("  --chroma-spill <n>     Chroma key: green spill suppression (default: 0.5)\n");
    printf("  --chroma-bg <r> <g> <b>  Chroma key: replacement background color (default: 0 0 0)\n");
    printf("  --bg-blur [str]        Background blur -- blurs everything outside center\n");
    printf("                           eg: kagerou.exe in.mp4 --bg-blur 12\n");
    printf("  --temporal-denoise [s] Inter-frame temporal denoise (s=0..1, default: 0.25)\n");
    printf("                           eg: kagerou.exe in.mp4 --temporal-denoise 0.3\n");
    printf("  --temporal-stab        Temporal stabilization (optical-flow warp)\n");
    printf("                           eg: kagerou.exe in.mp4 --temporal-stab\n");
    printf("  --hdr [method]         HDR tone map (0=Reinhard, 1=ACES, default: 0)\n");
    printf("                           eg: kagerou.exe in.mp4 --hdr 1\n");
    printf("  --peak-nits <n>        HDR peak brightness (default: 100 for SDR)\n");
    printf("                           eg: kagerou.exe in.mp4 --hdr 1 --peak-nits 1000\n");
    printf("  --all                  Enable common creative filters bundle\n");
    printf("\nAI Filters (require ONNX Runtime GPU -- build with onnx flag):\n");
    printf("  --ai-denoise          AI video denoise (FastDVDNet) -- needs 5-frame window\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-denoise\n");
    printf("  --ai-pose             AI pose estimation (MediaPipe 33 landmarks)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-pose\n");
    printf("  --ai-flow             AI optical flow (RAFT) -- motion visualization\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-flow\n");
    printf("  --ai-depth            AI depth estimation (Depth Anything V2)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-depth\n");
    printf("  --ai-matting          AI person matting (RVM) -- background-blur composite\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-matting\n");
    printf("  --ai-face             AI face detection (YuNet) -- draws boxes\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-face\n");
    printf("  --ai-hands            AI hand tracking (BlazePalm + landmarks) -- draws joints\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-hands\n");
    printf("  --ai-gaze             AI gaze (face + iris) -- draws iris + direction\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-gaze\n");
    printf("  --ai-lowlight         AI low-light enhance (Zero-DCE)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-lowlight\n");
    printf("  --ai-anime            AI anime stylization (AnimeGANv3 Hayao)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-anime\n");
    printf("  --ai-detect           AI object detection (YOLOv8n, 80 COCO classes)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-detect\n");
    printf("  --ai-autoframe        AI auto framing (face-tracked crop follows you)\n");
    printf("                           eg: kagerou.exe in.mp4 --ai-autoframe\n");
    printf("\nBatch examples:\n");
    printf("  kagerou.exe video.mp4\n");
    printf("  kagerou.exe video.mp4 --denoise --lut cinematic\n");
    printf("  kagerou.exe video.mp4 --clahe --scale 1920x1080\n");
    printf("  kagerou.exe video.mp4 --codec h265 --bitrate 10000\n");
    printf("  kagerou.exe video.mp4 --temporal-denoise 0.3 --sharpen 1.2\n");
    printf("  kagerou.exe video.mp4 --bg-blur 12 --chroma-key --saturation 1.3\n");
    printf("  kagerou.exe video.mp4 --crop 640x480+0+0 --flip h --hdr 1\n");
}

// ---- Benchmark mode ----

template<typename Func>
double benchmark_filter(Func fn, int iterations, const char* name) {
    for (int i = 0; i < 5; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_frame = ms / iterations;
    printf("  %-35s %8.3f ms/frame  (%7.1f fps)\n", name, per_frame, 1000.0 / per_frame);
    return per_frame;
}

int run_benchmark() {
    printf("=== Kagerou GPU Benchmark ===\n\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const uint32_t W = 1920, H = 1080;
    const int CHANNELS = 3;
    const int ITERS = 50;

    uint8_t *d_src, *d_dst;
    size_t src_size = W * H * CHANNELS;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_src, src_size));
    std::vector<uint8_t> h_src(src_size);
    for (size_t i = 0; i < src_size; ++i) h_src[i] = (uint8_t)(i * 7 + 13);
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), src_size, cudaMemcpyHostToDevice));

    printf("\nResolution: %ux%u, %d channels\n", W, H, CHANNELS);
    printf("Iterations: %d\n", ITERS);

    printf("\n[Color Conversion]\n");
    {
        size_t nv12_size = W * H + W * (H / 2);
        uint8_t* d_nv12;
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_nv12, nv12_size));
        benchmark_filter([&]() { kagerou::filters::nv12_to_rgb(d_nv12, d_src, W, H); }, ITERS, "NV12 -> RGB (1080p)");
        benchmark_filter([&]() { kagerou::filters::rgb_to_nv12(d_src, d_nv12, W, H); }, ITERS, "RGB -> NV12 (1080p)");
        cudaFree(d_nv12);
    }

    printf("\n[Resize]\n");
    {
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, (W/2)*(H/2)*CHANNELS));
        benchmark_filter([&]() { kagerou::filters::resize_bilinear(d_src, d_dst, W, H, W/2, H/2, CHANNELS); }, ITERS, "Bilinear 1080p -> 540p");
        benchmark_filter([&]() { kagerou::filters::resize_bicubic(d_src, d_dst, W, H, W/2, H/2, CHANNELS); }, ITERS, "Bicubic 1080p -> 540p");
        cudaFree(d_dst);
    }

    printf("\n[Denoise]\n");
    {
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, src_size));
        benchmark_filter([&]() { kagerou::filters::denoise_bilateral(d_src, d_dst, W, H, CHANNELS, 15.0f, 25.0f, 5); }, ITERS, "Bilateral 5x5 1080p");
        cudaFree(d_dst);
    }

    printf("\n[Super Resolution]\n");
    {
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, (W*2)*(H*2)*CHANNELS));
        benchmark_filter([&]() { kagerou::filters::super_res_2x(d_src, d_dst, W, H, CHANNELS, 0.5f); }, ITERS, "2x SR 1080p -> 4K");
        cudaFree(d_dst);
    }

    printf("\n[Frame Interpolation]\n");
    {
        uint8_t* d_frame_b;
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_frame_b, src_size));
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, src_size));
        benchmark_filter([&]() { kagerou::filters::frame_blend(d_src, d_frame_b, d_dst, W, H, CHANNELS, 0.5f); }, ITERS, "Alpha blend 1080p");
        cudaFree(d_frame_b);
        cudaFree(d_dst);
    }

    printf("\n[Creative Filters]\n");
    {
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, src_size));
        benchmark_filter([&]() { kagerou::filters::gaussian_blur(d_src, d_dst, W, H, CHANNELS, 2.0f); }, ITERS, "Gaussian blur 1080p");
        benchmark_filter([&]() { kagerou::filters::sharpen_rgb(d_src, d_dst, W, H, 1.0f); }, ITERS, "Sharpen 1080p");
        benchmark_filter([&]() { kagerou::filters::brightness_contrast(d_src, d_dst, W, H, 20.0f, 1.2f); }, ITERS, "Brightness/Contrast 1080p");
        benchmark_filter([&]() { kagerou::filters::saturation_rgb(d_src, d_dst, W, H, 1.4f); }, ITERS, "Saturation 1080p");
        benchmark_filter([&]() { kagerou::filters::gamma_rgb(d_src, d_dst, W, H, 0.8f); }, ITERS, "Gamma 1080p");
        benchmark_filter([&]() { kagerou::filters::vignette_rgb(d_src, d_dst, W, H, 0.6f); }, ITERS, "Vignette 1080p");
        benchmark_filter([&]() { kagerou::filters::film_grain_rgb(d_src, W, H, 25.0f, 42); }, ITERS, "Film Grain 1080p");
        benchmark_filter([&]() { kagerou::filters::edge_detect_rgb(d_src, d_dst, W, H); }, ITERS, "Edge Detect 1080p");
        benchmark_filter([&]() { kagerou::filters::white_balance_rgb(d_src, d_dst, W, H, 15.0f, 5.0f); }, ITERS, "White Balance 1080p");
        benchmark_filter([&]() { kagerou::filters::lens_distortion(d_src, d_dst, W, H, CHANNELS, -0.3f); }, ITERS, "Lens Distortion 1080p");
        benchmark_filter([&]() { kagerou::filters::directional_blur(d_src, d_dst, W, H, CHANNELS, 45.0f, 10); }, ITERS, "Directional Blur 1080p");
        benchmark_filter([&]() { kagerou::filters::flip_horizontal_gpu(d_src, d_dst, W, H); }, ITERS, "Flip Horizontal 1080p");
        benchmark_filter([&]() { kagerou::filters::flip_vertical_gpu(d_src, d_dst, W, H); }, ITERS, "Flip Vertical 1080p");
        cudaFree(d_dst);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}

// ---- Batch mode ----

struct BatchRun {
    const char* name;
    uint32_t bitrate;
    uint32_t fps;
    bool denoise;
    bool scale;
    uint32_t scale_w;
    uint32_t scale_h;
};

int run_batch(const char* input_path) {
    printf("=== Kagerou Batch Transcode ===\n\n");
    if (!input_path) { fprintf(stderr, "error: batch mode requires an input file\n"); return 1; }

    BatchRun runs[] = {
        { "1080p-5Mbps-nofilter",  5000, 30, false, false, 0,    0    },
        { "1080p-8Mbps-denoise",   8000, 30, true,  false, 0,    0    },
        { "720p-3Mbps-scale",      3000, 30, false, true,  1280, 720  },
        { "720p-5Mbps-both",       5000, 30, true,  true,  1280, 720  },
    };
    int num_runs = 4;

    for (int r = 0; r < num_runs; ++r) {
        printf("--- Run %d: %s ---\n", r, runs[r].name);

        kagerou::fileio::VideoFile video;
        kagerou::Error e = video.load(input_path);
        if (e != kagerou::Error::kOk) {
            fprintf(stderr, "error: cannot open input: %s\n", kagerou::error_string(e));
            return 1;
        }

        kagerou::PipelineConfig cfg;
        cfg.verbose = false;
        cfg.decoder.codec = kagerou::VideoCodec::kH264;
        cfg.decoder.max_width  = 7680;
        cfg.decoder.max_height = 4320;
        cfg.encoder.codec = kagerou::VideoCodec::kH264;
        cfg.encoder.fps    = runs[r].fps;
        cfg.encoder.bitrate_kbps = runs[r].bitrate;
        cfg.encoder.rc = kagerou::RateControl::kCQP; // historical behavior (CQP ignores bitrate)
        cfg.encoder.gop_size = runs[r].fps;
        cfg.encoder.width  = 0;
        cfg.encoder.height = 0;
        cfg.denoise.enabled = runs[r].denoise;
        cfg.scale.enabled   = runs[r].scale;
        cfg.scale.target_width  = runs[r].scale_w;
        cfg.scale.target_height = runs[r].scale_h;

        kagerou::Pipeline pipeline;
        e = pipeline.init(cfg);
        if (e != kagerou::Error::kOk) {
            fprintf(stderr, "  init failed: %s\n", kagerou::error_string(e));
            continue;
        }

        std::vector<uint8_t> output_data;
        int frame_count = 0;
        for (const auto& nalu : video.nalus) {
            std::vector<uint8_t> encoded;
            e = pipeline.process_frame(nalu.data, nalu.size, encoded);
            if (e == kagerou::Error::kOk) {
                output_data.insert(output_data.end(), encoded.begin(), encoded.end());
                frame_count++;
            }
        }

        std::vector<uint8_t> flush_data;
        pipeline.flush(flush_data);
        output_data.insert(output_data.end(), flush_data.begin(), flush_data.end());

        if (!output_data.empty()) {
            std::string base = kagerou::fileio::get_basename(input_path);
            std::string out_path = std::string("output_") + runs[r].name + ".h264";
            kagerou::fileio::write_file(out_path.c_str(), output_data);
            printf("  %d frames -> %s\n", frame_count, out_path.c_str());
        }

        pipeline.print_stats();
        pipeline.destroy();
        printf("\n");
    }
    return 0;
}

// ---- Transcode mode ----

int run_transcode(int argc, char** argv) {
    const char* input_path  = nullptr;
    const char* output_dir  = "output";
    bool denoise = false, scale_enabled = false, super_res = false;
    bool frame_interp = false, verbose = false, clahe_enabled = false;
    bool lut_enabled = false;
    bool blur_enabled = false, sharpen_enabled = false;
    bool bright_enabled = false, sat_enabled = false, gamma_enabled = false;
    bool vignette_enabled = false, grain_enabled = false, edge_enabled = false;
    bool wb_enabled = false, lens_enabled = false, flip_enabled = false;
    bool dirblur_enabled = false;
    bool crop_enabled = false, chromakey_enabled = false, bgblur_enabled = false;
    bool temporal_denoise_enabled = false, temporal_stab_enabled = false;
    bool hdr_enabled = false;
    bool ai_denoise_enabled = false;
    bool ai_depth_enabled = false, ai_flow_enabled = false;
    bool ai_pose_enabled = false;
    bool ai_matting_enabled = false, ai_face_enabled = false;
    bool ai_hands_enabled = false, ai_gaze_enabled = false;
    bool ai_lowlight_enabled = false, ai_anime_enabled = false;
    bool ai_detect_enabled = false;
    bool ai_autoframe_enabled = false;
    const char* ai_denoise_model = "models/denoise/fastdvdnet.onnx";
    const char* ai_depth_model = "models/depth/depth_anything_v2.onnx";
    const char* ai_flow_model = "models/flow/raft_small.onnx";
    const char* ai_pose_model = "models/pose/pose_landmark.onnx";
    const char* ai_matting_model = "models/matting/rvm_mobilenetv3.onnx";
    const char* ai_face_model = "models/face/yunet_2023mar.onnx";
    const char* ai_palm_model = "models/hands/palm_detection_lite.onnx";
    const char* ai_handlm_model = "models/hands/hand_landmark_lite.onnx";
    const char* ai_iris_model = "models/gaze/iris_landmark.onnx";
    const char* ai_lowlight_model = "models/lowlight/zero_dce.onnx";
    const char* ai_anime_model = "models/style/animeganv3_hayao.onnx";
    const char* ai_detect_model = "models/detect/yolov8n.onnx";
    uint32_t crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
    float key_h_min = 60.0f, key_h_max = 160.0f, key_sat_min = 0.3f, key_val_min = 0.2f;
    float key_spill = 0.5f;
    int key_bg_r = 0, key_bg_g = 0, key_bg_b = 0;
    float bg_blur_radius = 0.3f, bg_blur_str = 8.0f;
    float temporal_str = 0.25f;
    int hdr_method = 0;
    float hdr_peak_nits = 100.0f;
    uint32_t scale_w = 0, scale_h = 0;
    uint32_t fps = 30, bitrate = 0;
    bool fps_set_by_user = false;
    kagerou::VideoCodec codec = kagerou::VideoCodec::kH264;
    kagerou::LUTPreset lut_preset = kagerou::LUTPreset::kNone;
    float blur_sigma = 2.0f, sharpen_str = 1.0f;
    float brightness = 0.0f, contrast = 1.0f, saturation_f = 1.4f;
    float gamma_f = 0.8f, vignette_f = 0.6f, grain_f = 25.0f;
    float lens_k1 = -0.3f, dirblur_angle = 0.0f;
    int dirblur_len = 10;
    int flip_mode = 1; // 1=h, 2=v

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(); return 0; }
        else if (arg == "-v") verbose = true;
        else if (arg == "-o" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) { fps = (uint32_t)atoi(argv[++i]); fps_set_by_user = true; }
        else if (arg == "--bitrate" && i + 1 < argc) bitrate = (uint32_t)atoi(argv[++i]);
        else if (arg == "--codec" && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "h265") == 0 || strcmp(argv[i], "hevc") == 0)
                codec = kagerou::VideoCodec::kH265;
            else
                codec = kagerou::VideoCodec::kH264;
        }
        else if (arg == "--denoise") denoise = true;
        else if (arg == "--super-res") super_res = true;
        else if (arg == "--frame-interp") frame_interp = true;
        else if (arg == "--clahe") clahe_enabled = true;
        else if (arg == "--blur") { blur_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') blur_sigma = (float)atof(argv[++i]); }
        else if (arg == "--sharpen") { sharpen_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') sharpen_str = (float)atof(argv[++i]); }
        else if (arg == "--brightness") { bright_enabled = true; brightness = (float)atof(argv[++i]); }
        else if (arg == "--contrast") { bright_enabled = true; contrast = (float)atof(argv[++i]); }
        else if (arg == "--saturation") { sat_enabled = true; saturation_f = (float)atof(argv[++i]); }
        else if (arg == "--gamma") { gamma_enabled = true; gamma_f = (float)atof(argv[++i]); }
        else if (arg == "--vignette") { vignette_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') vignette_f = (float)atof(argv[++i]); }
        else if (arg == "--grain") { grain_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') grain_f = (float)atof(argv[++i]); }
        else if (arg == "--edge-detect") edge_enabled = true;
        else if (arg == "--white-balance") wb_enabled = true;
        else if (arg == "--lens-distort") { lens_enabled = true; if (i+1 < argc && (argv[i+1][0] != '-' || argv[i+1][1] == '.' || argv[i+1][1] >= '0')) lens_k1 = (float)atof(argv[++i]); }
        else if (arg == "--flip") {
            flip_enabled = true;
            if (i+1 < argc) {
                i++;
                if (argv[i][0] == 'v' || argv[i][0] == 'V') flip_mode = 2;
                else flip_mode = 1;
            }
        }
        else if (arg == "--dir-blur") { dirblur_enabled = true; if (i+2 < argc) { dirblur_angle = (float)atof(argv[++i]); dirblur_len = atoi(argv[++i]); } }
        else if (arg == "--crop") { crop_enabled = true; if (i+1 < argc) { sscanf(argv[++i], "%ux%u+%u+%u", &crop_w, &crop_h, &crop_x, &crop_y); } }
        else if (arg == "--chroma-key") { chromakey_enabled = true; }
        else if (arg == "--chroma-sat" && i + 1 < argc) { key_sat_min = (float)atof(argv[++i]); }
        else if (arg == "--chroma-hue" && i + 1 < argc) {
            // format: min:max  e.g. --chroma-hue 50:170
            float hmin = 60.0f, hmax = 160.0f;
            if (sscanf(argv[++i], "%f:%f", &hmin, &hmax) == 2) { key_h_min = hmin; key_h_max = hmax; }
        }
        else if (arg == "--chroma-spill" && i + 1 < argc) { key_spill = (float)atof(argv[++i]); }
        else if (arg == "--chroma-bg" && i + 3 < argc) { key_bg_r = atoi(argv[++i]); key_bg_g = atoi(argv[++i]); key_bg_b = atoi(argv[++i]); }
        else if (arg == "--bg-blur") { bgblur_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') bg_blur_str = (float)atof(argv[++i]); }
        else if (arg == "--temporal-denoise") { temporal_denoise_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') temporal_str = (float)atof(argv[++i]); }
        else if (arg == "--temporal-stab") { temporal_stab_enabled = true; }
        else if (arg == "--hdr") { hdr_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') hdr_method = atoi(argv[++i]); }
        else if (arg == "--peak-nits" && i + 1 < argc) { hdr_peak_nits = (float)atof(argv[++i]); }
        else if (arg == "--ai-denoise") { ai_denoise_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_denoise_model = argv[++i]; }
        else if (arg == "--ai-depth") { ai_depth_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_depth_model = argv[++i]; }
        else if (arg == "--ai-flow") { ai_flow_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_flow_model = argv[++i]; }
        else if (arg == "--ai-pose") { ai_pose_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_pose_model = argv[++i]; }
        else if (arg == "--ai-matting") { ai_matting_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_matting_model = argv[++i]; }
        else if (arg == "--ai-face") { ai_face_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_face_model = argv[++i]; }
        else if (arg == "--ai-hands") { ai_hands_enabled = true; }
        else if (arg == "--ai-palm") { ai_hands_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_palm_model = argv[++i]; }
        else if (arg == "--ai-handlm") { ai_hands_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_handlm_model = argv[++i]; }
        else if (arg == "--ai-gaze") { ai_gaze_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_iris_model = argv[++i]; }
        else if (arg == "--ai-lowlight") { ai_lowlight_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_lowlight_model = argv[++i]; }
        else if (arg == "--ai-anime") { ai_anime_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_anime_model = argv[++i]; }
        else if (arg == "--ai-detect") { ai_detect_enabled = true; if (i+1 < argc && argv[i+1][0] != '-') ai_detect_model = argv[++i]; }
        else if (arg == "--ai-autoframe") { ai_autoframe_enabled = true; }
        else if (arg == "--lut" && i + 1 < argc) {
            i++;
            lut_enabled = true;
            if (strcmp(argv[i], "warm") == 0) lut_preset = kagerou::LUTPreset::kWarm;
            else if (strcmp(argv[i], "cool") == 0) lut_preset = kagerou::LUTPreset::kCool;
            else if (strcmp(argv[i], "cinematic") == 0) lut_preset = kagerou::LUTPreset::kCinematic;
            else if (strcmp(argv[i], "vintage") == 0) lut_preset = kagerou::LUTPreset::kVintage;
            else if (strcmp(argv[i], "contrast") == 0) lut_preset = kagerou::LUTPreset::kHighContrast;
            else if (strcmp(argv[i], "desat") == 0) lut_preset = kagerou::LUTPreset::kDesaturate;
            else { fprintf(stderr, "error: unknown LUT preset '%s'\n", argv[i]); return 1; }
        }
        else if (arg == "--all") { denoise = true; super_res = true; frame_interp = true; clahe_enabled = true; lut_enabled = true; lut_preset = kagerou::LUTPreset::kCinematic; blur_enabled = true; sharpen_enabled = true; bright_enabled = true; sat_enabled = true; gamma_enabled = true; vignette_enabled = true; }
        else if (arg == "--scale" && i + 1 < argc) {
            i++;
            if (sscanf(argv[i], "%ux%u", &scale_w, &scale_h) != 2) {
                fprintf(stderr, "error: invalid scale format '%s' (expected WxH)\n", argv[i]);
                return 1;
            }
            scale_enabled = true;
        }
        else if (input_path == nullptr) input_path = argv[i];
        else { fprintf(stderr, "error: unknown argument '%s'\n", argv[i]); return 1; }
    }

    if (!input_path) { fprintf(stderr, "error: no input file\n"); print_usage(); return 1; }

    printf("=== Kagerou GPU Transcoder ===\n\n");
    // Resolve default relative model paths against the exe directory so
    // the CLI works from any working directory (bin/ has no models/).
    // Only untouched defaults are rewritten — explicit user paths stay.
    {
        char exepath[MAX_PATH] = {};
        GetModuleFileNameA(NULL, exepath, MAX_PATH);
        std::string exedir = exepath;
        size_t pp = exedir.find_last_of("\\/");
        std::string mroot = (pp == std::string::npos) ? "" : exedir.substr(0, pp) + "\\..\\";
        static std::string resolved[12];
        const char* dflts[12] = {
            "models/denoise/fastdvdnet.onnx", "models/depth/depth_anything_v2.onnx",
            "models/flow/raft_small.onnx", "models/pose/pose_landmark.onnx",
            "models/matting/rvm_mobilenetv3.onnx", "models/face/yunet_2023mar.onnx",
            "models/hands/palm_detection_lite.onnx", "models/hands/hand_landmark_lite.onnx",
            "models/gaze/iris_landmark.onnx", "models/lowlight/zero_dce.onnx",
            "models/style/animeganv3_hayao.onnx", "models/detect/yolov8n.onnx"};
        const char** slots[12] = {&ai_denoise_model, &ai_depth_model, &ai_flow_model,
            &ai_pose_model, &ai_matting_model, &ai_face_model, &ai_palm_model,
            &ai_handlm_model, &ai_iris_model, &ai_lowlight_model, &ai_anime_model,
            &ai_detect_model};
        for (int k = 0; k < 12; k++) {
            if (strcmp(*slots[k], dflts[k]) == 0 && !mroot.empty()) {
                resolved[k] = mroot + dflts[k];
                *slots[k] = resolved[k].c_str();
            }
        }
    }

    printf("Input:  %s\n", input_path);
    printf("Output: %s/\n", output_dir);
    if (denoise)       printf("Filter: denoise\n");
    if (scale_enabled) printf("Filter: scale -> %ux%u\n", scale_w, scale_h);
    if (super_res)     printf("Filter: super-res (2x)\n");
    if (frame_interp)  printf("Filter: frame-interp (%u -> %u fps)\n", fps, fps * 2);
    if (clahe_enabled) printf("Filter: CLAHE contrast\n");
    if (lut_enabled)   printf("Filter: LUT grading\n");
    if (blur_enabled)  printf("Filter: Gaussian blur (sigma=%.1f)\n", blur_sigma);
    if (sharpen_enabled) printf("Filter: sharpen (str=%.1f)\n", sharpen_str);
    if (bright_enabled) printf("Filter: brightness=%.0f contrast=%.1f\n", brightness, contrast);
    if (sat_enabled)   printf("Filter: saturation=%.1f\n", saturation_f);
    if (gamma_enabled) printf("Filter: gamma=%.1f\n", gamma_f);
    if (vignette_enabled) printf("Filter: vignette (str=%.1f)\n", vignette_f);
    if (grain_enabled) printf("Filter: film grain (amt=%.0f)\n", grain_f);
    if (edge_enabled)  printf("Filter: edge detect\n");
    if (wb_enabled)    printf("Filter: white balance\n");
    if (lens_enabled)  printf("Filter: lens distortion (k1=%.1f)\n", lens_k1);
    if (flip_enabled)  printf("Filter: flip %s\n", flip_mode==1?"horizontal":"vertical");
    if (crop_enabled)  printf("Filter: crop %ux%u+%u+%u\n", crop_w, crop_h, crop_x, crop_y);
    if (chromakey_enabled) printf("Filter: chroma key\n");
    if (bgblur_enabled) printf("Filter: background blur (radius=%.1f strength=%.0f)\n", bg_blur_radius, bg_blur_str);
    if (temporal_denoise_enabled) printf("Filter: temporal denoise (str=%.2f)\n", temporal_str);
    if (temporal_stab_enabled) printf("Filter: temporal stabilization\n");
    if (hdr_enabled)    printf("Filter: HDR tone map (%s)\n", hdr_method==0?"Reinhard":"ACES");
    if (dirblur_enabled) printf("Filter: directional blur (angle=%.0f len=%d)\n", dirblur_angle, dirblur_len);
    if (ai_denoise_enabled) printf("Filter: AI denoise (FastDVDNet)\n");
    if (ai_depth_enabled) printf("Filter: AI depth estimation (Depth Anything V2)\n");
    if (ai_flow_enabled) printf("Filter: AI optical flow (RAFT)\n");
    if (ai_pose_enabled) printf("Filter: AI pose estimation (MediaPipe Pose)\n");
    if (ai_matting_enabled) printf("Filter: AI matting (RVM)\n");

    if (ai_autoframe_enabled) printf("Filter: AI auto framing\n");
    if (ai_face_enabled) printf("Filter: AI face detection (YuNet)\n");
    if (ai_hands_enabled) printf("Filter: AI hand tracking (BlazePalm)\n");
    if (ai_gaze_enabled) printf("Filter: AI gaze (YuNet + iris)\n");
    if (ai_lowlight_enabled) printf("Filter: AI low-light (Zero-DCE)\n");
    if (ai_anime_enabled) printf("Filter: AI anime (AnimeGANv3)\n");
    if (ai_detect_enabled) printf("Filter: AI object detection (YOLOv8n)\n");
    printf("\n");

    kagerou::fileio::VideoFile video;
    kagerou::Error e = video.load(input_path);
    if (e != kagerou::Error::kOk) {
        fprintf(stderr, "error: cannot open input: %s\n", kagerou::error_string(e));
        return 1;
    }

    // Auto-detect decoder codec from the loaded file
    kagerou::VideoCodec decoder_codec = kagerou::VideoCodec::kH264;
    if (video.codec == 0x21) decoder_codec = kagerou::VideoCodec::kH264;
    else if (video.codec == 0x23) decoder_codec = kagerou::VideoCodec::kH265;
    else if (video.codec == 0x31) decoder_codec = kagerou::VideoCodec::kAV1;  // av01
    else if (video.codec == 0x00) decoder_codec = kagerou::VideoCodec::kAV1;  // ffmpeg fallback
    else {
        fprintf(stderr, "warning: unknown codec 0x%02X, assuming H.264\n", video.codec);
    }
    printf("[codec] detected decoder: %s (mp4 codec=0x%02X)\n",
           decoder_codec == kagerou::VideoCodec::kH264 ? "H.264" :
           decoder_codec == kagerou::VideoCodec::kH265 ? "H.265" :
           decoder_codec == kagerou::VideoCodec::kAV1  ? "AV1" : "unknown",
           video.codec);

    // Auto-detect FPS from input file if user didn't specify --fps
    if (!fps_set_by_user && video.fps > 0) {
        fps = (uint32_t)(video.fps + 0.5);
        printf("[fps] auto-detected from input: %u fps\n", fps);
    }

    // Auto-scale bitrate based on resolution if not explicitly set
    // Target ~0.07 bits per pixel at target fps (decent quality for streaming)
    if (bitrate == 0) {
        // If minimp4 didn't detect dimensions (AV1, etc.), use ffprobe
        if (video.width == 0 || video.height == 0) {
            char probe_cmd[512];
            snprintf(probe_cmd, sizeof(probe_cmd),
                "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 \"%s\"",
                input_path);
            FILE* pp = _popen(probe_cmd, "r");
            if (pp) {
                int pw = 0, ph = 0;
                if (fscanf(pp, "%d,%d", &pw, &ph) == 2 && pw > 0 && ph > 0) {
                    video.width = pw;
                    video.height = ph;
                    printf("[bitrate] ffprobe: %dx%d\n", pw, ph);
                }
                _pclose(pp);
            }
        }
        if (video.width > 0 && video.height > 0) {
            bitrate = (uint32_t)((double)video.width * video.height * fps * 0.07 / 1000.0);
            if (bitrate < 500)  bitrate = 500;
            if (bitrate > 50000) bitrate = 50000;
            printf("[bitrate] auto: %ux%u@%ufps -> %ukbps\n", video.width, video.height, fps, bitrate);
        } else {
            bitrate = 5000;
            printf("[bitrate] fallback: %ukbps (could not detect resolution)\n", bitrate);
        }
    }

    kagerou::PipelineConfig cfg;
    cfg.verbose = verbose;
    cfg.decoder.codec = decoder_codec;
    cfg.decoder.max_width  = 7680;
    cfg.decoder.max_height = 4320;
    cfg.encoder.codec = codec;
    cfg.encoder.fps    = fps;
    cfg.encoder.bitrate_kbps = bitrate;
    cfg.encoder.rc = kagerou::RateControl::kCQP; // historical behavior (CQP ignores bitrate)
    cfg.encoder.gop_size = fps;
    cfg.encoder.width  = 0;
    cfg.encoder.height = 0;

    cfg.denoise.enabled = denoise;
    cfg.super_res.enabled = super_res;
    cfg.frame_interp.enabled = frame_interp;
    cfg.clahe.enabled = clahe_enabled;
    cfg.lut.enabled = lut_enabled;
    cfg.lut.preset = lut_preset;
    cfg.blur.enabled = blur_enabled;
    cfg.blur.sigma = blur_sigma;
    cfg.sharpen.enabled = sharpen_enabled;
    cfg.sharpen.strength = sharpen_str;
    cfg.bright_contrast.enabled = bright_enabled;
    cfg.bright_contrast.brightness = brightness;
    cfg.bright_contrast.contrast = contrast;
    cfg.saturation.enabled = sat_enabled;
    cfg.saturation.factor = saturation_f;
    cfg.gamma.enabled = gamma_enabled;
    cfg.gamma.gamma = gamma_f;
    cfg.vignette.enabled = vignette_enabled;
    cfg.vignette.strength = vignette_f;
    cfg.film_grain.enabled = grain_enabled;
    cfg.film_grain.amount = grain_f;
    cfg.edge_detect.enabled = edge_enabled;
    cfg.white_balance.enabled = wb_enabled;
    cfg.lens_distortion.enabled = lens_enabled;
    cfg.lens_distortion.k1 = lens_k1;
    cfg.flip.enabled = flip_enabled;
    cfg.flip.mode = flip_mode;
    cfg.dir_blur.enabled = dirblur_enabled;
    cfg.dir_blur.angle = dirblur_angle;
    cfg.dir_blur.length = dirblur_len;
    cfg.crop.enabled = crop_enabled;
    cfg.crop.x = (int)crop_x;
    cfg.crop.y = (int)crop_y;
    cfg.crop.width = crop_w;
    cfg.crop.height = crop_h;
    cfg.chroma_key.enabled = chromakey_enabled;
    cfg.chroma_key.hue_min = key_h_min;
    cfg.chroma_key.hue_max = key_h_max;
    cfg.chroma_key.sat_min = key_sat_min;
    cfg.chroma_key.val_min = key_val_min;
    cfg.chroma_key.spill_suppress = key_spill;
    cfg.chroma_key.bg_r = (uint8_t)key_bg_r;
    cfg.chroma_key.bg_g = (uint8_t)key_bg_g;
    cfg.chroma_key.bg_b = (uint8_t)key_bg_b;
    cfg.bg_blur.enabled = bgblur_enabled;
    cfg.bg_blur.focus_radius = bg_blur_radius;
    cfg.bg_blur.blur_strength = bg_blur_str;
    cfg.temporal_denoise.enabled = temporal_denoise_enabled;
    cfg.temporal_denoise.strength = temporal_str;
    cfg.temporal_stab.enabled = temporal_stab_enabled;
    cfg.hdr.enabled = hdr_enabled;
    cfg.hdr.method = hdr_method;
    cfg.hdr.peak_nits = hdr_peak_nits;
    cfg.ai_denoise_enabled = ai_denoise_enabled;
    cfg.ai_denoise_model = ai_denoise_model;
    cfg.ai_depth_enabled = ai_depth_enabled;
    cfg.ai_depth_model = ai_depth_model;
    cfg.ai_flow_enabled = ai_flow_enabled;
    cfg.ai_flow_model = ai_flow_model;
    cfg.ai_pose_enabled = ai_pose_enabled;
    cfg.ai_pose_model = ai_pose_model;
    cfg.ai_matting_enabled = ai_matting_enabled;
    cfg.ai_matting_model = ai_matting_model;
    cfg.ai_face_enabled = ai_face_enabled;
    cfg.ai_face_model = ai_face_model;
    cfg.ai_hands_enabled = ai_hands_enabled;
    cfg.ai_palm_model = ai_palm_model;
    cfg.ai_handlm_model = ai_handlm_model;
    cfg.ai_gaze_enabled = ai_gaze_enabled;
    cfg.ai_iris_model = ai_iris_model;
    cfg.ai_lowlight_enabled = ai_lowlight_enabled;
    cfg.ai_lowlight_model = ai_lowlight_model;
    cfg.ai_anime_enabled = ai_anime_enabled;
    cfg.ai_anime_model = ai_anime_model;
    cfg.ai_detect_enabled = ai_detect_enabled;
    cfg.ai_detect_model = ai_detect_model;

    cfg.ai_autoframe_enabled = ai_autoframe_enabled;
    if (scale_enabled) {
        cfg.scale.enabled = true;
        cfg.scale.target_width  = scale_w;
        cfg.scale.target_height = scale_h;
    }

    kagerou::Pipeline pipeline;
    e = pipeline.init(cfg);
    if (e != kagerou::Error::kOk) {
        fprintf(stderr, "error: pipeline init failed: %s\n", kagerou::error_string(e));
        return 1;
    }

    kagerou::fileio::create_directory(output_dir);

    std::vector<uint8_t> output_data;
    int frame_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Count VCL (video coding layer) NALUs = actual frames
    int vcl_nalus = 0;
    for (auto& n : video.nalus) if (n.is_vcl) vcl_nalus++;
    printf("[input] %zu total NALUs, %d VCL (frames)\n", video.nalus.size(), vcl_nalus);

    if (pipeline.sw_decode_path) {
        // Software decode path: decode entire file via ffmpeg, process all frames
        e = pipeline.load_sw_input(input_path);
        if (e != kagerou::Error::kOk) {
            fprintf(stderr, "error: software decode failed: %s\n", kagerou::error_string(e));
            return 1;
        }
        e = pipeline.process_sw(output_data);
        if (e != kagerou::Error::kOk) {
            fprintf(stderr, "error: process_sw failed: %s\n", kagerou::error_string(e));
        }
        frame_count = (int)(output_data.size() > 0 ? 1 : 0); // count set by process_sw
    } else {
        // Hardware decode path: feed NALUs through CUVID parser
        for (const auto& nalu : video.nalus) {
            std::vector<uint8_t> encoded;
            e = pipeline.process_frame(nalu.data, nalu.size, encoded);
            if (e != kagerou::Error::kOk) {
                fprintf(stderr, "frame %d decode failed: %s\n", frame_count, kagerou::error_string(e));
                continue;
            }
            output_data.insert(output_data.end(), encoded.begin(), encoded.end());
            frame_count++;
            if (frame_count % 30 == 0)
                printf("\r  processed %d frames   ", frame_count);
        }

        std::vector<uint8_t> flush_data;
        e = pipeline.flush(flush_data);
        output_data.insert(output_data.end(), flush_data.begin(), flush_data.end());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("\r  done (%.1f MB output, %.1f s elapsed)              \n",
           output_data.size() / 1048576.0, elapsed / 1000.0);
    printf("  VCL frames in: %d  |  H.264 bytes out: %zu\n", vcl_nalus, output_data.size());

    if (!output_data.empty()) {
        std::string base = kagerou::fileio::get_basename(input_path);
        std::string h264_path = std::string(output_dir) + "/" + base + ".h264";
        std::string mp4_path  = std::string(output_dir) + "/" + base + ".mp4";

        kagerou::fileio::write_file(h264_path.c_str(), output_data);
        printf("  H.264: %s (%zu bytes)\n", h264_path.c_str(), output_data.size());

        printf("  Muxing to MP4...\n");
        e = kagerou::fileio::mux_to_mp4(output_data.data(), output_data.size(),
                                         mp4_path.c_str(), fps);
        if (e == kagerou::Error::kOk) {
            printf("  MP4:   %s\n", mp4_path.c_str());
            char tmp_probe[512];
            snprintf(tmp_probe, sizeof(tmp_probe), "_kagerou_probe_%d.txt", (int)GetCurrentProcessId());
            char probe_cmd[1024];
            // Convert forward slashes to backslashes for cmd.exe compatibility
            std::string mp4_win = mp4_path;
            for (auto& c : mp4_win) if (c == '/') c = '\\';
            snprintf(probe_cmd, sizeof(probe_cmd),
                "ffprobe -v error -select_streams v:0 -show_entries stream=duration,nb_frames,r_frame_rate,avg_frame_rate -of default=noprint_wrappers=1 \"%s\" 1>%s 2>nul",
                mp4_win.c_str(), tmp_probe);
            system(probe_cmd);
            FILE* pf = fopen(tmp_probe, "r");
            if (pf) {
                printf("  [ffprobe] ");
                char line[256];
                while (fgets(line, sizeof(line), pf)) {
                    line[strcspn(line, "\r\n")] = 0;
                    printf("%s | ", line);
                }
                printf("\n");
                fclose(pf);
            }
            remove(tmp_probe);
        } else
            fprintf(stderr, "  MP4 mux failed (is ffmpeg in PATH?)\n");
    }

    printf("\n");
    pipeline.print_stats();
    pipeline.destroy();
    return 0;
}

// ---- Main dispatcher ----

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (argc < 2) { print_usage(); return 0; }

    std::string cmd = argv[1];

    if (cmd == "-h" || cmd == "--help") { print_usage(); return 0; }
    if (cmd == "benchmark") return run_benchmark();
    if (cmd == "batch") return run_batch(argc > 2 ? argv[2] : nullptr);
    if (cmd == "test") { fprintf(stderr, "Use: kagerou_test.exe\n"); return 1; }

    // Default: transcode mode (first arg is input file)
    return run_transcode(argc, argv);
}
