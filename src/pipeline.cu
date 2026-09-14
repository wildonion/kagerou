// Kagerou SDK — pipeline orchestrator.
// Connects decoder -> filters -> encoder in a streaming fashion.
// NV12-native filters (denoise, clahe, scale, super_res, frame_interp) run
// directly on NV12. RGB conversion only for LUT (the only RGB-dependent filter).

#include "kagerou/common.h"
#include "kagerou/config.h"
#include "kagerou/filters.h"
#include "decoder.cu"
#include "encoder.cu"

#include "filters/color_convert.cu"
#include "filters/scale.cu"
#include "filters/denoise.cu"
#include "filters/super_res.cu"
#include "filters/frame_interp.cu"
#include "filters/clahe.cu"
#include "filters/lut.cu"
#include "filters/creative.cu"
#include "filters/compositor.cu"
#include "filters/transforms.cu"

#ifdef KAGEROU_USE_ONNX
#include "ai/ort_wrapper.cu"
#include "ai/ai_preprocess.cu"
#include "ai/ai_denoise.cu"
#include "ai/ai_depth.cu"
#include "ai/ai_flow.cu"
#include "ai/ai_pose_estimation.cu"
#include "ai/ai_lowlight.cu"
#include "ai/ai_face.cu"
#include "ai/ai_yolo.cu"
#include "ai/ai_anime.cu"
#include "ai/ai_matting.cu"
#include "ai/ai_hands.cu"
#include "ai/ai_gaze.cu"
#include "kagerou/ai/ai_filters.h"
#endif

#include <vector>
#include <cstdio>
#include <chrono>

namespace kagerou {

struct PipelineStats {
    uint32_t frames_processed = 0;
    uint32_t frames_decoded   = 0;
    uint32_t frames_encoded   = 0;
    uint32_t frames_flushed   = 0;
    uint32_t nalus_fed        = 0;
    double   total_ms         = 0.0;
    double   fps              = 0.0;
};

// Simple GPU buffer pool — reuses Frame allocations to avoid malloc/free per frame.
// Tracks in-use state so acquire() never returns the same buffer twice.
struct FramePool {
    static const int MAX_BUFFERS = 16;
    Frame buffers[MAX_BUFFERS];
    bool in_use[MAX_BUFFERS] = {};
    int count = 0;
    Frame overflow;  // fallback when pool exhausted (per-instance, not static)

    Frame& acquire(uint32_t w, uint32_t h, PixelFormat fmt) {
        for (int i = 0; i < count; ++i) {
            if (!in_use[i] && buffers[i].d_data && buffers[i].width == w &&
                buffers[i].height == h && buffers[i].fmt == fmt) {
                in_use[i] = true;
                return buffers[i];
            }
        }
        if (count < MAX_BUFFERS) {
            alloc_frame_gpu(buffers[count], w, h, fmt);
            in_use[count] = true;
            return buffers[count++];
        }
        alloc_frame_gpu(overflow, w, h, fmt);
        return overflow;
    }

    void release(Frame& f) {
        if (!f.d_data) return;
        for (int i = 0; i < count; ++i) {
            if (buffers[i].d_data == f.d_data) {
                in_use[i] = false;
                f.d_data = nullptr;
                return;
            }
        }
    }

    void free_all() {
        for (int i = 0; i < count; ++i) {
            buffers[i].free_gpu();
            in_use[i] = false;
        }
        count = 0;
        overflow.free_gpu();
    }
};

struct Pipeline {
    PipelineConfig cfg;
    Decoder        decoder;
    Encoder        encoder;
    cudaStream_t   stream = nullptr;
    bool           initialized = false;
    bool           encoder_initialized = false;
    bool           auto_detect_resolution = false;

    // Buffer pool for intermediate frames (avoids cudaMalloc/cudaFree per frame)
    FramePool      pool;

    // Per-instance LUT cache (not static — safe for multi-stream)
    uint8_t*       cached_lut = nullptr;
    int            cached_preset = -1;

    PipelineStats stats;
    bool           sw_decode_path = false;

    // CUDA graph: captures fixed filter chain for replay (reduces launch overhead)
    cudaGraph_t    graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool           graph_captured = false;

    // Ring buffer for temporal filters (denoise, stabilization)
    static const int TEMPORAL_BUF_SIZE = 4;
    Frame           temporal_buf[TEMPORAL_BUF_SIZE];
    int             temporal_idx = 0;
    int             temporal_count = 0;

    // Stabilization state
    int*            d_motion_accum = nullptr;  // 3 ints: [sum_dx, sum_dy, count]
    int             prev_warp_dx = 0, prev_warp_dy = 0;

    // Pre-allocated AI filter scratch buffers (avoid cudaMalloc per frame)
    float*          d_ai_depth = nullptr;

    // Persistent previous frame for optical flow (must survive across process_single_frame calls)
    Frame           flow_prev_frame;

    // Auto-frame state (smoothed crop window + face cache)
    ai::FaceBox    af_boxes[4];
    float          af_x = 0, af_y = 0, af_w = 0, af_h = 0;
    uint32_t       af_fw = 0, af_fh = 0;
    bool           af_init = false;

    bool has_active_filters() const {
        return cfg.denoise.enabled || cfg.scale.enabled ||
               cfg.super_res.enabled || cfg.frame_interp.enabled ||
               cfg.clahe.enabled || cfg.lut.enabled ||
               cfg.blur.enabled || cfg.sharpen.enabled ||
               cfg.bright_contrast.enabled || cfg.saturation.enabled ||
               cfg.gamma.enabled || cfg.vignette.enabled ||
               cfg.film_grain.enabled || cfg.edge_detect.enabled ||
               cfg.white_balance.enabled || cfg.lens_distortion.enabled ||
               cfg.flip.enabled || cfg.dir_blur.enabled ||
               cfg.crop.enabled || cfg.pad.enabled ||
               cfg.chroma_key.enabled || cfg.temporal_denoise.enabled ||
               cfg.temporal_stab.enabled || cfg.bg_blur.enabled ||
                cfg.hdr.enabled ||
                cfg.ai_denoise_enabled || cfg.ai_flow_enabled ||
                cfg.ai_pose_enabled || cfg.ai_depth_enabled ||
                cfg.ai_matting_enabled || cfg.ai_face_enabled ||
                cfg.ai_hands_enabled || cfg.ai_gaze_enabled ||
                cfg.ai_lowlight_enabled || cfg.ai_anime_enabled ||
                cfg.ai_detect_enabled || cfg.ai_autoframe_enabled;
    }

    Error init(const PipelineConfig& c) {
        cfg = c;
        cudaSetDevice(cfg.device_id);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cfg.device_id);
        if (cfg.verbose) {
            printf("[pipeline] GPU: %s (cc %d.%d, %d SMs)\n",
                   prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        }

        Error e = decoder.init(cfg.decoder);
        if (e != Error::kOk) return e;

        // For AV1/VP9/other non-H264/H265: use software decode path via ffmpeg
        if (cfg.decoder.codec != VideoCodec::kH264 && cfg.decoder.codec != VideoCodec::kH265) {
            printf("[pipeline] codec not H.264/H.265 -- using software decode path\n");
            sw_decode_path = true;
        }

        // If encoder width/height are 0, auto-detect from decoder on first frame
        auto_detect_resolution = (cfg.encoder.width == 0 || cfg.encoder.height == 0);
        encoder_initialized = false;

        if (!auto_detect_resolution) {
            e = init_encoder(cfg.encoder.width, cfg.encoder.height);
            if (e != Error::kOk) return e;
        }

        initialized = true;

#ifdef KAGEROU_USE_ONNX
        // Init AI models
        if (cfg.ai_denoise_enabled) {
            if (!ai::ai_denoise_init(cfg.ai_denoise_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI denoise model failed to load\n");
        }
        if (cfg.ai_depth_enabled) {
            if (!ai::ai_depth_init(cfg.ai_depth_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI depth model failed to load\n");
        }
        if (cfg.ai_flow_enabled) {
            if (!ai::ai_flow_init(cfg.ai_flow_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI flow model failed to load\n");
        }
        if (cfg.ai_pose_enabled) {
            if (!ai::ai_pose_init(cfg.ai_pose_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI pose model failed to load\n");
        }
        if (cfg.ai_autoframe_enabled) {
            if (!ai::ai_face_init(cfg.ai_face_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI autoframe (face) model failed to load\n");
        }
        if (cfg.ai_matting_enabled) {
            // CUDA EP only: TensorRT rejects RVM's symbolic dims, and the
            // static-shape build is baked for 640x480 (demo) while files
            // can be any resolution.
            if (!ai::ai_matting_init(cfg.ai_matting_model.c_str(), cfg.device_id, false))
                fprintf(stderr, "[pipeline] WARNING: AI matting model failed to load\n");
        }
        if (cfg.ai_face_enabled) {
            if (!ai::ai_face_init(cfg.ai_face_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI face model failed to load\n");
        }
        if (cfg.ai_detect_enabled) {
            if (!ai::ai_detect_init(cfg.ai_detect_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI detect model failed to load\n");
        }
        if (cfg.ai_hands_enabled) {
            if (!ai::ai_hands_init(cfg.ai_palm_model.c_str(), cfg.ai_handlm_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI hands models failed to load\n");
        }
        if (cfg.ai_gaze_enabled) {
            if (!ai::ai_face_init(cfg.ai_face_model.c_str(), cfg.device_id) ||
                !ai::ai_gaze_init(cfg.ai_iris_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI gaze models failed to load\n");
        }
        if (cfg.ai_lowlight_enabled) {
            if (!ai::ai_lowlight_init(cfg.ai_lowlight_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI lowlight model failed to load\n");
        }
        if (cfg.ai_anime_enabled) {
            if (!ai::ai_anime_init(cfg.ai_anime_model.c_str(), cfg.device_id))
                fprintf(stderr, "[pipeline] WARNING: AI anime model failed to load\n");
        }
        // Pre-allocate AI scratch buffers (avoid per-frame cudaMalloc)
        if (cfg.ai_depth_enabled && cfg.encoder.width > 0 && cfg.encoder.height > 0) {
            cudaMalloc(&d_ai_depth, cfg.encoder.width * cfg.encoder.height * sizeof(float));
        }
#else
        if (cfg.ai_denoise_enabled) {
            fprintf(stderr, "[pipeline] WARNING: AI filters requested but ONNX Runtime not available (build with onnx flag)\n");
        }
#endif

        // Create CUDA stream AFTER AI model inits (ORT CUDA EP invalidates streams created before it)
        KAGEROU_CUDA_CHECK(cudaStreamCreate(&stream));

        // Allocate temporal ring buffer for temporal denoise / stabilization
        // / AI denoise (only if resolution is known; deferred if auto-detect)
        if ((cfg.temporal_denoise.enabled || cfg.temporal_stab.enabled ||
             cfg.ai_denoise_enabled) &&
            cfg.encoder.width > 0 && cfg.encoder.height > 0) {
            for (int i = 0; i < TEMPORAL_BUF_SIZE; i++) {
                alloc_frame_gpu(temporal_buf[i], cfg.encoder.width, cfg.encoder.height,
                                PixelFormat::kRGB);
            }
            temporal_count = 0;
            temporal_idx = 0;
        }
        if (cfg.temporal_stab.enabled && cfg.encoder.width > 0 && cfg.encoder.height > 0) {
            cudaMalloc(&d_motion_accum, 3 * sizeof(int));
            cudaMemset(d_motion_accum, 0, 3 * sizeof(int));
        }

        return Error::kOk;
    }

    Error init_encoder(uint32_t w, uint32_t h) {
        cfg.encoder.width  = w;
        cfg.encoder.height = h;

        // if scale is enabled, use scale target dimensions instead
        if (cfg.scale.enabled) {
            cfg.encoder.width  = cfg.scale.target_width;
            cfg.encoder.height = cfg.scale.target_height;
        }

        // Share decoder's CUDA context — single context avoids WDDM scheduling glitches
#if !defined(KAGEROU_NO_VCODEC_SDK)
        Error e = encoder.init(cfg.encoder, decoder.cu_ctx);
#else
        Error e = encoder.init(cfg.encoder);
#endif
        if (e != Error::kOk) return e;
        encoder_initialized = true;
        printf("[pipeline] encoder initialized: %ux%u\n", cfg.encoder.width, cfg.encoder.height);
        return Error::kOk;
    }

    // ---- CUDA Graph: capture fixed filter chain for replay --------------------
    // Call after init() and init_encoder(). Captures the active filter chain
    // as a CUDA graph. Subsequent process_single_frame() calls replay the graph
    // instead of launching kernels individually — reduces launch overhead.
    //
    // Limitation: the graph captures a fixed filter chain. If filters are
    // toggled on/off at runtime, re-capture with capture_graph().
    Error capture_graph() {
        if (!initialized || !encoder_initialized) return Error::kFilterError;
        if (graph_captured) { destroy_graph(); }

        // Allocate dummy input for graph capture
        uint32_t w = cfg.encoder.width;
        uint32_t h = cfg.encoder.height;
        if (cfg.scale.enabled) { w = cfg.scale.target_width; h = cfg.scale.target_height; }

        Frame dummy_in;
        alloc_frame_gpu(dummy_in, w, h, PixelFormat::kNV12);
        // Fill with zeros so kernels don't read garbage
        cudaMemsetAsync(dummy_in.d_data, 128, w * h * 3 / 2, stream);

        Frame dummy_out;
        alloc_frame_gpu(dummy_out, w, h, PixelFormat::kNV12);

        // Begin capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Run the filter chain (same logic as process_single_frame, without encode)
        Frame* work = &dummy_in;
        Frame filtered = pool.acquire(work->width, work->height, PixelFormat::kNV12);
        Frame rgb_stage = pool.acquire(work->width, work->height, PixelFormat::kRGB);
        Frame rgb_out = pool.acquire(work->width, work->height, PixelFormat::kRGB);

        // NV12-native filters
        if (cfg.denoise.enabled) {
            Frame f = pool.acquire(work->width, work->height, PixelFormat::kNV12);
            filters::denoise_nv12_bilateral(work->d_data, f.d_data,
                work->width, work->height,
                cfg.denoise.sigma_spatial, cfg.denoise.sigma_color,
                cfg.denoise.kernel_size, stream);
            pool.release(*work);
            work = &f;
        }
        if (cfg.clahe.enabled) {
            Frame f = pool.acquire(work->width, work->height, PixelFormat::kNV12);
            filters::clahe_nv12(work->d_data, f.d_data,
                work->width, work->height,
                cfg.clahe.clip_limit, cfg.clahe.tile_size, stream);
            pool.release(*work);
            work = &f;
        }
        if (cfg.scale.enabled) {
            Frame f = pool.acquire(cfg.scale.target_width, cfg.scale.target_height, PixelFormat::kNV12);
            if (cfg.scale.interpolation == 0)
                filters::resize_nv12_bilinear(work->d_data, f.d_data,
                    work->width, work->height, cfg.scale.target_width, cfg.scale.target_height, stream);
            else
                filters::resize_nv12_bicubic(work->d_data, f.d_data,
                    work->width, work->height, cfg.scale.target_width, cfg.scale.target_height, stream);
            pool.release(*work);
            work = &f;
        }
        if (cfg.super_res.enabled) {
            uint32_t sr_w = work->width * 2;
            uint32_t sr_h = work->height * 2;
            Frame f = pool.acquire(sr_w, sr_h, PixelFormat::kNV12);
            // Y plane super-res
            filters::super_res_2x(work->d_data, f.d_data,
                work->width, work->height, 1, cfg.super_res.sharpen_strength, stream);
            // UV plane bilinear
            filters::resize_nv12_bilinear(work->d_data, f.d_data,
                work->width, work->height, sr_w, sr_h, stream);
            pool.release(*work);
            work = &f;
        }
        if (cfg.frame_interp.enabled) {
            // Frame interpolation needs two frames — skip in graph capture
            // (will be handled in process_frame_pair)
        }

        // RGB filters: only if LUT or creative RGB filters are active
        bool need_rgb = cfg.lut.enabled || cfg.blur.enabled || cfg.sharpen.enabled ||
                        cfg.bright_contrast.enabled || cfg.saturation.enabled ||
                        cfg.gamma.enabled || cfg.vignette.enabled ||
                        cfg.film_grain.enabled || cfg.edge_detect.enabled ||
                        cfg.white_balance.enabled || cfg.lens_distortion.enabled;
        if (need_rgb) {
            filters::nv12_to_rgb_limited(work->d_data, rgb_stage.d_data,
                work->width, work->height, stream);
            pool.release(*work);
            work = &rgb_stage;

            if (cfg.lut.enabled) {
                if (!cached_lut || cached_preset != (int)cfg.lut.preset) {
                    if (cached_lut) cudaFree(cached_lut);
                    cached_lut = filters::generate_builtin_lut((int)cfg.lut.preset, stream);
                    cached_preset = (int)cfg.lut.preset;
                }
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::lut3d_rgb(work->d_data, f.d_data,
                    work->width, work->height, cached_lut,
                    filters::LUT_RES, cfg.lut.strength, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.blur.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::gaussian_blur(work->d_data, f.d_data,
                    work->width, work->height, 3, cfg.blur.sigma, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.sharpen.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::sharpen_rgb(work->d_data, f.d_data,
                    work->width, work->height, cfg.sharpen.strength, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.bright_contrast.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::brightness_contrast(work->d_data, f.d_data,
                    work->width, work->height,
                    cfg.bright_contrast.brightness, cfg.bright_contrast.contrast, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.saturation.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::saturation_rgb(work->d_data, f.d_data,
                    work->width, work->height, cfg.saturation.factor, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.gamma.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::gamma_rgb(work->d_data, f.d_data,
                    work->width, work->height, cfg.gamma.gamma, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.vignette.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::vignette_rgb(work->d_data, f.d_data,
                    work->width, work->height, cfg.vignette.strength, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.film_grain.enabled) {
                filters::film_grain_rgb(work->d_data,
                    work->width, work->height,
                    cfg.film_grain.amount, cfg.film_grain.seed, stream);
            }
            if (cfg.edge_detect.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::edge_detect_rgb(work->d_data, f.d_data,
                    work->width, work->height, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.white_balance.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::white_balance_rgb(work->d_data, f.d_data,
                    work->width, work->height,
                    cfg.white_balance.temperature, cfg.white_balance.tint, stream);
                pool.release(*work);
                work = &f;
            }
            if (cfg.lens_distortion.enabled) {
                Frame f = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                filters::lens_distortion(work->d_data, f.d_data,
                    work->width, work->height, 3, cfg.lens_distortion.k1, stream);
                pool.release(*work);
                work = &f;
            }
            // Convert back to NV12 for encode
            Frame nv12_back = pool.acquire(work->width, work->height, PixelFormat::kNV12);
            filters::rgb_to_nv12_limited(work->d_data, nv12_back.d_data,
                work->width, work->height, stream);
            pool.release(*work);
            work = &nv12_back;
        }

        // Copy result to dummy_out (simulates the encode input)
        uint32_t total = work->width * work->height * 3 / 2;
        cudaMemcpyAsync(dummy_out.d_data, work->d_data, total,
                        cudaMemcpyDeviceToDevice, stream);
        pool.release(*work);

        // End capture
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

        // Cleanup capture temporaries
        dummy_in.free_gpu();
        dummy_out.free_gpu();

        graph_captured = true;
        printf("[pipeline] CUDA graph captured (%ux%u, %d filters)\n",
               w, h, has_active_filters() ? 1 : 0);
        return Error::kOk;
    }

    // Replay the captured graph (faster than individual kernel launches)
    Error replay_graph() {
        if (!graph_captured || !graph_exec) return Error::kFilterError;
        cudaGraphLaunch(graph_exec, stream);
        cudaStreamSynchronize(stream);
        return Error::kOk;
    }

    void destroy_graph() {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        graph_captured = false;
    }

    // Load input file for software decode path (AV1, VP9, etc.)
    // Decodes entire file to NV12 frames stored in decoder's frame buffer.
    Error load_sw_input(const char* input_path) {
        uint32_t w = 0, h = 0;
        Error e = decoder.load_sw_frames(input_path, w, h);
        if (e != Error::kOk) return e;

        // Auto-detect encoder dimensions from decoded frames
        if (auto_detect_resolution) {
            uint32_t enc_w = w, enc_h = h;
            if (cfg.scale.enabled) { enc_w = cfg.scale.target_width; enc_h = cfg.scale.target_height; }
            e = init_encoder(enc_w, enc_h);
            if (e != Error::kOk) return e;
        }
        return Error::kOk;
    }

    // Process one NALU. May produce multiple encoded frames if the parser
    // queued multiple display callbacks from a single parse.
    Error process_frame(const uint8_t* bitstream, size_t bs_size,
                        std::vector<uint8_t>& encoded_out) {
        if (!initialized) return Error::kFilterError;
        encoded_out.clear();

        auto t0 = std::chrono::high_resolution_clock::now();

        stats.nalus_fed++;

        // First call feeds the NALU to the parser. This may trigger multiple
        // display callbacks that populate the decoder's display_queue.
        Frame decoded;
        Error e = decoder.decode(bitstream, bs_size, decoded);
        if (e != Error::kOk) return e;

        // Now drain ALL queued frames (from this parse + any backlog)
        do {
            if (!decoded.d_data) {
                // decode() returned no frame — check if queue has more
                if (!decoder.has_queued_frames()) break;
                e = decoder.decode(nullptr, 0, decoded);
                if (e != Error::kOk) break;
                if (!decoded.d_data) break;
            }

            stats.frames_decoded++;

            // Auto-detect: init encoder on first decoded frame
                if (auto_detect_resolution && !encoder_initialized) {
                    Error ie = init_encoder(decoded.width, decoded.height);
                    if (ie != Error::kOk) return ie;

                    // Allocate temporal buffers now that resolution is known
                    if (cfg.temporal_denoise.enabled || cfg.temporal_stab.enabled || cfg.ai_denoise_enabled) {
                    if (!temporal_buf[0].d_data) {
                        for (int i = 0; i < TEMPORAL_BUF_SIZE; i++) {
                            alloc_frame_gpu(temporal_buf[i], cfg.encoder.width, cfg.encoder.height,
                                            PixelFormat::kRGB);
                        }
                        temporal_count = 0;
                        temporal_idx = 0;
                    }
                }
                if (cfg.temporal_stab.enabled && !d_motion_accum) {
                    cudaMalloc(&d_motion_accum, 3 * sizeof(int));
                    cudaMemset(d_motion_accum, 0, 3 * sizeof(int));
                }
            }

            std::vector<uint8_t> frame_encoded;
            e = process_single_frame(decoded, frame_encoded);
            if (e == Error::kOk && !frame_encoded.empty()) {
                encoded_out.insert(encoded_out.end(), frame_encoded.begin(), frame_encoded.end());
                stats.frames_encoded++;
            }

            decoded.free_gpu();
            decoded.d_data = nullptr;

            // Check for more queued frames
            if (decoder.has_queued_frames()) {
                e = decoder.decode(nullptr, 0, decoded);
                if (e != Error::kOk) break;
            } else {
                break;
            }
        } while (true);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.frames_processed++;
        stats.total_ms += ms;
        stats.fps = (stats.total_ms > 0) ?
            (double)stats.frames_processed * 1000.0 / stats.total_ms : 0.0;

        return Error::kOk;
    }

    // Process a single already-decoded frame through filters + encoder.
    // Filter ordering: NV12-native first (denoise, clahe, scale, super_res,
    // frame_interp), then RGB conversion only if LUT is active.
    // This eliminates the NV12->RGB->NV12 round-trip for all filters except LUT.
    Error process_single_frame(Frame& decoded, std::vector<uint8_t>& encoded_out) {

        Frame to_rgb;
        Frame filtered;
        Frame scaled;
        Frame sr_output;
        Frame nv12_out;

        Frame* work = &decoded;
        bool filters_active = has_active_filters();

        // Crop must run first (changes resolution)
        if (cfg.crop.enabled && cfg.crop.width > 0 && cfg.crop.height > 0) {
            Frame cropped = pool.acquire(cfg.crop.width, cfg.crop.height, work->fmt);
            if (work->fmt == PixelFormat::kNV12)
                filters::crop_nv12(work->d_data, cropped.d_data,
                                   work->width, work->height,
                                   cfg.crop.width, cfg.crop.height,
                                   cfg.crop.x, cfg.crop.y, stream);
            else
                filters::crop_rgb(work->d_data, cropped.d_data,
                                  work->width, work->height,
                                  cfg.crop.width, cfg.crop.height,
                                  cfg.crop.x, cfg.crop.y, 3, stream);
            pool.release(*work);
            work = &cropped;
            if (auto_detect_resolution && !encoder_initialized) {
                Error ie = init_encoder(work->width, work->height);
                if (ie != Error::kOk) return ie;
            }
        }

        // Pad adds border (changes resolution)
        if (cfg.pad.enabled && (cfg.pad.top > 0 || cfg.pad.bottom > 0 ||
                                cfg.pad.left > 0 || cfg.pad.right > 0)) {
            uint32_t padded_w = work->width + cfg.pad.left + cfg.pad.right;
            uint32_t padded_h = work->height + cfg.pad.top + cfg.pad.bottom;
            Frame padded = pool.acquire(padded_w, padded_h, work->fmt);
            if (work->fmt == PixelFormat::kRGB) {
                // NV12 pad not supported yet — convert to RGB first if needed
                filters::pad_rgb(work->d_data, padded.d_data,
                                 work->width, work->height,
                                 padded_w, padded_h,
                                 cfg.pad.left, cfg.pad.top,
                                 cfg.pad.pad_r, cfg.pad.pad_g, cfg.pad.pad_b, stream);
            } else {
                // For NV12, just copy with offset (no border)
                padded = pool.acquire(work->width, work->height, work->fmt);
                cudaMemcpyAsync(padded.d_data, work->d_data,
                    work->width * work->height * 3 / 2, cudaMemcpyDeviceToDevice, stream);
            }
            pool.release(*work);
            work = &padded;
            if (auto_detect_resolution && !encoder_initialized) {
                Error ie = init_encoder(work->width, work->height);
                if (ie != Error::kOk) return ie;
            }
        }

        if (filters_active) {
            // ==== PHASE 1: NV12-native filters (no RGB conversion) ====

            if (cfg.denoise.enabled && work->fmt == PixelFormat::kNV12) {
                filtered = pool.acquire(work->width, work->height, PixelFormat::kNV12);
                filters::denoise_nv12_bilateral(work->d_data, filtered.d_data,
                                                work->width, work->height,
                                                cfg.denoise.sigma_spatial,
                                                cfg.denoise.sigma_color,
                                                cfg.denoise.kernel_size, stream);
                pool.release(*work);
                work = &filtered;
            }

            if (cfg.clahe.enabled && work->fmt == PixelFormat::kNV12) {
                Frame clahe_out = pool.acquire(work->width, work->height, PixelFormat::kNV12);
                filters::clahe_nv12(work->d_data, clahe_out.d_data,
                                     work->width, work->height,
                                     cfg.clahe.clip_limit, cfg.clahe.tile_size, stream);
                pool.release(*work);
                filtered = clahe_out;
                work = &filtered;
            }

            if (cfg.scale.enabled && work->fmt == PixelFormat::kNV12) {
                scaled = pool.acquire(cfg.scale.target_width, cfg.scale.target_height, PixelFormat::kNV12);
                if (cfg.scale.interpolation == 0)
                    filters::resize_nv12_bilinear(work->d_data, scaled.d_data,
                                                  work->width, work->height,
                                                  cfg.scale.target_width, cfg.scale.target_height,
                                                  stream);
                else
                    filters::resize_nv12_bicubic(work->d_data, scaled.d_data,
                                                 work->width, work->height,
                                                 cfg.scale.target_width, cfg.scale.target_height,
                                                 stream);
                pool.release(*work);
                work = &scaled;
            }

            if (cfg.super_res.enabled && work->fmt == PixelFormat::kNV12) {
                // NV12-native super-res: upscale Y with bicubic+sharpen, bilinear upscale UV
                uint32_t new_w = work->width * (uint32_t)cfg.super_res.scale_factor;
                uint32_t new_h = work->height * (uint32_t)cfg.super_res.scale_factor;
                sr_output = pool.acquire(new_w, new_h, PixelFormat::kNV12);

                // Y plane: super-res (bicubic upscale + unsharp mask sharpen)
                filters::super_res_2x(work->d_data, sr_output.d_data,
                                      work->width, work->height, 1,
                                      cfg.super_res.sharpen_strength, stream);

                // UV plane: bilinear upscale using generic 2-channel resize
                // UV is interleaved U,V at half resolution, treat as 2-channel data
                const uint8_t* src_uv = work->d_data + (size_t)work->width * work->height;
                uint8_t* dst_uv = sr_output.d_data + (size_t)new_w * new_h;
                if (cfg.scale.interpolation == 0)
                    filters::resize_bilinear(src_uv, dst_uv,
                                             work->width / 2, work->height / 2,
                                             new_w / 2, new_h / 2,
                                             2, stream);
                else
                    filters::resize_bicubic(src_uv, dst_uv,
                                            work->width / 2, work->height / 2,
                                            new_w / 2, new_h / 2,
                                            2, stream);

                pool.release(*work);
                work = &sr_output;
            }

            if (cfg.frame_interp.enabled && work->fmt == PixelFormat::kNV12) {
                // Frame interpolation is already format-agnostic.
                // Blend entire NV12 buffer as raw bytes (Y + UV interleaved).
                // Alpha blending is linear, so it's mathematically correct for NV12.
                // Note: frame_interp is handled separately in process_sw_batch
                // (needs two frames). Here we just mark it as NV12-compatible.
            }

            // ==== PHASE 2: RGB conversion + RGB-only filters ====
            // Convert to RGB if LUT or any creative filter is active

            bool need_rgb = cfg.lut.enabled || cfg.blur.enabled || cfg.sharpen.enabled ||
                            cfg.bright_contrast.enabled || cfg.saturation.enabled ||
                            cfg.gamma.enabled || cfg.vignette.enabled || cfg.film_grain.enabled ||
                            cfg.edge_detect.enabled || cfg.white_balance.enabled ||
                            cfg.lens_distortion.enabled || cfg.flip.enabled ||
                            cfg.dir_blur.enabled || cfg.chroma_key.enabled ||
                            cfg.bg_blur.enabled || cfg.hdr.enabled ||
                            cfg.temporal_denoise.enabled || cfg.temporal_stab.enabled ||
                            cfg.ai_denoise_enabled || cfg.ai_flow_enabled ||
                            cfg.ai_pose_enabled || cfg.ai_depth_enabled ||
                            cfg.ai_matting_enabled || cfg.ai_face_enabled ||
                            cfg.ai_hands_enabled || cfg.ai_gaze_enabled ||
                            cfg.ai_lowlight_enabled || cfg.ai_anime_enabled ||
                            cfg.ai_detect_enabled || cfg.ai_autoframe_enabled;

            if (need_rgb && work->fmt == PixelFormat::kNV12) {
                // Convert NV12 -> RGB
                to_rgb = pool.acquire(work->width, work->height, PixelFormat::kRGB);
                if (cfg.color_range == ColorRange::kLimited)
                    filters::nv12_to_rgb_limited(work->d_data, to_rgb.d_data,
                                                 work->width, work->height, stream);
                else
                    filters::nv12_to_rgb(work->d_data, to_rgb.d_data,
                                         work->width, work->height, stream);
                pool.release(*work);
                work = &to_rgb;

                // Apply LUT in RGB space
                if (cfg.lut.enabled) {
                    if (!cached_lut || cached_preset != (int)cfg.lut.preset) {
                        if (cached_lut) cudaFree(cached_lut);
                        cached_lut = filters::generate_builtin_lut((int)cfg.lut.preset, stream);
                        cached_preset = (int)cfg.lut.preset;
                    }
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::lut3d_rgb(work->d_data, sr_output.d_data,
                                        work->width, work->height,
                                        cached_lut, filters::LUT_RES,
                                        cfg.lut.strength, stream);
                    pool.release(*work);
                    work = &sr_output;
                }

                // Apply creative filters in RGB space.
                // CRITICAL: save the source frame BY VALUE before assigning sr_output,
                // because work may already point to sr_output from a previous filter.
                // A pointer won't work (prev->d_data changes when sr_output is reassigned).
                // A value copy preserves the d_data pointer and pool.release() target.
                Frame prev_frame;
                if (cfg.bright_contrast.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::brightness_contrast(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height,
                        cfg.bright_contrast.brightness, cfg.bright_contrast.contrast, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.saturation.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::saturation_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, cfg.saturation.factor, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.gamma.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::gamma_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, cfg.gamma.gamma, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.white_balance.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::white_balance_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height,
                        cfg.white_balance.temperature, cfg.white_balance.tint, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.blur.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::gaussian_blur(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, 3, cfg.blur.sigma, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.sharpen.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::sharpen_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, cfg.sharpen.strength, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.vignette.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::vignette_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, cfg.vignette.strength, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.lens_distortion.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::lens_distortion(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, 3, cfg.lens_distortion.k1, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.dir_blur.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::directional_blur(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, 3,
                        cfg.dir_blur.angle, cfg.dir_blur.length, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.edge_detect.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::edge_detect_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.film_grain.enabled) {
                    filters::film_grain_rgb(work->d_data,
                        work->width, work->height,
                        cfg.film_grain.amount, cfg.film_grain.seed, stream);
                }
                if (cfg.flip.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    if (cfg.flip.mode == 1)
                        filters::flip_horizontal_gpu(prev_frame.d_data, sr_output.d_data,
                            work->width, work->height, stream);
                    else
                        filters::flip_vertical_gpu(prev_frame.d_data, sr_output.d_data,
                            work->width, work->height, stream);
                    pool.release(prev_frame); work = &sr_output;
                }

                // ---- New filters (transforms.cu) ----
                if (cfg.chroma_key.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::chroma_key_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height,
                        cfg.chroma_key.hue_min, cfg.chroma_key.hue_max,
                        cfg.chroma_key.sat_min, cfg.chroma_key.val_min,
                        cfg.chroma_key.spill_suppress,
                        cfg.chroma_key.bg_r, cfg.chroma_key.bg_g, cfg.chroma_key.bg_b,
                        1.0f, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.bg_blur.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::bg_blur_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height,
                        work->width * 0.5f, work->height * 0.5f,
                        cfg.bg_blur.focus_radius, cfg.bg_blur.blur_strength, stream);
                    pool.release(prev_frame); work = &sr_output;
                }

                // ---- AI Filters (ONNX Runtime GPU) ----
#ifdef KAGEROU_USE_ONNX
                if (cfg.ai_denoise_enabled && temporal_count >= 4) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    Frame& f4 = temporal_buf[(temporal_idx - 4 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    Frame& f3 = temporal_buf[(temporal_idx - 3 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    Frame& f2 = temporal_buf[(temporal_idx - 2 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    Frame& f1 = temporal_buf[(temporal_idx - 1 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    // Cap AI working resolution: FastDVDNet at 4K needs ~3.6GB
                    // TRT workspace and OOMs 6GB cards. Denoise small, upscale.
                    uint64_t pix = (uint64_t)work->width * work->height;
                    const uint64_t AI_PIX_BUDGET = 1280ull * 720ull;
                    bool ai_ok = false;
                    if (pix > AI_PIX_BUDGET) {
                        double sc = sqrt((double)AI_PIX_BUDGET / (double)pix);
                        uint32_t dw = (uint32_t)(work->width * sc) & ~1u;
                        uint32_t dh = (uint32_t)(work->height * sc) & ~1u;
                        if (dw < 64) dw = 64;
                        if (dh < 64) dh = 64;
                        fprintf(stderr, "[pipeline] AI denoise: %ux%u too big, working at %ux%u\n",
                                work->width, work->height, dw, dh);
                        Frame din[5], dout;
                        const uint8_t* srcs[5] = {f4.d_data, f3.d_data, f2.d_data,
                                                  f1.d_data, prev_frame.d_data};
                        for (int k = 0; k < 5; k++) {
                            din[k] = pool.acquire(dw, dh, work->fmt);
                            filters::resize_bilinear(srcs[k], din[k].d_data,
                                work->width, work->height, dw, dh, 3, stream);
                        }
                        dout = pool.acquire(dw, dh, work->fmt);
                        if (ai::ai_denoise(din[0].d_data, din[1].d_data, din[2].d_data,
                                           din[3].d_data, din[4].d_data, dout.d_data,
                                           dw, dh, stream)) {
                            filters::resize_bilinear(dout.d_data, sr_output.d_data,
                                dw, dh, work->width, work->height, 3, stream);
                            ai_ok = true;
                        }
                        for (int k = 0; k < 5; k++) pool.release(din[k]);
                        pool.release(dout);
                    } else {
                        ai_ok = ai::ai_denoise(f4.d_data, f3.d_data, f2.d_data, f1.d_data,
                                           prev_frame.d_data, sr_output.d_data,
                                           work->width, work->height, stream);
                    }
                    if (ai_ok) {
                        pool.release(prev_frame); work = &sr_output;
                    } else {
                        pool.release(sr_output);
                        fprintf(stderr, "[pipeline] AI denoise failed on frame\n");
                    }
                }
                if (cfg.ai_depth_enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    if (!d_ai_depth) {
                        cudaMalloc(&d_ai_depth, work->width * work->height * sizeof(float));
                    }
                    if (ai::ai_depth(prev_frame.d_data, d_ai_depth,
                                     work->width, work->height, stream)) {
                        ai::launch_normalize_depth(d_ai_depth,
                                                   work->width * work->height, stream);
                        ai::launch_depth_to_rgb(d_ai_depth, reinterpret_cast<uint8_t*>(sr_output.d_data),
                                                work->width * work->height, stream);
                        pool.release(prev_frame); work = &sr_output;
                    } else {
                        pool.release(sr_output);
                        fprintf(stderr, "[pipeline] AI depth failed on frame\n");
                    }
                }
                if (cfg.ai_flow_enabled) {
                    if (flow_prev_frame.d_data) {
                        uint32_t fw = work->width, fh = work->height;
                        float* d_flow_buf = nullptr;
                        cudaMalloc(&d_flow_buf, fw * fh * 2 * sizeof(float));
                        if (d_flow_buf) {
                            if (ai::ai_flow(flow_prev_frame.d_data, work->d_data,
                                            d_flow_buf, fw, fh, stream)) {
                                sr_output = pool.acquire(fw, fh, work->fmt);
                                ai::launch_flow_to_rgb(d_flow_buf,
                                    sr_output.d_data, fw, fh, stream);
                                work = &sr_output;
                            }
                            cudaFree(d_flow_buf);
                        }
                    }
                    flow_prev_frame = *work;
                }
                if (cfg.ai_pose_enabled) {
                    ai::PoseLandmark pose_landmarks[33];
                    if (ai::ai_pose(work->d_data, work->width, work->height,
                                    pose_landmarks, stream)) {
                        prev_frame = *work;
                        sr_output = pool.acquire(work->width, work->height, work->fmt);
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                        for (int li = 0; li < 33; li++) {
                            if (pose_landmarks[li].visibility > 0.5f) {
                                float px = pose_landmarks[li].x * work->width;
                                float py = pose_landmarks[li].y * work->height;
                                ai::launch_draw_circle(sr_output.d_data, work->width, work->height,
                                    px, py, 4.0f, 255, 0, 0, stream);
                            }
                        }
                        pool.release(prev_frame); work = &sr_output;
                    }
                }
                if (cfg.ai_lowlight_enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    if (ai::ai_lowlight(prev_frame.d_data, sr_output.d_data,
                                        work->width, work->height, stream)) {
                        pool.release(prev_frame); work = &sr_output;
                    } else {
                        pool.release(sr_output);
                        fprintf(stderr, "[pipeline] AI lowlight failed on frame\n");
                    }
                }
                if (cfg.ai_anime_enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    if (ai::ai_anime(prev_frame.d_data, sr_output.d_data,
                                     work->width, work->height, stream)) {
                        pool.release(prev_frame); work = &sr_output;
                    } else {
                        pool.release(sr_output);
                        fprintf(stderr, "[pipeline] AI anime failed on frame\n");
                    }
                }
                if (cfg.ai_matting_enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    if (ai::ai_matting(prev_frame.d_data, sr_output.d_data,
                                       work->width, work->height, stream)) {
                        pool.release(prev_frame); work = &sr_output;
                    } else {
                        pool.release(sr_output);
                        fprintf(stderr, "[pipeline] AI matting failed on frame\n");
                    }
                }
                if (cfg.ai_face_enabled) {
                    ai::FaceBox face_boxes[8];
                    int nface = ai::ai_face(work->d_data, work->width, work->height,
                                            face_boxes, 8, stream);
                    if (nface > 0) {
                        prev_frame = *work;
                        sr_output = pool.acquire(work->width, work->height, work->fmt);
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                        for (int fi = 0; fi < nface; fi++) {
                            ai::launch_draw_rect(sr_output.d_data, work->width, work->height,
                                (int)face_boxes[fi].x1, (int)face_boxes[fi].y1,
                                (int)face_boxes[fi].x2, (int)face_boxes[fi].y2,
                                0, 255, 0, 2, stream);
                        }
                        pool.release(prev_frame); work = &sr_output;
                    }
                }
                if (cfg.ai_detect_enabled) {
                    ai::DetectBox det_boxes[16];
                    int ndet = ai::ai_detect(work->d_data, work->width, work->height,
                                             det_boxes, 16, stream);
                    if (ndet > 0) {
                        prev_frame = *work;
                        sr_output = pool.acquire(work->width, work->height, work->fmt);
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                        for (int di = 0; di < ndet; di++) {
                            uint8_t dc[3];
                            ai::ai_detect_color(det_boxes[di].cls, dc);
                            int bx1 = (int)det_boxes[di].x1, by1 = (int)det_boxes[di].y1;
                            int bx2 = (int)det_boxes[di].x2, by2 = (int)det_boxes[di].y2;
                            ai::launch_draw_rect(sr_output.d_data, work->width, work->height,
                                bx1, by1, bx2, by2, dc[0], dc[1], dc[2], 2, stream);
                            int ty = by1 - 12;
                            if (ty < 0) ty = by2 + 2;
                            if (ty < 0) ty = 0;
                            ai::launch_draw_text(sr_output.d_data, work->width, work->height,
                                bx1 < 0 ? 0 : bx1, ty,
                                ai::ai_detect_class_name(det_boxes[di].cls),
                                2, 255, 255, 255, dc[0], dc[1], dc[2], stream);
                        }
                        pool.release(prev_frame); work = &sr_output;
                    }
                }
                if (cfg.ai_hands_enabled) {
                    ai::HandJoints hands[2];
                    int nhands = ai::ai_hands(work->d_data, work->width, work->height,
                                              hands, 2, stream);
                    if (nhands > 0) {
                        prev_frame = *work;
                        sr_output = pool.acquire(work->width, work->height, work->fmt);
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                        float jr = (float)(work->width > work->height ? work->width : work->height) / 200.0f;
                        if (jr < 2.0f) jr = 2.0f;
                        for (int hi2 = 0; hi2 < nhands; hi2++) {
                            if (hands[hi2].score <= 0.5f) continue;
                            for (int ji = 0; ji < 21; ji++)
                                ai::launch_draw_circle(sr_output.d_data, work->width, work->height,
                                    hands[hi2].x[ji], hands[hi2].y[ji], jr, 0, 255, 255, stream);
                        }
                        pool.release(prev_frame); work = &sr_output;
                    }
                }
                if (cfg.ai_gaze_enabled) {
                    ai::GazeEye eyes[2];
                    int neyes = ai::ai_gaze(work->d_data, work->width, work->height,
                                            eyes, 2, stream);
                    if (neyes > 0) {
                        prev_frame = *work;
                        sr_output = pool.acquire(work->width, work->height, work->fmt);
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                        for (int ei = 0; ei < neyes; ei++) {
                            if (eyes[ei].score <= 0.5f) continue;
                            ai::launch_draw_circle(sr_output.d_data, work->width, work->height,
                                eyes[ei].cx, eyes[ei].cy, 4.0f, 0, 255, 0, stream);
                            ai::launch_draw_circle(sr_output.d_data, work->width, work->height,
                                eyes[ei].cx + eyes[ei].dx * 24.0f,
                                eyes[ei].cy + eyes[ei].dy * 24.0f, 2.0f, 255, 255, 0, stream);
                        }
                        pool.release(prev_frame); work = &sr_output;
                    }
                }
                if (cfg.ai_autoframe_enabled) {
                    ai::FaceBox abox[4];
                    int nbox = ai::ai_face(work->d_data, work->width, work->height,
                                           abox, 4, stream);
                    int bi = -1; float bs = 0.4f; // measured max ~0.5 on real faces
                    for (int i = 0; i < nbox; i++)
                        if (abox[i].score > bs) { bs = abox[i].score; bi = i; }
                    if (bi >= 0) {
                        float fw = abox[bi].x2 - abox[bi].x1;
                        float fh = abox[bi].y2 - abox[bi].y1;
                        float tw = fw * 2.0f, th = fh * 2.0f;
                        if (tw > (float)work->width) tw = (float)work->width;
                        if (th > (float)work->height) th = (float)work->height;
                        float tcx = (abox[bi].x1 + abox[bi].x2) * 0.5f;
                        float tcy = (abox[bi].y1 + abox[bi].y2) * 0.5f - 0.1f * th;
                        float tx = tcx - tw * 0.5f, ty = tcy - th * 0.5f;
                        if (!af_init || af_fw != work->width || af_fh != work->height) {
                            af_x = tx; af_y = ty; af_w = tw; af_h = th;
                            af_fw = work->width; af_fh = work->height; af_init = true;
                        } else {
                            float dx = tx - af_x, dy = ty - af_y;
                            float dw = tw - af_w, dh = th - af_h;
                            if (fabsf(dx) < 5) dx = 0; if (fabsf(dy) < 5) dy = 0;
                            if (fabsf(dw) < 4) dw = 0; if (fabsf(dh) < 4) dh = 0;
                            af_x += dx * 0.12f; af_y += dy * 0.12f;
                            af_w += dw * 0.12f; af_h += dh * 0.12f;
                        }
                        if (af_w > work->width) af_w = (float)work->width;
                        if (af_h > work->height) af_h = (float)work->height;
                        if (af_x < 0) af_x = 0; if (af_y < 0) af_y = 0;
                        if (af_x + af_w > work->width) af_x = (float)work->width - af_w;
                        if (af_y + af_h > work->height) af_y = (float)work->height - af_h;
                        uint32_t cw = ((uint32_t)af_w) & ~1u, ch = ((uint32_t)af_h) & ~1u;
                        if (cw >= 16 && ch >= 16) {
                            prev_frame = *work;
                            sr_output = pool.acquire(work->width, work->height, work->fmt);
                            Frame crop = pool.acquire(cw, ch, work->fmt);
                            filters::crop_rgb(prev_frame.d_data, crop.d_data,
                                work->width, work->height, cw, ch,
                                (int)af_x, (int)af_y, 3, stream);
                            filters::resize_bilinear(crop.d_data, sr_output.d_data,
                                cw, ch, work->width, work->height, 3, stream);
                            pool.release(crop);
                            pool.release(prev_frame); work = &sr_output;
                        }
                    }
                }
#endif
                if (cfg.hdr.enabled) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    filters::hdr_tone_map_rgb(prev_frame.d_data, sr_output.d_data,
                        work->width, work->height,
                        cfg.hdr.method, cfg.hdr.peak_nits, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.temporal_denoise.enabled && temporal_count > 0) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    Frame& prev_temp = temporal_buf[(temporal_idx - 1 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    filters::temporal_denoise_rgb(prev_temp.d_data, prev_frame.d_data,
                        sr_output.d_data, work->width, work->height,
                        cfg.temporal_denoise.strength, stream);
                    pool.release(prev_frame); work = &sr_output;
                }
                if (cfg.temporal_stab.enabled && temporal_count > 0) {
                    prev_frame = *work;
                    sr_output = pool.acquire(work->width, work->height, work->fmt);
                    Frame& prev_temp = temporal_buf[(temporal_idx - 1 + TEMPORAL_BUF_SIZE) % TEMPORAL_BUF_SIZE];
                    filters::temporal_stabilize_rgb(prev_temp.d_data, prev_frame.d_data,
                        sr_output.d_data, work->width, work->height,
                        d_motion_accum, cfg.temporal_stab.block_size,
                        cfg.temporal_stab.search_range, stream);
                    // Read motion vector and apply smoothed warp
                    int h_acc[3];
                    cudaMemcpy(h_acc, d_motion_accum, 3 * sizeof(int), cudaMemcpyDeviceToHost);
                    int raw_dx = h_acc[2] > 0 ? h_acc[0] / h_acc[2] : 0;
                    int raw_dy = h_acc[2] > 0 ? h_acc[1] / h_acc[2] : 0;
                    float sf = cfg.temporal_stab.smooth_factor;
                    int smooth_dx = (int)(sf * prev_warp_dx + (1.0f - sf) * raw_dx);
                    int smooth_dy = (int)(sf * prev_warp_dy + (1.0f - sf) * raw_dy);
                    prev_warp_dx = smooth_dx;
                    prev_warp_dy = smooth_dy;
                    // Warp the CURRENT frame (prev_frame). sr_output was never
                    // written by the estimator (it only fills d_motion_accum),
                    // so warping from it produced garbage/black frames.
                    if (smooth_dx != 0 || smooth_dy != 0) {
                        filters::warp_translate_rgb(prev_frame.d_data, sr_output.d_data,
                            work->width, work->height, smooth_dx, smooth_dy, stream);
                    } else {
                        KAGEROU_CUDA_CHECK(cudaMemcpyAsync(sr_output.d_data, prev_frame.d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream));
                    }
                    pool.release(prev_frame); work = &sr_output;
                }

                // Store frame in temporal ring buffer (also feeds AI denoise)
                if (cfg.temporal_denoise.enabled || cfg.temporal_stab.enabled ||
                    cfg.ai_denoise_enabled) {
                    Frame& slot = temporal_buf[temporal_idx];
                    if (slot.d_data && slot.width == work->width && slot.height == work->height) {
                        cudaMemcpyAsync(slot.d_data, work->d_data,
                            work->width * work->height * 3, cudaMemcpyDeviceToDevice, stream);
                    }
                    temporal_idx = (temporal_idx + 1) % TEMPORAL_BUF_SIZE;
                    if (temporal_count < TEMPORAL_BUF_SIZE) temporal_count++;
                }

                // Convert RGB -> NV12 for encoder
                nv12_out = pool.acquire(work->width, work->height, PixelFormat::kNV12);
                if (cfg.color_range == ColorRange::kLimited)
                    filters::rgb_to_nv12_limited(work->d_data, nv12_out.d_data,
                                                 work->width, work->height, stream);
                else
                    filters::rgb_to_nv12(work->d_data, nv12_out.d_data,
                                         work->width, work->height, stream);
                pool.release(*work);
                work = &nv12_out;
            }
        }

        // Ensure all filter kernels complete before encoder reads frame data
        cudaStreamSynchronize(stream);

        if (cfg.verbose) {
            fprintf(stderr, "[pipeline] encode: work=%p d_data=%p w=%u h=%u fmt=%d\n",
                    (void*)work, (void*)work->d_data, work->width, work->height, (int)work->fmt);
        }

        Error e = encoder.encode(*work, encoded_out);

        // Release all intermediate buffers back to pool (must happen even on error)
        pool.release(to_rgb);
        pool.release(filtered);
        pool.release(scaled);
        pool.release(sr_output);
        pool.release(nv12_out);

        if (e != Error::kOk) return e;

        return Error::kOk;
    }

    // Process a raw NV12 frame directly (bypasses decoder).
    // Used for camera capture, RTSP streams, or any source that provides raw frames.
    Error process_raw_frame(const uint8_t* nv12_data, uint32_t width, uint32_t height,
                            std::vector<uint8_t>& encoded_out) {
        if (!initialized) return Error::kFilterError;
        encoded_out.clear();

        auto t0 = std::chrono::high_resolution_clock::now();

        // Auto-detect: init encoder on first frame
        if (auto_detect_resolution && !encoder_initialized) {
            Error ie = init_encoder(width, height);
            if (ie != Error::kOk) return ie;
        }

        // Create a Frame wrapping the input data (non-owning — caller manages memory)
        Frame decoded;
        decoded.d_data = const_cast<uint8_t*>(nv12_data);
        decoded.width = width;
        decoded.height = height;
        decoded.stride = width;
        decoded.fmt = PixelFormat::kNV12;
        decoded.owned = false;

        stats.frames_decoded++;
        stats.nalus_fed++;

        std::vector<uint8_t> frame_encoded;
        Error e = process_single_frame(decoded, frame_encoded);
        if (e == Error::kOk && !frame_encoded.empty()) {
            encoded_out.insert(encoded_out.end(), frame_encoded.begin(), frame_encoded.end());
            stats.frames_encoded++;
        }

        // Don't free — we don't own this data
        decoded.d_data = nullptr;

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.frames_processed++;
        stats.total_ms += ms;
        stats.fps = (stats.total_ms > 0) ?
            (double)stats.frames_processed * 1000.0 / stats.total_ms : 0.0;

        return Error::kOk;
    }

    Error process_frame_pair(const uint8_t* bs_a, size_t bs_a_size,
                             const uint8_t* bs_b, size_t bs_b_size,
                             std::vector<std::vector<std::uint8_t>>& frames_out) {
        frames_out.clear();

        std::vector<uint8_t> frame_a, frame_b;
        Error e = process_frame(bs_a, bs_a_size, frame_a);
        if (e != Error::kOk) return e;
        e = process_frame(bs_b, bs_b_size, frame_b);
        if (e != Error::kOk) return e;

        frames_out.push_back(frame_a);

        if (cfg.frame_interp.enabled && cfg.encoder.fps > 0) {
            Frame decoded_a, decoded_b;
            decoder.decode(bs_a, bs_a_size, decoded_a);
            decoder.decode(bs_b, bs_b_size, decoded_b);

            if (decoded_a.d_data && decoded_b.d_data) {
                Frame blended;
                alloc_frame_gpu(blended, decoded_a.width, decoded_a.height, decoded_a.fmt);

                for (uint32_t i = 1; i <= 1; ++i) {
                    float alpha = (float)i / 2.0f;
                    filters::frame_blend(decoded_a.d_data, decoded_b.d_data,
                                         blended.d_data,
                                         decoded_a.width, decoded_a.height,
                                         3, alpha, stream);

                    std::vector<uint8_t> interp_frame;
                    encoder.encode(blended, interp_frame);
                    frames_out.push_back(interp_frame);
                }

                blended.free_gpu();
            }

            // Free decoded frames regardless of whether both succeeded
            if (decoded_a.d_data) decoded_a.free_gpu();
            if (decoded_b.d_data) decoded_b.free_gpu();
        }

        frames_out.push_back(frame_b);
        return Error::kOk;
    }

    // Software decode path: iterate over all pre-decoded frames, apply filters, encode.
    // Used for AV1/VP9 input where CUVID parser doesn't work with our extraction.
    Error process_sw(std::vector<uint8_t>& encoded_out) {
        encoded_out.clear();
        if (!decoder.sw_decode_mode) return Error::kDecodeError;

        auto t0 = std::chrono::high_resolution_clock::now();
        int frame_count = 0;

        while (true) {
            Frame decoded;
            Error e = decoder.decode(nullptr, 0, decoded);
            if (e != Error::kOk) break;
            if (!decoded.d_data) break;

            std::vector<uint8_t> frame_encoded;
            e = process_single_frame(decoded, frame_encoded);
            decoded.free_gpu();
            if (!frame_encoded.empty())
                encoded_out.insert(encoded_out.end(), frame_encoded.begin(), frame_encoded.end());
            frame_count++;
            if (frame_count % 30 == 0)
                printf("\r  processed %d frames   ", frame_count);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("\r  processed %d frames in %.1f s (%.1f fps)     \n",
               frame_count, elapsed / 1000.0, frame_count * 1000.0 / std::max(elapsed, 1.0));
        stats.frames_processed += frame_count;
        stats.total_ms += elapsed;
        stats.fps = (stats.total_ms > 0) ?
            (double)stats.frames_processed * 1000.0 / stats.total_ms : 0.0;
        return Error::kOk;
    }

    // Flush decoder DPB + encoder buffer. Returns all remaining frames.
    Error flush(std::vector<uint8_t>& final_data) {
        final_data.clear();

        // Flush decoder DPB — route through process_single_frame for filter pipeline
        for (int i = 0; i < 512; ++i) {
            Frame remaining;
            decoder.flush(remaining);
            if (!remaining.d_data) break;

            stats.frames_flushed++;
            std::vector<uint8_t> encoded;
            Error e = process_single_frame(remaining, encoded);
            remaining.free_gpu();
            if (e != Error::kOk) continue;
            if (!encoded.empty())
                final_data.insert(final_data.end(), encoded.begin(), encoded.end());
        }

        // Flush encoder buffer
        if (encoder_initialized) {
            std::vector<std::vector<uint8_t>> remaining;
            Error e = encoder.flush(remaining);
            for (auto& f : remaining)
                final_data.insert(final_data.end(), f.begin(), f.end());
        }

        return Error::kOk;
    }

    void destroy() {
        printf("[pipeline] destroy: start\n");
        destroy_graph();
        printf("[pipeline] destroy: graph destroyed\n");
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); stream = nullptr; }
        printf("[pipeline] destroy: stream destroyed\n");
        pool.free_all();
        printf("[pipeline] destroy: pool freed\n");
        if (cached_lut) { cudaFree(cached_lut); cached_lut = nullptr; cached_preset = -1; }
        for (int i = 0; i < TEMPORAL_BUF_SIZE; i++) temporal_buf[i].free_gpu();
        flow_prev_frame.free_gpu();
        if (d_ai_depth) { cudaFree(d_ai_depth); d_ai_depth = nullptr; }
        if (d_motion_accum) { cudaFree(d_motion_accum); d_motion_accum = nullptr; }
        printf("[pipeline] destroy: GPU buffers freed\n");
        if (encoder_initialized) encoder.destroy();
        printf("[pipeline] destroy: encoder destroyed\n");
        decoder.destroy();
        printf("[pipeline] destroy: decoder destroyed\n");
#ifdef KAGEROU_USE_ONNX
        ai::AiInference::get().shutdown();
        printf("[pipeline] destroy: AI engine shutdown\n");
#endif
        initialized = false;
        encoder_initialized = false;
        printf("[pipeline] destroy: done\n");
    }

    void print_stats() const {
        printf("[pipeline] nalus_fed=%u decoded=%u encoded=%u flushed=%u total=%u (%.1f ms, %.1f fps)\n",
               stats.nalus_fed, stats.frames_decoded, stats.frames_encoded,
               stats.frames_flushed, stats.frames_processed, stats.total_ms, stats.fps);
    }
};

} // namespace kagerou
