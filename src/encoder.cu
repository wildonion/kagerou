// Kagerou SDK — NVENC hardware video encoder.
// Full integration with NVIDIA Video Codec SDK.
//
// Requires:
//   - NVIDIA Video Codec SDK headers (nvEncodeAPI.h)
//   - nvEncodeAPI64.dll loaded at runtime (LoadLibrary/GetProcAddress)
//   - CUDA runtime API

#include "kagerou/common.h"
#include "kagerou/config.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#if !defined(KAGEROU_NO_VCODEC_SDK)

#include <windows.h>
#include "nvEncodeAPI.h"
#include <cuda.h>

namespace kagerou {

static uint32_t nvenc_struct_ver(uint32_t struct_num, uint32_t major, uint32_t minor, bool ext = false) {
    uint32_t v = major | (minor << 24) | (struct_num << 16) | (0x7u << 28);
    if (ext) v |= (1u << 31);
    return v;
}

struct Encoder {
    HMODULE                     nvenc_dll        = nullptr;
    void*                       nvenc_encoder    = nullptr;
    NV_ENCODE_API_FUNCTION_LIST nvenc_fn         = {};
    NV_ENC_INITIALIZE_PARAMS    init_params      = {};
    NV_ENC_CONFIG               encode_config    = {};
    NV_ENC_OUTPUT_PTR           output_buffer    = nullptr;
    NV_ENC_INPUT_PTR            nvenc_input_buf  = nullptr;
    CUcontext                   nvenc_cu_ctx     = nullptr;
    bool                        owns_cu_ctx      = false;
    EncoderConfig               cfg;
    uint32_t                    frame_idx        = 0;
    uint32_t                    api_major        = 0;
    uint32_t                    api_minor        = 0;
    bool                        initialized      = false;
    uint32_t                    total_encoded    = 0;
    uint32_t                    total_submitted  = 0;

    // GPU-resident staging buffer for D2D copy into NVENC input buffer
    uint8_t*                    gpu_input_buf    = nullptr;
    size_t                      gpu_input_size   = 0;

    // Track whether NVENC has a buffered frame that needs draining
    bool                        has_buffered_frame = false;

    Error init(const EncoderConfig& c, CUcontext shared_ctx = nullptr) {
        cfg = c;

        nvenc_dll = LoadLibraryA("nvEncodeAPI64.dll");
        if (!nvenc_dll) nvenc_dll = LoadLibraryA("nvEncodeAPI.dll");
        if (!nvenc_dll) {
            fprintf(stderr, "[NVENC] cannot load nvEncodeAPI64.dll\n");
            return Error::kCudaError;
        }

        typedef NVENCSTATUS (NVENCAPI *PFNNVENCAPICREATE)(NV_ENCODE_API_FUNCTION_LIST*);
        auto pfnCreate = (PFNNVENCAPICREATE)GetProcAddress(nvenc_dll, "NvEncodeAPICreateInstance");
        if (!pfnCreate) {
            fprintf(stderr, "[NVENC] NvEncodeAPICreateInstance not found\n");
            return Error::kCudaError;
        }

        memset(&nvenc_fn, 0, sizeof(nvenc_fn));
        nvenc_fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        NVENCSTATUS st = pfnCreate(&nvenc_fn);
        if (st != NV_ENC_SUCCESS) {
            fprintf(stderr, "[NVENC] NvEncodeAPICreateInstance failed: %d\n", st);
            return Error::kEncodeError;
        }

        uint32_t max_ver = 0;
        typedef NVENCSTATUS (NVENCAPI *PFNNVENCGETMAXVERSION)(uint32_t*);
        auto pfnGetMaxVer = (PFNNVENCGETMAXVERSION)GetProcAddress(nvenc_dll, "NvEncodeAPIGetMaxSupportedVersion");
        if (pfnGetMaxVer) {
            st = pfnGetMaxVer(&max_ver);
            if (st == NV_ENC_SUCCESS) {
                uint32_t max_major = max_ver >> 4;
                uint32_t max_minor = max_ver & 0xF;
                printf("[NVENC] driver supports max API %u.%u (header=%u.%u)\n",
                       max_major, max_minor, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
            }
        }

        if (shared_ctx) {
            nvenc_cu_ctx = shared_ctx;
            owns_cu_ctx = false;
        } else {
            CUcontext cu_ctx = nullptr;
            CUdevice dev;
            cuInit(0);
            cuDeviceGet(&dev, 0);
            cuCtxCreate(&cu_ctx, 0, dev);
            nvenc_cu_ctx = cu_ctx;
            owns_cu_ctx = true;
        }

        if (cfg.codec == VideoCodec::kH264)
            init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
        else if (cfg.codec == VideoCodec::kH265)
            init_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
        else if (cfg.codec == VideoCodec::kAV1)
            init_params.encodeGUID = NV_ENC_CODEC_AV1_GUID;
        else {
            fprintf(stderr, "[NVENC] unsupported codec\n");
            return Error::kUnsupportedCodec;
        }
        // P4 quality preset (was P1 fastest). Takes/transcodes are quality-
        // bound, not latency-bound; P4 removes most CQP banding on gradients
        // and text at trivial GPU cost for <=1080p.
        init_params.presetGUID = NV_ENC_PRESET_P4_GUID;

        struct ApiVer { uint32_t major; uint32_t minor; };
        ApiVer versions[] = {
            { 12, 2 },
            { 12, 1 },
            { 11, 1 },
            { NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION },
        };
        if (max_ver) {
            versions[0] = { max_ver & 0xFF, (max_ver >> 24) & 0xFF };
        }

        NV_ENC_PRESET_CONFIG preset_cfg = {};
        bool got_preset = false;

        for (auto& v : versions) {
            if (nvenc_fn.nvEncOpenEncodeSessionEx) {
                NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sp = {};
                sp.version    = nvenc_struct_ver(1, v.major, v.minor, false);
                sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
                sp.device     = nvenc_cu_ctx;
                sp.apiVersion = v.major | (v.minor << 24);
                st = nvenc_fn.nvEncOpenEncodeSessionEx(&sp, &nvenc_encoder);
                if (st != NV_ENC_SUCCESS || !nvenc_encoder) continue;
            } else {
                continue;
            }

            api_major = v.major;
            api_minor = v.minor;
            printf("[NVENC] session opened -- API %u.%u  fn_list_ver=0x%08X\n", api_major, api_minor, nvenc_fn.version);

            printf("[NVENC] fn nvEncGetEncodePresetConfigEx=%p nvEncGetEncodePresetConfig=%p\n",
                   (void*)nvenc_fn.nvEncGetEncodePresetConfigEx,
                   (void*)nvenc_fn.nvEncGetEncodePresetConfig);

            for (int try_ext = 0; try_ext < 2 && !got_preset; try_ext++) {
                bool ext = (try_ext == 0);
                for (uint32_t snum = 5; snum >= 1 && !got_preset; snum--) {
                    memset(&preset_cfg, 0, sizeof(preset_cfg));
                    preset_cfg.version           = nvenc_struct_ver(snum, api_major, api_minor, ext);
                    preset_cfg.presetCfg.version = nvenc_struct_ver(9, api_major, api_minor, ext);
                    printf("[NVENC] trying preset: ver=0x%08X cfg_ver=0x%08X (snum=%u ext=%d)\n",
                           preset_cfg.version, preset_cfg.presetCfg.version, snum, ext);

                    st = nvenc_fn.nvEncGetEncodePresetConfigEx(nvenc_encoder,
                        init_params.encodeGUID, init_params.presetGUID,
                        NV_ENC_TUNING_INFO_LOW_LATENCY, &preset_cfg);
                    if (st == NV_ENC_SUCCESS) {
                        printf("[NVENC] preset config OK (Ex) -- struct_ver=%u ext=%d\n", snum, ext);
                        got_preset = true;
                    } else {
                        printf("[NVENC] -> err %d\n", st);
                    }
                }
            }
            if (!got_preset && nvenc_fn.nvEncGetEncodePresetConfig) {
                for (int try_ext = 0; try_ext < 2 && !got_preset; try_ext++) {
                    bool ext = (try_ext == 0);
                    for (uint32_t snum = 5; snum >= 1 && !got_preset; snum--) {
                        memset(&preset_cfg, 0, sizeof(preset_cfg));
                        preset_cfg.version           = nvenc_struct_ver(snum, api_major, api_minor, ext);
                        preset_cfg.presetCfg.version = nvenc_struct_ver(9, api_major, api_minor, ext);
                        st = nvenc_fn.nvEncGetEncodePresetConfig(nvenc_encoder,
                            init_params.encodeGUID, init_params.presetGUID, &preset_cfg);
                        if (st == NV_ENC_SUCCESS) {
                            printf("[NVENC] preset config OK (legacy) -- struct_ver=%u ext=%d\n", snum, ext);
                            got_preset = true;
                        }
                    }
                }
            }

            if (got_preset) break;

            printf("[NVENC] API %u.%u failed all preset configs, trying next...\n", api_major, api_minor);
            nvenc_fn.nvEncDestroyEncoder(nvenc_encoder);
            nvenc_encoder = nullptr;
        }

        if (!got_preset && nvenc_fn.nvEncOpenEncodeSession) {
            nvenc_encoder = nullptr;
            st = nvenc_fn.nvEncOpenEncodeSession((void*)nvenc_cu_ctx, 1, &nvenc_encoder);
            if (st == NV_ENC_SUCCESS && nvenc_encoder) {
                api_major = NVENCAPI_MAJOR_VERSION;
                api_minor = NVENCAPI_MINOR_VERSION;
                printf("[NVENC] session opened -- legacy\n");
                for (int try_ext = 0; try_ext < 2 && !got_preset; try_ext++) {
                    bool ext = (try_ext == 0);
                    for (uint32_t snum = 5; snum >= 1 && !got_preset; snum--) {
                        memset(&preset_cfg, 0, sizeof(preset_cfg));
                        preset_cfg.version           = nvenc_struct_ver(snum, api_major, api_minor, ext);
                        preset_cfg.presetCfg.version = nvenc_struct_ver(9, api_major, api_minor, ext);
                        st = nvenc_fn.nvEncGetEncodePresetConfig(nvenc_encoder,
                            init_params.encodeGUID, init_params.presetGUID, &preset_cfg);
                        if (st == NV_ENC_SUCCESS) {
                            printf("[NVENC] preset config OK (legacy, struct_ver=%u ext=%d)\n", snum, ext);
                            got_preset = true;
                        }
                    }
                }
            }
        }

        if (!got_preset || !nvenc_encoder) {
            fprintf(stderr, "[NVENC] all attempts failed -- falling back to stub\n");
            if (nvenc_encoder) { nvenc_fn.nvEncDestroyEncoder(nvenc_encoder); nvenc_encoder = nullptr; }
            FreeLibrary(nvenc_dll); nvenc_dll = nullptr;
            return Error::kEncodeError;
        } else {

        memcpy(&encode_config, &preset_cfg.presetCfg, sizeof(NV_ENC_CONFIG));
        encode_config.version = NV_ENC_CONFIG_VER;
        // Honor the requested keyframe interval (was: infinite GOP, which
        // makes output unseekable and starves segment/cut logic of IDRs).
        encode_config.gopLength        = cfg.gop_size ? cfg.gop_size : 60;

        // HEVC requires encoder-managed picture types (PTD=1). H.264 works with either.
        bool is_hevc = (cfg.codec == VideoCodec::kH265);
        encode_config.frameIntervalP   = 1;

        // Honor the requested rate control. CBR makes the bitrate flag real
        // (ideal for screen content: exact file sizes, quality where it
        // matters). CQP keeps consistent quality per frame for camera work.
        if (cfg.rc == RateControl::kCBR && cfg.bitrate_kbps > 0) {
            memset(&encode_config.rcParams, 0, sizeof(encode_config.rcParams));
            encode_config.rcParams.version         = NV_ENC_RC_PARAMS_VER;
            encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
            encode_config.rcParams.averageBitRate  = cfg.bitrate_kbps * 1000;
            encode_config.rcParams.maxBitRate      = cfg.bitrate_kbps * 1200;
            printf("[NVENC] rate control: CBR (%ukbps)\n", cfg.bitrate_kbps);
        } else if (!is_hevc) {
            memset(&encode_config.rcParams, 0, sizeof(encode_config.rcParams));
            encode_config.rcParams.version           = NV_ENC_RC_PARAMS_VER;
            encode_config.rcParams.rateControlMode   = NV_ENC_PARAMS_RC_CONSTQP;
            encode_config.rcParams.constQP.qpInterP  = cfg.qp;
            encode_config.rcParams.constQP.qpIntra   = cfg.qp;
            encode_config.rcParams.constQP.qpInterB  = cfg.qp;
            printf("[NVENC] rate control: CQP (qp=%u)\n", cfg.qp);
        } else {
            // HEVC: use preset's rate control (don't override rcParams)
            printf("[NVENC] rate control: preset default (HEVC)\n");
        }

        memset(&init_params, 0, sizeof(init_params));
        init_params.version      = NV_ENC_INITIALIZE_PARAMS_VER;
        init_params.enableEncodeAsync = 0;
        init_params.tuningInfo   = NV_ENC_TUNING_INFO_LOW_LATENCY;
        init_params.enablePTD    = is_hevc ? 1 : 0;  // HEVC needs encoder-managed PTD
        init_params.encodeGUID   = cfg.codec == VideoCodec::kH264 ? NV_ENC_CODEC_H264_GUID :
                                   cfg.codec == VideoCodec::kH265 ? NV_ENC_CODEC_HEVC_GUID :
                                   NV_ENC_CODEC_AV1_GUID;
        init_params.presetGUID   = NV_ENC_PRESET_P1_GUID;
        init_params.encodeWidth  = cfg.width;
        init_params.encodeHeight = cfg.height;
        init_params.darWidth     = cfg.width;
        init_params.darHeight    = cfg.height;
        init_params.frameRateNum = cfg.fps;
        init_params.frameRateDen = 1;
        init_params.encodeConfig = &encode_config;
        init_params.maxEncodeWidth  = cfg.width;
        init_params.maxEncodeHeight = cfg.height;

        st = nvenc_fn.nvEncInitializeEncoder(nvenc_encoder, &init_params);
        if (st != NV_ENC_SUCCESS) {
            fprintf(stderr, "[NVENC] nvEncInitializeEncoder failed: %d\n", st);
            return Error::kEncodeError;
        }

        // Always allocate GPU-resident input buffer for zero-copy encoding
        gpu_input_size = (size_t)cfg.width * cfg.height * 3 / 2;
        cudaError_t cu_err = cudaMalloc(&gpu_input_buf, gpu_input_size);
        if (cu_err != cudaSuccess) {
            fprintf(stderr, "[NVENC] cudaMalloc for input buffer failed: %s\n",
                    cudaGetErrorName(cu_err));
            return Error::kCudaError;
        }
        printf("[NVENC] zero-copy input allocated (%zu MB)\n", gpu_input_size >> 20);

        // Create NVENC-owned input buffer (avoids NvEncRegisterResource which
        // conflicts with ORT CUDA EP on some drivers)
        {
            NV_ENC_CREATE_INPUT_BUFFER ib = {};
            ib.version  = NV_ENC_CREATE_INPUT_BUFFER_VER;
            ib.width    = cfg.width;
            ib.height   = cfg.height;
            ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
            NVENCSTATUS ib_st = nvenc_fn.nvEncCreateInputBuffer(nvenc_encoder, &ib);
            if (ib_st != NV_ENC_SUCCESS) {
                fprintf(stderr, "[NVENC] NvEncCreateInputBuffer failed: %d\n", ib_st);
                return Error::kEncodeError;
            }
            nvenc_input_buf = ib.inputBuffer;
        }

        NV_ENC_CREATE_BITSTREAM_BUFFER bs = {};
        bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        st = nvenc_fn.nvEncCreateBitstreamBuffer(nvenc_encoder, &bs);
        if (st != NV_ENC_SUCCESS) {
            fprintf(stderr, "[NVENC] NvEncCreateBitstreamBuffer failed: %d\n", st);
            return Error::kEncodeError;
        }
        output_buffer = bs.bitstreamBuffer;

        printf("[NVENC] initialized -- %s %ux%u@%ufps %ukbps\n",
               cfg.codec == VideoCodec::kH264 ? "H.264" :
               cfg.codec == VideoCodec::kH265 ? "H.265" : "AV1",
               cfg.width, cfg.height, cfg.fps, cfg.bitrate_kbps);

        initialized = true;
        has_buffered_frame = false;
        frame_idx = 0;
        return Error::kOk;
        }
    }

    // Read back a single bitstream from the encoder output buffer.
    // Returns true if a frame was read, false if no bitstream available.
    bool drain_one_bitstream(std::vector<uint8_t>& bitstream) {
        if (!nvenc_encoder) return false;

        NV_ENC_LOCK_BITSTREAM lock = {};
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = output_buffer;
        NVENCSTATUS st = nvenc_fn.nvEncLockBitstream(nvenc_encoder, &lock);
        if (st != NV_ENC_SUCCESS) return false;

        if (lock.bitstreamSizeInBytes > 0) {
            bitstream.resize(lock.bitstreamSizeInBytes);
            memcpy(bitstream.data(), lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes);
        } else {
            bitstream.clear();
        }
        nvenc_fn.nvEncUnlockBitstream(nvenc_encoder, output_buffer);
        return true;
    }

    // Submit one frame for encoding and read back the previous frame's bitstream.
    // Pure GPU zero-copy path: D2D copy into NVENC input buffer, lock, copy, encode.
    Error encode(const Frame& input, std::vector<uint8_t>& bitstream) {
        bitstream.clear();
        if (!initialized) return Error::kEncodeError;

        if (!nvenc_encoder) {
            uint32_t frame_size = input.plane_size[0] + input.plane_size[1];
            bitstream.resize(frame_size);
            KAGEROU_CUDA_CHECK(cudaMemcpy(bitstream.data(), input.d_data,
                                          frame_size, cudaMemcpyDeviceToHost));
            frame_idx++;
            return Error::kOk;
        }

        uint32_t src_w = input.width;
        uint32_t src_h = input.height;
        uint32_t src_pitch = input.stride ? input.stride : src_w;
        uint32_t dst_w = cfg.width;
        uint32_t dst_h = cfg.height;

        // D2D copy from filter output into our staging GPU buffer
        if (src_w == dst_w && src_h == dst_h && src_pitch == dst_w) {
            size_t frame_bytes = (size_t)dst_w * dst_h * 3 / 2;
            if (frame_bytes > gpu_input_size) return Error::kEncodeError;
            if (!input.d_data) return Error::kEncodeError;
            KAGEROU_CUDA_CHECK(cudaMemcpy(gpu_input_buf, input.d_data,
                                          frame_bytes, cudaMemcpyDeviceToDevice));
        } else {
            uint32_t copy_w = (src_w < dst_w) ? src_w : dst_w;
            uint32_t copy_h = (src_h < dst_h) ? src_h : dst_h;
            KAGEROU_CUDA_CHECK(cudaMemcpy2D(
                gpu_input_buf, dst_w,
                input.d_data, src_pitch,
                copy_w, copy_h,
                cudaMemcpyDeviceToDevice));
            const uint8_t* src_uv = (const uint8_t*)input.d_data + (size_t)src_pitch * src_h;
            uint8_t* dst_uv = gpu_input_buf + (size_t)dst_w * dst_h;
            KAGEROU_CUDA_CHECK(cudaMemcpy2D(
                dst_uv, dst_w,
                src_uv, src_pitch,
                copy_w, copy_h / 2,
                cudaMemcpyDeviceToDevice));
        }

        // Lock NVENC input buffer and GPU-memcpy into it
        NV_ENC_LOCK_INPUT_BUFFER lock = {};
        lock.version    = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lock.inputBuffer = nvenc_input_buf;
        NVENCSTATUS lst = nvenc_fn.nvEncLockInputBuffer(nvenc_encoder, &lock);
        if (lst != NV_ENC_SUCCESS) {
            fprintf(stderr, "[NVENC] NvEncLockInputBuffer failed: %d\n", lst);
            return Error::kEncodeError;
        }

        // GPU→GPU: pitch-corrected copy from our staging buffer into NVENC's locked buffer
        if (src_w == dst_w && src_h == dst_h) {
            KAGEROU_CUDA_CHECK(cudaMemcpy2D(
                lock.bufferDataPtr, lock.pitch,
                gpu_input_buf, dst_w,
                dst_w, dst_h * 3 / 2,
                cudaMemcpyDeviceToDevice));
        } else {
            uint32_t copy_w = (src_w < dst_w) ? src_w : dst_w;
            uint32_t copy_h = (src_h < dst_h) ? src_h : dst_h;
            KAGEROU_CUDA_CHECK(cudaMemcpy2D(
                lock.bufferDataPtr, lock.pitch,
                gpu_input_buf, dst_w,
                copy_w, copy_h,
                cudaMemcpyDeviceToDevice));
            KAGEROU_CUDA_CHECK(cudaMemcpy2D(
                (uint8_t*)lock.bufferDataPtr + lock.pitch * dst_h, lock.pitch,
                gpu_input_buf + (size_t)dst_w * dst_h, dst_w,
                copy_w, copy_h / 2,
                cudaMemcpyDeviceToDevice));
        }

        nvenc_fn.nvEncUnlockInputBuffer(nvenc_encoder, nvenc_input_buf);

        // If NVENC has a buffered frame from the previous call, read it
        if (has_buffered_frame) {
            drain_one_bitstream(bitstream);
            has_buffered_frame = false;
        }

        NV_ENC_PIC_PARAMS pic = {};
        pic.version         = NV_ENC_PIC_PARAMS_VER;
        pic.inputWidth      = dst_w;
        pic.inputHeight     = dst_h;
        pic.inputPitch      = lock.pitch;
        pic.bufferFmt       = NV_ENC_BUFFER_FORMAT_NV12;
        pic.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;
        pic.pictureType     = (frame_idx == 0) ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_P;
        pic.frameIdx        = frame_idx;
        pic.inputTimeStamp  = frame_idx;
        pic.inputBuffer     = nvenc_input_buf;
        pic.outputBitstream = output_buffer;

        NVENCSTATUS st = nvenc_fn.nvEncEncodePicture(nvenc_encoder, &pic);

        if (st == NV_ENC_ERR_NEED_MORE_INPUT) {
            has_buffered_frame = true;
            frame_idx++;
            total_submitted++;
            return Error::kOk;
        }

        if (st != NV_ENC_SUCCESS) {
            fprintf(stderr, "[NVENC] NvEncEncodePicture failed: %d (frame %u)\n", st, frame_idx);
            return Error::kEncodeError;
        }

        drain_one_bitstream(bitstream);
        total_submitted++;
        total_encoded++;
        frame_idx++;
        return Error::kOk;
    }

    Error flush(std::vector<std::vector<std::uint8_t>>& frames) {
        frames.clear();
        if (!nvenc_encoder || !initialized) return Error::kOk;

        // If there's a buffered frame that was submitted but not yet drained,
        // read it before sending EOS
        if (has_buffered_frame) {
            std::vector<uint8_t> last_frame;
            if (drain_one_bitstream(last_frame) && !last_frame.empty())
                frames.push_back(std::move(last_frame));
            has_buffered_frame = false;
        }

        // Signal end-of-stream.
        // NOTE: Do NOT call nvEncLockBitstream after EOS — it blocks indefinitely
        // when there's no bitstream ready. With frameIntervalP=1 (no B-frames),
        // NVENC outputs bitstream immediately per encode() call, so there is
        // nothing left to drain after the has_buffered_frame check above.
        NV_ENC_PIC_PARAMS pic = {};
        pic.version        = NV_ENC_PIC_PARAMS_VER;
        pic.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        pic.inputTimeStamp = frame_idx;
        nvenc_fn.nvEncEncodePicture(nvenc_encoder, &pic);

        frame_idx = 0;
        return Error::kOk;
    }

    // Fetch the SPS/PPS (+VPS) sequence header straight from the encoder.
    // Independent of stream position: every segment file (rotation, post-CUT
    // resume) can be prefixed with it and stays independently decodable.
    // Must run on the same thread that owns the session.
    Error get_sequence_header(std::vector<uint8_t>& spspps) {
        spspps.clear();
        if (!nvenc_encoder || !initialized) return Error::kEncodeError;
        spspps.assign(512, 0);
        uint32_t out_size = 0;
        NV_ENC_SEQUENCE_PARAM_PAYLOAD sp = {};
        sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        sp.inBufferSize = (uint32_t)spspps.size();
        sp.spsppsBuffer = spspps.data();
        sp.outSPSPPSPayloadSize = &out_size;
        NVENCSTATUS st = nvenc_fn.nvEncGetSequenceParams(nvenc_encoder, &sp);
        if (st != NV_ENC_SUCCESS || out_size == 0 || out_size > spspps.size()) {
            spspps.clear();
            return Error::kEncodeError;
        }
        spspps.resize(out_size);
        return Error::kOk;
    }

    void destroy() {
        printf("[NVENC] stats: %u submitted, %u encoded, %u buffered at end\n",
               total_submitted, total_encoded, has_buffered_frame ? 1 : 0);
        if (gpu_input_buf) {
            cudaFree(gpu_input_buf);
            gpu_input_buf = nullptr;
            gpu_input_size = 0;
        }
        if (nvenc_encoder && nvenc_input_buf) {
            nvenc_fn.nvEncDestroyInputBuffer(nvenc_encoder, nvenc_input_buf);
        }
        if (nvenc_encoder && output_buffer)
            nvenc_fn.nvEncDestroyBitstreamBuffer(nvenc_encoder, output_buffer);
        if (nvenc_encoder)
            nvenc_fn.nvEncDestroyEncoder(nvenc_encoder);
        if (nvenc_dll) { FreeLibrary(nvenc_dll); nvenc_dll = nullptr; }
        if (owns_cu_ctx && nvenc_cu_ctx) { cuCtxDestroy(nvenc_cu_ctx); nvenc_cu_ctx = nullptr; }
        nvenc_encoder = nullptr;
        nvenc_input_buf = nullptr;
        output_buffer = nullptr;
        has_buffered_frame = false;
        initialized = false;
    }
};

} // namespace kagerou

#else

namespace kagerou {

struct Encoder {
    EncoderConfig  cfg;
    uint32_t       frame_idx = 0;
    bool           initialized = false;

    Error init(const EncoderConfig& c) {
        cfg = c;
        initialized = true;
        frame_idx = 0;
        printf("[encoder] init (stub mode) -- codec=%d %ux%u@%ufps %ukbps\n",
               (int)c.codec, c.width, c.height, c.fps, c.bitrate_kbps);
        return Error::kOk;
    }

    Error encode(const Frame& input, std::vector<uint8_t>& bitstream) {
        if (!initialized) return Error::kEncodeError;

        uint32_t frame_size = 0;
        switch (input.fmt) {
            case PixelFormat::kNV12:
                frame_size = input.plane_size[0] + input.plane_size[1];
                break;
            case PixelFormat::kRGB:
                frame_size = input.width * input.height * 3;
                break;
            default:
                frame_size = input.width * input.height * 3;
                break;
        }

        bitstream.resize(frame_size);
        KAGEROU_CUDA_CHECK(cudaMemcpy(bitstream.data(), input.d_data,
                                      frame_size, cudaMemcpyDeviceToHost));

        frame_idx++;
        return Error::kOk;
    }

    Error flush(std::vector<std::vector<std::uint8_t>>& frames) {
        frames.clear();
        frame_idx = 0;
        return Error::kOk;
    }

    // Stub has no NVENC session: no sequence header available.
    Error get_sequence_header(std::vector<uint8_t>& spspps) {
        spspps.clear();
        (void)spspps;
        return Error::kEncodeError;
    }

    void destroy() {
        initialized = false;
    }
};

} // namespace kagerou
#endif
