// Kagerou SDK — NVDEC hardware video decoder.
// Full integration with NVIDIA Video Codec SDK.
//
// Uses CUVID video parser to properly feed NALUs to the hardware decoder.
// The parser handles sequence detection, decoder creation, and frame scheduling.

#include "kagerou/common.h"
#include "kagerou/config.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _WIN32
  #include <windows.h>
#endif

#if !defined(KAGEROU_NO_VCODEC_SDK)
  #include "cuviddec.h"
  #include "nvcuvid.h"
#endif

namespace kagerou {

struct Decoder {
#if !defined(KAGEROU_NO_VCODEC_SDK)
    CUcontext           cu_ctx         = nullptr;
    CUvideodecoder      nvdec_decoder  = nullptr;
    CUvideoctxlock      ctx_lock       = nullptr;
    CUvideoparser       nvdec_parser   = nullptr;
#endif
    DecoderConfig       cfg;
    uint32_t            width          = 0;
    uint32_t            height         = 0;
    uint32_t            coded_width    = 0;
    uint32_t            coded_height   = 0;
    int32_t             display_left   = 0;
    int32_t             display_top    = 0;
    bool                initialized    = false;
    VideoCodec          codec          = VideoCodec::kH264;
    // Queue for ALL display callbacks — a single cuvidParseVideoData() call can
    // trigger multiple display callbacks (B-frames, display delay). We queue them
    // all and pop in CUVID's display order (FIFO). CUVID with ulMaxDisplayDelay>0
    // fires callbacks in the correct display order — no sorting needed.
#if !defined(KAGEROU_NO_VCODEC_SDK)
    struct DisplayEntry {
        int picture_index;
        unsigned long long timestamp;
    };
    std::vector<DisplayEntry> display_queue;
#endif
    uint32_t            decode_count   = 0;
    uint32_t            display_count  = 0;
    uint32_t            frame_timestamp = 0; // monotonic PTS fed to CUVID parser

    // Software decode path (for AV1, VP9, etc. where CUVID fails or isn't available)
    bool                sw_decode_mode = false;
    std::vector<uint8_t> sw_frame_buf;     // concatenated NV12 frames
    uint32_t            sw_frame_size     = 0; // bytes per frame (Y + UV)
    uint32_t            sw_frame_idx      = 0; // next frame to return

#if !defined(KAGEROU_NO_VCODEC_SDK)
    static cudaVideoCodec to_cuda_codec(VideoCodec c) {
        switch (c) {
            case VideoCodec::kH264: return cudaVideoCodec_H264;
            case VideoCodec::kH265: return cudaVideoCodec_HEVC;
            case VideoCodec::kVP9:  return cudaVideoCodec_VP9;
            case VideoCodec::kAV1:  return cudaVideoCodec_AV1;
            default:                return cudaVideoCodec_H264;
        }
    }

    static int CUDAAPI handle_video_sequence(void* user_data, CUVIDEOFORMAT* fmt) {
        Decoder* dec = (Decoder*)user_data;
        if (!dec) return 0;

        printf("[NVDEC] sequence callback: %ux%u codec=%d chroma=%d display_area=[%d,%d,%d,%d]\n",
               fmt->coded_width, fmt->coded_height, (int)fmt->codec, (int)fmt->chroma_format,
               fmt->display_area.left, fmt->display_area.top,
               fmt->display_area.right, fmt->display_area.bottom);

        dec->coded_width  = fmt->coded_width;
        dec->coded_height = fmt->coded_height;
        // Use display dimensions if available, else coded
        dec->display_left = fmt->display_area.left;
        dec->display_top  = fmt->display_area.top;
        dec->width  = fmt->display_area.right  - fmt->display_area.left;
        dec->height = fmt->display_area.bottom - fmt->display_area.top;
        if (dec->width == 0 || dec->height == 0) {
            dec->width  = fmt->coded_width;
            dec->height = fmt->coded_height;
        }

        if (dec->nvdec_decoder) {
            cuvidDestroyDecoder(dec->nvdec_decoder);
            dec->nvdec_decoder = nullptr;
        }

        CUcontext prev_ctx = nullptr;
        cuCtxPushCurrent(dec->cu_ctx);

        CUVIDDECODECREATEINFO dcinfo = {};
        dcinfo.CodecType = fmt->codec;
        dcinfo.ChromaFormat = fmt->chroma_format;
        dcinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
        dcinfo.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
        dcinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
        dcinfo.ulNumOutputSurfaces = 4;
        // Bump decode surfaces well beyond min to give CUVID enough DPB slots
        // for display-delay reordering (ulMaxDisplayDelay=16). H.264 HP needs
        // up to 16 ref surfaces; add display delay on top.
        dcinfo.ulNumDecodeSurfaces = fmt->min_num_decode_surfaces + 16;
        if (dcinfo.ulNumDecodeSurfaces < 16) dcinfo.ulNumDecodeSurfaces = 16;
        dcinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        dcinfo.ulWidth = fmt->coded_width;
        dcinfo.ulHeight = fmt->coded_height;
        dcinfo.ulTargetWidth = fmt->coded_width;
        dcinfo.ulTargetHeight = fmt->coded_height;

        CUresult res = cuvidCreateDecoder(&dec->nvdec_decoder, &dcinfo);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuvidCreateDecoder failed: %d\n", res);
            return 0;
        }
        printf("[NVDEC] decoder created: %ux%u (codec=%d)\n",
               dec->width, dec->height, (int)fmt->codec);
        return 1;
    }

    static int CUDAAPI handle_picture_decode(void* user_data, CUVIDPICPARAMS* pic_params) {
        Decoder* dec = (Decoder*)user_data;
        if (!dec || !dec->nvdec_decoder) return 0;

        CUcontext prev_ctx = nullptr;
        cuCtxPushCurrent(dec->cu_ctx);

        CUresult res = cuvidDecodePicture(dec->nvdec_decoder, pic_params);

        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuvidDecodePicture failed: %d\n", res);
            return 0;
        }
        dec->decode_count++;
        return 1;
    }

    static int CUDAAPI handle_picture_display(void* user_data, CUVIDPARSERDISPINFO* disp) {
        Decoder* dec = (Decoder*)user_data;
        if (!dec || !disp) return 0;

        dec->display_count++;
        DisplayEntry entry;
        entry.picture_index = disp->picture_index;
        entry.timestamp = (unsigned long long)disp->timestamp;
        dec->display_queue.push_back(entry);
        return 1;
    }
#endif

    Error init(const DecoderConfig& c) {
        cfg = c;
        codec = c.codec;
        width  = c.max_width;
        height = c.max_height;

#if !defined(KAGEROU_NO_VCODEC_SDK)
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuInit failed: %d\n", res);
            return Error::kCudaError;
        }

        CUdevice dev;
        res = cuDeviceGet(&dev, 0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuDeviceGet failed: %d\n", res);
            return Error::kCudaError;
        }

        res = cuCtxCreate(&cu_ctx, 0, dev);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuCtxCreate failed: %d\n", res);
            return Error::kCudaError;
        }

        res = cuvidCtxLockCreate(&ctx_lock, cu_ctx);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuvidCtxLockCreate failed: %d\n", res);
            return Error::kCudaError;
        }

        CUVIDPARSERPARAMS parser_params = {};
        parser_params.CodecType = to_cuda_codec(codec);
        parser_params.ulMaxNumDecodeSurfaces = 16;
        parser_params.ulClockRate = 1000;
        parser_params.ulMaxDisplayDelay = 16; // enough buffer for B-frame reordering (broadcast standard)
        parser_params.pUserData = this;
        parser_params.pfnSequenceCallback = handle_video_sequence;
        parser_params.pfnDecodePicture = handle_picture_decode;
        parser_params.pfnDisplayPicture = handle_picture_display;

        res = cuvidCreateVideoParser(&nvdec_parser, &parser_params);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuvidCreateVideoParser failed: %d\n", res);
            return Error::kCudaError;
        }

        printf("[NVDEC] initialized -- max %ux%u (HW decode via CUVID parser)\n", width, height);
#else
        printf("[NVDEC] initialized -- max %ux%u (stub mode, test patterns)\n", width, height);
#endif
        initialized = true;
        return Error::kOk;
    }

    // Software decode path: use ffmpeg to decode the entire file to raw NV12
    // frames, stored in sw_frame_buf. This bypasses CUVID for codecs like AV1
    // where CUVID parser OBU framing doesn't work correctly via our extraction.
    Error load_sw_frames(const char* input_path, uint32_t& out_w, uint32_t& out_h) {
        sw_decode_mode = true;

        // Always use ffprobe to get actual video dimensions (not max from config)
        {
            char probe_cmd[1024];
            char tmp_probe[512];
            snprintf(tmp_probe, sizeof(tmp_probe), "_kagerou_probe_%d.txt", (int)GetCurrentProcessId());
            snprintf(probe_cmd, sizeof(probe_cmd),
                "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 \"%s\" >\"%s\" 2>nul",
                input_path, tmp_probe);
            system(probe_cmd);
            FILE* pf = fopen(tmp_probe, "r");
            if (pf) {
                if (fscanf(pf, "%u,%u", &width, &height) != 2) {
                    fprintf(stderr, "[NVDEC] sw decode: ffprobe failed to parse dimensions\n");
                }
                fclose(pf);
            }
            remove(tmp_probe);
        }

        if (width == 0 || height == 0) {
            fprintf(stderr, "[NVDEC] sw decode: cannot determine dimensions\n");
            return Error::kDecodeError;
        }

        // Decode to raw NV12 using ffmpeg with hardware decode if available
        char tmp_raw[512];
        snprintf(tmp_raw, sizeof(tmp_raw), "_kagerou_sw_%d.raw", (int)GetCurrentProcessId());
        char decode_cmd[2048];

        // Try av1_cuvid first, fall back to software (libdav1d)
        snprintf(decode_cmd, sizeof(decode_cmd),
            "ffmpeg -y -c:v av1_cuvid -i \"%s\" -pix_fmt nv12 -f rawvideo \"%s\" 2>nul",
            input_path, tmp_raw);
        int ret = system(decode_cmd);
        if (ret != 0) {
            snprintf(decode_cmd, sizeof(decode_cmd),
                "ffmpeg -y -i \"%s\" -pix_fmt nv12 -f rawvideo \"%s\" 2>nul",
                input_path, tmp_raw);
            ret = system(decode_cmd);
        }

        if (ret != 0) {
            fprintf(stderr, "[NVDEC] sw decode: ffmpeg decode failed\n");
            return Error::kDecodeError;
        }

        // Read raw NV12 frames
        FILE* rawf = fopen(tmp_raw, "rb");
        if (!rawf) return Error::kFileError;
        fseek(rawf, 0, SEEK_END);
        long raw_size = ftell(rawf);
        fseek(rawf, 0, SEEK_SET);

        sw_frame_size = width * height * 3 / 2; // NV12: Y + UV
        if (sw_frame_size == 0 || raw_size % sw_frame_size != 0) {
            fprintf(stderr, "[NVDEC] sw decode: raw size %ld not divisible by frame size %u\n",
                    raw_size, sw_frame_size);
            fclose(rawf);
            remove(tmp_raw);
            return Error::kDecodeError;
        }

        sw_frame_buf.resize(raw_size);
        if (fread(sw_frame_buf.data(), 1, raw_size, rawf) != (size_t)raw_size) {
            fclose(rawf);
            remove(tmp_raw);
            return Error::kFileError;
        }
        fclose(rawf);
        remove(tmp_raw);

        out_w = width;
        out_h = height;
        sw_frame_idx = 0;
        printf("[NVDEC] software decode: %zu frames (%ux%u, NV12, %zu bytes/frame)\n",
               sw_frame_buf.size() / sw_frame_size, width, height, (size_t)sw_frame_size);
        return Error::kOk;
    }

    // Returns:
    //   kOk + output allocated = decoded a frame
    //   kOk + output not allocated = NALU consumed (SPS/PPS/SEI), no frame yet
    //   kDecodeError = parse or decode failure
    //
    // NOTE: A single cuvidParseVideoData() call can trigger multiple display
    // callbacks (B-frames becoming displayable). We queue them all and return
    // one per decode() call. Call decode() in a loop until it returns no frame.
    Error decode(const uint8_t* bitstream, size_t bs_size, Frame& output) {
        if (!initialized) return Error::kDecodeError;

        // Software decode path: return pre-decoded frames from buffer
        if (sw_decode_mode) {
            if (sw_frame_idx >= sw_frame_buf.size() / sw_frame_size) {
                output.d_data = nullptr;
                return Error::kOk;
            }
        alloc_frame_gpu(output, width, height, PixelFormat::kNV12);

            const uint8_t* frame_ptr = sw_frame_buf.data() + sw_frame_idx * sw_frame_size;
            size_t y_size = (size_t)width * height;
            size_t uv_size = (size_t)width * height / 2;
            cudaMemcpy(output.d_data, frame_ptr, y_size, cudaMemcpyHostToDevice);
            cudaMemcpy(output.d_data + y_size, frame_ptr + y_size, uv_size, cudaMemcpyHostToDevice);
            sw_frame_idx++;
            return Error::kOk;
        }

#if !defined(KAGEROU_NO_VCODEC_SDK)
        if (!nvdec_parser) return Error::kDecodeError;

        // If queue already has frames from a previous parse, pop the next one.
        // CUVID display callbacks fire in display order — no sorting needed.
        if (!display_queue.empty()) {
            DisplayEntry e = display_queue.front();
            display_queue.erase(display_queue.begin());
            return map_and_copy(e.picture_index, output);
        }

        // null/empty bitstream: only valid in software decode path.
        // In HW decode mode, nullptr means "drain queue" — never generate test patterns here.
        if (!bitstream || bs_size == 0) {
            output.d_data = nullptr;
            return Error::kOk;
        }

        // For H.264/H.265: prepend Annex-B start code if not already present.
        // For AV1: CUVID expects raw OBUs — do NOT add start codes.
        std::vector<uint8_t> packet_buf;
        const uint8_t* payload;
        unsigned long payload_size;
        if (codec == VideoCodec::kAV1) {
            // AV1: pass raw bitstream directly (OBUs)
            payload = bitstream;
            payload_size = (unsigned long)bs_size;
        } else if (bs_size >= 4 && bitstream[0] == 0 && bitstream[1] == 0 &&
            (bitstream[2] == 1 || (bitstream[2] == 0 && bitstream[3] == 1))) {
            payload = bitstream;
            payload_size = (unsigned long)bs_size;
        } else {
            packet_buf.resize(4 + bs_size);
            packet_buf[0] = 0; packet_buf[1] = 0;
            packet_buf[2] = 0; packet_buf[3] = 1;
            memcpy(packet_buf.data() + 4, bitstream, bs_size);
            payload = packet_buf.data();
            payload_size = (unsigned long)(4 + bs_size);
        }

        CUVIDSOURCEDATAPACKET packet = {};
        packet.payload = payload;
        packet.payload_size = payload_size;
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.timestamp = frame_timestamp++;

        CUresult res = cuvidParseVideoData(nvdec_parser, &packet);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[NVDEC] cuvidParseVideoData failed: %d (size=%zu)\n", res, bs_size);
            return Error::kDecodeError;
        }

        // Parse may have triggered display callbacks — pop in CUVID display order (FIFO)
        if (display_queue.empty()) {
            output.d_data = nullptr;
            return Error::kOk;
        }

        DisplayEntry e = display_queue.front();
        display_queue.erase(display_queue.begin());
        return map_and_copy(e.picture_index, output);
#else
        alloc_frame_gpu(output, width, height, PixelFormat::kNV12);

        std::vector<uint8_t> y_data(width * height);
        std::vector<uint8_t> uv_data(width * (height / 2), 128);

        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                y_data[y * width + x] = (uint8_t)((x + y) & 0xFF);

        KAGEROU_CUDA_CHECK(cudaMemcpy(output.d_data, y_data.data(),
                                      y_data.size(), cudaMemcpyHostToDevice));
        KAGEROU_CUDA_CHECK(cudaMemcpy(output.d_data + width * height,
                                      uv_data.data(), uv_data.size(),
                                      cudaMemcpyHostToDevice));

        return Error::kOk;
#endif
    }

    void destroy() {
#if !defined(KAGEROU_NO_VCODEC_SDK)
        if (nvdec_parser) {
            cuvidDestroyVideoParser(nvdec_parser);
            nvdec_parser = nullptr;
        }
        if (nvdec_decoder) {
            cuvidCtxLock(ctx_lock, 0);
            cuvidDestroyDecoder(nvdec_decoder);
            cuvidCtxUnlock(ctx_lock, 0);
            nvdec_decoder = nullptr;
        }
        if (ctx_lock) {
            cuvidCtxLockDestroy(ctx_lock);
            ctx_lock = nullptr;
        }
        if (cu_ctx) {
            cuCtxDestroy(cu_ctx);
            cu_ctx = nullptr;
        }
#endif
        initialized = false;
    }

    // Flush all remaining frames from DPB (decoded picture buffer).
    // A single null-packet flush triggers display callbacks for ALL buffered frames.
    // Call in a loop until it returns with output.d_data == nullptr.
    Error flush(Frame& output) {
        output.d_data = nullptr;
#if !defined(KAGEROU_NO_VCODEC_SDK)
        if (!nvdec_parser) return Error::kOk;

        // If queue has frames from a previous call, pop the next one (CUVID display order = FIFO)
        if (!display_queue.empty()) {
            DisplayEntry e = display_queue.front();
            display_queue.erase(display_queue.begin());
            return map_and_copy(e.picture_index, output);
        }

        // Send null payload with CUVID_PKT_ENDOFSTREAM to flush DPB.
        // NVIDIA SDK: "MUST be set with last packet. Parser triggers display
        // callback for all pending buffers in the display queue."
        CUVIDSOURCEDATAPACKET packet = {};
        packet.payload = nullptr;
        packet.payload_size = 0;
        packet.flags = CUVID_PKT_ENDOFSTREAM;
        packet.timestamp = 0;

        CUresult res = cuvidParseVideoData(nvdec_parser, &packet);

        printf("[NVDEC] flush: decoded=%u display=%u queue=%zu\n",
               decode_count, display_count, display_queue.size());

        if (res != CUDA_SUCCESS || display_queue.empty()) return Error::kOk;

        DisplayEntry fe = display_queue.front();
        display_queue.erase(display_queue.begin());
        return map_and_copy(fe.picture_index, output);
#else
        (void)output;
#endif
        return Error::kOk;
    }

    // Check if flush has more frames queued
    bool has_queued_frames() const {
#if !defined(KAGEROU_NO_VCODEC_SDK)
        return !display_queue.empty();
#else
        return false;
#endif
    }

private:
    // Map a CUVID frame and copy it to output with display_area offset applied
    Error map_and_copy(int pic_idx, Frame& output) {
        output.d_data = nullptr;
#if !defined(KAGEROU_NO_VCODEC_SDK)
        if (!nvdec_decoder || pic_idx < 0) return Error::kOk;

        CUVIDPROCPARAMS proc = {};
        proc.progressive_frame = 1;
        proc.output_stream = 0;

        cuvidCtxLock(ctx_lock, 0);
        cuCtxPushCurrent(cu_ctx);

        unsigned long long dev_ptr = 0;
        unsigned int pitch = 0;
        CUresult res = cuvidMapVideoFrame(nvdec_decoder, pic_idx, &dev_ptr, &pitch, &proc);
        if (res != CUDA_SUCCESS) {
            CUcontext popped;
            cuCtxPopCurrent(&popped);
            cuvidCtxUnlock(ctx_lock, 0);
            fprintf(stderr, "[NVDEC] cuvidMapVideoFrame failed: %d pic=%d\n", res, pic_idx);
            return Error::kDecodeError;
        }

        alloc_frame_gpu(output, width, height, PixelFormat::kNV12);

        const uint8_t* src_y = (const uint8_t*)dev_ptr + display_top * pitch + display_left;
        const uint8_t* src_uv = (const uint8_t*)dev_ptr + pitch * coded_height
                                + (display_top / 2) * pitch + display_left;

        KAGEROU_CUDA_CHECK(cudaMemcpy2D(
            output.d_data, width,
            src_y, pitch,
            width, height,
            cudaMemcpyDeviceToDevice));

        KAGEROU_CUDA_CHECK(cudaMemcpy2D(
            output.d_data + width * height, width,
            src_uv, pitch,
            width, height / 2,
            cudaMemcpyDeviceToDevice));

        cuvidUnmapVideoFrame(nvdec_decoder, dev_ptr);

        CUcontext popped;
        cuCtxPopCurrent(&popped);
        cuvidCtxUnlock(ctx_lock, 0);
#endif
        return Error::kOk;
    }
};

} // namespace kagerou
