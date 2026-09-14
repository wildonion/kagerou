
<p align="center">
    <img src="https://github.com/wildonion/elyon/blob/kagerou/src/HardLab/Kagerou/kagerou.png">
</p>


# Kagerou (熱叡) — GPU Video Transcoding SDK

A CUDA C++ SDK for real-time video/image transcoding with hardware-accelerated decode/encode and custom GPU filter pipeline. Built for the streaming, broadcast, and media processing industry.

---

## Table of Contents

1. [What Is Kagerou?](#what-is-kagerou)
2. [What Makes Kagerou Special?](#what-makes-kagerou-special)
3. [Video Concepts Glossary](#video-concepts-glossary)
4. [Architecture](#architecture)
5. [Build & Run](#build--run)
6. [AI Features](#ai-features)
7. [Virtual Camera](#virtual-camera)
8. [Enable Full NVDEC/NVENC](#enable-full-nvdecnvenc)
9. [API Reference](#api-reference)
10. [Real-Time GPU Filtering & Streaming](#real-time-gpu-filtering--streaming)
11. [Project Structure](#project-structure)
12. [Complete Filter Reference](#complete-filter-reference)

---

## What Is Kagerou?

Kagerou is a GPU-accelerated video transcoding pipeline. It takes video in, runs it through a chain of CUDA filters, and outputs the processed result — all on the GPU without CPU roundtrips.

```
Input File → [NVDEC Decode] → [CUDA Filters] → [NVENC Encode] → Output File
                (hardware)       (GPU kernels)      (hardware)
```

**What you can do with it:**
- Decode and transcode H.264, H.265, and AV1 video
- Denoise video frames in real-time
- Resize, crop, color-convert between NV12/RGB/YUV420P
- Insert interpolated frames for FPS conversion (30fps → 60fps)
- Tone-map HDR content to SDR
- AI depth estimation, optical flow, pose estimation, denoising

---

## What Makes Kagerou Special?

### The Problem With Existing Solutions

Every other video processing tool (FFmpeg, OBS, GStreamer) does this:

```
Decode (GPU) → Copy to CPU → Process on CPU → Copy to GPU → Encode (GPU)
              ↑                      ↑              ↑
           slow                 slowest          slow
```

The data bounces between GPU and CPU memory. Each copy costs time and bus bandwidth.

### How Kagerou Does It

```
Decode (NVDEC) → Filter (CUDA) → Encode (NVENC)
     ↓                ↓               ↓
   GPU memory     GPU memory      GPU memory
     ~0.5ms         ~2-3ms          ~0.5ms
```

**Zero copies. Everything stays on GPU.** Data never touches CPU memory between decode and encode.

### Head-to-Head Comparison

| | Kagerou | FFmpeg | OBS | GStreamer |
|--|---------|--------|-----|----------|
| **Architecture** | GPU-native | CPU-based | CPU-based | Mixed |
| **GPU pipeline** | ✓ Zero-copy | ✗ CPU middleman | ✗ CPU middleman | Partial |
| **SDK (embed in your app)** | ✓ Single header | ✗ CLI only | ✗ App only | ✓ Library |
| **1080p latency** | **~5 ms** | ~30ms+ | ~15ms | ~10ms |
| **Decoding** | NVDEC (dedicated HW) | CPU / NVDEC | CPU / NVDEC | CPU / NVDEC |
| **Encoding** | NVENC (dedicated HW) | CPU / NVENC | CPU / NVENC | CPU / NVENC |
| **Filter processing** | CUDA cores | CPU threads | CPU threads | CPU threads |
| **Dependencies** | **None** | Many libs | Many libs | Many libs |
| **Compilation** | Single .cu file | Multi-file build | Complex | Complex |
| **Integration** | `#include` + call API | Shell pipes | Plugin system | Pipeline graphs |

### Why This Matters

**For video call apps:**
- Sub-4ms processing = more time budget for network, less lag
- Zero CPU usage = CPU available for audio, UI, encoding
- Embed as SDK = works in any app (Zoom, Teams, custom)

**For live streaming:**
- GPU filters don't drop frames under load
- Multiple NVDEC sessions = process N camera feeds simultaneously
- NVENC = hardware encode — dedicated chip, zero CUDA core usage

**For video editing:**
- Preview plays at full FPS (filters run in real-time)
- No proxy files needed — edit directly on GPU
- Export uses same pipeline — no re-encode overhead

**For surveillance / IP cameras:**
- Process 16+ camera streams on one GPU
- Denoise low-light cameras in real-time
- Detect objects (future AI feature) while streaming

### The Numbers

| Metric | Kagerou | Typical CPU Solution |
|--------|---------|---------------------|
| 1080p filter latency | **2-3 ms** | 15-30 ms |
| Total pipeline latency | **~5 ms** | 30-50 ms |
| CPU usage during processing | **~0%** (GPU-bound) | 80-100% |
| Simultaneous streams (RTX 3050) | **16+** | 2-3 |
| Max throughput (1080p, no filters) | **210 fps** | 30-60 fps |
| Max throughput (1080p, denoise) | **191 fps** | 15-30 fps |
| Max throughput (1080p, crop) | **305 fps** | — |
| Max throughput (1080p, chroma key) | **195 fps** | — |

### GPU Efficiency

Kagerou's pipeline keeps data on GPU between decode → filter → encode. Two GPU→GPU copies exist for pitch stripping (NVDEC output has stride padding) — each ~0.1ms for 1080p.

| Stage | Copy? | Why |
|-------|-------|-----|
| NVDEC decode | GPU→GPU | CUVID output has pitch padding; filters need contiguous NV12 |
| Filter chain | **Zero-copy** | Filters read/write GPU buffers directly, no CPU involvement |
| NVENC encode | GPU→GPU | Registered buffer needs contiguous NV12 matching encoder dimensions |
| CUDA graphs | **Zero overhead** | Captured filter chains eliminate per-kernel launch cost |
| GPU compositor | **Zero-copy** | Composites N frames on GPU — no staging through CPU |

**Total GPU→GPU overhead:** ~0.2ms for 1080p NV12 (at 300 GB/s bandwidth). The two copies are required for format compatibility between NVDEC's padded output and NVENC's contiguous input requirement.

---

## Supported Codecs

### Input (Decode)

Kagerou supports decoding all major video codecs. H.264 and H.265 use **NVDEC hardware decode** (dedicated chip, zero CUDA core usage). AV1 uses **FFmpeg software decode** (via av1_cuvid or libdav1d), then feeds raw NV12 frames through the GPU filter pipeline.

| Codec | Decode Method | Speed | Notes |
|-------|--------------|-------|-------|
| **H.264 (AVC)** | NVDEC hardware | 200+ fps | Full hardware decode via CUVID parser. Best performance. |
| **H.265 (HEVC)** | NVDEC hardware | 200+ fps | Full hardware decode via CUVID parser. |
| **AV1** | FFmpeg (SW) | ~90 fps | Software decode via ffmpeg (av1_cuvid HW or libdav1d SW). Filters + encode still on GPU. |

### Output (Encode)

The encoder outputs **H.264** or **H.265** via NVENC hardware. This is a hardware limitation — NVIDIA's NVENC chip on RTX 20/30-series GPUs only supports H.264 and H.265 encoding. AV1 encoding requires RTX 40-series (Ada Lovelace) or newer.

| Codec | Encode Method | Why This Codec |
|-------|--------------|----------------|
| **H.264** | NVENC hardware | Universal compatibility — plays everywhere. Default output. |
| **H.265** | NVENC hardware | 50% smaller than H.264 at same quality. Supported by most modern players. |

### Why H.264/H.265 Instead of AV1?

**Short answer:** NVENC hardware on your GPU only encodes H.264 and H.265.

**Detailed explanation:**
- NVENC is a **dedicated hardware chip** on the GPU — separate from CUDA cores. It encodes H.264/H.265 at high speed and uses zero CUDA core resources.
- On RTX 20-series (Turing) and RTX 30-series (Ampere) GPUs, NVENC supports only **H.264 and H.265** encoding.
- **AV1 encoding** via NVENC requires RTX 40-series (Ada Lovelace) or newer.
- For AV1 input, Kagerou **decodes** via FFmpeg (software or av1_cuvid), runs GPU filters, then **re-encodes as H.264/H.265** via NVENC. The filters + encode are still fully on GPU — only the decode step uses CPU/ffmpeg.
- If you need AV1 output, you can use FFmpeg as a post-processing step: `ffmpeg -i output.mp4 -c:v libsvtav1 output.av1`

### The Two Decode Paths

```
┌─────────────────────────────────────────────────────────┐
│  H.264 / H.265 Input                                    │
│                                                         │
│  MP4/MKV → minimp4/ffmpeg demux → Annex-B NALUs         │
│      → NVDEC hardware decode (CUVID parser)             │
│      → NV12 frames on GPU                               │
│      → CUDA filters (GPU)                               │
│      → NVENC hardware encode → H.264/H.265 output       │
│                                                         │
│  Performance: 210+ fps (full hardware path)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  AV1 / Other Codec Input                                │
│                                                         │
│  MP4 → ffmpeg demux → raw NV12 frames (CPU/GPU decode)  │
│      → stored in frame buffer                           │
│      → CUDA filters (GPU)                               │
│      → NVENC hardware encode → H.264/H.265 output       │
│                                                         │
│  Performance: ~90 fps (decode is SW, filters+encode HW) │
│  Note: Decode uses ffmpeg (av1_cuvid or libdav1d).      │
│        Filters and encode remain fully on GPU.          │
└─────────────────────────────────────────────────────────┘
```

---

## Video Concepts Glossary

Everything you need to understand before reading the rest of this doc.

### The Fundamental Problem

A single 1080p frame = **6.2 MB** of raw pixel data. At 30fps = **186 MB/sec**. A 1-minute video = **11 GB**. This is impossible to store or stream. Video compression solves this.

### Pixel Formats — How Colors Are Stored

```
RGB (what screens and images use):
  Each pixel = 3 bytes: Red, Green, Blue
  ┌─────┬─────┬─────┐
  │ R=  │ G=  │ B=  │  ← one pixel
  │ 255 │ 128 │ 0   │     = orange
  └─────┴─────┴─────┘
  1080p frame = 1920 × 1080 × 3 = 6.2 MB

YUV / NV12 (what video codecs use):
  Y = brightness (luma), U/V = color (chroma)
  Human eyes are more sensitive to brightness than color,
  so we store full brightness but half color resolution.

  NV12 format (used by NVDEC/NVENC):
  ┌───────────────────────────────┐
  │ Y Y Y Y Y Y Y Y Y Y Y Y ...   │  ← full resolution (2.1 MB)
  │ Y Y Y Y Y Y Y Y Y Y Y Y ...   │
  │ Y Y Y Y Y Y Y Y Y Y Y Y ...   │
  ├───────────────────────────────┤
  │ U V U V U V U V U V U V ...   │  ← half resolution (1.0 MB)
  │ U V U V U V U V U V U V ...   │
  └───────────────────────────────┘
  1080p NV12 = 1920 × 1080 × 1.5 = 3.1 MB (2× smaller than RGB)
```

### Compression — How Video Gets Small

```
Raw frame (6.2 MB)
    ↓  NVENC encodes
Bitstream (30-100 KB)  ← 60-200× smaller!
    ↓  NVDEC decodes
Raw frame again (6.2 MB)
```

| Format | What it is | Size per 1080p frame |
|--------|-----------|---------------------|
| **RGB** | Raw pixels, 3 bytes per pixel | 6.2 MB |
| **NV12** | YUV with half-color chroma | 3.1 MB |
| **H.264** | Compressed (codec, most compatible) | 30-100 KB |
| **H.265** | Compressed (better, 50% smaller than H.264) | 15-50 KB |
| **AV1** | Compressed (newest, best compression) | 10-40 KB |

### Key Terms

| Term | Simple explanation |
|------|-------------------|
| **Bitstream** | A stream of compressed bytes — the encoded output. A `.h264` file IS a bitstream. |
| **NALU** | Network Abstraction Layer Unit — one compressed chunk inside the bitstream. One NALU ≈ one frame (or one slice of a frame). |
| **Codec** | CO-DEC — algorithm that encodes (compresses) and decodes (decompresses). H.264, H.265, AV1 are codecs. |
| **Container** | File format that wraps bitstream + audio + metadata. MP4, MKV, WebM are containers. |
| **Encode** | Compress raw pixels → bitstream. Done by NVENC (hardware) or x264/x265 (software). |
| **Decode** | Decompress bitstream → raw pixels. Done by NVDEC (hardware) or FFmpeg (software). |
| **NVDEC** | NVIDIA's hardware decoder chip — decodes video without using CUDA cores. |
| **NVENC** | NVIDIA's hardware encoder chip — encodes video without using CUDA cores. |
| **Frame** | One image in the video. A 30fps video has 30 frames per second. |
| **Keyframe (IDR)** | A complete frame that doesn't reference other frames. Needed for seeking and stream start. |
| **P-frame** | Predicted frame — references a previous frame (smaller than keyframe). |
| **B-frame** | Bi-predicted frame — references past AND future frames (smallest, but adds latency). |
| **GOP** | Group of Pictures — distance between keyframes. GOP=30 means keyframe every 30 frames (1 second at 30fps). |
| **Bitrate** | How many bits per second the compressed stream uses. 5 Mbps = 5 megabits/sec. |
| **Resolution** | Frame dimensions in pixels. 1920×1080 = 1080p, 3840×2160 = 4K. |
| **FPS** | Frames per second. 30fps = standard video, 60fps = smooth motion. |

### MP4 Container — What's Inside

```
video.mp4
├── moov (movie header)
│   └── trak (track)
│       └── avcC (AVC decoder config)
│           ├── SPS (Sequence Parameter Set) ← resolution, profile, level
│           └── PPS (Picture Parameter Set)  ← quantization params
│
├── mdat (media data)
│   └── compressed samples (H.264 NALUs)
│       ├── Sample 0: [len][NALU][len][NALU]...  ← AVCC format
│       ├── Sample 1: [len][NALU][len][NALU]...
│       └── ...
│
└── Track metadata (duration, FPS, codec info)
```

**SPS/PPS** are like the "settings" for the video — they tell the decoder: "this video is 1920×1080, H.264 High Profile, Level 4.1, with these quantization parameters." Without them, the decoder can't start.

### Stream Formats — How NALUs Are Packaged

```
Annex-B (used by CUVID parser, RTP streaming):
┌──────────┬──────────────────┬──────────┬──────────────────┐
│ 00 00 00 │                  │ 00 00 00 │                  │
│ 01       │ NALU payload     │ 01       │ NALU payload     │
└──────────┴──────────────────┴──────────┴──────────────────┘
Start codes separate each NALU. Parser finds NALUs by scanning for 00 00 01.

AVCC (used inside MP4 files):
┌──────────────────┬──────────────────┬──────────────────┐
│ 4-byte length    │ 4-byte length    │ 4-byte length    │
│ (big-endian)     │ (big-endian)     │ (big-endian)     │
│ NALU payload     │ NALU payload     │ NALU payload     │
└──────────────────┴──────────────────┴──────────────────┘
Length prefix tells you how big each NALU is. No start codes needed.
```

### The Kagerou Pipeline — What Happens to Your Data

```
INPUT                          GPU PROCESSING                    OUTPUT
──────                         ──────────────                    ──────

Compressed bitstream     →    NVDEC decode      →    Raw NV12 frame
(H.264 NALUs)                 (hardware chip)        (on GPU memory)
                                                        ↓
                                                    CUDA filters
                                                    (denoise/resize/etc)
                                                        ↓
                                                Raw NV12/RGB frame
                                                (filtered, on GPU)
                                                        ↓
                                                   NVENC encode
                                                   (hardware chip)
                                                        ↓
Compressed bitstream     ←    ← ← ← ← ← ← ← ←    Compressed output
(H.264/H.265 NALUs)                                  (ready to send/store)
```

**Key insight:** NVDEC and NVENC run on **dedicated hardware chips** — they don't use your CUDA cores. So decode + filter + encode all happen in parallel, giving you maximum throughput.

---

## Architecture

The SDK is a single compilation unit — every `.cu` file is `#include`d into one master file (`pipeline.cu`). This means:
- Zero link-time dependencies between filters
- CUDA compiles everything in one pass
- All kernel code is visible for optimization

```
pipeline.cu
├── decoder.cu          NVDEC wrapper
├── encoder.cu          NVENC wrapper
├── ai/
│   ├── ort_wrapper.cu       ORT 1.20.1 wrapper (CUDA EP, session management)
│   ├── ai_preprocess.cu     Shared kernels: resize, normalize, NCHW, draw, letterbox
│   ├── ai_pose_estimation.cu MediaPipe Pose 33 landmarks
│   ├── ai_depth.cu          Depth Anything V2 depth estimation
│   ├── ai_flow.cu           RAFT optical flow
│   └── ai_denoise.cu        FastDVDNet temporal denoising
└── filters/
    ├── color_convert.cu   NV12↔RGB, YUV420P↔RGB, HDR PQ→SDR
    ├── scale.cu           Bilinear, bicubic resize (RGB + NV12-native)
    ├── denoise.cu         Bilateral filter (RGB + NV12-native)
    ├── super_res.cu       2x bicubic upscale + unsharp mask
    ├── frame_interp.cu    Alpha blending for frame interpolation
    ├── clahe.cu           CLAHE adaptive contrast (RGB + NV12-native)
    ├── lut.cu             3D LUT color grading (6 presets)
    ├── creative.cu        Blur, sharpen, B/C, sat, gamma, vignette, grain, edge, WB, lens, flip
    ├── transforms.cu      Crop, pad, chroma key, bg blur, temporal denoise/stab, HDR tone map, warp
    └── compositor.cu      GPU multi-stream composite (NV12 tiled layout)
```

**Data flow:**
1. Demuxer reads MP4/MKV container → extracts compressed bitstream
2. AVCC → Annex-B conversion (SPS/PPS from avcC box, start codes added)
3. CUVID parser parses bitstream, creates decoder, schedules frames
4. NVDEC hardware decodes pictures → NV12 frame on GPU memory
5. Each enabled filter processes the frame in-place or to a new buffer
6. NVENC hardware encodes the final processed frame → compressed output
7. Output written to file

All GPU buffers are allocated with `cudaMalloc` and stay on-device. No CPU copies happen between filter stages.

**Multi-stream:** Each `Pipeline` instance is fully independent — own decoder, encoder, filter pool, CUDA stream, and LUT cache. No static mutable state in Pipeline. Safe to create multiple instances for parallel camera processing (e.g., group video calls with N participants). Note: CLAHE uses a shared GPU buffer — concurrent CLAHE across instances may have minor contention.

---

### Decoder (NVDEC)

**Without SDK (stub mode):**
Generates synthetic test patterns — a gradient Y plane with neutral UV. Useful for pipeline testing without hardware decode.

**With SDK (full NVDEC via CUVID parser):**

The decoder uses NVIDIA's CUVID video parser API, which handles the entire decode workflow: parsing the H.264 bitstream, detecting sequence changes, managing decoder lifecycle, and scheduling frame display.

**Initialization flow:**
```
Decoder::init()
├── cuInit(0)                        → initialize CUDA driver API
├── cuDeviceGet() / cuCtxCreate()    → create CUDA context on GPU
├── cuvidCtxLockCreate()             → create lock for thread-safe decoder access
└── cuvidCreateVideoParser()         → create CUVID parser with callback functions
```

The parser is configured with three callback functions that NVIDIA's driver calls at the appropriate times:
- `handle_video_sequence` — called when a new SPS (sequence parameter set) is detected
- `handle_picture_decode` — called when a complete picture is ready to be decoded
- `handle_picture_display` — called when a decoded picture is available for output

**MP4 input handling (minimp4):**

MP4 containers store video in AVCC format — each frame's NALUs are prefixed with 4-byte big-endian length fields. CUVID expects Annex-B format — NALUs prefixed with `00 00 00 01` start codes. The decoder performs three critical conversions:

1. **SPS/PPS extraction** from the MP4's `avcC` box via `MP4D_read_sps()` / `MP4D_read_pps()`. These parameter sets describe the video profile, level, and resolution — CUVID needs them before it can decode any frames.

2. **AVCC → Annex-B conversion** for each frame sample. Each frame's raw data is parsed: the 4-byte length prefix is stripped and a `00 00 00 01` start code is prepended.

3. **Start code re-insertion** when feeding NALUs to the parser. Individual NALUs from `split_nalus_h264()` have start codes stripped — the decoder prepends them back before calling `cuvidParseVideoData()`.

**Per-frame decode flow:**
```
Decoder::decode(nalu_data, nalu_size)
├── Prepend Annex-B start code (00 00 00 01) if not present
├── cuCtxPushCurrent(cu_ctx)         → push CUDA context for parser callbacks
├── cuvidParseVideoData()            → feed bitstream to parser
│   ├── Parser detects SPS → handle_video_sequence()
│   │   └── cuvidCreateDecoder()     → create HW decoder (640x368, H.264)
│   ├── Parser detects picture → handle_picture_decode()
│   │   └── cuvidDecodePicture()     → submit to NVDEC hardware
│   └── Parser ready → handle_picture_display()
│       └── Sets got_frame=true + picture_index
├── cuCtxPopCurrent()                → restore previous context
├── cuvidCtxLock()                   → lock decoder for thread-safe access
├── cuCtxPushCurrent(cu_ctx)         → push context for map operation
├── cuvidMapVideoFrame()             → map decoded NV12 to CUDA memory
├── cudaMemcpy2D() × 2              → copy Y plane + UV plane to pipeline buffer
├── cuvidUnmapVideoFrame()           → release decoder's reference
├── cuCtxPopCurrent()                → restore context
└── cuvidCtxUnlock()                 → release lock
```

**Why CUDA context management matters:**

The CUVID parser callbacks (`handle_video_sequence`, `handle_picture_decode`) run on the parser's internal thread, not the calling thread. Each callback must push the CUDA context (`cuCtxPushCurrent`) before calling NVDEC functions and pop it afterward — otherwise `cuvidCreateDecoder` returns error 201 (`CUDA_ERROR_INVALID_CONTEXT`) and `cuvidDecodePicture` fails silently.

**Error handling and graceful degradation:**

The decoder returns a three-state result:
- `kOk` + output frame allocated → frame decoded successfully (NV12 on GPU)
- `kOk` + output frame empty → NALU consumed but no frame produced (SPS/PPS/SEI NALUs)
- `kDecodeError` → actual decode failure

This allows the pipeline to process all NALUs without failing on non-VCL NALUs (which don't contain picture data).

**Performance:**
- NVDEC runs on **dedicated hardware** — zero CUDA core usage during decode
- 1080p H.264 decode + encode (no filters): **~210 fps** on RTX 3050 6GB Laptop
- Decode happens in parallel with filter processing on CUDA cores
- The CUVID parser handles reference frame management, reordering, and display scheduling automatically

---

### Encoder (NVENC)

**Without SDK (stub mode):**
Copies raw frame data as-is (no compression). Useful for testing filter output. Returns the raw NV12 frame bytes as if they were "encoded" — the pipeline works but output is uncompressed.

**With SDK (full NVENC via NvEncodeAPI):**

The encoder loads NVIDIA's NVENC library dynamically at runtime and uses the official `nvEncodeAPI.h` interface. It supports multiple API versions for maximum driver compatibility.

**Initialization flow:**
```
Encoder::init()
├── LoadLibrary("nvEncodeAPI64.dll")        → load NVENC from NVIDIA driver
├── GetProcAddress("NvEncodeAPICreateInstance") → get factory function
├── NvEncodeAPICreateInstance()             → fill NV_ENCODE_API_FUNCTION_LIST
├── Version fallback loop:                  → try API versions until one works
│   ├── v13.1 (SDK 13.x headers)
│   ├── v12.2 (driver 520+)
│   ├── v12.1 (driver 510+)
│   ├── v12.0 (driver 500+)
│   └── v11.1 (driver 470+)
├── NvEncOpenEncodeSessionEx()              → create encoder session
│   └── Uses CUDA context + NVENC device type
├── NvEncGetEncodePresetConfigEx()          → get P1 preset (lowest latency)
├── NvEncInitializeEncoder()                → configure:
│   ├── Codec: H.264 or H.265
│   ├── Resolution: target width × height
│   ├── Framerate: FPS for VBV timing
│   ├── Bitrate: CBR / VBR / CQP
│   ├── GOP: keyframe interval
│   ├── Profile: Baseline / Main / High
│   └── Low latency: tune for real-time
└── Ready to encode frames
```

**Why version fallback matters:**

The NVIDIA Video Codec SDK headers (version 13.1) define API structures that may be newer than what your installed driver supports. For example:
- SDK 13.1 headers require driver **531+** for full NVENC support
- SDK 12.1 headers work with driver **510+**
- SDK 11.1 headers work with driver **470+**

The encoder tries each API version in order. When `NvEncOpenEncodeSessionEx` succeeds, it means the driver supports that version. If all versions fail, the encoder falls back to stub mode gracefully.

```
Driver 528.97 (your current):
  v13.1 → ERROR 15 (INVALID_VERSION)  ✗
  v12.2 → ERROR 15 (INVALID_VERSION)  ✗
  v12.1 → ERROR 15 (INVALID_VERSION)  ✗
  v11.1 → ERROR 15 (INVALID_VERSION)  ✗
  → Falls back to stub mode

Driver 531.18+ (after update):
  v13.1 → SUCCESS  ✓  ← uses this
  → Full NVENC encode
```

**Per-frame encode flow (zero-copy path):**
```
Encoder::encode(input_frame)
├── cudaStreamSynchronize(0)               → ensure filter kernels complete
├── cudaMemcpy(gpu_input_buf, frame)       → GPU→GPU copy to registered NV12 buffer
├── NvEncMapInputResource()                → map registered resource for this encode
├── NvEncEncodePicture()                   → submit frame to NVENC hardware
│   └── NVENC reads directly from GPU memory (~0.5ms, zero CUDA core usage)
├── NvEncUnmapInputResource()              → release mapping
├── NvEncLockBitstream()                   → lock output for CPU read
├── memcpy(bitstream_out, locked_buffer)   → copy compressed NALUs
├── NvEncUnlockBitstream()                 → release output lock
└── Returns compressed H.264/H.265 bitstream

Encoder::encode(input_frame) — CPU staging fallback (if NvEncRegisterResource fails)
├── cudaDeviceSynchronize()                → ensure all GPU work completes
├── cudaMemcpy(staging_buf, frame)         → GPU→CPU copy
├── NvEncLockInputBuffer()                 → lock NVENC input for CPU write
├── memcpy(nvenc_buf, staging_buf)         → CPU→NVENC copy
├── NvEncUnlockInputBuffer()               → release lock
├── NvEncEncodePicture()                   → submit to NVENC
└── ... same as above
```

**Graceful degradation:**
The encoder returns a three-state result:
- `kOk` + output filled → frame encoded successfully (H.264/H.265 NALUs)
- `kOk` + output empty → encoder is in stub mode (raw frame passthrough)
- `kEncodeError` → actual encode failure (rare — usually driver issue)

This means the pipeline always works — with or without compatible drivers.

**Performance:**
- NVENC runs on **dedicated hardware** — zero CUDA core usage during encode
- 1080p H.264 encode + decode (no filters): **~210 fps** on RTX 3050 6GB Laptop
- With denoise: **~191 fps** — filters add ~0.4ms overhead per frame
- Encode happens in parallel with filter processing on CUDA cores
- Supports H.264 (Baseline/Main/High) and H.265 (Main/Main10) profiles
- Bitrate control: CBR (constant), VBR (variable), CQP (constant quality)

---

## Build & Run

### Dependencies

You need these installed before building:

| Dependency | Version | Why | Install |
|-----------|---------|-----|---------|
| **Windows** | 10/11, 64-bit | Virtual camera is a 64-bit DirectShow filter | — |
| **NVIDIA GPU** | RTX 20xx+ (sm_86+) | CUDA compute capability | Already have it |
| **CUDA Toolkit** | 12.x | nvcc compiler, CUDA runtime | `winget install NVIDIA.CUDA` or [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **MSVC Build Tools** | VS 2019+ | C++ linker + Windows SDK (both required) | `winget install Microsoft.VisualStudio.2022.BuildTools` — during install select **"Desktop development with C++"** workload |
| **FFmpeg** | 8.x+ (any recent) | Webcam capture + MP4 mux + file probing | `winget install Gyan.FFmpeg` or [ffmpeg.org/download](https://ffmpeg.org/download.html) — add `bin\` to PATH |
| **Git** | any | Clone the repo | `winget install Git.Git` |
| **Admin rights** | one-time | Registering the virtual camera writes HKLM | Right-click → Run as administrator (only for `vcam_dll` registration) |
| **Python 3 + torch** | optional | Exporting models (Zero-DCE) + diagnostics only — never needed to build or run | `pip install torch onnx` |

No Visual C++ Redistributable needed (static CRT + system DLLs only). No CMake, no vcpkg, no UI libraries. Disk: ~250MB models + ~1GB ORT/cuDNN/TensorRT DLLs in `bin/`.

### NVIDIA Driver

You need an updated NVIDIA GPU driver with NVDEC/NVENC support. Download from:
https://www.nvidia.com/en-us/drivers/

Verify after install:
```powershell
nvidia-smi        # should show your GPU + driver version
```

**UI-only dependencies** (no extra install needed — already on every Windows system):

| Component | Why | Notes |
|-----------|-----|-------|
| **Win32 API** (`user32.dll`, `kernel32.dll`) | Window creation, message loop, keyboard input | Part of Windows SDK |
| **OpenGL** (`opengl32.dll`) | GPU-accelerated texture display | Ships with every Windows install since XP |
| **GDI32** (`gdi32.dll`) | Text rendering (Consolas font on screen) | Ships with Windows |
| **ComDlg32** (`comdlg32.dll`) | File open dialog for picking video files | Ships with Windows |

The virtual camera UI uses **zero external UI libraries** — no ImGui, no GLFW, no SDL. It's pure Win32 + GDI, which means no additional downloads and no ABI conflicts with CUDA.

**Quick check** — run these in PowerShell to verify everything is installed:
```powershell
nvcc --version          # should show CUDA 12.x
cl                      # should show MSVC compiler version
ffmpeg -version         # should show ffmpeg version
```

### Install Commands (one-liner)

```powershell
winget install NVIDIA.CUDA; winget install Microsoft.VisualStudio.2022.BuildTools; winget install Gyan.FFmpeg; winget install Git.Git
```

After VS Build Tools install, open **"Developer Command Prompt for VS 2022"** or **"x64 Native Tools Command Prompt"** — this sets up `cl.exe` and linker paths.

### Build (stub mode — no SDK needed)

```batch
cd src\HardLab\Kagerou
build.bat                    # auto-detect GPU arch
build.bat sm_86              # RTX 30xx
build.bat sm_89              # RTX 40xx
build.bat sm_120             # RTX 50xx
```

This builds with stub decoder/encoder (generates test patterns). All filters work fully on GPU.

### Build (with MP4/MKV container support)

```batch
build.bat minimp4            # adds MP4 demuxing via minimp4
build.bat mp4                # alias for minimp4
build.bat all minimp4        # build everything with MP4 support
```

MP4 support uses [minimp4](https://github.com/lieff/minimp4) (public domain, single-header).
MKV support uses FFmpeg CLI as fallback (requires FFmpeg in PATH).

### Build (all targets)

```batch
build.bat all                # build kagerou.exe + tests
build.bat test               # build and run tests only
build.bat all sdk            # build everything with SDK
build.bat all sdk minimp4    # everything with SDK + MP4 support
build.bat all sdk minimp4 onnx   # everything with SDK + MP4 + AI - run this as admin for vcam dll
```

### Run

```batch
# Basic transcode (auto-detects resolution and codec from input)
kagerou.exe video.mp4
kagerou.exe video.mp4 -o my_output
kagerou.exe bunny.mp4               # H.264 input → H.264 output (full HW decode + GPU filters)

# With filters
kagerou.exe video.mp4 --denoise
kagerou.exe video.mp4 --scale 1920x1080
kagerou.exe video.mp4 --super-res
kagerou.exe video.mp4 --frame-interp
kagerou.exe video.mp4 --all                     # all filters at once

# Combine filters
kagerou.exe video.mp4 --denoise --scale 1280x720
kagerou.exe video.mp4 --denoise --super-res -o output_hq

# CLAHE and LUT
kagerou.exe video.mp4 --clahe
kagerou.exe video.mp4 --lut cinematic
kagerou.exe video.mp4 --denoise --lut warm

# Transform filters
kagerou.exe video.mp4 --crop 640x480+100+50
kagerou.exe video.mp4 --chroma-key
kagerou.exe video.mp4 --chroma-key --chroma-sat 0.08 --chroma-hue 50:170
kagerou.exe video.mp4 --temporal-denoise 0.3
kagerou.exe video.mp4 --temporal-stab
kagerou.exe video.mp4 --hdr 1                   # ACES tone map
kagerou.exe video.mp4 --bg-blur 12

# Codec and quality options
kagerou.exe video.mp4 --codec h265 --bitrate 10000
kagerou.exe video.mp4 --fps 60 --bitrate 8000

# Other modes
kagerou.exe benchmark                                     # benchmark all filters
kagerou.exe test                                          # run unit tests (12 tests)
kagerou.exe batch video.mp4                               # batch: 4 quality configs
```

### CLI Reference

```
kagerou.exe <input> [options]     Transcode video
kagerou.exe benchmark             Run filter benchmarks
kagerou.exe test                  Run unit tests
kagerou.exe batch <input>         Batch transcode (multiple quality configs)

General options:
  -o <path>              Output folder (default: output/)
  --fps <n>              Output FPS (default: 30)
  --bitrate <kbps>       Output bitrate in kbps (default: auto based on resolution)
  --codec <h264|h265>    Output codec (default: h264)
  -v                     Verbose output
  -h                     Show help

Core filters:
  --denoise              Enable bilateral denoise filter
  --scale WxH            Enable scale filter (e.g. --scale 1280x720)
  --super-res            Enable 2x super resolution filter
  --frame-interp         Enable frame interpolation (2x FPS)
  --clahe                Enable CLAHE contrast enhancement
  --lut <preset>         Enable LUT color grading (warm|cool|cinematic|vintage|contrast|desat)

Creative filters:
  --blur [sigma]         Gaussian blur (sigma=0.5..20, default: 2.0)
  --sharpen [str]        Unsharp mask sharpen (str=0.1..5.0, default: 1.0)
  --brightness <n>       Brightness offset (-255..255, default: 0)
  --contrast <n>         Contrast multiplier (0.1..3.0, default: 1.0)
  --saturation [n]       Color saturation (0=gray, 1=normal, default: 1.4)
  --gamma [n]            Gamma (<1=brighten, >1=darken, default: 0.8)
  --vignette [str]       Lens darkening (0=none, 1=heavy, default: 0.6)
  --grain [amt]          Film grain noise (0..100, default: 25)
  --edge-detect          Sobel edge detection
  --white-balance        Warm/cool color temperature shift
  --lens-distort [k]     Barrel/pincushion distortion (k<0=barrel, default: -0.3)
  --flip <h|v>           Flip horizontal or vertical
  --dir-blur <ang> <len> Directional blur: angle degrees + pixel length

Transform filters:
  --crop WxH+X+Y         Crop region from frame (e.g. --crop 640x480+100+50)
  --chroma-key           Green screen removal (tunable via --chroma-sat, --chroma-hue)
  --chroma-sat <n>       Chroma key: min saturation (default: 0.3, lower=catches more)
  --chroma-hue <min:max> Chroma key: hue range in degrees (default: 60:160)
  --chroma-spill <n>     Chroma key: green spill suppression (default: 0.5)
  --chroma-bg <r> <g> <b>  Chroma key: replacement bg color (default: 0 0 0)
  --bg-blur [str]        Background blur — center-weighted depth-of-field (default: 8)
  --temporal-denoise [s] Inter-frame temporal denoise (s=0..1, default: 0.25)
  --temporal-stab        Temporal stabilization (block-matching motion estimation + warp)
  --hdr [method]         HDR tone map (0=Reinhard, 1=ACES, default: 0)
  --peak-nits <n>        HDR peak brightness (default: 100 for SDR)
  --all                  Enable common creative filters bundle

AI filters (require ORT GPU + models):
  --ai-pose              AI pose estimation (MediaPipe 33 landmarks)
  --ai-depth             AI depth estimation (Depth Anything V2)
  --ai-flow              AI optical flow (RAFT) — motion visualization
  --ai-denoise           AI video denoise (FastDVDNet) — needs 5-frame window

LUT presets:
  warm, cool, cinematic, vintage, contrast, desat

Examples:
  kagerou.exe video.mp4
  kagerou.exe video.mp4 --denoise --lut cinematic
  kagerou.exe video.mp4 --clahe --scale 1920x1080
  kagerou.exe video.mp4 --codec h265 --bitrate 10000
  kagerou.exe video.mp4 --temporal-denoise 0.3 --sharpen 1.2
  kagerou.exe video.mp4 --bg-blur 12 --chroma-key --saturation 1.3
  kagerou.exe video.mp4 --crop 640x480+0+0 --flip h --hdr 1
  kagerou.exe video.mp4 --chroma-key --chroma-sat 0.08 --chroma-hue 50:170
  kagerou.exe video.mp4 --all -o my_output
  kagerou.exe video.mp4 --ai-pose
  kagerou.exe video.mp4 --ai-denoise --denoise
  kagerou.exe video.mp4 --ai-depth --lut cinematic
  kagerou.exe video.mp4 --ai-flow
```

**Output structure:**
```
output/
├── video.h264    # Raw H.264 bitstream (Annex-B)
└── video.mp4     # Muxed MP4 container (via FFmpeg)
```

**Dynamic resolution:** The pipeline auto-detects input resolution from the video stream. You do NOT need to specify width/height -- just pass the input file and the encoder initializes at the correct size automatically.

### What Happens When You Run `kagerou.exe video.mp4`

```
video.mp4
    |
    v
[1] Minimp4 demux (CPU)
    └─ Reads MP4 container, extracts compressed video samples
    └─ Each sample contains AVCC-format NALUs (4-byte length-prefixed)
    |
    v
[2] AVCC → Annex-B conversion (CPU)
    └─ Extract SPS/PPS from avcC decoder config box
    └─ Strip 4-byte length prefixes, add 00 00 00 01 start codes
    └─ Result: standard Annex-B H.264 bitstream
    |
    v
[3] CUVID parser + NVDEC hardware decode (GPU — dedicated chip)
    └─ cuvidParseVideoData() — parses bitstream, detects sequences
    └─ cuvidCreateDecoder() — creates HW decoder on sequence detection
    └─ cuvidDecodePicture() — submits pictures to hardware decoder
    └─ cuvidMapVideoFrame() — maps decoded NV12 frame to GPU memory
    └─ cudaMemcpy2D() — copies Y + UV planes to pipeline buffer
    └─ Output: NV12 frame on GPU (Y plane + UV interleaved)
    |
    v
[4] CUDA filter kernels (GPU — runs on CUDA cores)
    └─ denoise_bilateral() — edge-preserving 5x5 bilateral filter
    └─ resize_bilinear/bicubic() — scale to target resolution
    └─ super_res_2x() — bicubic upscale + unsharp mask
    └─ nv12_to_rgb() / rgb_to_nv12() — color space conversion
    └─ crop_rgb() / pad_rgb() — geometry transforms
    └─ chroma_key_rgb() — HSV green screen removal
    └─ bg_blur_rgb() — center-weighted depth blur
    └─ temporal_denoise_rgb() — inter-frame noise reduction
    └─ temporal_stabilize_rgb() — motion estimation + warp
    └─ hdr_tone_map_rgb() — Reinhard/ACES tone mapping
    └─ All filters stay on GPU — no CPU copies between stages
    |
    v
[5] NVENC hardware encode (GPU — dedicated chip, zero CUDA core usage)
    └─ NvEncEncodePicture() — submits filtered frame to hardware encoder
    └─ NvEncLockBitstream() — reads compressed H.264/H.265 output
    └─ Output: compressed bitstream on GPU
    |
    v
[6] Write output.h264
    └─ Compressed bitstream written to file
```

**Key facts:**
- NVDEC + NVENC run on **dedicated hardware chips** — they don't use CUDA cores
- CUDA filters run on **CUDA cores** — parallel with decode/encode
- The entire pipeline is **GPU-native** — data stays on GPU between decode → filter → encode
- Only two CPU touches: read input file, write output file
- Without `sdk` flag: stub mode generates test patterns (no real decode/encode)
- The CUVID parser handles reference frame management and display scheduling automatically

---

## AI Features

Kagerou includes 12 AI-powered video filters running fully on GPU via ONNX Runtime (TensorRT EP primary, CUDA EP fallback). Frames stay on the device from decode to encode — zero CPU roundtrips in the frame path; only compact result tensors (boxes, scores, landmarks) download for CPU decode/NMS.

```
Input → [NVDEC] → [RGB Convert] → [AI Model GPU Inference] → [Post-Process GPU] → [NVENC] → Output
                                ↑                                                          ↑
                     ORT TensorRT EP (FP16,                          NVENC HW
                     engines cached to trt_engines/)                  (dedicated chip)
```

### Available AI Filters

| Filter | CLI Flag | Model | Description |
|--------|----------|-------|-------------|
| **Pose Estimation** | `--ai-pose` | `pose_landmark.onnx` | MediaPipe Pose 33 landmarks — draws body skeleton |
| **Depth Estimation** | `--ai-depth` | `depth_anything_v2.onnx` | Depth Anything V2 — grayscale depth heatmap |
| **Optical Flow** | `--ai-flow` | `raft_small.onnx` | RAFT — Middlebury color-coded motion visualization |
| **AI Denoise** | `--ai-denoise` | `fastdvdnet.onnx` | FastDVDNet — temporal AI denoising (needs 5-frame window) |
| **AI Matting** | `--ai-matting` | `rvm_mobilenetv3_static.onnx` | RVM MobileNetV3 — background segmentation with frame recurrence |
| **AI Face** | `--ai-face` | `yunet_2023mar.onnx` | YuNet face detection — boxes (also drives autoframe + gaze) |
| **AI Hands** | `--ai-hands` | `palm_detection_lite.onnx` + `hand_landmark_lite.onnx` | BlazePalm + 21-joint landmarks (also drives gesture control) |
| **AI Gaze** | `--ai-gaze` | `iris_landmark.onnx` | MediaPipe iris — gaze vectors per eye |
| **AI LowLight** | `--ai-lowlight` | `zero_dce.onnx` | Zero-DCE single-frame low-light enhance |
| **AI Anime** | `--ai-anime` | `animeganv3_hayao.onnx` | AnimeGANv3 Hayao stylization |
| **AI Detect** | `--ai-detect` | `yolov8n.onnx` | YOLOv8n object detection — 80 COCO classes, boxes + labels |
| **AI AutoFrame** | `--ai-autoframe` | (uses YuNet face) | Face-tracked auto-framing crop that follows you (file + live `F7`) |

### Dependencies (AI Features)

AI features require additional dependencies beyond the base SDK:

| Dependency | Version | Purpose | Install |
|------------|---------|---------|---------|
| **ONNX Runtime GPU** | 1.20.1 | AI model inference on GPU | Place in `sdk/onnxruntime/` (see below) |
| **cuDNN** | 9.x | Required by ORT CUDA EP | Place DLLs in `bin/` (see below) |
| **CUDA Runtime DLLs** | 12.x | cuBLAS, cuFFT for ORT | Already have with CUDA Toolkit |
| **ONNX Models** | — | Pre-trained AI models | Place in `models/` directory |

**Base SDK dependencies** (same as without AI):
- NVIDIA GPU (RTX 20xx+ / sm_86+)
- CUDA Toolkit 12.x
- MSVC Build Tools (VS 2019+)
- FFmpeg (for MKV fallback)

### Installing AI Dependencies

#### 1. ONNX Runtime GPU (1.20.1)

Download ONNX Runtime GPU 1.20.1 for Windows:

```powershell
# Option A: Download via PowerShell
cd sdk\onnxruntime
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip" -OutFile "onnxruntime.zip"
Expand-Archive -Path "onnxruntime.zip" -DestinationPath "." -Force
# Move contents to correct structure:
# sdk/onnxruntime/include/   ← headers
# sdk/onnxruntime/lib/       ← .lib and .dll files
```

**Option B:** Manual download from https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1
- Download `onnxruntime-win-x64-gpu-1.20.1.zip`
- Extract `include/` → `sdk/onnxruntime/include/`
- Extract `lib/` → `sdk/onnxruntime/lib/`

Required structure:
```
sdk/onnxruntime/
├── include/
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_c_api.h
│   └── ...
└── lib/
    ├── onnxruntime.lib
    ├── onnxruntime.dll
    ├── onnxruntime_providers_cuda.lib
    ├── onnxruntime_providers_cuda.dll
    ├── onnxruntime_providers_shared.dll
    ├── onnxruntime_providers_tensorrt.lib
    └── onnxruntime_providers_tensorrt.dll   ← required for the TensorRT EP
```

#### 2. cuDNN 9

Download cuDNN 9 from https://developer.nvidia.com/cudnn

Place all cuDNN DLLs in `bin/`:
```
bin/
├── cudnn64_9.dll
├── cudnn_adv64_9.dll
├── cudnn_cnn64_9.dll
├── cudnn_engines_precompiled64_9.dll
├── cudnn_engines_runtime_compiled64_9.dll
├── cudnn_engines_tensor_ir64_9.dll
├── cudnn_ext64_9.dll
├── cudnn_graph64_9.dll
├── cudnn_heuristic64_9.dll
└── cudnn_ops64_9.dll
```

#### 3. TensorRT 10.x (for the TensorRT execution provider)

Without this, AI still works (CUDA EP fallback) but slower and noisier in the log. Download from https://developer.nvidia.com/tensorrt (free login) — any **10.x** works with ORT 1.20.1. From the zip's `lib/` folder:

```
sdk/tensorrt/            <- create this; the app adds it to the DLL search path
├── nvinfer.dll
├── nvinfer_plugin.dll
├── nvonnxparser.dll
└── nvinfer_builder_resource.dll
bin/                     <- copy the same four here (loader checks here first)
├── nvinfer.dll
├── nvinfer_plugin.dll
├── nvonnxparser.dll
└── nvinfer_builder_resource.dll
```

(The `*_10.dll` suffixed twins some zips ship are the same binaries — either name works. `onnxruntime_providers_tensorrt.dll` itself comes inside the ORT GPU zip from step 1.)

#### 4. CUDA Runtime DLLs

Copy from your CUDA Toolkit installation (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`):
```
bin/
├── cublas64_12.dll
└── cufft64_10.dll
```

#### 4. ONNX Models

Place pre-trained ONNX models in the `models/` directory (~250MB total):
```
models/
├── pose/       └── pose_landmark.onnx
├── depth/      └── depth_anything_v2.onnx (+ .onnx.data weights)
├── flow/       └── raft_small.onnx
├── denoise/    └── fastdvdnet.onnx (+ .onnx.data weights)
├── matting/    └── rvm_mobilenetv3.onnx (+ rvm_mobilenetv3_static.onnx, auto-built)
├── face/       └── yunet_2023mar.onnx
├── hands/      ├── palm_detection_{full,lite}.onnx
│               └── hand_landmark_{full,lite}.onnx (lite is wired)
├── gaze/       └── iris_landmark.onnx
├── lowlight/   └── zero_dce.onnx (+ Epoch99.pth source weights)
└── style/      └── animeganv3_hayao.onnx
```
Sources: RVM + AnimeGANv3 from their GitHub releases, YuNet from opencv_zoo, iris/hands from PINTO_model_zoo bundles, Zero-DCE exported from Epoch99 (see `build.bat ai_diag` probe for verification). RNNoise (audio) has no model file — it's compiled C code.

### Building with AI Support

```batch
# Build everything with AI + SDK + MP4 support
build.bat all sdk minimp4 onnx

# Just the CLI with AI support
build.bat onnx

# Full build: CLI + tests + virtual camera with AI
build.bat all sdk minimp4 onnx
```

The `onnx` flag:
- Compiles all AI filter source files (`src/ai/*.cu`)
- Links against ONNX Runtime GPU libraries
- Defines `KAGEROU_USE_ONNX` preprocessor macro
- Includes the AI .cu files in the build

**Without `onnx` flag:** AI filter code is excluded from the build. The binary has zero AI dependencies.

### Running AI Filters

```batch
# Pose estimation — draws body skeleton with 33 landmarks
kagerou.exe video.mp4 --ai-pose

# Depth estimation — grayscale depth heatmap
kagerou.exe video.mp4 --ai-depth

# Optical flow — Middlebury color-coded motion visualization
kagerou.exe video.mp4 --ai-flow

# AI denoise — temporal AI denoising (needs 5+ frames)
kagerou.exe video.mp4 --ai-denoise

# Combine AI + non-AI filters
kagerou.exe video.mp4 --ai-denoise --denoise --lut cinematic
kagerou.exe video.mp4 --ai-depth --scale 1280x720
```

### Custom Model Paths

Override default model paths via CLI or API:

```batch
# CLI: pass model path after the flag
kagerou.exe video.mp4 --ai-depth models/custom/depth_v2.onnx
```

```cpp
// API: set in PipelineConfig
cfg.ai_depth_enabled = true;
cfg.ai_depth_model = "models/custom/depth_v2.onnx";
```

### AI Filter Details

#### Pose Estimation (MediaPipe 33)
- **Model:** MediaPipe Pose (256x256 input)
- **Output:** 33 body landmarks with visibility scores
- **Draws:** Red circles at joints, lines connecting skeleton
- **Visibility filter:** Only draws landmarks with confidence > 0.5

#### Depth Estimation (Depth Anything V2)
- **Model:** Depth Anything V2 (518x518 input)
- **Output:** Per-pixel depth [0,1] → grayscale heatmap
- **Draws:** White=near, Black=far depth visualization

#### Optical Flow (RAFT)
- **Model:** RAFT-Small (512x320 input)
- **Requires:** Two consecutive frames (previous + current)
- **Output:** Per-pixel motion vectors (dx, dy)
- **Draws:** Middlebury color-coded visualization

#### AI Denoise (FastDVDNet)
- **Model:** FastDVDNet (5-frame temporal window)
- **Requires:** 5 consecutive frames for temporal denoising
- **Pipeline:** Collects frames → batch inference → output denoised

### Performance (AI Filters)

Measured on RTX 3050 6GB Laptop, 1080p input:

| AI Filter | Per-Frame | Notes |
|-----------|-----------|-------|
| Pose Estimation | ~10ms | Single ORT inference |
| Depth Estimation | ~15ms | Single ORT inference |
| Optical Flow | ~20ms | Two-frame inference + colorize |
| AI Denoise | ~30ms | 5-frame batch inference |
| AI Detect (YOLOv8n) | ~3ms | TRT FP16; file transcode with `--ai-detect` measured 68.7fps @640x480 |

**Note:** AI inference runs on CUDA cores (shared with other filters). AI denoise will reduce pipeline throughput. NVDEC/NVENC remain on dedicated hardware.

### Architecture (AI Pipeline)

```
NVDEC decode
    ↓
NV12 → RGB conversion (GPU kernel)
    ↓
AI Preprocessing (GPU kernels):
  ├── Center-crop / resize / letterbox (aspect-preserving for detectors)
  ├── NCHW float conversion
  ├── Normalize ([-1,1], [0,1], or raw — per model)
  └── Upload to ORT GPU tensor
    ↓
ORT Inference (GPU):
  ├── TensorRT EP primary (FP16, engines built once at warmup, cached)
  ├── CUDA EP fallback (e.g. RVM, whose symbolic dims TRT rejects)
  ├── Output tensors on GPU; frames never leave the device
  └── Only compact results (boxes/scores/landmarks) download for CPU decode + NMS
    ↓
AI Post-Processing (GPU kernels):
  ├── Draw rectangles / circles / lines / text labels
  ├── Depth-to-RGB conversion
  ├── Flow-to-RGB color coding
  └── Blur / composite operations
    ↓
RGB → NV12 conversion (GPU kernel)
    ↓
NVENC encode
```

### Troubleshooting AI Features

**"Cannot load model" error:**
- Check model file exists at `models/...` path
- Verify ONNX Runtime DLLs are in `bin/`
- Run `bin\kagerou_test.exe` to verify ORT initialization

**"ORT CUDA EP not available" warning:**
- Ensure cuDNN 9 DLLs are in `bin/`
- Check CUDA Toolkit DLLs (cublas64_12.dll, cufft64_10.dll) are in `bin/`
- Verify GPU supports CUDA (sm_60+)

**AI denoise requires 5 frames:**
- FastDVDNet uses temporal information from 5 consecutive frames
- First 4 frames may have reduced quality
- Use `--ai-denoise` on videos with 5+ frames

---

## Virtual Camera

Live webcam → full Kagerou GPU filter pipeline → virtual webcam device for Zoom/Teams/Meet/OBS. Like NVIDIA Broadcast, but you own the code.

### Build

```batch
cd src\HardLab\Kagerou
build.bat virtualcam sdk minimp4 onnx
```

This builds **both** binaries and registers the device (a UAC prompt appears — admin is required so browsers can see it):

| Output | What it is |
|--------|-----------|
| `bin\kagerou_virtualcam.exe` | Capture + GPU filter app (the engine) |
| `bin\kagerou_virtualcam.dll` | DirectShow source filter (the device) |

Other useful targets:

```batch
build.bat vcam_dll    # rebuild + re-register only the DLL
build.bat vcam_test   # verify: lists devices, builds a live graph, streams 3s
```

To unregister: run `virtualcam\unregister.bat` as Administrator.

### Run

1. Run `bin\kagerou_virtualcam.exe` — a preview window opens instantly (camera auto-detected, test pattern if no camera). AI models load in the background.
2. Open Zoom/Teams/Meet/OBS → camera list → **"Kagerou Virtual Camera"**.
3. Toggle filters in the preview window (click or shortcut keys) — the virtual output follows live.

The EXE **must be running** while you use the virtual camera — it produces the frames. If it isn't running, the device emits a test pattern so apps still see a live signal.

### Record Tab — Live Cut Recorder

The third tab (`RECORD`, hotkey `F9` = start/stop) records the filtered feed to MP4 via NVENC — what you see is what gets saved, filters included:

| Control | What it does |
|---------|--------------|
| **Source `CAM / SCREEN / TUTORIAL`** (`F5` cycles) | Camera, screen capture (up to 1080p), or screen + camera picture-in-picture tutorial mode. The virtual cam mirrors the same feed. |
| **CUT LAST ~4s** (`F8`) | Drops the in-progress segment — the sneeze killer. Cut ranges land in `edit.csv` next to the MP4. |
| **CUT PREV / PAUSE / MARK (`4`)** | Drop the previous segment, freeze the take, drop a chapter marker. |
| **MIC chip + meter** | Click for `OFF / default / each mic by name`. Live level meter; `GAIN 1x/2x/4x/8x` boosts quiet mics. Voice is auto-leveled to broadcast loudness at save; silence stays silent. |
| **Codec / BR chips** | H264/H265, 5M–40M. Screen takes use CBR at the selected bitrate; camera takes use CQP. |

Gesture director (needs Hands AI `J` on): hold ✋ = CUT, 👍 = MARK, ✌ = pause/resume, hold 👊 2s = stop. Amber preview border = hold counting, green flash = fired.

Takes land in `recordings\take_<date>_<time>.mp4` (+ `edit.csv` markers/cuts, `take.log` diagnostics). Cutting is lossless (segments start on keyframes and concatenate with no re-encode); voice stays lip-synced across cuts by construction. Every take carries H264 video + AAC voice.

### Game Tab — Gesture Game Control

The fourth tab (`GAME`) turns your hands into a mouse + keyboard for real PC games — or a built-in shooting gallery for practice:

| Control | What it does |
|---------|--------------|
| **MODE `OFF / EXTERNAL / TRAINER`** | External injects inputs into a detected game window; trainer runs the in-app gallery on the camera feed. |
| **START** | Begins tracking (auto-switches to CAM feed). Needs the Hands model. |
| **TARGET + GRAB GAME** | `ANY` window, or lock the foreground game's exe (click GRAB while the game is focused). Inputs never go to Kagerou itself. |
| **AIM `ABS / REL`, SENS, DEADZONE, SMOOTH** | Absolute (hand = cursor, default) or relative (trackpad-style) aim mapping. |
| **MOVE `HAND / FACE`** | Left-hand zones or face-lean for WASD; left pinch = sprint; face drop = crouch. |
| **FIRE / JUMP / RELOAD / PAUSE / USE** | Click to cycle bindings. Defaults: ✌/pinch = fire (auto while held), flick-up = jump, 👊 = reload, ✋ tap = pause, 👍 = use. |
| **CALIB** | Hold still 2s, click: sets aim center + face reference. |
| **VCAM VIEW** | Game canvas (crosshair/targets/HUD) rides `g_d_rgb`, so preview + virtual cam + record tap carry it automatically. |

Safety: `F12` or 2s ✋ hold kills injection instantly and releases every held key/button; injection pauses when the target window loses focus; `--gametest` runs a headless SendInput self-check (mouse square + `HI` keystrokes). Needs Hands AI model; game tab forces hand inference every 2nd frame while on.

How to play an external game (e.g. CoD):
1. Open Kagerou, go to the GAME tab, set MODE to EXTERNAL.
2. Launch the game, alt-tab back to Kagerou, click GRAB GAME (locks the game's exe — it grabs the last non-Kagerou window, since Kagerou is focused at click time).
3. Click CALIBRATE while holding your aim hand still in a comfortable center, then START.
4. Click the game window to focus it — crosshair follows your palm; ✌ hold fires, flick up jumps, 👊 reloads, ✋ tap pauses, left hand zones move (WASD), sitting crouches.
5. `F12` anytime to kill control instantly. Practice first in TRAINER mode (no game needed — 👍 starts waves).

### AutoFrame

Face-tracked auto-framing (`F7` cycles Off / Wide 2.5x / Med 2.0x / Tight 1.5x; `--ai-autoframe` for files). The crop follows your face around the frame. Camera mode only. In tutorial mode face/hands/pose/gaze run on the camera feed instead and their overlays map into your PiP box — tracking follows you, not the recorded screen — while takes stay clean (overlays are preview/virtual-cam only during recording).

### Live Pipeline — Streamer to App

```
Physical webcam
    │  DirectShow capture via FFmpeg (raw BGR24, no codec involved)
    ▼
kagerou_virtualcam.exe (GPU thread, ~30fps)
    │  H2D upload → RGB on GPU
    │  non-AI filters: denoise, CLAHE, SR, LUT, blur, sharpen,
    │      B/C, sat, gamma, WB, vignette, grain, edge, flip, lens
    │  AI filters (ORT TensorRT EP → CUDA EP fallback):
    │      FastDVDNet denoise, Depth Anything V2, RAFT flow, MediaPipe pose
    │  RGB → NV12 convert (GPU kernel), locked 640x480
    ▼  shared memory (Local\KagerouVirtualCam, ~0.5ms handoff)
kagerou_virtualcam.dll (DirectShow source filter, loaded by the call app)
    │  reads NV12 → negotiates NV12/YUY2 + size with the app
    │  pushes ~30fps into the app's filter graph
    ▼
Zoom / Teams / Google Meet / OBS
```

**Where do NVENC/NVDEC and AI fit here?**

| Component | Used in virtual camera? | Why |
|-----------|------------------------|-----|
| **Non-AI CUDA filters** | Yes, all 17 | The actual per-frame pipeline |
| **AI filters (ORT TensorRT/CUDA EP)** | Yes, optional toggles | Run on CUDA cores / TensorRT, not on codec hardware |
| **NVDEC / NVENC (Video Codec SDK)** | **No** | Nothing to decode (webcam is already raw) or encode (apps take raw NV12/YUY2). The codec SDK only matters for file transcoding (`kagerou.exe` Pipeline). AI inference never touches NVENC/NVDEC. |

Kagerou is two products sharing one GPU filter core: **file transcoding** (`kagerou.exe`: compressed file in → NVDEC → filters → NVENC → compressed file out, codecs essential) and **virtual camera** (raw webcam in → filters → raw frames out to apps, codecs uninvolved — encoding for the network happens inside Meet/Zoom itself).

### What each AI filter looks like

AI buttons live in the sidebar under **AI MODELS** (scroll down — there are 10). They start dimmed with a `...` pill and the header reads `AI MODELS ...warming up` — clicks/keys are ignored until warmup finishes (watch the console for `Warmup 1/10 → 10/10`, then `[AI] Ready`). A button showing `N/A` means its model file is missing. What you see in the preview is exactly what the meeting receives:

| Button | Model | What you should see |
|--------|-------|---------------------|
| **AI Denoise [7]** | FastDVDNet (5-frame temporal) | Same picture, visibly cleaner/smoother in low light — noise grain melts away while edges stay sharp. Needs 5 frames after toggling before the effect kicks in. Heaviest model; expect lower fps while on. |
| **AI Depth [8]** | Depth Anything V2 | Grayscale heatmap blended over your video (near = bright, far = dark). Your silhouette pops white against a darker room. Infers every 3rd frame; the overlay itself refreshes every frame so motion stays smooth. |
| **AI Flow [9]** | RAFT-Small | Motion paint: moving things bloom into color (hue = direction, brightness = speed), still things stay normal. Wave your hand to see it. Infers every 2nd frame. |
| **AI Pose [0]** | MediaPipe Pose (33 landmarks) | Red dots on your joints — shoulders, elbows, wrists, hips, knees, ankles — tracked live. Lightest model, runs every frame at full fps. Dots only appear where confidence > 0.5. |
| **AI Matting [A]** | RVM MobileNetV3 | You stay sharp, background melts into blur. True frame-to-frame memory (no flicker). Runs every 2nd frame; heaviest after denoise. |
| **AI Face [P]** | YuNet | Green box around each detected face. Infers every 2nd frame. |
| **AI Hands [J]** | BlazePalm + landmarks | Cyan dots on all 21 finger joints per hand. Infers every 3rd frame. |
| **AI Gaze [Z]** | YuNet + iris | Green dot on each iris + yellow gaze-direction dot. Needs a visible face. Every 3rd frame. |
| **AI LowLight [5]** | Zero-DCE | Dark rooms brighten up, same picture otherwise. Cheapest AI model — full fps. |
| **AI Anime [6]** | AnimeGANv3 Hayao | Whole picture restyled Hayao-like, every frame. |
| **AI Detect [F10]** | YOLOv8n (80 COCO classes) | Per-class colored boxes + name labels (person, laptop, chair…). Infers every 2nd frame. Works in camera + tutorial (PiP-mapped) modes. |
| **AutoFrame [F7]** | YuNet (same face boxes) | Crop window follows your face: Wide 2.5x → Med 2.0x → Tight 1.5x. Camera mode only. |

First run after a fresh build warms each model once (one-time TensorRT engine build, cached to `trt_engines/` — denoise takes the longest). Later runs start fast.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Camera not listed | Re-run `build.bat vcam_dll` as admin (DLL path is baked into the registry at registration time — rebuild = re-register), then fully restart the browser |
| "Camera unavailable" on select | Make sure `kagerou_virtualcam.exe` is running; restart the browser; try OBS to isolate (OBS uses DirectShow in-process) |
| `LNK1104 cannot open DLL/EXE` at build | Close Chrome (it holds the DLL) and the preview EXE, then rebuild |
| Depth `.onnx.data` errors | Fixed by absolute model paths; if a model file is missing the filter is skipped with a console message |
| AI filters slow | Expected on RTX 3050 (depth/flow/denoise are heavy); non-AI filters stay at full fps |

### Compatibility (Windows)

Registration is one-time and lives under `HKLM\SOFTWARE\Classes\CLSID` (filter CLSID + `VideoInputDeviceCategory` instance with `FriendlyName`, `CLSID`, `DevicePath`, `Description`). Rebuilding overwrites the DLL in place, so no re-registration is needed unless the DLL moves. Verified working:

| App | Status | Notes |
|-----|--------|-------|
| **Chrome / Edge** (Meet etc.) | ✓ verified | Needs full browser restart after (re-)registration; EXE must be running |
| **OBS Studio** | ✓ verified | Sources → + → Video Capture Device → Device dropdown |
| **Zoom / Teams / Discord** | Should work | Standard DirectShow capture clients, same API as OBS |

Known boundaries (by design, not bugs):

- **Windows 10/11 64-bit only.** The filter is a 64-bit in-proc COM server: 32-bit apps can't load it (WOW64 can't cross the bitness), and there is no 32-bit build.
- **No UWP / Microsoft Store apps** (Windows Camera app, etc.). Those enumerate cameras via Media Foundation, which doesn't see DirectShow filters. A Media Foundation virtual camera is a separate driver model and out of scope.
- **Admin required once** for the HKLM registration. Unregister anytime with `virtualcam\unregister.bat` (admin).

### Files

```
virtualcam/
├── kagerou_vcam_shm.h      # shared-memory frame protocol (writer + reader)
├── kagerou_vcam_filter.h/.cpp  # DirectShow source filter (pin, allocator, caps)
├── kagerou_vcam_dll.cpp    # COM exports + HKLM registration
├── kagerou_vcam.def        # DLL export table (DllRegisterServer, ...)
└── unregister.bat          # removes registry entries (run as admin)
examples/02_virtualcam_demo.cu  # capture + GPU pipeline + preview UI
tests/test_vcam_enum.cpp    # device enumeration + live graph test
```

---

## Enable Full NVDEC/NVENC

The SDK supports full hardware decode/encode through the NVIDIA Video Codec SDK. This gives you:
- **NVDEC:** Hardware H.264/H.265 decode (doesn't use CUDA cores). AV1 decode via FFmpeg (software).
- **NVENC:** Hardware H.264/H.265 encode (doesn't use CUDA cores)
- **Zero CUDA core usage** for H.264/H.265 decode/encode (dedicated silicon)

### Steps

1. **Download the SDK:**
   - Go to: https://developer.nvidia.com/video-codec-sdk
   - Free registration required
   - Download "NVIDIA Video Codec SDK" (ZIP file)

2. **Extract headers:**
   ```
   Extract to: src/HardLab/Kagerou/sdk/nvidia_video_codec_sdk/
   Ensure these files exist:
     sdk/nvidia_video_codec_sdk/Interface/nvEncodeAPI.h
     sdk/nvidia_video_codec_sdk/Interface/nvcuvid.h
     sdk/nvidia_video_codec_sdk/Interface/cuviddec.h
   ```

3. **Build with SDK:**
   ```batch
   build.bat sdk                # builds with NVDEC/NVENC enabled
   build.bat all sdk            # builds everything with SDK
   ```

4. **What changes:**
   - Without SDK: `KAGEROU_NO_VCODEC_SDK` is defined → stub mode
   - With SDK: Real NVDEC/NVENC initialization and hardware encode/decode
   - The decoder/encoder automatically fall back to stub if SDK headers aren't found

### NVDEC/NVENC Code Path

When built with `build.bat sdk`:

```
Decoder::init()
├── cuInit(0)                         → init CUDA driver API
├── cuDeviceGet() + cuCtxCreate()     → create CUDA context
├── cuvidCtxLockCreate()              → thread-safe decoder access
└── cuvidCreateVideoParser()          → CUVID parser with 3 callbacks

Decoder::decode(nalu)
├── Prepend start code (00 00 00 01)  → Annex-B format for parser
├── cuCtxPushCurrent(cu_ctx)          → context for parser callbacks
├── cuvidParseVideoData()             → feed to CUVID parser
│   ├── [SPS detected]  → handle_video_sequence → cuvidCreateDecoder
│   ├── [Picture ready] → handle_picture_decode → cuvidDecodePicture
│   └── [Display ready] → handle_picture_display → set got_frame
├── cuvidCtxLock / cuvidMapVideoFrame → map NV12 to GPU memory
├── cudaMemcpy2D × 2                  → Y + UV planes to pipeline buffer
└── cuvidUnmapVideoFrame + unlock     → release decoder resources

Encoder::init()
├── LoadLibrary("nvEncodeAPI64.dll")   → load NVENC from driver
├── NvEncodeAPICreateInstance()        → get API function table
├── Version fallback loop              → try API versions for driver compatibility
├── NvEncOpenEncodeSessionEx()         → create encoder session
├── NvEncGetEncodePresetConfigEx()     → get P1 preset (lowest latency)
└── NvEncInitializeEncoder()           → configure codec/bitrate/GOP

Encoder::encode(frame)
├── cudaStreamSynchronize(0)               → ensure filter kernels complete
├── cudaMemcpy(gpu_input_buf, frame)       → GPU→GPU copy to registered buffer
├── NvEncMapInputResource()                → map for this encode
├── NvEncEncodePicture()                   → NVENC reads directly from GPU
├── NvEncUnmapInputResource()              → release mapping
├── NvEncLockBitstream()                   → lock output for read
├── memcpy(bitstream, output)              → read compressed H.264/H.265
└── NvEncUnlockBitstream()                 → release output
```

### MP4 Input Format Conversion

MP4 containers use AVCC format. CUVID parser expects Annex-B. The decoder handles this transparently:

```
MP4 AVCC format (from minimp4):
┌──────────────────┬──────────────────┬──────────────────┐
│ 4-byte length    │ 4-byte length    │ 4-byte length    │
│ (big-endian)     │ (big-endian)     │ (big-endian)     │
│ [NALU payload]   │ [NALU payload]   │ [NALU payload]   │
└──────────────────┴──────────────────┴──────────────────┘

Converted to Annex-B for CUVID parser:
┌──────────┬──────────────────┬──────────┬──────────────────┐
│ 00 00 00 │                  │ 00 00 00 │                  │
│ 01       │ [NALU payload]   │ 01       │ [NALU payload]   │
└──────────┴──────────────────┴──────────┴──────────────────┘

SPS + PPS extracted from avcC box:
┌──────────┬──────────────────┬──────────┬──────────────────┐
│ 00 00 00 │ SPS (profile,    │ 00 00 00 │ PPS (quantization│
│ 01       │ level, res)      │ 01       │ params)          │
└──────────┴──────────────────┴──────────┴──────────────────┘
```

---

## API Reference

### Pipeline API

```cpp
#include "pipeline.cu"
#include "fileio.h"

// configure
kagerou::PipelineConfig cfg;
cfg.decoder.codec = kagerou::VideoCodec::kH264;   // or kH265, kAV1
cfg.encoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.width = 1920;
cfg.encoder.height = 1080;
cfg.encoder.fps = 30;
cfg.encoder.bitrate_kbps = 5000;
cfg.encoder.gop_size = 30;       // keyframe every 30 frames
cfg.encoder.rc = kagerou::RateControl::kVBR;  // or kCBR, kCQP
cfg.color_range = kagerou::ColorRange::kLimited;  // NVDEC outputs limited-range

// enable filters
cfg.denoise.enabled = true;
cfg.denoise.sigma_spatial = 15.0f;
cfg.denoise.sigma_color = 25.0f;
cfg.denoise.kernel_size = 5;

cfg.scale.enabled = true;
cfg.scale.target_width = 1280;
cfg.scale.target_height = 720;
cfg.scale.interpolation = 1;    // 0=bilinear, 1=bicubic

cfg.super_res.enabled = true;
cfg.super_res.scale_factor = 2.0f;
cfg.super_res.sharpen_strength = 0.5f;

// init and run
kagerou::Pipeline pipeline;
pipeline.init(cfg);

kagerou::fileio::BitstreamReader reader;
reader.open("input.h264");

size_t nalu_size;
const uint8_t* nalu;
while ((nalu = reader.next_nalu(nalu_size)) != nullptr) {
    std::vector<uint8_t> encoded;
    pipeline.process_frame(nalu, nalu_size, encoded);
    // encoded contains compressed output
}

pipeline.print_stats();
pipeline.destroy();
```

### Pipeline Methods

| Method | Description |
|--------|-------------|
| `init(config)` | Initialize decoder, encoder, CUDA stream. Returns `Error::kOk` on success. |
| `process_frame(nalu, size, out)` | Full pipeline: NVDEC decode → GPU filters → NVENC encode. Input is H.264/H.265 Annex-B NALU. Output is re-encoded bitstream. Handles frame queuing internally (one NALU may produce multiple decoded frames). |
| `process_raw_frame(nv12, w, h, out)` | **Raw frame path (Path B):** Skip NVDEC decode. Takes a raw NV12 buffer (e.g. from camera), runs GPU filters, encodes via NVENC. Use this for USB cameras, capture cards, screen capture. |
| `process_frame_pair(bs_a, sz_a, bs_b, sz_b, out)` | **Frame interpolation:** Decode two consecutive NALUs, apply frame_blend() between them (if `frame_interp.enabled`), output: frame_a, blended, frame_b. For 2x FPS conversion (30→60). |
| `process_sw(out)` | Software decode path for AV1/VP9. Iterates pre-decoded frames from `load_sw_input()`, applies filters, encodes. |
| `flush(out)` | Drain decoder DPB + encoder buffer. Call at end of stream to get all remaining frames. |
| `print_stats()` | Print decoded/encoded/flushed counts, total time, FPS. |
| `destroy()` | Free all GPU memory, CUDA streams, decoder/encoder resources. |

### Direct Filter API

```cpp
#include "filters.h"

// All functions take GPU pointers (uint8_t*) and a CUDA stream
kagerou::filters::nv12_to_rgb(d_nv12, d_rgb, w, h, stream);
kagerou::filters::nv12_to_rgb_limited(d_nv12, d_rgb, w, h, stream);  // BT.709 limited-range
kagerou::filters::rgb_to_nv12(d_rgb, d_nv12, w, h, stream);
kagerou::filters::rgb_to_nv12_limited(d_rgb, d_nv12, w, h, stream);  // BT.709 limited-range
kagerou::filters::resize_bilinear(d_src, d_dst, sw, sh, dw, dh, channels, stream);
kagerou::filters::resize_bicubic(d_src, d_dst, sw, sh, dw, dh, channels, stream);
kagerou::filters::resize_nv12_bilinear(d_src, d_dst, sw, sh, dw, dh, stream);  // NV12-native
kagerou::filters::resize_nv12_bicubic(d_src, d_dst, sw, sh, dw, dh, stream);  // NV12-native
kagerou::filters::denoise_bilateral(d_src, d_dst, w, h, ch, sigma_s, sigma_c, kernel, stream);
kagerou::filters::denoise_nv12_bilateral(d_src, d_dst, w, h, sigma_s, sigma_c, kernel, stream);  // NV12-native
kagerou::filters::super_res_2x(d_src, d_dst, sw, h, channels, sharpen, stream);
kagerou::filters::frame_blend(d_a, d_b, d_out, w, h, channels, alpha, stream);

// Creative / artistic filters
kagerou::filters::clahe_nv12(d_src, d_dst, w, h, clip_limit, tile_size, stream);   // NV12-native
kagerou::filters::clahe_rgb(d_src, d_dst, w, h, clip_limit, tile_size, stream);
kagerou::filters::lut3d_rgb(d_src, d_dst, w, h, d_lut, lut_res, strength, stream);
kagerou::filters::gaussian_blur(d_src, d_dst, w, h, channels, sigma, stream);
kagerou::filters::sharpen_rgb(d_src, d_dst, w, h, strength, stream);
kagerou::filters::brightness_contrast(d_src, d_dst, w, h, brightness, contrast, stream);
kagerou::filters::saturation_rgb(d_src, d_dst, w, h, factor, stream);
kagerou::filters::gamma_rgb(d_src, d_dst, w, h, gamma, stream);
kagerou::filters::vignette_rgb(d_src, d_dst, w, h, strength, stream);
kagerou::filters::film_grain_rgb(d_data, w, h, amount, seed, stream);
kagerou::filters::directional_blur(d_src, d_dst, w, h, channels, angle_deg, length, stream);
kagerou::filters::edge_detect_rgb(d_src, d_dst, w, h, stream);
kagerou::filters::white_balance_rgb(d_src, d_dst, w, h, temperature, tint, stream);
kagerou::filters::lens_distortion(d_src, d_dst, w, h, channels, k1, stream);
kagerou::filters::flip_horizontal_gpu(d_src, d_dst, w, h, stream);
kagerou::filters::flip_vertical_gpu(d_src, d_dst, w, h, stream);

// Transform filters
kagerou::filters::crop_rgb(d_src, d_dst, src_w, src_h, dst_w, dst_h, crop_x, crop_y, 3, stream);
kagerou::filters::crop_nv12(d_src, d_dst, src_w, src_h, dst_w, dst_h, crop_x, crop_y, stream);
kagerou::filters::pad_rgb(d_src, d_dst, src_w, src_h, dst_w, dst_h, pad_x, pad_y, pad_r, pad_g, pad_b, stream);
kagerou::filters::chroma_key_rgb(d_src, d_dst, w, h,
    hue_min, hue_max, sat_min, val_min, spill_suppress, bg_r, bg_g, bg_b, blend_edge, stream);
kagerou::filters::bg_blur_rgb(d_src, d_dst, w, h, center_x, center_y, focus_radius, blur_strength, stream);
kagerou::filters::temporal_denoise_rgb(d_prev, d_curr, d_dst, w, h, strength, stream);
kagerou::filters::temporal_denoise_nv12(d_prev, d_curr, d_dst, w, h, strength, stream);
kagerou::filters::hdr_tone_map_rgb(d_src, d_dst, w, h, method, peak_nits, stream);
kagerou::filters::temporal_stabilize_rgb(d_prev, d_curr, d_dst, w, h, d_accum, block_size, search_range, stream);
kagerou::filters::warp_translate_rgb(d_src, d_dst, w, h, warp_dx, warp_dy, stream);
```

### GpuBuffer Helper

```cpp
#include "common.h"

kagerou::GpuBuffer buf;
buf.alloc(1920 * 1080 * 3);     // cudaMalloc
buf.upload(host_ptr, size);       // cudaMemcpy H2D
buf.download(host_ptr, size);     // cudaMemcpy D2H
buf.free();                        // cudaFree
```

### GPU Compositor

Composites N NV12 source frames into a single output grid on GPU. For multi-stream video calls, surveillance grids, or picture-in-picture layouts.

```cpp
#include "filters.h"

// Each tile describes one source frame's position in the output grid
kagerou::filters::CompositeTile tiles[4];
for (int i = 0; i < 4; i++) {
    tiles[i].d_src = d_peer_frames[i];  // device NV12 pointer
    tiles[i].src_w = 1920;
    tiles[i].src_h = 1080;
    tiles[i].dst_x = (i % 2) * 960;    // 2x2 grid in 1920x1080
    tiles[i].dst_y = (i / 2) * 540;
    tiles[i].tile_w = 960;
    tiles[i].tile_h = 540;
    tiles[i].active = true;
}

uint8_t* d_composite;
cudaMalloc(&d_composite, 1920 * 1080 * 3 / 2);

kagerou::filters::composite_nv12(tiles, 4, d_composite, 1920, 1080, stream);

// Or auto-compute grid layout:
int cols, rows;
uint32_t tile_w, tile_h;
kagerou::filters::compute_composite_grid(4, 1920, 1080, cols, rows, tile_w, tile_h);
// cols=2, rows=2, tile_w=960, tile_h=540
```

### CUDA Graphs (Fixed Filter Chains)

Capture a fixed filter chain as a CUDA graph once, then replay it on every frame. Eliminates kernel launch overhead (~5-10μs per launch). Best for pipelines where the filter chain doesn't change at runtime.

```cpp
kagerou::Pipeline pipeline;
pipeline.init(cfg);

// After init, capture the active filter chain as a graph
pipeline.capture_graph();

// On each frame: decode normally, then replay the graph for filters
std::vector<uint8_t> encoded;
pipeline.process_frame(nalu, nalu_size, encoded);
// The filter chain runs via graph replay (faster than individual launches)

// To re-capture (e.g., after toggling filters):
pipeline.capture_graph();

pipeline.destroy();
```

### Configuration Types

```cpp
enum class VideoCodec   { kH264, kH265, kVP9, kAV1 };
enum class AudioCodec   { kAAC, kMP3, kOpus, kFLAC, kNone };
enum class RateControl  { kCBR, kVBR, kCQP, kCRF };
enum class PixelFormat  { kRGB, kRGBA, kNV12, kYUV420P, kP010 };
enum class ColorRange   { kLimited, kFull };  // BT.709 limited (NVDEC) vs full range
```

---

## Real-Time GPU Filtering & Streaming

Kagerou provides two integration paths for real-time video processing. Both run filters entirely on GPU — zero CPU copies between stages.

### Path A: Compressed Stream → Decode → Filter → Encode

Feed H.264/H.265 NALUs (from RTSP camera, file, network stream). NVDEC decodes on dedicated hardware, CUDA filters process on GPU cores, NVENC re-encodes.

```cpp
#include "pipeline.cu"

kagerou::PipelineConfig cfg;
cfg.decoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.width = 1920;
cfg.encoder.height = 1080;
cfg.encoder.fps = 30;
cfg.encoder.bitrate_kbps = 5000;
cfg.denoise.enabled = true;
cfg.scale.enabled = true;
cfg.scale.target_width = 1280;
cfg.scale.target_height = 720;

kagerou::Pipeline pipeline;
pipeline.init(cfg);

// Each call: NVDEC decode → GPU filters → NVENC encode
while (has_nalu_data) {
    std::vector<uint8_t> encoded;
    pipeline.process_frame(nalu_data, nalu_size, encoded);
    // encoded = filtered + re-encoded H.264 bitstream
    send_to_network(encoded);
}

pipeline.flush(remaining);  // drain decoder + encoder
pipeline.destroy();
```

**Latency budget (1080p):**

| Stage | Time |
|-------|------|
| NVDEC decode | ~0.5ms |
| GPU filters (denoise + scale) | ~2-3ms |
| NVENC encode | ~0.5ms |
| **Total** | **~4-5ms** |

**Measured throughput (RTX 3050 6GB, 1080p 60fps H.264, 600 frames):**

| Filter combination | FPS | Per-frame |
|--------------------|-----|-----------|
| No filters (plain encode) | 210 | 4.8 ms |
| Crop | 305 | 3.3 ms |
| Denoise | 191 | 5.2 ms |
| Chroma key | 195 | 5.1 ms |
| HDR tone map (ACES) | 217 | 4.6 ms |
| Temporal denoise | 188 | 5.3 ms |
| Background blur | 51 | 19.7 ms |

### Path B: Raw Camera Frame → Filter → Encode

For USB cameras, capture cards, screen capture — anything that outputs raw NV12/RGB. Skip NVDEC, upload directly to GPU.

```cpp
#include "pipeline.cu"

kagerou::PipelineConfig cfg;
cfg.encoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.width = 1920;
cfg.encoder.height = 1080;
cfg.encoder.fps = 30;
cfg.denoise.enabled = true;
cfg.super_res.enabled = true;

kagerou::Pipeline pipeline;
pipeline.init(cfg);

while (camera.is_open()) {
    camera.capture_frame();  // fills camera.frame_data (NV12)

    std::vector<uint8_t> encoded;
    pipeline.process_raw_frame(
        camera.frame_data, 1920, 1080, encoded);
    // encoded = filtered + NVENC-encoded H.264
    stream_or_display(encoded);
}

pipeline.destroy();
```

### Path C: Direct Filter Functions (No Encode)

Use individual filter functions on GPU buffers without the pipeline. For custom processing, display-only, or integration with existing encode pipelines.

```cpp
#include "filters.h"
#include <cuda_runtime.h>

// Allocate GPU buffers
uint8_t *d_input, *d_rgb, *d_filtered;
cudaMalloc(&d_input,  1920 * 1080 * 3 / 2);  // NV12
cudaMalloc(&d_rgb,    1920 * 1080 * 3);       // RGB
cudaMalloc(&d_filtered, 1920 * 1080 * 3);     // RGB

cudaStream_t stream;
cudaStreamCreate(&stream);

// Upload camera frame to GPU
cudaMemcpy(d_input, camera_frame, 1920*1080*3/2, cudaMemcpyHostToDevice);

// Convert NV12 → RGB
kagerou::filters::nv12_to_rgb_limited(d_input, d_rgb, 1920, 1080, stream);

// Chain filters on RGB
kagerou::filters::denoise_bilateral(d_rgb, d_filtered, 1920, 1080, 3,
    15.0f, 25.0f, 5, stream);
kagerou::filters::sharpen_rgb(d_filtered, d_filtered, 1920, 1080,
    0.8f, stream);
kagerou::filters::brightness_contrast(d_filtered, d_filtered, 1920, 1080,
    10.0f, 1.2f, stream);

// Read back to display or encode
cudaStreamSynchronize(stream);
// d_filtered now contains the processed RGB frame
```

### Multi-Stream Processing

Each `Pipeline` instance is fully independent — safe for parallel use across multiple camera streams. Use the GPU compositor to tile decoded frames into a single output for display/encoding.

```cpp
const int NUM_CAMERAS = 4;
kagerou::Pipeline pipelines[NUM_CAMERAS];
uint8_t* d_peer_frames[NUM_CAMERAS];

for (int i = 0; i < NUM_CAMERAS; i++) {
    kagerou::PipelineConfig cfg;
    cfg.decoder.codec = kagerou::VideoCodec::kH264;
    cfg.encoder.codec = kagerou::VideoCodec::kH264;
    cfg.encoder.width = 1920;
    cfg.encoder.height = 1080;
    cfg.denoise.enabled = true;
    pipelines[i].init(cfg);
    cudaMalloc(&d_peer_frames[i], 1920 * 1080 * 3 / 2);
}

// GPU compositor: tile all peers into a 2x2 grid
uint8_t* d_composite;
cudaMalloc(&d_composite, 1920 * 1080 * 3 / 2);

int cols, rows;
uint32_t tile_w, tile_h;
kagerou::filters::compute_composite_grid(NUM_CAMERAS, 1920, 1080, cols, rows, tile_w, tile_h);

// Each camera thread decodes to its own frame
void camera_thread(int cam_id) {
    while (running) {
        std::vector<uint8_t> nalu = receive_nalu(cam_id);
        std::vector<uint8_t> encoded;
        pipelines[cam_id].process_frame(nalu.data(), nalu.size(), encoded);

        // Decode-only: get NV12 frame for compositor
        kagerou::Frame decoded;
        pipelines[cam_id].decoder.decode(nalu.data(), nalu.size(), decoded);
        cudaMemcpy(d_peer_frames[cam_id], decoded.d_data,
                   1920 * 1080 * 3 / 2, cudaMemcpyDeviceToDevice);
        decoded.free_gpu();
    }
}

// Main thread: composite + encode
while (running) {
    kagerou::filters::CompositeTile tiles[NUM_CAMERAS];
    for (int i = 0; i < NUM_CAMERAS; i++) {
        tiles[i].d_src = d_peer_frames[i];
        tiles[i].src_w = 1920; tiles[i].src_h = 1080;
        tiles[i].dst_x = (i % cols) * tile_w;
        tiles[i].dst_y = (i / cols) * tile_h;
        tiles[i].tile_w = tile_w; tiles[i].tile_h = tile_h;
        tiles[i].active = true;
    }
    kagerou::filters::composite_nv12(tiles, NUM_CAMERAS, d_composite, 1920, 1080);
    // d_composite is the tiled output — encode or display
}
```

### Frame Interpolation (FPS Conversion)

Convert 30fps to 60fps by blending consecutive frames on GPU.

```cpp
cfg.frame_interp.enabled = true;
cfg.encoder.fps = 60;  // target output FPS

pipeline.init(cfg);

// Feed pairs of consecutive NALUs
while (has_frame_pair) {
    std::vector<std::vector<uint8_t>> frames;
    pipeline.process_frame_pair(
        nalu_a, nalu_a_size,
        nalu_b, nalu_b_size,
        frames);
    // frames[0] = original frame A (encoded)
    // frames[1] = blended interpolated frame (encoded)
    // frames[2] = original frame B (encoded)
    for (auto& f : frames) send_to_display(f);
}
```

---

## Project Structure

```
Kagerou/
├── include/kagerou/
│   ├── common.h            # Frame struct, Error codes, GpuBuffer, CUDA macros
│   ├── config.h            # PipelineConfig, DecoderConfig, EncoderConfig, filter configs
│   ├── filters.h           # Filter function declarations
│   ├── filters_common.h    # Shared block_2d/grid_2d helpers
│   ├── fileio.h            # BitstreamReader, MP4Demuxer, VideoFile, file I/O
│   └── ai/
│       ├── ort_wrapper.h   # ORT 1.20.1 wrapper (TrtSession, AiInference singleton)
│       └── ai_filters.h   # AI filter declarations + FaceBox/FaceLandmark structs
├── src/
│   ├── decoder.cu          # NVDEC hardware decoder (full SDK + stub fallback)
│   ├── encoder.cu          # NVENC hardware encoder (full SDK + stub fallback)
│   ├── pipeline.cu         # Pipeline orchestrator (decode→filter→encode chain)
│   ├── ai/
│   │   ├── ort_wrapper.cu  # ORT session management, CUDA EP setup
│   │   ├── ai_preprocess.cu # Shared kernels: resize, normalize, NCHW, draw
│   │   ├── ai_pose_estimation.cu # MediaPipe Pose 33 landmarks
│   │   ├── ai_depth.cu          # Depth Anything V2 depth estimation
│   │   ├── ai_flow.cu           # RAFT optical flow
│   │   └── ai_denoise.cu        # FastDVDNet temporal denoising
│   └── filters/
│       ├── color_convert.cu   # NV12↔RGB, YUV420P↔RGB, HDR PQ→SDR
│       ├── scale.cu           # Bilinear + bicubic resize (RGB + NV12-native)
│       ├── denoise.cu         # Bilateral edge-preserving denoise (RGB + NV12-native)
│       ├── super_res.cu       # 2x bicubic upscale + unsharp mask
│       ├── frame_interp.cu    # Alpha blending for frame interpolation
│       ├── clahe.cu           # CLAHE adaptive contrast (RGB + NV12-native)
│       ├── lut.cu             # 3D LUT color grading (6 presets)
│       ├── creative.cu        # Blur, sharpen, B/C, sat, gamma, vignette, grain, edge, WB, lens, flip
│       ├── transforms.cu      # Crop, pad, chroma key, bg blur, temporal denoise/stab, HDR tone map, warp
│       └── compositor.cu      # GPU multi-stream composite (NV12 tiled layout)
├── examples/
│   ├── 00_kagerou.cu         # Unified CLI: transcode, batch, benchmark, test
│   └── 02_virtualcam_demo.cu # Virtual camera: capture + GPU pipeline + preview UI
├── virtualcam/
│   ├── kagerou_vcam_shm.h      # shared-memory frame protocol
│   ├── kagerou_vcam_filter.h/.cpp  # DirectShow source filter
│   ├── kagerou_vcam_dll.cpp    # COM exports + registration
│   └── kagerou_vcam.def        # DLL export table
├── tests/
│   ├── test_filters.cu       # 12 unit tests
│   └── test_vcam_enum.cpp    # virtualcam enumeration + live graph test
├── models/                   # ONNX AI models (not in git — user downloads)
│   ├── pose/                 # pose_landmark.onnx
│   ├── depth/                # depth_anything_v2.onnx
│   ├── flow/                 # raft_small.onnx
│   └── denoise/              # fastdvdnet.onnx
├── sdk/
│   ├── nvidia_video_codec_sdk/ # NVIDIA Video Codec SDK (user downloads)
│   │   └── Interface/
│   │       ├── nvEncodeAPI.h
│   │       ├── nvcuvid.h
│   │       └── cuviddec.h
│   └── onnxruntime/           # ONNX Runtime GPU 1.20.1 (user downloads)
│       ├── include/           # ORT C++ headers
│       └── lib/               # ORT libraries (.lib + .dll)
├── docs/
│   ├── ENCODING_PROTOCOLS.md # Full guide: codecs, containers, color spaces
│   └── QUICKSTART.md         # Quick start guide
├── build.bat                 # Build script (stub + SDK + ONNX modes)
└── README.md
```

---

## GPU Optimizations Applied

All of these are already implemented in the codebase. The goal: zero CPU involvement between decode and encode — everything on GPU.

| Optimization | What it does | Status |
|---|---|---|
| **GPU-native pipeline** | Decode → filter → encode all stay on GPU memory. No CPU copies between stages. | Done |
| **NVDEC hardware decode** | CUVID parser with 3 callbacks, H.264 HW decode on dedicated silicon | Done |
| **NVENC hardware encode** | NvEncodeAPI with version fallback (13.1→12.2→12.1→12.0→11.1), H.264/H.265 HW encode | Done |
| **Zero-copy encoder input** | NvEncRegisterResource on CUDA device pointer — NVENC reads directly from GPU memory. No CPU staging. | Done |
| **Zero-copy filter chain** | Filters read/write GPU buffers directly. NV12→RGB→filter→RGB→NV12 all on device | Done |
| **NV12-native denoise** | Bilateral filter on NV12 directly (Y bilateral + UV box blur). Skip RGB round-trip. | Done |
| **NV12-native scale** | Bilinear/bicubic resize on NV12 Y+UV planes separately. Skip RGB round-trip. | Done |
| **NV12-native super-res** | Y plane super-res (bicubic+sharpen), bilinear UV upscale. No RGB conversion. | Done |
| **NV12-native frame-interp** | Alpha blend entire NV12 buffer as raw bytes (linear blend is format-agnostic). | Done |
| **Limited-range color** | BT.709 limited-range (Y:16-235) conversion for NVDEC output compatibility | Done |
| **Auto-resolution detection** | Encoder initializes at detected resolution from CUVID sequence callback | Done |
| **Frame-by-frame encode** | `frameIntervalP=1` (no B-frames in encode), LOW_LATENCY tuning, P1 preset | Done |
| **Flush-on-demand** | Encoder flush method drains NVENC buffer on stream end | Done |
| **Decoder flush** | DPB drain via null-packet flush at end-of-stream (recovers B-frames) | Done |
| **Dynamic resolution** | Pipeline handles any input resolution without hardcoding | Done |
| **Per-frame bitstream** | Each encoded frame produces independent bitstream (no buffering delay) | Done |
| **GPU compositor** | Multi-stream NV12 composite kernel: N decoded frames → tiled output on GPU. Bilinear scale + blit, zero CPU. | Done |
| **CUDA graphs** | Capture fixed filter chains as CUDA graph, replay per frame. Eliminates kernel launch overhead. | Done |
| **Transform filters** | Crop, pad, chroma key, bg blur, temporal denoise/stab, HDR tone map on GPU. | Done |

### Measured Performance (RTX 3050 6GB Laptop, 1080p 60fps H.264, 600 frames)

| Pipeline | FPS | Per-frame |
|----------|-----|-----------|
| No filters (plain encode) | 210 | 4.8 ms |
| Denoise only | 191 | 5.2 ms |
| Crop only | 305 | 3.3 ms |
| Chroma key only | 195 | 5.1 ms |
| Directional blur only | 234 | 4.3 ms |
| White balance only | 194 | 5.2 ms |
| Lens distortion only | 180 | 5.6 ms |
| Flip only | 195 | 5.1 ms |
| Edge detect only | 132 | 7.6 ms |
| Film grain only | 117 | 8.6 ms |
| HDR tone map (Reinhard) | 333 | 3.0 ms |
| HDR tone map (ACES) | 217 | 4.6 ms |
| Temporal denoise (0.3) | 188 | 5.3 ms |
| Temporal stab | 39 | 25.4 ms |
| Background blur (12) | 51 | 19.7 ms |

---

## What's Next

### Priority 1: Fix known issues

- [x] **Studio-range color conversion** — Fixed: `ColorRange::kLimited` default, `nv12_to_rgb_limited` / `rgb_to_nv12_limited` kernels
- [x] **Decoder flush** — Fixed: `ulMaxDisplayDelay = 16`, `ulNumDecodeSurfaces = 16`, `flush()` method drains DPB (600/600 frames at 44+ dB PSNR)
- [x] **NV12-native denoise** — Fixed: `denoise_nv12_bilateral()` runs bilateral on Y plane + box blur on UV (no RGB conversion needed)
- [x] **Decoder flush fix** — Fixed: CUVID parser returns B-frames with display delay + null-packet flush

### Priority 2: Performance optimizations

- [x] **NV12-native scale** — Fixed: `resize_nv12_bilinear()` / `resize_nv12_bicubic()` resize Y+UV planes separately
- [x] **Zero-copy encoder input** — Fixed: `NvEncRegisterResource` on CUDA device pointer, NVENC reads directly from GPU (2.1x speedup)
- [x] **Wire up FramePool** — Fixed: `pool.acquire()`/`pool.release()` with in-use tracking, avoids cudaMalloc/cudaFree per frame
- [x] **Remove CUDA stream sync points** — Fixed: removed 8 redundant `cudaStreamSynchronize` calls (same-stream serialization handles ordering)
- [x] **NV12-native super-res** — Fixed: Y plane super-res (bicubic+sharpen), bilinear UV upscale. No RGB conversion needed (1.77x speedup)
- [x] **Eliminate RGB round-trip** — Fixed: denoise/clahe/scale/super-res/frame-interp all run on NV12 directly. RGB conversion only for LUT. ALL filters 2.89x faster
- [x] **Multi-stream support** — Fixed: eliminated all static mutable state (`static Frame temp`, `static cached_lut`). Each Pipeline instance is fully independent and safe for parallel use. Note: CLAHE uses a shared GPU buffer (`g_tile_cdf`) — concurrent CLAHE across multiple Pipeline instances may have minor contention.
- [x] **GPU compositor** — Fixed: `composite_nv12()` kernel composites N NV12 frames into a tiled output on GPU. Bilinear scale + blit, zero CPU involvement.
- [x] **CUDA graphs** — Fixed: `capture_graph()` / `replay_graph()` captures fixed filter chains as CUDA graph, eliminating kernel launch overhead.

### Priority 3: Non-AI filters

- [x] **Color grading / LUT** — 3D LUT application on GPU (film looks, color correction). 6 built-in presets: warm, cool, cinematic, vintage, contrast, desat.
- [x] **CLAHE** — Contrast Limited Adaptive Histogram Equalization on GPU (NV12 + RGB). Per-tile histogram, clip, CDF, bilinear interpolation.
- [x] **Gaussian Blur** — variable sigma, separable 2-pass on GPU
- [x] **Unsharp Mask Sharpen** — standalone sharpen kernel
- [x] **Exposure / Brightness / Contrast** — per-pixel offset + multiply
- [x] **Saturation Control** — luminance-weighted saturation adjust
- [x] **Gamma Correction** — power-law gamma on GPU
- [x] **Vignette** — radial lens darkening effect
- [x] **Film Grain** — PRNG noise overlay for film look
- [x] **Directional Blur** — motion blur simulation at configurable angle + length
- [x] **Edge Detect** — Sobel gradient magnitude (grayscale output)
- [x] **White Balance** — temperature / tint color shift
- [x] **Lens Distortion** — barrel / pincushion distortion
- [x] **GPU Flip** — horizontal / vertical flip on GPU
- [x] **Crop** — extract sub-region from frame (RGB + NV12). `--crop WxH+X+Y`
- [x] **Pad** — add border padding around frame. API: `pad_rgb()`
- [x] **Chroma Key** — green screen removal via HSV keying with soft edge blending and spill suppression. Tunable hue/sat/val thresholds. `--chroma-key`
- [x] **Background Blur** — center-weighted depth-of-field blur (non-AI heuristic). `--bg-blur`
- [x] **Temporal Denoise** — inter-frame temporal averaging for noise reduction. Ring buffer of previous frames. `--temporal-denoise`
- [x] **Temporal Stabilization** — block-matching motion estimation + warp translation. `--temporal-stab`
- [x] **HDR Tone Map** — Reinhard and ACES filmic tone mapping operators. SDR→HDR normalization with configurable peak nits. `--hdr`
- [x] **LUT Interpolation** — custom .cube files (API available, 6 built-in presets via CLI)

All 24 non-AI GPU filters are **complete and verified**.

### Priority 4: AI features (ONNX Runtime GPU)

- [x] **ORT wrapper** — ONNX Runtime 1.20.1 with CUDA EP, session management, IoBinding
- [x] **TensorRT EP** — primary provider (FP16, engine cache in `trt_engines/`), CUDA EP fallback per model
- [x] **Pose estimation** — MediaPipe Pose 33 landmarks (256x256 input), normalized x/y + sigmoid visibility
- [x] **Depth estimation** — Depth Anything V2 (518x518 input), per-frame normalize + grayscale overlay
- [x] **Optical flow** — RAFT-Small (360x480 input, two-frame), Middlebury color overlay
- [x] **AI denoise** — FastDVDNet temporal (5-frame window batch, calibrated 0.05 noise map)
- [x] **Temporal processing** — FastDVDNet 5-frame ring buffer + optical-flow frame pair
- [x] **Shared preprocessing** — resize, normalize, NCHW/NHWC, draw kernels (all GPU)
- [x] **GPU visualization** — draw_rect, draw_circle, flow_to_rgb, depth_to_rgb + frame_blend overlays
- [x] **Pipeline dispatch** — AI filters wired into file pipeline loop
- [x] **CLI flags** — AI filters exposed via `--ai-*` flags
- [x] **Virtual camera UI** — all 11 AI filters toggleable live with load/warmup gating + overlays
- [x] **Warmup + diagnostics** — background one-time TRT engine build, `build.bat ai_diag` model probe
- [x] **Background removal** — covered by RVM matting (`ai_matting()` + bg-blur composite); `ai_bg_removal()` header stub superseded
- [x] **Face detection (YuNet)** — `ai_face()`, drives autoframe + gaze base
- [x] **Object detection** — YOLOv8n (`ai_detect()`): 80 COCO classes, boxes + labels, live/record/transcode
- [x] **Gaze estimation** — MediaPipe iris (`ai_gaze()`): iris dots + gaze vectors (eye-contact redirect removed by design)
- [x] **Auto framing** — face-tracked crop (Wide 2.5x / Med 2.0x / Tight 1.5x), camera + tutorial modes
- [x] **Hand tracking + gestures** — BlazePalm + landmarks (`ai_hands()`), heuristic CUT/MARK/PAUSE/STOP
- [x] **Low-light / anime style** — Zero-DCE (`ai_lowlight()`), AnimeGANv3 Hayao (`ai_anime()`)
- [x] **Record burn-in** — tracking overlays drawn before the take tap, burned into saved MP4
- [ ] **Single-frame denoise (NAFNet)** — cheaper than FastDVDNet for weak GPUs
- [ ] **Studio lighting** — portrait relighting (key light, fill light, rim light simulation)
- [ ] **Audio enhancement** — RNNoise / DPRNN noise suppression, echo cancellation, voice isolation
- [ ] **RIFE interpolation** — real-time intermediate frame estimation for 2x/4x FPS conversion
- [ ] **MiDaS / DPT-Large** — alternative high-accuracy depth estimation (Depth Anything V2 covers this today)
- [ ] **YOLOv8-seg** — person mask + mask-driven auto-frame (bbox detection already done)
- [ ] **Gamers Mode** — Virtual gamepad in LIVE tab driven by hand tracking: hand position = joystick axis, gestures = buttons (open = FIRE hold,  = pause,  = A...). Exposed as XInput/DirectInput virtual controller so any game sees a standard pad. Sensitivity, deadzone, calibration.
---

## Complete Filter Reference

All GPU filters available in Kagerou. Every filter runs on CUDA cores — zero CPU involvement.

### Core Pipeline Filters

| Filter | Key | Function | Description |
|--------|-----|----------|-------------|
| **Denoise** | `D` | `denoise_bilateral()` | Bilateral edge-preserving denoise. Reduces noise while keeping edges sharp. |
| **NV12 Denoise** | — | `denoise_nv12_bilateral()` | NV12-native bilateral filter (Y plane only, UV light blur). Skips RGB round-trip. |
| **CLAHE** | `C` | `clahe_rgb()` / `clahe_nv12()` | Adaptive contrast enhancement. Brightens dark areas without blowing out highlights. |
| **SuperRes** | `S` | `super_res_2x()` | 2x bicubic upscale + unsharp mask sharpening. Upscales resolution on GPU. |
| **LUT** | `L` | `lut3d_rgb()` | 3D color grading. 6 presets: Warm, Cool, Cinema, Vintage, Contrast, Desat. |
| **Resize** | — | `resize_bilinear()` / `resize_bicubic()` | Scale video to target resolution. Also NV12-native variants. |
| **Frame Blend** | — | `frame_blend()` | Alpha-blend two frames for frame interpolation / temporal effects. |
| **Color Convert** | — | `nv12_to_rgb()` / `rgb_to_nv12()` | BT.709 NV12↔RGB. Also `*_limited()` for limited-range (NVDEC default). |
| **YUV420P Convert** | — | `yuv420p_to_rgb()` / `rgb_to_yuv420p()` | BT.709 YUV420P↔RGB conversion. |

### Creative Filters

| Filter | Key | Function | Description |
|--------|-----|----------|-------------|
| **Gaussian Blur** | `B` | `gaussian_blur()` | Variable-sigma Gaussian blur. Separable 2-pass (horizontal + vertical). |
| **Sharpen** | `N` | `sharpen_rgb()` | Unsharp mask sharpen. Boosts edge contrast for crisp detail. |
| **Brightness/Contrast** | `1` | `brightness_contrast()` | Per-pixel brightness offset + contrast multiply. |
| **Saturation** | `2` | `saturation_rgb()` | Luminance-weighted saturation. 0=grayscale, 1=normal, >1=oversaturated. |
| **Gamma** | `3` | `gamma_rgb()` | Power-law gamma correction. <1=brighten, 1=identity, >1=darken. |
| **Vignette** | `V` | `vignette_rgb()` | Radial lens darkening. Creates film-style vignette effect. |
| **Film Grain** | `G` | `film_grain_rgb()` | PRNG noise overlay. Adds film-grain texture for cinematic look. |
| **Edge Detect** | `E` | `edge_detect_rgb()` | Sobel gradient magnitude. Outputs grayscale edge map. |
| **White Balance** | `W` | `white_balance_rgb()` | Temperature (warm/cool) and tint (green/magenta) color shift. |
| **Flip** | `F` | `flip_horizontal_gpu()` / `flip_vertical_gpu()` | GPU-accelerated flip. Cycles: Off → Horizontal → Vertical → Off. |
| **Lens Distortion** | `Q` | `lens_distortion()` | Barrel (negative k1) or pincushion (positive k1) lens distortion. |
| **Directional Blur** | — | `directional_blur()` | Motion blur simulation at configurable angle and length. |

### Transform Filters

| Filter | CLI | Function | Description |
|--------|-----|----------|-------------|
| **Crop** | `--crop WxH+X+Y` | `crop_rgb()` / `crop_nv12()` | Extract sub-region from frame. |
| **Pad** | — | `pad_rgb()` | Add border padding around frame with configurable fill color. |
| **Chroma Key** | `--chroma-key` | `chroma_key_rgb()` | HSV-based green screen removal with soft edge blending and spill suppression. Tunable hue/sat/val thresholds. |
| **Background Blur** | `--bg-blur` | `bg_blur_rgb()` | Center-weighted depth-of-field blur (non-AI heuristic). |
| **Temporal Denoise** | `--temporal-denoise` | `temporal_denoise_rgb()` / `temporal_denoise_nv12()` | Inter-frame temporal averaging using ring buffer of previous frames. |
| **Temporal Stabilize** | `--temporal-stab` | `temporal_stabilize_rgb()` | Block-matching global motion estimation + warp translation for video stabilization. |
| **HDR Tone Map** | `--hdr` | `hdr_tone_map_rgb()` | Reinhard and ACES filmic tone mapping. SDR→HDR normalization with configurable peak nits. |
| **Warp Translate** | — | `warp_translate_rgb()` | Sub-pixel translation warp (used internally by temporal stabilization). |

### Multi-Stream

| Filter | Function | Description |
|--------|----------|-------------|
| **GPU Compositor** | `composite_nv12()` | Multi-stream layout: composite N NV12 frames into a tiled output. Bilinear scale + blit. |

### AI Filters (require ORT GPU + models)

| Filter | CLI / UI | Function | Description |
|--------|----------|----------|-------------|
| **Pose Estimation** | `--ai-pose` / `0` | `ai_pose()` | MediaPipe Pose 33 body landmarks (256x256). Outputs normalized x/y + sigmoid visibility; UI draws red joint dots. |
| **Depth Estimation** | `--ai-depth` | `ai_depth()` | Depth Anything V2 (518x518). Per-frame min/max normalize → grayscale heatmap (white=near, black=far). |
| **Optical Flow** | `--ai-flow` | `ai_flow()` | RAFT-Small (360x480, two-frame). Interleaved dx/dy field → Middlebury color visualization. |
| **AI Denoise** | `--ai-denoise` | `ai_denoise()` | FastDVDNet temporal denoising. 5-frame ring buffer, calibrated 0.05 noise map, [0,1] value domain. |
| **AI Matting** | `--ai-matting` / `A` | `ai_matting()` | RVM MobileNetV3 with true frame-to-frame recurrence. Background-blur composite. Static-shape build for TRT. |
| **AI Face** | `--ai-face` / `P` | `ai_face()` | YuNet (640px, raw BGR). Point-prior decode + NMS; green boxes. |
| **AI Hands** | `--ai-hands` / `J` | `ai_hands()` | BlazePalm detect + 21-joint landmarks. Cyan joint dots. `--ai-palm` / `--ai-handlm` override models. |
| **AI Gaze** | `--ai-gaze` / `Z` | `ai_gaze()` | YuNet face box + MediaPipe iris per eye. Green iris dot + yellow gaze arrow. |
| **AI LowLight** | `--ai-lowlight` / `5` | `ai_lowlight()` | Zero-DCE single-frame enhance. Cheapest AI model here. |
| **AI Anime** | `--ai-anime` / `6` | `ai_anime()` | AnimeGANv3 Hayao (256px, NHWC ±1 domain), upscaled back. |
| **AI Detect** | `--ai-detect` / `F10` | `ai_detect()` | YOLOv8n (640px, RGB/255). 80 COCO classes, per-class colors + 3x5 bitmap name labels. Conf 0.35, NMS 0.45. |
| **AI AutoFrame** | `--ai-autoframe` / `F7` | (reuses `ai_face()` boxes) | Face-tracked crop (Wide 2.5x / Med 2.0x / Tight 1.5x). Camera mode only. |
| **Gesture Control** | `J` + hand pose | `classify_gesture()` (`include/kagerou/ai/gesture.h`) | Heuristic on 21 landmarks, no model: ✋ CUT, 👍 MARK, ✌ PAUSE, 👊 STOP. Unit-tested (`build.bat gesture_test`). |
| **Background Removal** | done (via RVM) | (`ai_matting()`) | RVM MobileNetV3 + bg-blur composite covers this; `ai_bg_removal()` stub superseded. |

---

## License

Internal use. See LICENSE file.
