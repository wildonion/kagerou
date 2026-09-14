# Kagerou SDK — Video Encoding & Decoding Protocols

A complete guide to video codecs, containers, color spaces, and hardware acceleration.

---

## Table of Contents

1. [What is Transcoding?](#1-what-is-transcoding)
2. [Container Formats](#2-container-formats)
3. [Video Codecs — The H.* Family](#3-video-codecs--the-h-family)
4. [Other Video Codecs](#4-other-video-codecs)
5. [Audio Codecs](#5-audio-codecs)
6. [Color Spaces & HDR](#6-color-spaces--hdr)
7. [Bitrate Control Modes](#7-bitrate-control-modes)
8. [Hardware Acceleration](#8-hardware-acceleration)
9. [Resolution & Aspect Ratios](#9-resolution--aspect-ratios)
10. [Codec Comparison Table](#10-codec-comparison-table)

---

## 1. What is Transcoding?

Transcoding is the process of converting video/audio from one format to another:

```
Input File → DECODE → PROCESS → ENCODE → Output File
  (any)       (raw)    (filters)   (codec)    (any)
```

**Why transcode?**
- Change resolution (4K → 1080p for mobile)
- Change codec (H.264 → H.265 for smaller file size)
- Change bitrate (reduce for streaming)
- Add filters (denoise, sharpen, color grade)
- Convert for compatibility (ProRes → H.264 for web)

**Two types:**
- **Lossless**: no quality loss, large files (ProRes, FFV1, Huffman lossless)
- **Lossy**: quality reduced, small files (H.264, H.265, AV1)

---

## 2. Container Formats

Containers wrap video + audio + metadata into a single file. They do NOT define the codec — they are just envelopes.

| Container | Extension | Notes |
|-----------|-----------|-------|
| **MP4** | `.mp4` | Most universal. Supports H.264, H.265, AAC. Web standard. |
| **MKV** | `.mkv` | Most flexible. Supports all codecs. Open source. |
| **MOV** | `.mov` | Apple QuickTime. Professional video (ProRes). |
| **AVI** | `.avi` | Old Microsoft format. Limited codec support. |
| **WebM** | `.webm` | Google's format. VP8/VP9 + Opus/VP8 audio. |
| **FLV** | `.flv` | Flash Video. Legacy, still used by some streams. |
| **TS** | `.ts` | MPEG Transport Stream. Broadcasting, live streaming. |
| **OGG** | `.ogg` | Xiph.org container. Vorbis/Opus audio + Theora/AV1 video. |
| **MPEG-PS** | `.mpg` | MPEG Program Stream. DVD, older broadcast. |

### Container vs Codec

```
Container:  MP4, MKV, MOV, AVI, WebM, TS
                ↓
Codec:      H.264, H.265, VP9, AV1, MPEG-2
```

A single container can hold different codecs:
- MP4 can contain H.264 OR H.265
- MKV can contain almost anything

---

## 3. Video Codecs — The H.* Family

### H.261 (1990)
- **Full name**: ITU-T H.261
- **Use**: Early video conferencing (ISDN lines)
- **Resolution**: QCIF (176x144), CIF (352x288)
- **Bitrate**: 64-1920 kbps
- **Status**: Obsolete. First codec to use motion compensation + DCT.

### H.262 / MPEG-2 Part 2 (1994)
- **Full name**: ITU-T H.262 / ISO/IEC 13818-2
- **Use**: DVD, Blu-ray (early), digital TV (DVB, ATSC, ISDB)
- **Resolution**: up to 4K (but rarely used above 1080i)
- **Bitrate**: 1-50 Mbps
- **Key features**:
  - I, P, B frames (intra, predicted, bidirectional)
  - Motion compensation with 16x16 macroblocks
  - DCT (Discrete Cosine Transform) compression
  - Variable-length coding (VLC)
- **Status**: Legacy, but still used in broadcasting.

### H.263 (1996)
- **Full name**: ITU-T H.263
- **Use**: Video conferencing (H.323, SIP), early mobile video
- **Resolution**: QCIF, CIF, 4CIF
- **Bitrate**: 20-3000 kbps
- **Key features**: Improved motion compensation over H.261
- **Status**: Largely replaced by H.264.

### H.263+ / H.263++ (1998/2000)
- **Full name**: ITU-T H.263v2 / H.263v3
- **Use**: Same as H.263 with better compression
- **Key features**: Optional modes for better quality
- **Status**: Obsolete.

### H.264 / AVC (2003) — THE MOST IMPORTANT CODEC
- **Full name**: ITU-T H.264 / ISO/IEC 14496-10 (Advanced Video Coding)
- **Use**: EVERYTHING — Blu-ray, streaming (YouTube, Netflix), video conferencing, surveillance, broadcasting
- **Resolution**: up to 8K (but practical up to 4K)
- **Bitrate**: 100 kbps (low-res) to 50+ Mbps (4K Blu-ray)
- **Key features**:
  - **Macroblock-based**: 16x16 pixel blocks (can use 8x8, 4x4 sub-blocks)
  - **Multiple reference frames**: can reference multiple past frames
  - **Variable block size**: motion compensation at different block sizes
  - **Quarter-pixel precision**: motion vectors at 1/4 pixel accuracy
  - **CABAC**: Context-Adaptive Binary Arithmetic Coding (better compression)
  - **CAVLC**: Simpler, faster entropy coding option
  - **Deblocking filter**: reduces block artifacts
  - **Slices**: parallel processing, error resilience
- **Profiles**:
  - **Baseline**: Low complexity, no B-frames. Mobile, video conferencing.
  - **Main**: B-frames, CABAC. Broadcast, streaming.
  - **High**: 8x8 transforms, custom quantization matrices. Blu-ray, HD broadcast.
- **Patent situation**: MPEG LA pool. Royalty required for commercial use (most end in 2023-2025).
- **Why it matters**: It's the "Hello World" of video codecs. Everything else is compared to it.

### H.265 / HEVC (2013)
- **Full name**: ITU-T H.265 / ISO/IEC 23008-2 (High Efficiency Video Coding)
- **Use**: 4K UHD streaming, Blu-ray, broadcasting
- **Resolution**: up to 8K
- **Bitrate**: 30-40% smaller than H.264 at same quality
- **Key features**:
  - **Coding Tree Units (CTU)**: up to 64x64 blocks (vs 16x16 in H.264)
  - **More intra prediction modes**: 35 modes (vs 9 in H.264)
  - **Advanced motion compensation**: larger block sizes, more partition modes
  - **Sample Adaptive Offset (SAO)**: reduces ringing artifacts
  - **Parallel processing tools**: tiles, WPP (Wavefront Parallel Processing)
- **Profiles**: Main, Main 10 (10-bit), Main 4:2:2, Main 4:4:4
- **Patent situation**: VERY complex. Multiple patent pools (MPEG LA, HEVC Advance, Velos Media). Royalty headaches.
- **vs H.264**: ~40% better compression, ~2x more complex to encode.

### H.266 / VVC (2020)
- **Full name**: ITU-T H.266 / ISO/IEC 23090-3 (Versatile Video Coding)
- **Use**: 8K, HDR, 360° video, screen content
- **Resolution**: up to 8K and beyond
- **Bitrate**: ~40% smaller than H.265 at same quality
- **Key features**:
  - **CTU up to 128x128**
  - **More partition modes**: binary/ternary split, affine motion compensation
  - **Palette mode**: for screen content
  - **Subpictures**: independent decoding of regions
- **Patent situation**: Even worse than HEVC. Multiple pools, uncertain licensing.
- **Status**: Slow adoption due to licensing fears. AV1 is its main competitor.

### H.267 / LCEVC (expected ~2025+)
- **Full name**: Low Complexity Enhancement Video Coding
- **Use**: Enhancement layer on top of existing codecs
- **Status**: Under development.

---

## 4. Other Video Codecs

### VP8 (2008)
- **Developer**: On2 Technologies (acquired by Google)
- **Use**: WebRTC, legacy WebM
- **Key features**: royalty-free, decent quality
- **Status**: Mostly replaced by VP9.

### VP9 (2013)
- **Developer**: Google
- **Use**: YouTube (default for 1440p+), WebM
- **Key features**:
  - CTU up to 64x64
  - 8 reference frames
  - Adaptive quantization
  - **Royalty-free**
- **vs H.265**: Similar compression, but free to use.
- **Status**: Strong in web/streaming. Being superseded by AV1.

### AV1 (2018)
- **Developer**: Alliance for Open Media (Google, Microsoft, Amazon, Mozilla, Netflix, etc.)
- **Use**: YouTube, Netflix, Twitch, future of streaming
- **Key features**:
  - CTU up to 128x128
  - 56 intra prediction modes
  - Advanced motion compensation with warping
  - **Royalty-free** (no patent pools)
  - Optimized for high-resolution content
- **vs H.265**: ~30% better compression, free to use, but encoding is slower.
- **vs VP9**: ~30% better compression.
- **Status**: The future. Being adopted rapidly. AV1 hardware encoders arriving in GPUs (RTX 40xx, Intel Arc).

### AV2 (in development)
- **Status**: Under development by Alliance for Open Media.
- **Expected**: ~2026-2027.

### MPEG-4 Part 2 / ASP (2001)
- **Use**: DivX, Xvid (early internet video)
- **Status**: Obsolete, but historically important.

### DivX / Xvid
- **Use**: Early 2000s video sharing
- **Status**: Dead, but the name stuck in pop culture.

### ProRes (2007)
- **Developer**: Apple
- **Use**: Professional video editing (Final Cut Pro, cinema)
- **Key features**:
  - Visually lossless at high bitrates
  - Very fast decode (edit-friendly)
  - Multiple quality levels: Proxy, LT, 422, HQ, 4444, 4444 XQ
- **Bitrate**: 100-500+ Mbps
- **Status**: Industry standard for production.

### DNxHR / DNxHD
- **Developer**: Avid
- **Use**: Professional video editing (Avid Media Composer)
- **Key features**: Similar to ProRes, OpenEXR support
- **Status**: Standard in broadcast.

### WMV / VC-1
- **Developer**: Microsoft
- **Use**: Legacy streaming, Windows Media
- **Status**: Obsolete.

### Theora
- **Developer**: Xiph.org
- **Use**: Legacy WebM/OGG
- **Status**: Obsolete, replaced by AV1.

---

## 5. Audio Codecs

| Codec | Bitrate | Use | Notes |
|-------|---------|-----|-------|
| **AAC** | 96-320 kbps | Default for MP4, streaming, iTunes | Lossy. Successor to MP3. |
| **MP3** | 128-320 kbps | Legacy music, podcasts | Lossy. MPEG-1 Audio Layer III. Still widely used. |
| **Opus** | 6-510 kbps | WebRTC, Discord, Twitch, WebM | Lossy. Best quality at low bitrate. Open source. |
| **FLAC** | ~1000+ kbps | Archiving, audiophiles | Lossless. Open source. |
| **ALAC** | ~1000+ kbps | Apple ecosystem | Lossless. Apple's FLAC equivalent. |
| **AC3/EAC3** | 192-640 kbps | DVDs, Blu-ray, streaming (Dolby Digital) | Lossy. Dolby surround sound. |
| **DTS** | 768-1500+ kbps | Blu-ray, cinema | Lossy/lossless variants. |
| **Vorbis** | 64-500 kbps | Legacy OGG/WebM | Lossy. Xiph.org. Being replaced by Opus. |
| **WMA** | 128-384 kbps | Legacy Windows | Lossy. Obsolete. |
| **PCM** | 1411+ kbps (CD) | Raw audio, WAV | Uncompressed. Lossless by definition. |

### Audio Codec Recommendations
- **Streaming**: AAC (universal) or Opus (best quality/size)
- **Archiving**: FLAC
- **Video conferencing**: Opus
- **Legacy compatibility**: MP3

---

## 6. Color Spaces & HDR

### BT.601 (SD)
- **Use**: Standard definition video (480i/576i)
- **Luma coefficients**: Y = 0.299R + 0.587G + 0.114B
- **Chroma subsampling**: 4:2:0
- **Bit depth**: 8-bit

### BT.709 (HD)
- **Use**: HD video (720p, 1080i/p), Blu-ray, streaming
- **Luma coefficients**: Y = 0.2126R + 0.7152G + 0.0722B
- **Chroma subsampling**: 4:2:0, 4:2:2
- **Bit depth**: 8-bit, 10-bit
- **Status**: Current standard for most video content.

### BT.2020 (UHD/HDR)
- **Use**: 4K/8K UHD, HDR10, Dolby Vision
- **Luma coefficients**: Y = 0.2627R + 0.6780G + 0.0593B
- **Color gamut**: ~75% of CIE 1931 (wider than BT.709)
- **Bit depth**: 10-bit, 12-bit
- **Status**: Required for HDR content.

### HDR Standards

| Standard | Format | Metadata | Peak Brightness |
|----------|--------|----------|-----------------|
| **HDR10** | Static | Single HDR10 metadata block | 1000-10000 nits |
| **HDR10+** | Dynamic | Per-scene metadata (Samsung) | 1000-4000 nits |
| **Dolby Vision** | Dynamic | Per-frame metadata (Dolby) | Up to 10000 nits |
| **HLG** | Static/Dynamic | Hybrid Log-Gamma (BBC/NHK) | 1000 nits |

### Transfer Functions (EOTF)
- **PQ (Perceptual Quantizer)**: SMPTE ST 2084. Used by HDR10, Dolby Vision.
- **HLG (Hybrid Log-Gamma)**: ARIB STD-B67. Used by HLG HDR.
- **sRGB**: Standard for SDR displays.

### Chroma Subsampling

| Format | Description | Bandwidth |
|--------|-------------|-----------|
| **4:4:4** | Full chroma resolution | 100% |
| **4:2:2** | Horizontal chroma subsampling | 67% |
| **4:2:0** | Both H and V chroma subsampling | 50% |

```
4:4:4:  Y U V Y U V Y U V Y U V    (every pixel has chroma)
4:2:2:  Y U V Y . . Y U V Y . .    (every other pixel shares chroma)
4:2:0:  Y U . . Y V . . Y U . .    (2x2 block shares chroma)
```

---

## 7. Bitrate Control Modes

### CBR (Constant Bitrate)
- Same bitrate regardless of content complexity
- **Use**: Live streaming, broadcasting
- **Pros**: Predictable bandwidth usage
- **Cons**: Wastes bits on simple scenes, starves complex scenes

### VBR (Variable Bitrate)
- Bitrate varies based on content complexity
- **Use**: File-based encoding, streaming (with buffer constraints)
- **Pros**: Better quality-per-bit
- **Cons**: Less predictable bandwidth

### CQP (Constant QP)
- Fixed quantization parameter
- **Use**: Testing, archival
- **Pros**: Consistent quality
- **Cons**: File size varies wildly

### CRF (Constant Rate Factor)
- Single-pass, quality-targeted encoding
- **Use**: Archival, local storage
- **Values**: 0 (lossless) → 51 (worst). Default: 23 (H.264), 28 (H.265)
- **Pros**: Best quality/size ratio, single pass
- **Cons**: File size unpredictable

### Bitrate Examples

| Resolution | H.264 CBR | H.265 CBR | AV1 CBR |
|------------|-----------|-----------|---------|
| 480p       | 1-2 Mbps  | 0.5-1 Mbps | 0.3-0.8 Mbps |
| 720p       | 2-5 Mbps  | 1-3 Mbps  | 0.8-2 Mbps |
| 1080p      | 5-10 Mbps | 3-6 Mbps  | 2-4 Mbps |
| 4K         | 20-50 Mbps | 10-25 Mbps | 8-15 Mbps |

---

## 8. Hardware Acceleration

### NVIDIA NVDEC (Decode)
- Hardware video decoder on NVIDIA GPUs
- Supports: H.264, H.265, VP8, VP9, AV1 (RTX 40xx+)
- **API**: NVIDIA Video Codec SDK (nvcuvid.h, cuviddec.h)
- **Performance**: 800+ fps for 1080p H.264 decode

### NVIDIA NVENC (Encode)
- Hardware video encoder on NVIDIA GPUs
- Supports: H.264, H.265, AV1 (RTX 40xx+)
- **API**: NVIDIA Video Codec SDK (nvEncodeAPI.h)
- **Performance**: 1000+ fps for 1080p H.264 encode
- **Quality**: Near-x265 quality at 10x speed

### Intel Quick Sync
- Hardware encode/decode on Intel iGPUs and Arc GPUs
- Supports: H.264, H.265, VP9, AV1
- **API**: Intel oneVPL, Media SDK

### AMD VCN
- Video Core Next on AMD GPUs
- Supports: H.264, H.265, VP9, AV1

### Software Encoders (CPU)

| Encoder | Codec | Speed | Quality | Notes |
|---------|-------|-------|---------|-------|
| **x264** | H.264 | Fast | Excellent | Best H.264 encoder |
| **x265** | H.265 | Slow | Excellent | Best H.265 encoder |
| **SVT-AV1** | AV1 | Medium | Excellent | Intel's AV1 encoder |
| **aomenc** | AV1 | Very slow | Best | Reference encoder, impractical for production |
| **libvpx-vp9** | VP9 | Slow | Good | Google's encoder |
| **rav1e** | AV1 | Medium | Good | Rust-based AV1 encoder |

---

## 9. Resolution & Aspect Ratios

| Name | Resolution | Aspect Ratio | Use |
|------|-----------|--------------|-----|
| QCIF | 176x144 | 11:9 | Legacy mobile |
| CIF | 352x288 | 11:9 | Legacy video conferencing |
| VGA | 640x480 | 4:3 | Legacy display |
| HD | 1280x720 | 16:9 | Streaming, broadcast |
| Full HD | 1920x1080 | 16:9 | Standard HD content |
| 2K | 2560x1440 | 16:9 | QHD monitors |
| 4K UHD | 3840x2160 | 16:9 | 4K streaming, Blu-ray |
| 5K | 5120x2880 | 16:9 | Apple iMac, displays |
| 8K UHD | 7680x4320 | 16:9 | Future broadcasting |

### Frame Rates

| FPS | Use |
|-----|-----|
| 24 | Cinema, film |
| 25 | PAL broadcast |
| 29.97 | NTSC broadcast |
| 30 | Standard video, streaming |
| 48 | HFR cinema |
| 50 | PAL HFR |
| 60 | Gaming, smooth motion |
| 120 | HFR gaming, sports |
| 240+ | Slow motion capture |

---

## 10. Codec Comparison Table

| Codec | Year | Compression | Speed | Royalty | Hardware |
|-------|------|------------|-------|---------|----------|
| **H.264** | 2003 | Baseline | Fast | Yes (mostly expired) | Universal |
| **H.265** | 2013 | -40% vs H.264 | Medium | Yes (complex) | Most GPUs |
| **H.266** | 2020 | -40% vs H.265 | Slow | Yes (unclear) | Almost none |
| **VP9** | 2013 | ~H.265 | Medium | **Free** | Most GPUs |
| **AV1** | 2018 | -30% vs H.265 | Slow | **Free** | RTX 40xx+, Intel Arc |
| **ProRes** | 2007 | Near-lossless | Very fast | Yes | Apple hardware |

---

## Kagerou SDK Mapping

Kagerou implements these codecs via the NVIDIA Video Codec SDK:

```
Kagerou Pipeline:
  Decoder (NVDEC) → CUDA Filters → Encoder (NVENC)
  ─────────────────────────────────────────────────
  H.264 Decode      Denoise (RGB+NV12)    H.264 Encode
  H.265 Decode      Scale/Resize (RGB+NV12)  H.265 Encode
  VP9 Decode        Super Res (2x)        AV1 Encode
  AV1 Decode        CLAHE (RGB+NV12)
                    LUT (6 presets)
                    Color Convert (NV12↔RGB)
                    Frame Interp
                    Tone Mapping (HDR→SDR)

  NV12-native path: denoise + CLAHE + scale can run
  directly on NV12 without RGB round-trip (faster).
```

---

*Last updated: 2026. Kagerou SDK v1.0*
