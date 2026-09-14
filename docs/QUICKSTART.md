# Kagerou SDK — Quick Start Guide

## Build

```batch
cd src\HardLab\Kagerou

# Build with stub decoder/encoder (no NVIDIA SDK needed)
build.bat

# Build with NVIDIA Video Codec SDK (NVDEC/NVENC hardware acceleration)
build.bat sdk

# Build with MP4 container support
build.bat minimp4

# Build everything
build.bat all sdk minimp4
```

Output: `bin\kagerou.exe`, `bin\kagerou_test.exe`

## Transcode Video

```batch
# Basic transcode
kagerou.exe video.mp4

# With filters
kagerou.exe video.mp4 --denoise
kagerou.exe video.mp4 --scale 1920x1080
kagerou.exe video.mp4 --super-res
kagerou.exe video.mp4 --clahe
kagerou.exe video.mp4 --lut cinematic

# Combine filters
kagerou.exe video.mp4 --denoise --lut warm --scale 1280x720
kagerou.exe video.mp4 --all   # all filters at once

# Codec and quality
kagerou.exe video.mp4 --codec h265 --bitrate 10000
kagerou.exe video.mp4 --fps 60
```

## Other Modes

```batch
# Benchmark filter performance
kagerou.exe benchmark

# Run unit tests
kagerou.exe test

# Batch transcode at multiple quality levels
kagerou.exe batch video.mp4
```

## LUT Presets

| Preset | Effect |
|--------|--------|
| `warm` | Boost reds/yellows |
| `cool` | Boost blues/cyans |
| `cinematic` | Orange-teal split tone |
| `vintage` | Faded blacks, warm midtones |
| `contrast` | Aggressive contrast boost |
| `desat` | Partial desaturation (bleach bypass) |

## Output

Output goes to `output/` folder by default:
```
output/
├── video.h264    # Raw H.264 bitstream (Annex-B)
└── video.mp4     # Muxed MP4 container (via FFmpeg)
```

## API (C++ Embed)

```cpp
#include "pipeline.cu"
#include "fileio.h"

kagerou::PipelineConfig cfg;
cfg.decoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.codec = kagerou::VideoCodec::kH264;
cfg.encoder.width = 1920;
cfg.encoder.height = 1080;
cfg.encoder.fps = 30;
cfg.encoder.bitrate_kbps = 5000;
cfg.denoise.enabled = true;
cfg.clahe.enabled = true;
cfg.lut.enabled = true;
cfg.lut.preset = kagerou::LUTPreset::kCinematic;

kagerou::Pipeline pipeline;
pipeline.init(cfg);

kagerou::fileio::VideoFile video;
video.load("input.mp4");

for (const auto& nalu : video.nalus) {
    std::vector<uint8_t> encoded;
    pipeline.process_frame(nalu.data(), nalu.size(), encoded);
    // use encoded output
}

std::vector<uint8_t> flush_data;
pipeline.flush(flush_data);
pipeline.destroy();
```
