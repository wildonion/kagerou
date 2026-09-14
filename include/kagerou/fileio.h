// Kagerou SDK — File I/O utilities for video files.
// Supports raw bitstream (.h264/.h265), MP4 containers via minimp4,
// and MKV via FFmpeg CLI fallback.

#pragma once

#include "common.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef KAGEROU_USE_MINIMP4
#define MINIMP4_IMPLEMENTATION
#include "third_party/minimp4.h"
#endif

namespace kagerou {
namespace fileio {

// ---- Read entire file into memory ------------------------------------------
inline Error read_file(const char* path, std::vector<uint8_t>& data) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[fileio] cannot open: %s\n", path);
        return Error::kFileError;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(size);
    size_t read = fread(data.data(), 1, size, f);
    fclose(f);
    if (read != size) {
        fprintf(stderr, "[fileio] read error: %s (%zu/%zu bytes)\n", path, read, size);
        return Error::kFileError;
    }
    return Error::kOk;
}

// ---- Write buffer to file --------------------------------------------------
inline Error write_file(const char* path, const uint8_t* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[fileio] cannot create: %s\n", path);
        return Error::kFileError;
    }
    fwrite(data, 1, size, f);
    fclose(f);
    return Error::kOk;
}

// ---- Write vector to file --------------------------------------------------
inline Error write_file(const char* path, const std::vector<uint8_t>& data) {
    return write_file(path, data.data(), data.size());
}

// ---- Get file extension (lowercase) ----------------------------------------
inline std::string get_extension(const char* path) {
    std::string s(path);
    auto pos = s.rfind('.');
    if (pos == std::string::npos) return "";
    std::string ext = s.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// ---- Get filename without extension -----------------------------------------
inline std::string get_basename(const char* path) {
    std::string s(path);
    auto pos = s.find_last_of("/\\");
    if (pos != std::string::npos) s = s.substr(pos + 1);
    auto dot = s.rfind('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
    return s;
}

// ---- Create directory (recursive) ------------------------------------------
inline bool create_directory(const char* path) {
#ifdef _WIN32
    // CreateDirectoryA is single-level: walk and create each prefix so
    // nested paths like "output\clip" succeed even if "output\" is missing.
    // (Previously a nested -o silently wrote nothing.)
    std::string p(path);
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i] == '/' || p[i] == '\\') {
            std::string prefix = p.substr(0, i);
            if (!prefix.empty() && prefix.back() != ':')
                CreateDirectoryA(prefix.c_str(), NULL);
        }
    }
    return CreateDirectoryA(path, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(path, 0755) == 0 || errno == EEXIST;
#endif
}

// ---- Mux raw H.264 bitstream to MP4 using FFmpeg ---------------------------
inline Error mux_to_mp4(const char* h264_path, const char* mp4_path, uint32_t fps = 30) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -nostdin -y -r %u -i \"%s\" -c:v copy -vsync cfr -movflags +faststart \"%s\" 2>nul",
        fps, h264_path, mp4_path);
    printf("[fileio] muxing %s -> %s (%u fps)\n", h264_path, mp4_path, fps);
    int ret = system(cmd);
    return (ret == 0) ? Error::kOk : Error::kFileError;
}

// ---- Mux H.264 bitstream data to MP4 (in-memory) ---------------------------
inline Error mux_to_mp4(const uint8_t* h264_data, size_t h264_size,
                        const char* mp4_path, uint32_t fps = 30) {
    // Write temp H.264 file, then mux with ffmpeg
    char tmp_h264[512];
    snprintf(tmp_h264, sizeof(tmp_h264), "_kagerou_tmp_%d.h264",
#ifdef _WIN32
             (int)GetCurrentProcessId());
#else
             (int)getpid());
#endif
    Error e = write_file(tmp_h264, h264_data, h264_size);
    if (e != Error::kOk) return e;

    e = mux_to_mp4(tmp_h264, mp4_path, fps);
    remove(tmp_h264);
    return e;
}

// ---- H.264/H.265 NALU splitter for Annex-B bitstreams ----------------------
struct NALU {
    const uint8_t* data;
    size_t         size;
    uint8_t        type;
    bool           is_vcl;
};

inline std::vector<NALU> split_nalus_h264(const uint8_t* data, size_t size) {
    std::vector<NALU> nalus;
    size_t i = 0;

    while (i < size) {
        while (i < size - 3) {
            if (data[i] == 0 && data[i+1] == 0) {
                if (data[i+2] == 1) { i += 3; goto found; }
                if (i < size - 4 && data[i+2] == 0 && data[i+3] == 1) { i += 4; goto found; }
            }
            i++;
        }
        break;

    found:
        size_t nalu_start = i;
        while (i < size - 3) {
            if (data[i] == 0 && data[i+1] == 0 &&
                (data[i+2] == 1 || (i < size - 4 && data[i+2] == 0 && data[i+3] == 1))) {
                break;
            }
            i++;
        }

        size_t nalu_size = i - nalu_start;
        if (nalu_size > 0) {
            NALU nalu;
            nalu.data = data + nalu_start;
            nalu.size = nalu_size;
            nalu.type = data[nalu_start] & 0x1F;
            nalu.is_vcl = (nalu.type >= 1 && nalu.type <= 5);
            nalus.push_back(nalu);
        }
    }
    return nalus;
}

// ---- MP4 demuxer (requires KAGEROU_USE_MINIMP4) ----------------------------
#ifdef KAGEROU_USE_MINIMP4

struct MP4Frame {
    std::vector<uint8_t> data;
    unsigned             timestamp;
    unsigned             duration;
    bool                 is_keyframe;
};

struct MP4Demuxer {
    std::vector<uint8_t> file_data;
    MP4D_demux_t         mp4;
    int                  video_track = -1;
    unsigned             frame_idx = 0;
    bool                 opened = false;
    double               detected_fps = 0.0;

    ~MP4Demuxer() { close(); }

    Error open(const char* path) {
        Error e = read_file(path, file_data);
        if (e != Error::kOk) return e;

        memset(&mp4, 0, sizeof(mp4));
        mp4.read_pos = 0;
        mp4.read_size = (int64_t)file_data.size();

        if (!MP4D_open(&mp4, mp4_read_callback, this, (int64_t)file_data.size())) {
            fprintf(stderr, "[mp4] failed to parse: %s\n", path);
            return Error::kFileError;
        }

        // find first video track
        for (unsigned t = 0; t < mp4.track_count; t++) {
            if (mp4.track[t].handler_type == MP4D_HANDLER_TYPE_VIDE) {
                video_track = (int)t;
                break;
            }
        }
        if (video_track < 0) {
            fprintf(stderr, "[mp4] no video track found: %s\n", path);
            MP4D_close(&mp4);
            return Error::kFileError;
        }

        frame_idx = 0;
        opened = true;

        // Compute duration and FPS
        MP4D_track_t& tr = mp4.track[video_track];
        double dur_sec = (4294967296.0 * mp4.duration_hi + mp4.duration_lo) / mp4.timescale;
        double track_dur = (4294967296.0 * tr.duration_hi + tr.duration_lo) / tr.timescale;
        if (track_dur > 0 && dur_sec <= 0) dur_sec = track_dur;
        double fps = (dur_sec > 0) ? tr.sample_count / dur_sec : 0;
        detected_fps = fps;

        // Codec name
        const char* codec_name = "unknown";
        switch (tr.object_type_indication) {
            case 0x21: codec_name = "H.264 (AVC)"; break;
            case 0x23: codec_name = "H.265 (HEVC)"; break;
            case 0x31: codec_name = "AV1"; break;
            case 0x10: codec_name = "MPEG-4"; break;
            default: break;
        }

        printf("[mp4] opened %s\n", path);
        printf("  codec    : %s (0x%02X)\n", codec_name, tr.object_type_indication);
        printf("  size     : %dx%d\n",
               tr.SampleDescription.video.width, tr.SampleDescription.video.height);
        printf("  frames   : %u\n", tr.sample_count);
        printf("  duration : %.2f s\n", dur_sec);
        printf("  fps      : %.2f\n", fps);
        printf("  track    : %d\n", video_track);
        return Error::kOk;
    }

    // Get total frame count
    unsigned frame_count() const {
        if (!opened || video_track < 0) return 0;
        return mp4.track[video_track].sample_count;
    }

    // Get video dimensions
    void get_dimensions(int& w, int& h) const {
        w = h = 0;
        if (!opened || video_track < 0) return;
        w = mp4.track[video_track].SampleDescription.video.width;
        h = mp4.track[video_track].SampleDescription.video.height;
    }

    // Get codec (0x21 = H.264, 0x23 = H.265)
    unsigned get_codec() const {
        if (!opened || video_track < 0) return 0;
        return mp4.track[video_track].object_type_indication;
    }

    // Read next frame (returns nullptr when exhausted)
    const uint8_t* next_frame(size_t& frame_bytes) {
        if (!opened || video_track < 0 || frame_idx >= mp4.track[video_track].sample_count) {
            frame_bytes = 0;
            return nullptr;
        }

        unsigned timestamp, duration;
        unsigned frame_bytes_u = 0;
        MP4D_file_offset_t offset = MP4D_frame_offset(&mp4, video_track, frame_idx,
                                                       &frame_bytes_u, &timestamp, &duration);
        frame_bytes = frame_bytes_u;

        frame_idx++;

        // Return pointer into file_data
        if (offset + frame_bytes > file_data.size()) {
            frame_bytes = 0;
            return nullptr;
        }
        return file_data.data() + offset;
    }

    // Read all frames into vector of MP4Frame
    std::vector<MP4Frame> read_all_frames() {
        std::vector<MP4Frame> frames;
        if (!opened || video_track < 0) return frames;

        unsigned count = mp4.track[video_track].sample_count;
        frames.reserve(count);

        for (unsigned i = 0; i < count; i++) {
            unsigned timestamp, duration;
            unsigned frame_bytes_u = 0;
            MP4D_file_offset_t offset = MP4D_frame_offset(&mp4, video_track, i,
                                                           &frame_bytes_u, &timestamp, &duration);

            MP4Frame f;
            f.timestamp = timestamp;
            f.duration = duration;
            f.is_keyframe = true; // minimp4 doesn't expose keyframe info directly

            if (offset + (MP4D_file_offset_t)frame_bytes_u <= file_data.size()) {
                f.data.assign(file_data.data() + offset, file_data.data() + offset + frame_bytes_u);
            }
            frames.push_back(std::move(f));
        }
        return frames;
    }

    void close() {
        if (opened) {
            MP4D_close(&mp4);
            opened = false;
        }
    }

private:
    static int mp4_read_callback(int64_t offset, void* buffer, size_t size, void* token) {
        MP4Demuxer* self = (MP4Demuxer*)token;
        if (offset + (int64_t)size > (int64_t)self->file_data.size()) return 1;
        memcpy(buffer, self->file_data.data() + offset, size);
        return 0;
    }
};

#endif // KAGEROU_USE_MINIMP4

// ---- MKV demuxer via FFmpeg CLI fallback ------------------------------------
// Converts MKV to raw H.264/H.265 Annex-B bitstream using FFmpeg.
// Requires FFmpeg to be installed and available in PATH.

inline Error mkv_to_annexb(const char* mkv_path, const char* out_h264_path) {
    char cmd[1024];
    // Try H.264 bitstream filter first
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -nostdin -y -i %s -c:v copy -bsf:v h264_mp4toannexb -an %s 2>nul",
        mkv_path, out_h264_path);
    printf("[mkv] converting %s -> %s\n", mkv_path, out_h264_path);
    int ret = system(cmd);
    if (ret != 0) {
        // try H.265
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -hide_banner -nostdin -y -i %s -c:v copy -bsf:v hevc_mp4toannexb -an %s 2>nul",
            mkv_path, out_h264_path);
        ret = system(cmd);
    }
    if (ret != 0) {
        // try AV1 — extract as IVF container
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -hide_banner -nostdin -y -i %s -c:v copy -an -f ivf %s 2>nul",
            mkv_path, out_h264_path);
        ret = system(cmd);
    }
    return (ret == 0) ? Error::kOk : Error::kFileError;
}

// ---- Auto-detect file type and load as Annex-B bitstream --------------------
struct VideoFile {
    std::vector<uint8_t> data;
    std::vector<NALU>    nalus;
    int                  width = 0;
    int                  height = 0;
    double               fps = 0.0;
    unsigned             codec = 0; // 0x21=H.264, 0x23=H.265
    std::string          type;

    Error load(const char* path) {
        std::string ext = get_extension(path);

        if (ext == ".mp4") {
#ifdef KAGEROU_USE_MINIMP4
            return load_mp4(path);
#else
            fprintf(stderr, "[fileio] MP4 support requires KAGEROU_USE_MINIMP4. "
                    "Rebuild with: build.bat minimp4\n");
            return Error::kFileError;
#endif
        }

        if (ext == ".mkv" || ext == ".webm") {
            return load_mkv(path);
        }

        // raw bitstream (.h264, .h265, .264, .265, or unknown)
        return load_raw(path);
    }

private:
    Error load_raw(const char* path) {
        Error e = read_file(path, data);
        if (e != Error::kOk) return e;
        nalus = split_nalus_h264(data.data(), data.size());
        type = "raw";
        printf("[fileio] loaded raw bitstream: %s (%zu bytes, %zu NALUs)\n",
               path, data.size(), nalus.size());
        return Error::kOk;
    }

#ifdef KAGEROU_USE_MINIMP4
    Error load_mp4(const char* path) {
        MP4Demuxer demux;
        Error e = demux.open(path);
        if (e != Error::kOk) return e;

        codec = demux.get_codec();
        demux.get_dimensions(width, height);
        fps = demux.detected_fps;

        // For non-H.264/H.265 codecs (AV1, VP9, etc), minimp4 can't extract
        // frame data properly. Fall back to ffmpeg CLI demuxer.
        if (codec != 0x21 && codec != 0x23) {
            printf("[fileio] MP4 codec 0x%02X not natively supported, using ffmpeg demux\n", codec);
            demux.close();
            return load_mkv(path);  // same ffmpeg-based extraction
        }

        data.clear();

        // Extract SPS/PPS from the avcC/hvcC box and prepend as Annex-B
        if (codec == 0x21) { // H.264
            int sps_idx = 0, pps_idx = 0;
            int sps_bytes = 0, pps_bytes = 0;
            const void* sps;
            const void* pps;

            while ((sps = MP4D_read_sps(&demux.mp4, demux.video_track, sps_idx, &sps_bytes)) != nullptr) {
                data.push_back(0x00); data.push_back(0x00);
                data.push_back(0x00); data.push_back(0x01);
                data.insert(data.end(), (const uint8_t*)sps, (const uint8_t*)sps + sps_bytes);
                sps_idx++;
            }
            while ((pps = MP4D_read_pps(&demux.mp4, demux.video_track, pps_idx, &pps_bytes)) != nullptr) {
                data.push_back(0x00); data.push_back(0x00);
                data.push_back(0x00); data.push_back(0x01);
                data.insert(data.end(), (const uint8_t*)pps, (const uint8_t*)pps + pps_bytes);
                pps_idx++;
            }
            printf("[fileio] extracted %d SPS, %d PPS from avcC\n", sps_idx, pps_idx);
        }

        // HEVC: parse hvcC box to extract VPS/SPS/PPS (minimp4 only supports H.264)
        if (codec == 0x23) {
            int vps_count = 0, sps_count = 0, pps_count = 0;
            const unsigned char* dsi = demux.mp4.track[demux.video_track].dsi;
            unsigned dsi_bytes = demux.mp4.track[demux.video_track].dsi_bytes;
            if (dsi && dsi_bytes > 22) {
                // hvcC box: skip fixed header (21 bytes) + numOfArrays (1 byte)
                const unsigned char* p = dsi;
                const unsigned char* end = dsi + dsi_bytes;
                // Skip to numOfArrays (byte 21)
                if (p[0] == 'h' && p[1] == 'v' && p[2] == 'c' && p[3] == 'C') {
                    p += 22; // skip box header (8) + config record (14+7=21+1)
                } else {
                    p += 21; // raw config record (no box header)
                }
                if (p < end) {
                    uint8_t num_arrays = *p++;
                    for (int arr = 0; arr < num_arrays && p + 3 <= end; arr++) {
                        uint8_t nal_type = p[1] & 0x3F; // bits 0-5 of second byte
                        uint16_t num_nalus = ((uint16_t)p[2] << 8) | p[3];
                        p += 4;
                        for (int n = 0; n < num_nalus && p + 2 <= end; n++) {
                            uint16_t nal_len = ((uint16_t)p[0] << 8) | p[1];
                            p += 2;
                            if (p + nal_len > end) break;
                            // VPS=32, SPS=33, PPS=34
                            if (nal_type == 32 || nal_type == 33 || nal_type == 34) {
                                data.push_back(0x00); data.push_back(0x00);
                                data.push_back(0x00); data.push_back(0x01);
                                data.insert(data.end(), p, p + nal_len);
                                if (nal_type == 32) vps_count++;
                                else if (nal_type == 33) sps_count++;
                                else pps_count++;
                            }
                            p += nal_len;
                        }
                    }
                }
            }
            printf("[fileio] extracted %d VPS, %d SPS, %d PPS from hvcC\n", vps_count, sps_count, pps_count);
        }

        // Read all frames and convert from AVCC to Annex-B
        auto frames = demux.read_all_frames();
        for (auto& f : frames) {
            const uint8_t* p = f.data.data();
            size_t remaining = f.data.size();
            while (remaining >= 4) {
                uint32_t nalu_len = ((uint32_t)p[0] << 24) |
                                    ((uint32_t)p[1] << 16) |
                                    ((uint32_t)p[2] << 8)  |
                                    (uint32_t)p[3];
                if (nalu_len + 4 > remaining) break;

                data.push_back(0x00); data.push_back(0x00);
                data.push_back(0x00); data.push_back(0x01);
                data.insert(data.end(), p + 4, p + 4 + nalu_len);

                p += 4 + nalu_len;
                remaining -= 4 + nalu_len;
            }
        }

        nalus = split_nalus_h264(data.data(), data.size());

        // If minimp4 extracted 0 VCL frames, fall back to ffmpeg
        bool has_vcl = false;
        for (auto& n : nalus) if (n.is_vcl) { has_vcl = true; break; }
        if (!has_vcl) {
            fprintf(stderr, "[fileio] minimp4 extracted 0 VCL frames, falling back to ffmpeg\n");
            demux.close();
            return load_mkv(path);
        }

        type = "mp4";
        printf("  nalus    : %zu (Annex-B)\n", nalus.size());
        printf("  data     : %.2f MB\n", data.size() / 1048576.0);
        return Error::kOk;
    }
#endif

    Error load_mkv(const char* path) {
        // Probe dimensions first using ffprobe
        {
            char probe_cmd[1024];
            char tmp_probe[512];
#ifdef _WIN32
            snprintf(tmp_probe, sizeof(tmp_probe), "_kagerou_probe_%d.txt", (int)GetCurrentProcessId());
#else
            snprintf(tmp_probe, sizeof(tmp_probe), "_kagerou_probe_%d.txt", (int)getpid());
#endif
            snprintf(probe_cmd, sizeof(probe_cmd),
                "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,codec_name -of csv=p=0 \"%s\" > \"%s\" 2>nul",
                path, tmp_probe);
            system(probe_cmd);

            FILE* pf = fopen(tmp_probe, "r");
            if (pf) {
                char codec_name[64] = {};
                if (fscanf(pf, "%d,%d,%63[^,\n]", &width, &height, codec_name) >= 2) {
                    if (strstr(codec_name, "av1")) codec = 0x31;
                    else if (strstr(codec_name, "hevc") || strstr(codec_name, "h265")) codec = 0x23;
                    else codec = 0x21;
                }
                fclose(pf);
            }
            remove(tmp_probe);
        }

        // Convert MKV/MP4 to temp Annex-B file using FFmpeg
        char tmp_path[512];
#ifdef _WIN32
        snprintf(tmp_path, sizeof(tmp_path), "_kagerou_tmp_%d.h264", (int)GetCurrentProcessId());
#else
        snprintf(tmp_path, sizeof(tmp_path), "_kagerou_tmp_%d.h264", (int)getpid());
#endif

        Error e = mkv_to_annexb(path, tmp_path);
        if (e != Error::kOk) {
            fprintf(stderr, "[fileio] MKV/MP4 conversion failed. Is FFmpeg in PATH?\n");
            return e;
        }

        // Check if the output is an IVF file (starts with 'DKIF')
        FILE* ftest = fopen(tmp_path, "rb");
        if (!ftest) return Error::kFileError;
        uint8_t magic[4];
        bool is_ivf = (fread(magic, 1, 4, ftest) == 4 &&
                       magic[0]=='D' && magic[1]=='K' && magic[2]=='I' && magic[3]=='F');
        fseek(ftest, 0, SEEK_END);
        long file_size = ftell(ftest);
        fclose(ftest);

        if (is_ivf && file_size > 32) {
            // Parse IVF: 32-byte header, then repeated [4-byte LE size][frame_data]
            FILE* ivf = fopen(tmp_path, "rb");
            fseek(ivf, 32, SEEK_SET);
            data.clear();
            nalus.clear();
            while (!feof(ivf)) {
                uint8_t sz_buf[4];
                if (fread(sz_buf, 1, 4, ivf) != 4) break;
                uint32_t frame_size = sz_buf[0] | (sz_buf[1]<<8) | (sz_buf[2]<<16) | (sz_buf[3]<<24);
                if (frame_size == 0 || frame_size > 10*1024*1024) break;
                size_t cur = data.size();
                data.resize(cur + frame_size);
                if (fread(data.data() + cur, 1, frame_size, ivf) != frame_size) break;
                NALU f;
                f.data = data.data() + cur;
                f.size = frame_size;
                f.type = 0;
                f.is_vcl = true;
                nalus.push_back(f);
            }
            fclose(ivf);
            remove(tmp_path);
            printf("[fileio] parsed IVF: %zu bytes, %zu frames\n", data.size(), nalus.size());
        } else {
            // H.264/H.265 Annex-B: read raw file and split into NALUs
            e = load_raw(tmp_path);
            remove(tmp_path);

            // For non-H.264/H.265 codecs that failed NALU splitting
            if (nalus.empty() && !data.empty() && (codec == 0x31 || codec == 0x00)) {
                NALU raw;
                raw.data = data.data();
                raw.size = data.size();
                raw.type = 0;
                raw.is_vcl = true;
                nalus.push_back(raw);
                printf("[fileio] AV1/raw: treating %zu bytes as single packet\n", data.size());
            }
        }

        if (e == Error::kOk) {
            type = "mkv";
            const char* cname = codec == 0x31 ? "AV1" : codec == 0x23 ? "H.265" : codec == 0x21 ? "H.264" : "unknown";
            printf("  codec    : %s (0x%02X)\n", cname, codec);
            printf("  size     : %dx%d\n", width, height);
            printf("  frames   : %zu\n", nalus.size());
            printf("  data     : %.2f MB\n", data.size() / 1048576.0);
            printf("  source   : FFmpeg decode\n");
        }
        return e;
    }
};

// ---- Bitstream file reader (frame-by-frame) --------------------------------
struct BitstreamReader {
    std::vector<uint8_t> file_data;
    size_t               offset = 0;
    std::vector<NALU>    nalus;
    size_t               nalu_idx = 0;

    Error open(const char* path) {
        Error e = read_file(path, file_data);
        if (e != Error::kOk) return e;

        if (file_data.size() > 4) {
            nalus = split_nalus_h264(file_data.data(), file_data.size());
        }

        nalu_idx = 0;
        printf("[bitstream] loaded %s: %zu bytes, %zu NALUs\n",
               path, file_data.size(), nalus.size());
        return Error::kOk;
    }

    const uint8_t* next_nalu(size_t& nalu_size) {
        if (nalu_idx >= nalus.size()) {
            nalu_size = 0;
            return nullptr;
        }
        const NALU& n = nalus[nalu_idx++];
        nalu_size = n.size;
        return n.data;
    }

    std::vector<NALU> get_vcl_nalus() const {
        std::vector<NALU> vcl;
        for (const auto& n : nalus)
            if (n.is_vcl) vcl.push_back(n);
        return vcl;
    }

    void reset() { nalu_idx = 0; }
};

} // namespace fileio
} // namespace kagerou
