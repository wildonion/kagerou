#pragma once
// ============================================================================
// Kagerou Virtual Camera — Shared Memory Protocol
// Frame transfer between Kagerou process and the virtual camera DLL
// ============================================================================

#include <windows.h>
#include <cstdint>

#define VIRTUALCAM_SHM_NAME L"Local\\KagerouVirtualCam"
#define VIRTUALCAM_MAX_W 3840
#define VIRTUALCAM_MAX_H 2160
// NV12: Y plane + UV plane = 1.5 bytes/pixel
#define VIRTUALCAM_MAX_FRAME_SIZE (VIRTUALCAM_MAX_W * VIRTUALCAM_MAX_H * 3 / 2)
#define VIRTUALCAM_SHM_SIZE (sizeof(VirtualCamHeader) + VIRTUALCAM_MAX_FRAME_SIZE + 4096)

#pragma pack(push, 1)
struct VirtualCamHeader {
    uint32_t magic;         // 0x5643414D ("VCAM")
    uint32_t width;
    uint32_t height;
    uint32_t fourcc;        // 'NV12' = 0x3231564E
    uint32_t frame_size;    // Y + UV plane size
    uint32_t frame_id;      // monotonically increasing; odd = writing, even = complete
    uint64_t timestamp_us;  // microsecond timestamp
};
#pragma pack(pop)

#define VIRTUALCAM_MAGIC 0x5643414D

// ============================================================================
// Writer side (Kagerou process)
// ============================================================================
class VirtualCamWriter {
    HANDLE hMapFile = nullptr;
    void* pBuf = nullptr;
    VirtualCamHeader* hdr = nullptr;

public:
    bool Open() {
        hMapFile = CreateFileMappingW(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
            0, VIRTUALCAM_SHM_SIZE, VIRTUALCAM_SHM_NAME);
        if (!hMapFile) {
            // Already exists? Open existing.
            hMapFile = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, VIRTUALCAM_SHM_NAME);
        }
        if (!hMapFile) return false;

        pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, VIRTUALCAM_SHM_SIZE);
        if (!pBuf) { CloseHandle(hMapFile); hMapFile = nullptr; return false; }

        hdr = (VirtualCamHeader*)pBuf;
        hdr->magic = VIRTUALCAM_MAGIC;
        hdr->width = 0;
        hdr->height = 0;
        hdr->fourcc = 0x3231564E; // 'NV12'
        hdr->frame_id = 0;
        hdr->timestamp_us = 0;
        return true;
    }

    // Write NV12 frame (Y plane followed by interleaved UV)
    void WriteFrame(const uint8_t* nv12_data, uint32_t w, uint32_t h, uint64_t timestamp_us) {
        uint32_t frame_size = w * h * 3 / 2;

        // Signal "writing in progress" (odd frame_id)
        hdr->frame_id++;
        MemoryBarrier();

        hdr->width = w;
        hdr->height = h;
        hdr->fourcc = 0x3231564E;
        hdr->frame_size = frame_size;
        hdr->timestamp_us = timestamp_us;

        uint8_t* frame_buf = (uint8_t*)pBuf + sizeof(VirtualCamHeader);
        memcpy(frame_buf, nv12_data, frame_size);

        // Signal "frame complete" (even frame_id)
        MemoryBarrier();
        hdr->frame_id++;
    }

    // Write RGB24 frame (converts to NV12 internally)
    void WriteFrameRGB(const uint8_t* rgb24, uint32_t w, uint32_t h, uint64_t timestamp_us) {
        // Signal "writing in progress"
        hdr->frame_id++;
        MemoryBarrier();

        hdr->width = w;
        hdr->height = h;
        hdr->fourcc = 0x3231564E;
        hdr->frame_size = w * h * 3 / 2;
        hdr->timestamp_us = timestamp_us;

        uint8_t* frame_buf = (uint8_t*)pBuf + sizeof(VirtualCamHeader);

        // Convert RGB24 -> NV12
        const uint8_t* rgb = rgb24;
        uint8_t* y_plane = frame_buf;
        uint8_t* uv_plane = frame_buf + w * h;

        for (uint32_t row = 0; row < h; row++) {
            for (uint32_t col = 0; col < w; col++) {
                uint8_t r = rgb[(row * w + col) * 3 + 0];
                uint8_t g = rgb[(row * w + col) * 3 + 1];
                uint8_t b = rgb[(row * w + col) * 3 + 2];
                uint8_t y_val = (uint8_t)(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
                y_plane[row * w + col] = y_val;

                // Subsample UV (2x2 block)
                if ((row & 1) == 0 && (col & 1) == 0) {
                    int32_t u = (-38 * r - 74 * g + 112 * b + 128) >> 8;
                    int32_t v = (112 * r - 94 * g - 18 * b + 128) >> 8;
                    uint32_t uv_idx = (row / 2) * w + col;
                    uv_plane[uv_idx + 0] = (uint8_t)(u + 128);
                    uv_plane[uv_idx + 1] = (uint8_t)(v + 128);
                }
            }
        }

        // Signal "frame complete"
        MemoryBarrier();
        hdr->frame_id++;
    }

    void Close() {
        if (pBuf) { UnmapViewOfFile(pBuf); pBuf = nullptr; }
        if (hMapFile) { CloseHandle(hMapFile); hMapFile = nullptr; }
    }

    ~VirtualCamWriter() { Close(); }
};

// ============================================================================
// Reader side (Virtual camera DLL, loaded by Zoom/Teams/OBS)
// ============================================================================
class VirtualCamReader {
    HANDLE hMapFile = nullptr;
    void* pBuf = nullptr;
    VirtualCamHeader* hdr = nullptr;
    uint32_t last_frame_id = 0;

public:
    bool Open() {
        hMapFile = OpenFileMappingW(FILE_MAP_READ, FALSE, VIRTUALCAM_SHM_NAME);
        if (!hMapFile) return false;

        pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, VIRTUALCAM_SHM_SIZE);
        if (!pBuf) { CloseHandle(hMapFile); hMapFile = nullptr; return false; }

        hdr = (VirtualCamHeader*)pBuf;
        last_frame_id = hdr->frame_id;
        return true;
    }

    // Returns true if a new frame is available, copies NV12 data to output buffer
    bool ReadFrame(uint8_t* nv12_out, uint32_t& w, uint32_t& h, uint64_t& timestamp_us) {
        if (!hdr || hdr->magic != VIRTUALCAM_MAGIC) return false;

        // Wait for frame to be complete (even frame_id means complete)
        uint32_t fid = hdr->frame_id;
        if (fid == last_frame_id || (fid & 1)) return false; // no new frame or still writing

        MemoryBarrier();
        w = hdr->width;
        h = hdr->height;
        timestamp_us = hdr->timestamp_us;

        uint32_t frame_size = w * h * 3 / 2;
        const uint8_t* frame_buf = (const uint8_t*)pBuf + sizeof(VirtualCamHeader);
        memcpy(nv12_out, frame_buf, frame_size);

        last_frame_id = fid;
        return true;
    }

    // Check if writer is still alive (heartbeat via frame_id progression)
    bool IsWriterAlive() {
        if (!hdr || hdr->magic != VIRTUALCAM_MAGIC) return false;
        uint32_t fid = hdr->frame_id;
        return fid != 0;
    }

    void Close() {
        if (pBuf) { UnmapViewOfFile(pBuf); pBuf = nullptr; }
        if (hMapFile) { CloseHandle(hMapFile); hMapFile = nullptr; }
    }

    ~VirtualCamReader() { Close(); }
};
