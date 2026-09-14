#pragma once
// Kagerou SDK — common types, error codes, CUDA helpers.

#include <cuda_runtime.h>
#include <cuda.h>        // CUDA driver API (cuInit, cuDeviceGet, cuCtxCreate)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace kagerou {

// ---- error codes ----------------------------------------------------------
enum class Error {
    kOk = 0,
    kCudaError,
    kInvalidArg,
    kUnsupportedCodec,
    kDecodeError,
    kEncodeError,
    kFilterError,
    kFileError,
    kOutOfMemory
};

inline const char* error_string(Error e) {
    switch (e) {
        case Error::kOk:               return "ok";
        case Error::kCudaError:        return "cuda error";
        case Error::kInvalidArg:       return "invalid argument";
        case Error::kUnsupportedCodec: return "unsupported codec";
        case Error::kDecodeError:      return "decode error";
        case Error::kEncodeError:      return "encode error";
        case Error::kFilterError:      return "filter error";
        case Error::kFileError:        return "file error";
        case Error::kOutOfMemory:      return "out of memory";
    }
    return "unknown";
}

// ---- CUDA error check -----------------------------------------------------
#define KAGEROU_CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(err));                                    \
        abort();                                                            \
    }                                                                       \
} while(0)

// ---- pixel formats --------------------------------------------------------
enum class PixelFormat {
    kRGB = 0,
    kRGBA,
    kNV12,       // Y plane + interleaved UV (NVDEC output)
    kYUV420P,    // Y + U + V planes (planar 4:2:0)
    kP010        // 10-bit NV12 (HDR)
};

// ---- image / frame --------------------------------------------------------
struct Frame {
    uint8_t*  d_data   = nullptr;   // device pointer
    uint32_t  width    = 0;
    uint32_t  height   = 0;
    uint32_t  stride   = 0;         // bytes per row (may include padding)
    PixelFormat fmt    = PixelFormat::kRGB;
    uint32_t  plane_count = 1;
    uint32_t  plane_size[3] = {};   // per-plane byte size
    bool      owned    = false;     // true = we allocated d_data

    void free_gpu() {
        if (owned && d_data) { cudaFree(d_data); d_data = nullptr; }
        owned = false;
    }
};

// ---- image dimensions helper -----------------------------------------------
inline uint32_t plane_size(PixelFormat fmt, uint32_t w, uint32_t h, int plane) {
    switch (fmt) {
        case PixelFormat::kRGB:  return w * h * 3;
        case PixelFormat::kRGBA: return w * h * 4;
        case PixelFormat::kNV12: {
            uint32_t y = w * h;
            uint32_t uv = w * (h / 2);
            return plane == 0 ? y : uv;
        }
        case PixelFormat::kYUV420P: {
            uint32_t y = w * h;
            uint32_t u = (w / 2) * (h / 2);
            return plane == 0 ? y : u;
        }
        case PixelFormat::kP010: {
            uint32_t y = w * h * 2;
            uint32_t uv = w * (h / 2) * 2;
            return plane == 0 ? y : uv;
        }
    }
    return 0;
}

// ---- allocation helpers ----------------------------------------------------
inline Error alloc_frame_gpu(Frame& f, uint32_t w, uint32_t h, PixelFormat fmt) {
    f.free_gpu();
    f.width  = w;
    f.height = h;
    f.fmt    = fmt;
    f.stride = w;   // simplified: stride = width in bytes

    switch (fmt) {
        case PixelFormat::kRGB:
            f.plane_count = 1;
            f.plane_size[0] = w * h * 3;
            KAGEROU_CUDA_CHECK(cudaMalloc(&f.d_data, f.plane_size[0]));
            break;
        case PixelFormat::kRGBA:
            f.plane_count = 1;
            f.plane_size[0] = w * h * 4;
            KAGEROU_CUDA_CHECK(cudaMalloc(&f.d_data, f.plane_size[0]));
            break;
        case PixelFormat::kNV12:
            f.plane_count = 2;
            f.plane_size[0] = w * h;
            f.plane_size[1] = w * (h / 2);
            KAGEROU_CUDA_CHECK(cudaMalloc(&f.d_data, f.plane_size[0] + f.plane_size[1]));
            break;
        case PixelFormat::kYUV420P:
            f.plane_count = 3;
            f.plane_size[0] = w * h;
            f.plane_size[1] = (w / 2) * (h / 2);
            f.plane_size[2] = (w / 2) * (h / 2);
            KAGEROU_CUDA_CHECK(cudaMalloc(&f.d_data, f.plane_size[0] + f.plane_size[1] + f.plane_size[2]));
            break;
        case PixelFormat::kP010:
            f.plane_count = 2;
            f.plane_size[0] = w * h * 2;
            f.plane_size[1] = w * (h / 2) * 2;
            KAGEROU_CUDA_CHECK(cudaMalloc(&f.d_data, f.plane_size[0] + f.plane_size[1]));
            break;
    }
    f.owned = true;
    return Error::kOk;
}

// ---- GPU buffer (generic) --------------------------------------------------
struct GpuBuffer {
    uint8_t* d_ptr = nullptr;
    size_t   size  = 0;

    Error alloc(size_t bytes) {
        free();
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        size = bytes;
        return Error::kOk;
    }
    void free() {
        if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; }
        size = 0;
    }
    void upload(const void* host, size_t bytes) {
        KAGEROU_CUDA_CHECK(cudaMemcpy(d_ptr, host, bytes, cudaMemcpyHostToDevice));
    }
    void download(void* host, size_t bytes) const {
        KAGEROU_CUDA_CHECK(cudaMemcpy(host, d_ptr, bytes, cudaMemcpyDeviceToHost));
    }
};

} // namespace kagerou
