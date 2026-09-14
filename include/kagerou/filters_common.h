#pragma once
// Kagerou SDK — shared CUDA filter helpers.

#include <cuda_runtime.h>
#include <cstdint>

namespace kagerou {
namespace filters {

inline dim3 block_2d(uint32_t /*w*/, uint32_t /*h*/) {
    return dim3(16, 16);
}

inline dim3 grid_2d(uint32_t w, uint32_t h) {
    return dim3((w + 15) / 16, (h + 15) / 16);
}

} // namespace filters
} // namespace kagerou
