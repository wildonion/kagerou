// Kagerou SDK — tutorial picture-in-picture compositor (CPU, RGB24).
// Pastes the camera frame scaled into the screen frame's bottom-right
// corner with an accent border. Pure buffer logic, no globals: unit-testable.

#pragma once

#include <cstdint>

namespace kagerou {
namespace pip {

// Value classes for tests/diagnostics: where the PiP landed.
struct PiPBox { int x, y, w, h; };

// PiP geometry shared by the compositor and overlay mapping below.
// Returns false when there is no room (caller skips).
inline bool box(uint32_t sw, uint32_t sh, PiPBox* out) {
    if (!out || !sw || !sh) return false;
    uint32_t PW = (sw / 6) & ~1u, PH = (sh / 6) & ~1u;
    if (PW < 96) PW = 96;
    if (PH < 72) PH = 72;
    if (PW > 320) PW = 320;
    if (PH > 240) PH = 240;
    out->x = (int)sw - (int)PW - 12;
    out->y = (int)sh - (int)PH - 12;
    out->w = (int)PW;
    out->h = (int)PH;
    return out->x >= 3 && out->y >= 3;
}

// Composite camera (cw x ch RGB) into screen (sw x sh RGB, in place).
// PiP is ~1/6 of screen size, clamped to [96x72, 320x240], 12px margin.
// Returns false when there is no room (tiny screen).
inline bool composite(const uint8_t* cam, uint32_t cw, uint32_t ch,
                      uint8_t* screen, uint32_t sw, uint32_t sh,
                      PiPBox* out_box = nullptr) {
    if (!cam || !screen || !cw || !ch || !sw || !sh) return false;
    const int B = 3;
    PiPBox b = {};
    if (!box(sw, sh, &b)) return false;
    int px = b.x, py = b.y;
    uint32_t PW = (uint32_t)b.w, PH = (uint32_t)b.h;
    for (uint32_t y = 0; y < PH; y++) {
        uint32_t sy = y * ch / PH;
        if (sy >= ch) sy = ch - 1;
        for (uint32_t x = 0; x < PW; x++) {
            uint32_t sx = x * cw / PW;
            if (sx >= cw) sx = cw - 1;
            for (int c = 0; c < 3; c++)
                screen[((size_t)(py + y) * sw + px + x) * 3 + c] =
                    cam[((size_t)sy * cw + sx) * 3 + c];
        }
    }
    for (int x = px - B; x < px + (int)PW + B; x++) {
        for (int bb = 0; bb < B; bb++) {
            if (x < 0 || x >= (int)sw) continue;
            int y1 = py - B + bb, y2 = py + (int)PH + bb;
            if (y1 >= 0) {
                size_t o = ((size_t)y1 * sw + x) * 3;
                screen[o] = 0; screen[o+1] = 180; screen[o+2] = 255;
            }
            if (y2 < (int)sh) {
                size_t o = ((size_t)y2 * sw + x) * 3;
                screen[o] = 0; screen[o+1] = 180; screen[o+2] = 255;
            }
        }
    }
    for (int y = py; y < py + (int)PH; y++) {
        for (int bb = 0; bb < B; bb++) {
            int x1 = px - B + bb, x2 = px + (int)PW + bb;
            if (x1 >= 0) {
                size_t o = ((size_t)y * sw + x1) * 3;
                screen[o] = 0; screen[o+1] = 180; screen[o+2] = 255;
            }
            if (x2 < (int)sw) {
                size_t o = ((size_t)y * sw + x2) * 3;
                screen[o] = 0; screen[o+1] = 180; screen[o+2] = 255;
            }
        }
    }
    if (out_box) { out_box->x = px; out_box->y = py; out_box->w = PW; out_box->h = PH; }
    return true;
}

} // namespace pip
} // namespace kagerou
