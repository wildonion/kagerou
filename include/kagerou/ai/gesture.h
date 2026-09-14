// Kagerou SDK — heuristic hand-gesture classifier on 21 hand landmarks.
// MediaPipe order: 0 wrist, thumb 1-4, index 5-8, middle 9-12,
// ring 13-16, pinky 17-20. Coordinates are pixels (any scale: all tests
// are ratios of squared distances from the wrist).
// No model, no state: feeds on ai_hands() output, runs every frame.

#pragma once

#include "ai_filters.h"

namespace kagerou {
namespace ai {

enum class HandGesture {
    kNone = 0,
    kPalm,    // all four fingers open  -> CUT
    kFist,    // everything folded       -> STOP (with hold)
    kThumb,   // thumb open, rest folded -> MARK
    kPeace    // index+middle open       -> PAUSE/RESUME
};

inline HandGesture classify_gesture(const HandJoints& j) {
    auto d2 = [&](int a, int b) {
        float dx = j.x[a] - j.x[b], dy = j.y[a] - j.y[b];
        return dx * dx + dy * dy;
    };
    // finger extended = tip clearly farther from wrist than its PIP joint
    bool idx = d2(8, 0) > d2(6, 0) * 1.21f;    // 1.1^2 margin
    bool mid = d2(12, 0) > d2(10, 0) * 1.21f;
    bool rng = d2(16, 0) > d2(14, 0) * 1.21f;
    bool pky = d2(20, 0) > d2(18, 0) * 1.21f;
    bool thb = d2(4, 0) > d2(3, 0) * 1.32f;    // 1.15^2 margin
    int open4 = (idx ? 1 : 0) + (mid ? 1 : 0) + (rng ? 1 : 0) + (pky ? 1 : 0);
    if (open4 == 4) return HandGesture::kPalm;
    if (open4 == 0 && !thb) return HandGesture::kFist;
    if (thb && open4 == 0) return HandGesture::kThumb;
    if (idx && mid && !rng && !pky) return HandGesture::kPeace;
    return HandGesture::kNone;
}

} // namespace ai
} // namespace kagerou
