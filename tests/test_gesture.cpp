// Gesture classifier unit test — synthetic landmark poses, no GPU needed.
// Build: cl /EHsc /I include /I "%CUDA_PATH%\include" tests\test_gesture.cpp
#include <cstdio>
#include "kagerou/ai/gesture.h"

using kagerou::ai::HandJoints;
using kagerou::ai::HandGesture;
using kagerou::ai::classify_gesture;

static int failures = 0;
static void check(HandGesture got, HandGesture want, const char* name) {
    if (got != want) { printf("FAIL %-12s got %d want %d\n", name, (int)got, (int)want); failures++; }
    else printf("ok   %s\n", name);
}

// pose builder: wrist + per-finger (pip, tip) positions; thumb (ip, tip)
static HandJoints pose(float wx, float wy,
                       float ipx, float ipy, float itx, float ity,
                       float mpx, float mpy, float mtx, float mty,
                       float rpx, float rpy, float rtx, float rty,
                       float ppx, float ppy, float ptx, float pty,
                       float tix, float tiy, float ttx, float tty) {
    HandJoints j = {};
    j.x[0] = wx; j.y[0] = wy;
    j.x[6] = ipx; j.y[6] = ipy; j.x[8] = itx; j.y[8] = ity;
    j.x[10] = mpx; j.y[10] = mpy; j.x[12] = mtx; j.y[12] = mty;
    j.x[14] = rpx; j.y[14] = rpy; j.x[16] = rtx; j.y[16] = rty;
    j.x[18] = ppx; j.y[18] = ppy; j.x[20] = ptx; j.y[20] = pty;
    j.x[3] = tix; j.y[3] = tiy; j.x[4] = ttx; j.y[4] = tty;
    j.score = 0.9f;
    return j;
}

int main() {
    // open palm: all tips far from wrist (320,400)
    HandJoints palm = pose(320, 400,
        300, 250, 300, 150,   // index open
        320, 245, 320, 140,   // middle open
        340, 250, 340, 155,   // ring open
        358, 262, 360, 175,   // pinky open
        350, 350, 380, 320);  // thumb open
    check(classify_gesture(palm), HandGesture::kPalm, "palm");

    // fist: all tips curled near wrist
    HandJoints fist = pose(320, 400,
        300, 320, 310, 375,
        320, 318, 320, 375,
        340, 320, 332, 375,
        356, 328, 348, 372,
        345, 370, 340, 385);
    check(classify_gesture(fist), HandGesture::kFist, "fist");

    // thumbs up: thumb out, rest curled
    HandJoints thumb = pose(320, 400,
        300, 320, 310, 375,
        320, 318, 320, 375,
        340, 320, 332, 375,
        356, 328, 348, 372,
        360, 330, 400, 260);
    check(classify_gesture(thumb), HandGesture::kThumb, "thumb");

    // peace: index+middle open, ring+pinky curled
    HandJoints peace = pose(320, 400,
        300, 250, 295, 150,
        325, 245, 328, 145,
        340, 320, 332, 375,
        356, 328, 348, 372,
        345, 370, 340, 385);
    check(classify_gesture(peace), HandGesture::kPeace, "peace");

    // garbage / partial (three fingers) -> none
    HandJoints three = pose(320, 400,
        300, 250, 300, 150,
        320, 245, 320, 140,
        340, 248, 342, 150,
        356, 328, 348, 372,
        345, 370, 340, 385);
    check(classify_gesture(three), HandGesture::kNone, "three-fingers");

    // scale invariance: same palm at 2x size, shifted
    HandJoints big = pose(100, 700,
        60, 400, 60, 200,
        100, 390, 100, 180,
        140, 400, 140, 210,
        176, 424, 180, 250,
        160, 600, 220, 540);
    check(classify_gesture(big), HandGesture::kPalm, "palm-big");

    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("ALL GESTURE TESTS PASSED\n");
    return 0;
}
