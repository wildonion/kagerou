// PiP compositor unit test — synthetic frames, no GPU needed.
// Build: build.bat pip_test
#include <cstdio>
#include <vector>
#include "kagerou/pip.h"

using kagerou::pip::PiPBox;

static int failures = 0;
static void check(bool ok, const char* name) {
    if (!ok) { printf("FAIL %s\n", name); failures++; }
    else printf("ok   %s\n", name);
}

int main() {
    // red-gradient camera 640x480, black 960x540 screen
    std::vector<uint8_t> cam(640 * 480 * 3), scr(960 * 540 * 3, 0);
    for (int y = 0; y < 480; y++)
        for (int x = 0; x < 640; x++) {
            size_t o = ((size_t)y * 640 + x) * 3;
            cam[o] = (uint8_t)(x * 255 / 639);
            cam[o+1] = (uint8_t)(y * 255 / 479);
            cam[o+2] = 40;
        }
    PiPBox box = {};
    bool ok = kagerou::pip::composite(cam.data(), 640, 480, scr.data(), 960, 540, &box);
    check(ok, "composite-returns-true");
    check(box.w == 160 && box.h == 90, "pip-size-160x90");
    check(box.x == 960 - 160 - 12 && box.y == 540 - 90 - 12, "pip-bottom-right");
    // PiP interior carries camera content (not black)
    {
        size_t o = ((size_t)(box.y + 10) * 960 + box.x + 10) * 3;
        check(scr[o] + scr[o+1] + scr[o+2] > 30, "pip-has-content");
    }
    // accent border present (orange 0,180,255)
    {
        int found = 0;
        for (size_t i = 0; i < scr.size(); i += 3)
            if (scr[i] < 60 && scr[i+1] > 140 && scr[i+1] < 220 && scr[i+2] > 200)
                found++;
        check(found > 500, "border-present");
    }
    // outside PiP untouched (still black)
    check(scr[100] == 0 && scr[101] == 0 && scr[102] == 0, "screen-untouched");
    // tiny screen refuses
    std::vector<uint8_t> tiny(100 * 60 * 3, 0);
    check(!kagerou::pip::composite(cam.data(), 640, 480, tiny.data(), 100, 60, nullptr),
          "tiny-screen-refused");
    // box() geometry shared with the overlay mapper
    {
        kagerou::pip::PiPBox b0 = {};
        check(kagerou::pip::box(960, 540, &b0), "box-960-ok");
        check(b0.w == 160 && b0.h == 90, "box-960-size");
        check(b0.x == 960 - 160 - 12 && b0.y == 540 - 90 - 12, "box-960-pos");
        kagerou::pip::PiPBox b1 = {};
        check(kagerou::pip::box(1920, 1080, &b1), "box-1080-ok");
        check(b1.w == 320 && b1.h == 180, "box-1080-size");
        check(!kagerou::pip::box(100, 60, &b1), "box-tiny-refused");
        check(!kagerou::pip::box(0, 0, nullptr), "box-null-refused");
    }
    // null guards
    check(!kagerou::pip::composite(nullptr, 640, 480, scr.data(), 960, 540, nullptr),
          "null-cam-refused");
    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("ALL PIP TESTS PASSED\n");
    return 0;
}
