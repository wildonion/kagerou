// Kagerou screen-capture shared protocol (helper <-> main app).
// Per-instance names ("Local\\KagerouScreen_<pid>" + "...Stop") so two app
// instances never share/stomp each other's feed (a global name made concurrent
// runs mutually destructive: stop events killed the wrong helper).
// Seqlock: even seq = consistent frame.

#pragma once

#include <cstdint>
#include <cstdio>

#define KSCR_MAGIC 0x4B534352u // 'KSCR'
#define KSCR_VERSION 1u
// Native capture cap (helper publishes FEED-box-fit RGB, not native:
// a 4K bilinear downscale on the reader would cap takes at ~20fps).
#define KSCR_MAX_W 3840
#define KSCR_MAX_H 2160
#define FEED_CAP_W 1920
#define FEED_CAP_H 1080
#define KSCR_MAX_BYTES ((size_t)FEED_CAP_W * FEED_CAP_H * 3)

#pragma pack(push, 1)
struct ScreenShm {
    uint32_t magic;   // KSCR_MAGIC
    uint32_t version; // KSCR_VERSION
    volatile long seq;   // seqlock: odd = writer active
    volatile long state; // 0 init, 1 ready(frame), 2 failed, 3 done
    uint32_t w, h;       // RGB24 payload dims (native desktop)
    // bytes follow: w*h*3 RGB24
};
#pragma pack(pop)

// Build the per-instance object names. tag = parent PID string.
inline void kscr_names(const char* tag, char* map, size_t mapn,
                       char* ev, size_t evn) {
    snprintf(map, mapn, "Local\\KagerouScreen_%s", tag);
    snprintf(ev, evn, "Local\\KagerouScreenStop_%s", tag);
}
