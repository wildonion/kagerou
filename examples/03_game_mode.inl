// ============================================================================
// Kagerou GAME tab — gesture game control (external games + built-in trainer)
// Included INTO 02_virtualcam_demo.cu (uses its globals + draw helpers).
// External mode: hand gestures -> SendInput mouse/keyboard to a detected
// game window. Trainer mode: in-app shooting gallery on the camera feed.
// Canvas draws into g_d_rgb, so preview + vcam + record tap carry it.
// ============================================================================
#include <mmsystem.h>
#include <cctype>
#include <cmath>
#include "kagerou/ai/gesture.h"

// ---- state ----
enum GameMode { GM_OFF = 0, GM_EXTERNAL, GM_TRAINER };
static int g_game_mode = GM_OFF;
static bool g_game_on = false;
static bool g_game_vcam_out = true;  // draw canvas (else clean feed, injector still runs)
static int g_game_aim_mode = 0;      // 0 absolute, 1 relative
static float g_game_sens_vals[] = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
static int g_game_sens_idx = 2;
static float g_game_dz_vals[] = {0.05f, 0.12f, 0.2f};
static int g_game_dz_idx = 1;
static float g_game_sm_vals[] = {0.2f, 0.35f, 0.5f};
static int g_game_sm_idx = 1;
static int g_game_move_src = 0;      // 0 left-hand zones, 1 face lean
static char g_game_target[128] = ""; // fg exe/title substring, empty = any window
static float g_game_cal_cx = 0.5f, g_game_cal_cy = 0.5f; // aim center (norm)
static float g_game_cal_fx = 0.5f, g_game_cal_fy = 0.5f; // face ref (norm)
static bool g_game_calibrated = false;
static bool g_game_ini_loaded = false;

// Bindings (click row to cycle). Mouse buttons encoded as VK + 0x100.
static const uint8_t BIND_LCLICK = 0xF0, BIND_RCLICK = 0xF1;
static uint8_t g_bind_fire = BIND_LCLICK;
static uint8_t g_bind_jump = VK_SPACE;
static uint8_t g_bind_reload = 'R';
static uint8_t g_bind_pause = VK_ESCAPE;
static uint8_t g_bind_interact = 'E';
static const uint8_t kFireOpts[] = {BIND_LCLICK, BIND_RCLICK, VK_SPACE};
static const uint8_t kJumpOpts[] = {VK_SPACE, 'W'};
static const uint8_t kReloadOpts[] = {'R', 'F'};
static const uint8_t kPauseOpts[] = {VK_ESCAPE, 'P'};
static const uint8_t kInteractOpts[] = {'E', 'F'};

// Runtime aim / gesture state
static float g_game_aimx = 0.5f, g_game_aimy = 0.5f; // smoothed cursor (norm)
static bool g_game_has_aim = false;
static float g_game_aim_cx = 0, g_game_aim_cy = 0;   // aim-hand centroid (px, disp coords)
static bool g_game_aim_valid = false;
static float g_game_last_cx = 0, g_game_last_cy = 0;
static bool g_game_has_last = false;
static float g_game_flick_hist[6] = {};
static int g_game_flick_n = 0;
static uint32_t g_game_flick_cool = 0;
static uint32_t g_game_palm_hold_t0 = 0;
static bool g_game_palm_holding = false;
static int g_game_crouch_n = 0;
static char g_game_gesture_txt[32] = "no hands";
static char g_game_fg_txt[128] = "-";
static char g_game_last_fg[128] = ""; // last non-Kagerou foreground exe (for GRAB)
static int g_game_hand_fps_n = 0;
static double g_game_hand_fps = 0;
static uint32_t g_game_fps_t0 = 0;

// Injected-held state (for release-all safety)
static bool g_inj_held_vk[256] = {};
static bool g_inj_lbutton = false, g_inj_rbutton = false;

// Debounce helper: returns 1 on rise, -1 on fall, 0 else; .state = level.
struct EdgeDeb { int on_n = 0, off_n = 0; bool state = false; };
static int deb_update(EdgeDeb& e, bool held) {
    if (held) {
        e.on_n++; e.off_n = 0;
        if (!e.state && e.on_n >= 2) { e.state = true; return 1; }
    } else {
        e.off_n++; e.on_n = 0;
        if (e.state && e.off_n >= 2) { e.state = false; return -1; }
    }
    return 0;
}
static EdgeDeb g_deb_fire;
static float g_game_rel_dx = 0, g_game_rel_dy = 0; // relative-mode per-frame deltas (norm)

// ---- trainer state ----
enum TrState { TR_IDLE = 0, TR_PLAY, TR_PAUSE, TR_OVER };
static int g_tr_state = TR_IDLE;
struct GTarget { float nx, ny, r; float vx, vy; int hp; bool alive; uint32_t born; int flash; };
static GTarget g_trg[12];
static int g_tr_score = 0, g_tr_combo = 0, g_tr_best = 0, g_tr_hp = 100;
static int g_tr_ammo = 12, g_tr_wave = 1, g_tr_kills = 0;
static double g_tr_time = 60.0;
static uint32_t g_tr_reload_t0 = 0;
static bool g_tr_reloading = false;

// ---- sounds (synthesized WAVs, no assets) ----
static std::vector<uint8_t> g_snd_shot, g_snd_hit, g_snd_reload, g_snd_click, g_snd_wave;
static std::vector<uint8_t> synth_wav(const std::vector<int16_t>& pcm) {
    std::vector<uint8_t> w(44 + pcm.size() * 2, 0);
    memcpy(&w[0], "RIFF", 4);
    uint32_t sz = (uint32_t)(36 + pcm.size() * 2);
    memcpy(&w[4], &sz, 4);
    memcpy(&w[8], "WAVEfmt ", 8);
    uint32_t f16 = 16; memcpy(&w[16], &f16, 4);
    uint16_t a1 = 1, ch1 = 1; memcpy(&w[20], &a1, 2); memcpy(&w[22], &ch1, 2);
    uint32_t sr = 22050; memcpy(&w[24], &sr, 4);
    uint32_t br = 22050 * 2; memcpy(&w[28], &br, 4);
    uint16_t ba = 2, bp = 16; memcpy(&w[32], &ba, 2); memcpy(&w[34], &bp, 2);
    memcpy(&w[36], "data", 4);
    memcpy(&w[40], &sz, 4); // reuse (close enough for PlaySound size field)
    uint32_t ds = (uint32_t)(pcm.size() * 2); memcpy(&w[40], &ds, 4);
    memcpy(&w[44], pcm.data(), pcm.size() * 2);
    return w;
}
static void game_init_sounds() {
    if (!g_snd_shot.empty()) return;
    { // gunshot: noise burst + low thump, 0.18s
        int n = 22050 * 18 / 100; std::vector<int16_t> p(n);
        unsigned s = 12345;
        for (int i = 0; i < n; i++) {
            s = s * 1103515245 + 12345;
            float nz = ((s >> 16) & 0x7FFF) / 32768.0f - 0.5f;
            float t = (float)i / n;
            float th = sinf(i * 6.28318f * 120.0f / 22050.0f) * expf(-t * 9.0f);
            p[i] = (int16_t)((nz * expf(-t * 14.0f) * 0.8f + th * 0.6f) * 30000);
        }
        g_snd_shot = synth_wav(p);
    }
    { // hit: 880->1320 sweep, 0.09s
        int n = 22050 * 9 / 100; std::vector<int16_t> p(n);
        for (int i = 0; i < n; i++) {
            float f = 880.0f + 440.0f * i / n;
            p[i] = (int16_t)(sinf(i * 6.28318f * f / 22050.0f) * 22000 * (1.0f - (float)i / n));
        }
        g_snd_hit = synth_wav(p);
    }
    { // reload: two clicks
        int n = 22050 * 14 / 100; std::vector<int16_t> p(n, 0);
        for (int k = 0; k < 2; k++) {
            int o = k * n / 2;
            for (int i = 0; i < n / 8 && o + i < n; i++)
                p[o + i] = (int16_t)(10000 * (1.0f - (float)i / (n / 8)) * ((i % 2) ? 1 : -1));
        }
        g_snd_reload = synth_wav(p);
    }
    { // click: 0.05s 1200Hz
        int n = 22050 * 5 / 100; std::vector<int16_t> p(n);
        for (int i = 0; i < n; i++)
            p[i] = (int16_t)(sinf(i * 6.28318f * 1200.0f / 22050.0f) * 16000 * (1.0f - (float)i / n));
        g_snd_click = synth_wav(p);
    }
    { // wave: 440->880, 0.2s
        int n = 22050 * 2 / 10; std::vector<int16_t> p(n);
        for (int i = 0; i < n; i++) {
            float f = 440.0f + 440.0f * i / n;
            p[i] = (int16_t)(sinf(i * 6.28318f * f / 22050.0f) * 18000);
        }
        g_snd_wave = synth_wav(p);
    }
}
static void game_snd(const std::vector<uint8_t>& w) {
    if (!w.empty()) PlaySoundA((LPCSTR)w.data(), NULL, SND_MEMORY | SND_ASYNC | SND_NODEFAULT);
}

// ---- ini persist ----
static std::string game_ini_path() { return get_exe_dir() + "\\game_binds.ini"; }
static void game_save_ini() {
    FILE* f = fopen(game_ini_path().c_str(), "w");
    if (!f) return;
    fprintf(f, "mode=%d\naim=%d\nsens=%d\ndz=%d\nsmooth=%d\nmove=%d\nvcam=%d\n",
            g_game_mode, g_game_aim_mode, g_game_sens_idx, g_game_dz_idx,
            g_game_sm_idx, g_game_move_src, g_game_vcam_out ? 1 : 0);
    fprintf(f, "fire=%u\njump=%u\nreload=%u\npause=%u\ninteract=%u\n",
            g_bind_fire, g_bind_jump, g_bind_reload, g_bind_pause, g_bind_interact);
    fprintf(f, "target=%s\n", g_game_target);
    fclose(f);
}
static void game_load_ini() {
    if (g_game_ini_loaded) return;
    g_game_ini_loaded = true;
    game_init_sounds();
    FILE* f = fopen(game_ini_path().c_str(), "r");
    if (!f) return;
    char k[64], v[128];
    while (fscanf(f, "%63[^=]=%127[^\n]\n", k, v) == 2) {
        if (!strcmp(k, "mode")) g_game_mode = atoi(v);
        else if (!strcmp(k, "aim")) g_game_aim_mode = atoi(v);
        else if (!strcmp(k, "sens")) g_game_sens_idx = atoi(v);
        else if (!strcmp(k, "dz")) g_game_dz_idx = atoi(v);
        else if (!strcmp(k, "smooth")) g_game_sm_idx = atoi(v);
        else if (!strcmp(k, "move")) g_game_move_src = atoi(v);
        else if (!strcmp(k, "vcam")) g_game_vcam_out = atoi(v) != 0;
        else if (!strcmp(k, "fire")) g_bind_fire = (uint8_t)atoi(v);
        else if (!strcmp(k, "jump")) g_bind_jump = (uint8_t)atoi(v);
        else if (!strcmp(k, "reload")) g_bind_reload = (uint8_t)atoi(v);
        else if (!strcmp(k, "pause")) g_bind_pause = (uint8_t)atoi(v);
        else if (!strcmp(k, "interact")) g_bind_interact = (uint8_t)atoi(v);
        else if (!strcmp(k, "target")) { strncpy(g_game_target, v, sizeof(g_game_target) - 1); }
    }
    fclose(f);
    if (g_game_mode < 0 || g_game_mode > 2) g_game_mode = GM_OFF;
}

// ---- foreground target detection ----
static std::string game_own_exe() {
    char p[MAX_PATH] = {};
    GetModuleFileNameA(NULL, p, MAX_PATH);
    std::string s(p);
    size_t q = s.find_last_of("\\/");
    return (q == std::string::npos) ? s : s.substr(q + 1);
}
static std::string game_fg_exe() {
    HWND fg = GetForegroundWindow();
    if (!fg) return "";
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    if (!pid) return "";
    HANDLE pr = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!pr) return "";
    char p[MAX_PATH] = {};
    DWORD n = MAX_PATH;
    std::string out;
    if (QueryFullProcessImageNameA(pr, 0, p, &n)) {
        out = p;
        size_t q = out.find_last_of("\\/");
        if (q != std::string::npos) out = out.substr(q + 1);
    }
    CloseHandle(pr);
    return out;
}
static std::string game_fg_title() {
    HWND fg = GetForegroundWindow();
    if (!fg) return "";
    char t[128] = {};
    GetWindowTextA(fg, t, sizeof(t) - 1);
    return std::string(t);
}
static bool game_target_ok() {
    std::string fg = game_fg_exe();
    if (fg.empty()) return false;
    // Never inject into ourselves (lets the user click our own UI safely).
    std::string own = game_own_exe();
    if (!_stricmp(fg.c_str(), own.c_str())) return false;
    if (g_game_target[0] == 0) return true; // any window
    std::string tgt(g_game_target), fgl = fg, ttl = game_fg_title();
    for (auto& c : tgt) c = (char)tolower(c);
    for (auto& c : fgl) c = (char)tolower(c);
    for (auto& c : ttl) c = (char)tolower(c);
    return fgl.find(tgt) != std::string::npos || ttl.find(tgt) != std::string::npos;
}

// ---- injector ----
static void inj_mouse_abs(float nx, float ny) {
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
    if (nx < 0) nx = 0; if (nx > 1) nx = 1;
    if (ny < 0) ny = 0; if (ny > 1) ny = 1;
    in.mi.dx = (LONG)(nx * 65535.0f);
    in.mi.dy = (LONG)(ny * 65535.0f);
    SendInput(1, &in, sizeof(in));
}
static void inj_mouse_rel(int dx, int dy) {
    if (!dx && !dy) return;
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_MOVE;
    in.mi.dx = dx; in.mi.dy = dy;
    SendInput(1, &in, sizeof(in));
}
static void inj_button(bool& held, DWORD down_flag, DWORD up_flag, bool want) {
    if (want == held) return;
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = want ? down_flag : up_flag;
    SendInput(1, &in, sizeof(in));
    held = want;
}
static void inj_key(uint8_t vk, bool want) {
    if (g_inj_held_vk[vk] == want) return;
    INPUT in = {};
    in.type = INPUT_KEYBOARD;
    in.ki.wVk = vk;
    in.ki.dwFlags = want ? 0 : KEYEVENTF_KEYUP;
    SendInput(1, &in, sizeof(in));
    g_inj_held_vk[vk] = want;
}
static void inj_bind(uint8_t b, bool want) {
    if (b == BIND_LCLICK) inj_button(g_inj_lbutton, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, want);
    else if (b == BIND_RCLICK) inj_button(g_inj_rbutton, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, want);
    else inj_key(b, want);
}
static void inj_release_all() {
    for (int i = 0; i < 256; i++)
        if (g_inj_held_vk[i]) inj_key((uint8_t)i, false);
    if (g_inj_lbutton) inj_button(g_inj_lbutton, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, false);
    if (g_inj_rbutton) inj_button(g_inj_rbutton, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, false);
}

// ---- hand analysis ----
struct GameHand {
    bool valid = false;
    float cx = 0, cy = 0;   // centroid px (disp coords)
    float pinch = 1.0f;     // thumb-index gap / palm size (small = pinched)
    ai::HandGesture g = ai::HandGesture::kNone;
    bool point = false;     // index only
};
static float gdist(const kagerou::ai::HandJoints& j, int a, int b) {
    float dx = j.x[a] - j.x[b], dy = j.y[a] - j.y[b];
    return sqrtf(dx * dx + dy * dy);
}
static GameHand game_analyze(const kagerou::ai::HandJoints& j) {
    GameHand gh;
    if (j.score <= 0.5f) return gh;
    gh.valid = true;
    gh.cx = (j.x[5] + j.x[9] + j.x[13] + j.x[17]) * 0.25f;
    gh.cy = (j.y[5] + j.y[9] + j.y[13] + j.y[17]) * 0.25f;
    float palm = gdist(j, 0, 9);
    if (palm < 1.0f) palm = 1.0f;
    gh.pinch = gdist(j, 4, 8) / palm;
    gh.g = ai::classify_gesture(j);
    auto ext = [&](int tip, int pip) {
        float dx1 = j.x[tip] - j.x[0], dy1 = j.y[tip] - j.y[0];
        float dx2 = j.x[pip] - j.x[0], dy2 = j.y[pip] - j.y[0];
        return (dx1 * dx1 + dy1 * dy1) > (dx2 * dx2 + dy2 * dy2) * 1.21f;
    };
    bool idx = ext(8, 6), mid = ext(12, 10), rng = ext(16, 14), pky = ext(20, 18);
    gh.point = idx && !mid && !rng && !pky;
    return gh;
}

// ---- trainer ----
static void trainer_reset() {
    g_tr_state = TR_IDLE;
    g_tr_score = 0; g_tr_combo = 0; g_tr_best = 0; g_tr_hp = 100;
    g_tr_ammo = 12; g_tr_wave = 1; g_tr_kills = 0; g_tr_time = 60.0;
    g_tr_reloading = false;
    for (auto& t : g_trg) t.alive = false;
}
static void trainer_spawn_wave() {
    int n = 3 + g_tr_wave;
    if (n > 12) n = 12;
    unsigned s = (unsigned)(GetTickCount() + g_tr_wave * 7919);
    for (int i = 0; i < n; i++) {
        auto& t = g_trg[i];
        t.alive = true;
        s = s * 1103515245 + 12345;
        t.nx = 0.1f + ((s >> 16) % 800) / 1000.0f;
        s = s * 1103515245 + 12345;
        t.ny = 0.15f + ((s >> 16) % 650) / 1000.0f;
        t.r = 30.0f - g_tr_wave * 1.5f;
        if (t.r < 14.0f) t.r = 14.0f;
        s = s * 1103515245 + 12345;
        float sp = (20.0f + g_tr_wave * 6.0f) * ((((s >> 16) % 2) ? 1.0f : -1.0f));
        s = s * 1103515245 + 12345;
        t.vx = sp * (0.5f + ((s >> 16) % 500) / 1000.0f);
        s = s * 1103515245 + 12345;
        t.vy = sp * 0.6f * ((((s >> 16) % 2) ? 1.0f : -1.0f));
        t.hp = 1; t.born = GetTickCount(); t.flash = 0;
    }
    for (int i = n; i < 12; i++) g_trg[i].alive = false;
}
static void trainer_start() {
    trainer_reset();
    g_tr_state = TR_PLAY;
    trainer_spawn_wave();
    game_snd(g_snd_wave);
}
static void trainer_update(double dt, float aim_nx, float aim_ny, bool fire_edge, int W, int H) {
    if (g_tr_state != TR_PLAY) return;
    g_tr_time -= dt;
    if (g_tr_time <= 0) { g_tr_time = 0; g_tr_state = TR_OVER; game_snd(g_snd_click); return; }
    if (g_tr_reloading) {
        if (GetTickCount() - g_tr_reload_t0 > 1200) {
            g_tr_reloading = false; g_tr_ammo = 12;
        }
    }
    bool any = false;
    uint32_t now = GetTickCount();
    for (auto& t : g_trg) {
        if (!t.alive) continue;
        any = true;
        t.nx += t.vx * (float)dt / W;
        t.ny += t.vy * (float)dt / H;
        if (t.nx < 0.05f || t.nx > 0.95f) t.vx = -t.vx;
        if (t.ny < 0.08f || t.ny > 0.92f) t.vy = -t.vy;
        if (t.flash > 0) t.flash--;
        if (now - t.born > 7000) { // expired: combo lost, hp chip
            t.alive = false; g_tr_combo = 0; g_tr_hp -= 5;
            if (g_tr_hp <= 0) { g_tr_hp = 0; g_tr_state = TR_OVER; }
        }
    }
    if (!any && g_tr_state == TR_PLAY) {
        g_tr_wave++;
        if (g_tr_wave > 8) { g_tr_state = TR_OVER; game_snd(g_snd_wave); return; }
        trainer_spawn_wave();
        game_snd(g_snd_wave);
    }
    if (fire_edge && !g_tr_reloading) {
        if (g_tr_ammo <= 0) {
            g_tr_reloading = true; g_tr_reload_t0 = now;
            game_snd(g_snd_reload);
            return;
        }
        g_tr_ammo--;
        game_snd(g_snd_shot);
        float ax = aim_nx * W, ay = aim_ny * H;
        bool hit = false;
        for (auto& t : g_trg) {
            if (!t.alive) continue;
            float dx = ax - t.nx * W, dy = ay - t.ny * H;
            if (dx * dx + dy * dy <= t.r * t.r) {
                t.hp--; t.flash = 6; hit = true;
                if (t.hp <= 0) {
                    t.alive = false;
                    g_tr_combo++;
                    if (g_tr_combo > g_tr_best) g_tr_best = g_tr_combo;
                    g_tr_score += 100 * g_tr_combo;
                    g_tr_kills++;
                }
                break;
            }
        }
        if (hit) game_snd(g_snd_hit);
        else { g_tr_combo = 0; }
        if (g_tr_ammo <= 0) { g_tr_reloading = true; g_tr_reload_t0 = now; }
    }
}

// ---- enable/stop/kill ----
static void game_release_all() { inj_release_all(); }
static void game_set_on(bool on) {
    if (on == g_game_on) return;
    if (on) {
        game_load_ini();
        if (g_game_mode == GM_OFF) return;
        if (!g_ai_hands_ready.load()) return; // hands model missing: can't play
        if (g_src_mode != 0) g_src_mode = 0;  // game needs the CAM feed
        g_game_has_aim = false; g_game_has_last = false;
        g_game_palm_holding = false;
        g_deb_fire = EdgeDeb();
        if (g_game_mode == GM_TRAINER) trainer_start();
        else g_tr_state = TR_IDLE;
        g_game_on = true;
    } else {
        game_release_all();
        g_game_on = false;
        if (g_tr_state == TR_PLAY) g_tr_state = TR_PAUSE;
    }
    game_save_ini();
}
static void game_kill() {
    game_release_all();
    g_game_on = false;
    if (g_tr_state == TR_PLAY) g_tr_state = TR_PAUSE;
}
static bool game_is_on() { return g_game_on; }
static void game_on_key(WPARAM w) {
    if (w == VK_F12) game_kill();
}
static void game_calibrate() {
    if (g_game_aim_valid) { g_game_cal_cx = g_game_aim_cx / g_disp_w; g_game_cal_cy = g_game_aim_cy / g_disp_h; }
    if (g_face_boxes[0].score > 0.5f) {
        g_game_cal_fx = (g_face_boxes[0].x1 + g_face_boxes[0].x2) * 0.5f / g_disp_w;
        g_game_cal_fy = (g_face_boxes[0].y1 + g_face_boxes[0].y2) * 0.5f / g_disp_h;
    }
    g_game_calibrated = true;
    g_game_has_aim = false;
}

// ---- per-frame tick (GPU thread, caches fresh by now) ----
static void game_tick_gpu() {
    // Remember the last external foreground window even when idle, so
    // GRAB GAME locks the game (not Kagerou, which is focused on click).
    {
        std::string fg = game_fg_exe();
        if (!fg.empty() && _stricmp(fg.c_str(), game_own_exe().c_str()))
            strncpy(g_game_last_fg, fg.c_str(), sizeof(g_game_last_fg) - 1);
    }
    if (!g_game_on) return;
    if (g_ai_warming.load() || g_src_mode != 0) return;
    bool hands_ready = g_ai_hands_ready.load();
    bool face_ready = g_ai_face_ready.load();
    if (!hands_ready) return;
    uint32_t W = g_disp_w, H = g_disp_h;
    if (!W || !H) return;
    // Ensure inference even when the user toggles are off (every 2nd frame).
    if (!g_ai_hands && (g_frame_count % 2) == 0)
        ai::ai_hands(g_d_rgb, W, H, g_hand_joints, 2, g_stream);
    bool want_face = (g_game_mode == GM_EXTERNAL) && face_ready && !g_ai_face;
    if (want_face && (g_frame_count % 2) == 0)
        ai::ai_face(g_d_rgb, W, H, g_face_boxes, 4, g_stream);

    // Role assignment: sticky aim hand by proximity, other = move hand.
    GameHand ah[2] = {game_analyze(g_hand_joints[0]), game_analyze(g_hand_joints[1])};
    int nvalid = (ah[0].valid ? 1 : 0) + (ah[1].valid ? 1 : 0);
    g_game_hand_fps_n++;
    uint32_t now = GetTickCount();
    if (!g_game_fps_t0) g_game_fps_t0 = now;
    if (now - g_game_fps_t0 >= 1000) {
        g_game_hand_fps = g_game_hand_fps_n * 1000.0 / (now - g_game_fps_t0);
        g_game_hand_fps_n = 0; g_game_fps_t0 = now;
    }
    int ai = -1, mi = -1;
    if (nvalid == 1) ai = ah[0].valid ? 0 : 1;
    else if (nvalid == 2) {
        float d0 = g_game_has_last ?
            (ah[0].cx - g_game_last_cx) * (ah[0].cx - g_game_last_cx) +
            (ah[0].cy - g_game_last_cy) * (ah[0].cy - g_game_last_cy) : 0;
        float d1 = g_game_has_last ?
            (ah[1].cx - g_game_last_cx) * (ah[1].cx - g_game_last_cx) +
            (ah[1].cy - g_game_last_cy) * (ah[1].cy - g_game_last_cy) : 1e30f;
        if (!g_game_has_last) {
            // Prefer right hand for aim (larger x)? No: prefer the
            // pointing/firing hand; default to hand 0.
            ai = 0; mi = 1;
        } else { ai = (d0 <= d1) ? 0 : 1; mi = 1 - ai; }
    }
    if (ai < 0) {
        snprintf(g_game_gesture_txt, sizeof(g_game_gesture_txt), "no hands");
        g_game_aim_valid = false;
        return;
    }
    GameHand& A = ah[ai];
    g_game_aim_cx = A.cx; g_game_aim_cy = A.cy;
    g_game_aim_valid = true;
    // Previous centroid for relative mode / flick (captured BEFORE update).
    float pcx = g_game_last_cx, pcy = g_game_last_cy;
    bool had_last = g_game_has_last;

    // Gesture text for readout
    const char* gn = "point/open";
    if (A.g == ai::HandGesture::kPalm) gn = "palm";
    else if (A.g == ai::HandGesture::kFist) gn = "fist";
    else if (A.g == ai::HandGesture::kThumb) gn = "thumb";
    else if (A.g == ai::HandGesture::kPeace) gn = "peace";
    else if (A.point) gn = "point";
    snprintf(g_game_gesture_txt, sizeof(g_game_gesture_txt), "%s%s",
             gn, A.pinch < 0.5f ? "+pinch" : "");

    // Aim (skip while calibrating target unknown — still track).
    {
        float sens = g_game_sens_vals[g_game_sens_idx];
        float dz = g_game_dz_vals[g_game_dz_idx];
        float sm = g_game_sm_vals[g_game_sm_idx];
        float nx = A.cx / W, ny = A.cy / H;
        if (g_game_aim_mode == 0) {
            float tx = (nx - g_game_cal_cx) * sens + 0.5f;
            float ty = (ny - g_game_cal_cy) * sens + 0.5f;
            if (fabsf(tx - 0.5f) < dz) tx = 0.5f;
            if (fabsf(ty - 0.5f) < dz) ty = 0.5f;
            if (tx < 0) tx = 0; if (tx > 1) tx = 1;
            if (ty < 0) ty = 0; if (ty > 1) ty = 1;
            if (!g_game_has_aim) { g_game_aimx = tx; g_game_aimy = ty; g_game_has_aim = true; }
            else { g_game_aimx += (tx - g_game_aimx) * (1.0f - sm); g_game_aimy += (ty - g_game_aimy) * (1.0f - sm); }
        } else {
            // Relative: displacement from last centroid -> cursor velocity.
            float dx = 0, dy = 0;
            if (had_last) {
                dx = (A.cx - pcx) / W * sens * 2.0f;
                dy = (A.cy - pcy) / H * sens * 2.0f;
                if (fabsf(dx) < dz * 0.2f) dx = 0;
                if (fabsf(dy) < dz * 0.2f) dy = 0;
            }
            g_game_rel_dx = dx * (1.0f - sm); g_game_rel_dy = dy * (1.0f - sm);
            g_game_aimx += g_game_rel_dx; g_game_aimy += g_game_rel_dy;
            if (g_game_aimx < 0) g_game_aimx = 0; if (g_game_aimx > 1) g_game_aimx = 1;
            if (g_game_aimy < 0) g_game_aimy = 0; if (g_game_aimy > 1) g_game_aimy = 1;
            g_game_has_aim = true;
        }
    }
    g_game_last_cx = A.cx; g_game_last_cy = A.cy;
    g_game_has_last = true;

    // Palm-hold kill (2s open palm quits to safety).
    if (A.g == ai::HandGesture::kPalm) {
        if (!g_game_palm_holding) { g_game_palm_holding = true; g_game_palm_hold_t0 = now; }
        else if (now - g_game_palm_hold_t0 > 2000) { game_kill(); return; }
    } else g_game_palm_holding = false;

    // Flick-up history (edge evaluated below).
    {
        for (int i = 5; i > 0; i--) g_game_flick_hist[i] = g_game_flick_hist[i - 1];
        g_game_flick_hist[0] = A.cy / H;
        if (g_game_flick_n < 6) g_game_flick_n++;
    }

    bool fire_held = (A.g == ai::HandGesture::kPeace);
    bool pinch_held = (A.pinch < 0.5f);
    int fire_ev = deb_update(g_deb_fire, fire_held || pinch_held);
    // Jump = flick-up edge -> 150ms key tap.
    static uint32_t jump_t0 = 0;
    static bool jump_down = false;
    bool flick_edge = false;
    if (g_game_flick_n >= 4 && now - g_game_flick_cool > 800) {
        float rise = g_game_flick_hist[3] - g_game_flick_hist[0];
        if (rise > 0.18f) { flick_edge = true; g_game_flick_cool = now; }
    }
    if (flick_edge && !jump_down) { inj_bind(g_bind_jump, true); jump_down = true; jump_t0 = now; }
    if (jump_down && now - jump_t0 > 150) { inj_bind(g_bind_jump, false); jump_down = false; }

    if (g_game_mode == GM_EXTERNAL) {
        if (!game_target_ok()) {
            // Wrong window: hold everything released (safety).
            inj_release_all();
            g_deb_fire.state = false;
            snprintf(g_game_fg_txt, sizeof(g_game_fg_txt), "waiting: %s",
                     g_game_target[0] ? g_game_target : "any");
            return;
        }
        snprintf(g_game_fg_txt, sizeof(g_game_fg_txt), "%s", game_fg_exe().c_str());
        // Mouse + fire
        if (g_game_has_aim) {
            if (g_game_aim_mode == 0) inj_mouse_abs(g_game_aimx, g_game_aimy);
            else inj_mouse_rel((int)(g_game_rel_dx * 3000.0f), (int)(g_game_rel_dy * 3000.0f));
        }
        inj_bind(g_bind_fire, g_deb_fire.state);
        // Reload / pause / interact on gesture edges (tap = 150ms press).
        // (Jump already handled as a timed tap above.)
        static uint32_t feat_t0[4] = {};
        static bool feat_on[4] = {};
        bool feats[4] = {
            A.g == ai::HandGesture::kFist,
            false, // filled below: palm tap = pause key
            A.g == ai::HandGesture::kThumb,
            false  // spare
        };
        // palm tap (short, not the 2s kill hold) = pause key
        feats[1] = (A.g == ai::HandGesture::kPalm) && g_game_palm_holding &&
                   (now - g_game_palm_hold_t0 < 600);
        uint8_t binds[4] = {g_bind_reload, g_bind_pause, g_bind_interact, g_bind_interact};
        for (int i = 0; i < 4; i++) {
            if (feats[i] && !feat_on[i]) { feat_on[i] = true; feat_t0[i] = now; inj_bind(binds[i], true); }
            if (feat_on[i] && (!feats[i] || now - feat_t0[i] > 150)) {
                if (now - feat_t0[i] > 150 || !feats[i]) { feat_on[i] = false; inj_bind(binds[i], false); }
            }
        }
        // Move: left-hand zones or face lean
        bool kW = false, kA = false, kS = false, kD = false, kShift = false;
        if (g_game_move_src == 0 && mi >= 0 && ah[mi].valid) {
            float lx = ah[mi].cx / W - g_game_cal_cx;
            float ly = ah[mi].cy / H - g_game_cal_cy;
            if (ly < -0.25f) kW = true; if (ly > 0.25f) kS = true;
            if (lx < -0.25f) kA = true; if (lx > 0.25f) kD = true;
            float lpalm = gdist(g_hand_joints[mi], 0, 9);
            if (lpalm < 1.0f) lpalm = 1.0f;
            if (gdist(g_hand_joints[mi], 4, 8) / lpalm < 0.5f) kShift = true;
        } else if (g_game_move_src == 1 && g_face_boxes[0].score > 0.5f) {
            float fx = (g_face_boxes[0].x1 + g_face_boxes[0].x2) * 0.5f / W;
            if (fx < g_game_cal_fx - 0.08f) kA = true;
            if (fx > g_game_cal_fx + 0.08f) kD = true;
        }
        inj_key('W', kW); inj_key('A', kA); inj_key('S', kS); inj_key('D', kD);
        inj_key(VK_SHIFT, kShift);
        // Crouch: face drops (sitting) sustained
        if (face_ready && g_face_boxes[0].score > 0.5f && g_game_calibrated) {
            float fy = (g_face_boxes[0].y1 + g_face_boxes[0].y2) * 0.5f / H;
            if (fy > g_game_cal_fy + 0.10f) { if (++g_game_crouch_n >= 3) inj_key('C', true); }
            else { g_game_crouch_n = 0; inj_key('C', false); }
        } else inj_key('C', false);
        (void)fire_ev;
    } else {
        // Trainer: fire edge drives hit-test; jump/flick pauses or restarts.
        snprintf(g_game_fg_txt, sizeof(g_game_fg_txt), "trainer");
        static uint32_t last_ms = 0;
        uint32_t ms = now;
        double dt = last_ms ? (ms - last_ms) / 1000.0 : 0.033;
        last_ms = ms;
        if (dt > 0.25) dt = 0.25;
        if (g_tr_state == TR_PLAY)
            trainer_update(dt, g_game_aimx, g_game_aimy, fire_ev == 1, W, H);
        else if (g_tr_state == TR_OVER) {
            if (A.g == ai::HandGesture::kThumb) { trainer_start(); }
        } else if (g_tr_state == TR_PAUSE) {
            if (A.g == ai::HandGesture::kThumb) { g_tr_state = TR_PLAY; }
            if (A.g == ai::HandGesture::kPalm && g_game_palm_holding &&
                (now - g_game_palm_hold_t0 > 2000)) { trainer_reset(); }
        } else if (g_tr_state == TR_IDLE) {
            if (A.g == ai::HandGesture::kThumb) trainer_start();
        }
        // Pause toggle on peace? No: peace = fire. Palm tap = pause.
        if (g_tr_state == TR_PLAY && A.g == ai::HandGesture::kPalm &&
            g_game_palm_holding && (now - g_game_palm_hold_t0 > 600) &&
            (now - g_game_palm_hold_t0 < 2000)) {
            static uint32_t last_pause = 0;
            if (now - last_pause > 1500) { last_pause = now; g_tr_state = TR_PAUSE; }
        }
    }
}

// ---- canvas draw (into g_d_rgb; vcam + preview + record carry it) ----
static void game_draw_canvas() {
    if (!g_game_on || !g_game_vcam_out) return;
    uint32_t W = g_disp_w, H = g_disp_h;
    if (!W || !H || !g_d_rgb) return;
    if (g_game_mode == GM_EXTERNAL) {
        // Crosshair at aim + tiny status.
        if (g_game_has_aim) {
            int cx = (int)(g_game_aimx * W), cy = (int)(g_game_aimy * H);
            ai::launch_draw_circle(g_d_rgb, W, H, (float)cx, (float)cy, 10.0f, 0, 255, 0, g_stream);
            ai::launch_draw_circle(g_d_rgb, W, H, (float)cx, (float)cy, 2.0f, 0, 255, 0, g_stream);
            ai::launch_draw_rect(g_d_rgb, W, H, cx - 16, cy, cx + 16, cy, 0, 255, 0, 1, g_stream);
            ai::launch_draw_rect(g_d_rgb, W, H, cx, cy - 16, cx, cy + 16, 0, 255, 0, 1, g_stream);
        }
        char st[96];
        snprintf(st, sizeof(st), "GAME EXT %s", g_game_fg_txt);
        ai::launch_draw_text(g_d_rgb, W, H, 8, 8, st, 2, 255, 255, 255, 0, 120, 0, g_stream);
        return;
    }
    // Trainer HUD
    char hud[128];
    snprintf(hud, sizeof(hud), "SC %d HP %d AMMO %d T %ds W%d",
             g_tr_score, g_tr_hp, g_tr_reloading ? 0 : g_tr_ammo,
             (int)g_tr_time, g_tr_wave);
    ai::launch_draw_text(g_d_rgb, W, H, 8, 8, hud, 2, 255, 255, 255, 0, 0, 0, g_stream);
    if (g_tr_combo > 1) {
        char cb[32];
        snprintf(cb, sizeof(cb), "COMBO X%d", g_tr_combo);
        ai::launch_draw_text(g_d_rgb, W, H, 8, 30, cb, 2, 255, 220, 0, 0, 0, 0, g_stream);
    }
    for (auto& t : g_trg) {
        if (!t.alive) continue;
        int cx = (int)(t.nx * W), cy = (int)(t.ny * H), r = (int)t.r;
        uint8_t rr = t.flash > 0 ? 255 : 230, gg = t.flash > 0 ? 255 : 40, bb = 40;
        ai::launch_draw_circle(g_d_rgb, W, H, (float)cx, (float)cy, (float)r, rr, gg, bb, g_stream);
        ai::launch_draw_circle(g_d_rgb, W, H, (float)cx, (float)cy, (float)r - 4 > 2 ? (float)r - 4 : 2, 255, 255, 255, g_stream);
    }
    if (g_game_has_aim) {
        int cx = (int)(g_game_aimx * W), cy = (int)(g_game_aimy * H);
        ai::launch_draw_circle(g_d_rgb, W, H, (float)cx, (float)cy, 12.0f, 0, 255, 255, g_stream);
        ai::launch_draw_rect(g_d_rgb, W, H, cx - 18, cy, cx + 18, cy, 0, 255, 255, 1, g_stream);
        ai::launch_draw_rect(g_d_rgb, W, H, cx, cy - 18, cx, cy + 18, 0, 255, 255, 1, g_stream);
    }
    if (g_tr_state == TR_IDLE) {
        ai::launch_draw_text(g_d_rgb, W, H, W / 2 - 90, H / 2 - 10,
                             "THUMB UP TO START", 2, 255, 255, 255, 0, 0, 0, g_stream);
    } else if (g_tr_state == TR_PAUSE) {
        ai::launch_draw_text(g_d_rgb, W, H, W / 2 - 120, H / 2 - 10,
                             "PAUSED - THUMB RESUME", 2, 255, 220, 0, 0, 0, 0, g_stream);
    } else if (g_tr_state == TR_OVER) {
        char ov[64];
        snprintf(ov, sizeof(ov), "GAME OVER SC %d BEST X%d", g_tr_score, g_tr_best);
        ai::launch_draw_text(g_d_rgb, W, H, W / 2 - 130, H / 2 - 10, ov, 2, 255, 80, 80, 0, 0, 0, g_stream);
        ai::launch_draw_text(g_d_rgb, W, H, W / 2 - 110, H / 2 + 14,
                             "THUMB TO RETRY", 2, 255, 255, 255, 0, 0, 0, g_stream);
    }
    if (g_tr_reloading)
        ai::launch_draw_text(g_d_rgb, W, H, W / 2 - 50, H - 30, "RELOADING", 2, 255, 220, 0, 0, 0, 0, g_stream);
}

// ---- panel UI ----
static const char* game_bind_name(uint8_t b) {
    if (b == BIND_LCLICK) return "LClick";
    if (b == BIND_RCLICK) return "RClick";
    if (b == VK_SPACE) return "Space";
    if (b == VK_ESCAPE) return "Esc";
    if (b == VK_SHIFT) return "Shift";
    static char s[2] = {};
    s[0] = (char)b; s[1] = 0;
    return s;
}
static RECT game_row_rect(int row) {
    int y = TAB_H + SIDE_TOP + 8 + row * 30;
    RECT r = {12, y, SIDE_W - 12, y + 28};
    return r;
}
static void render_game_panel(HDC hdc, int W, int H) {
    (void)W; (void)H;
    game_load_ini();
    SelectObject(hdc, g_font_ui);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, CLR_TEXT);
    RECT tr = {14, TAB_H + 8 - 22, SIDE_W - 14, TAB_H + 8 - 2};
    DrawTextA(hdc, "Gesture Game", -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    struct Row { const char* label; char val[64]; };
    Row rows[16];
    int n = 0;
    snprintf(rows[n].val, 64, "%s", g_game_mode == GM_OFF ? "OFF" : (g_game_mode == GM_EXTERNAL ? "EXTERNAL" : "TRAINER"));
    rows[n++].label = "MODE";
    snprintf(rows[n].val, 64, "%s", g_game_on ? "STOP" : "START");
    rows[n++].label = g_game_on ? "RUNNING" : "START";
    snprintf(rows[n].val, 64, "%s", g_game_target[0] ? g_game_target : "ANY");
    rows[n++].label = "TARGET";
    snprintf(rows[n].val, 64, "LOCK FG");
    rows[n++].label = "GRAB GAME";
    snprintf(rows[n].val, 64, "%s", g_game_aim_mode == 0 ? "ABS" : "REL");
    rows[n++].label = "AIM";
    snprintf(rows[n].val, 64, "%.2f", g_game_sens_vals[g_game_sens_idx]);
    rows[n++].label = "SENS";
    snprintf(rows[n].val, 64, "%.2f", g_game_dz_vals[g_game_dz_idx]);
    rows[n++].label = "DEADZONE";
    snprintf(rows[n].val, 64, "%.2f", g_game_sm_vals[g_game_sm_idx]);
    rows[n++].label = "SMOOTH";
    snprintf(rows[n].val, 64, "%s", g_game_move_src == 0 ? "HAND" : "FACE");
    rows[n++].label = "MOVE";
    snprintf(rows[n].val, 64, "%s", g_game_calibrated ? "REDO" : "CALIBRATE");
    rows[n++].label = "CALIB";
    snprintf(rows[n].val, 64, "%s", g_game_vcam_out ? "ON" : "OFF");
    rows[n++].label = "VCAM VIEW";
    snprintf(rows[n].val, 64, "%s", game_bind_name(g_bind_fire));
    rows[n++].label = "FIRE";
    snprintf(rows[n].val, 64, "%s", game_bind_name(g_bind_jump));
    rows[n++].label = "JUMP";
    snprintf(rows[n].val, 64, "%s", game_bind_name(g_bind_reload));
    rows[n++].label = "RELOAD";
    snprintf(rows[n].val, 64, "%s", game_bind_name(g_bind_pause));
    rows[n++].label = "PAUSE";
    snprintf(rows[n].val, 64, "%s", game_bind_name(g_bind_interact));
    rows[n++].label = "USE";
    SelectObject(hdc, g_font_st);
    for (int i = 0; i < n; i++) {
        RECT r = game_row_rect(i + 1);
        bool hov = (g_hover == 3000 + i + 1);
        HBRUSH bb = CreateSolidBrush(hov ? RGB(42, 42, 58) : CLR_PANEL2);
        draw_rounded_rect(hdc, r.left, r.top, r.right - r.left, r.bottom - r.top, 5, bb, CLR_LINE);
        DeleteObject(bb);
        SetTextColor(hdc, CLR_TEXT);
        RECT lr = {r.left + 8, r.top, r.right - 70, r.bottom};
        DrawTextA(hdc, rows[i].label, -1, &lr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        bool hot = (i == 1); // START/STOP row highlighted
        draw_pill(hdc, r.right - 64, r.top + 6, 58, 16, rows[i].val,
                  hot ? (g_game_on ? RGB(150, 30, 30) : RGB(20, 110, 60)) : RGB(35, 35, 50),
                  hot ? RGB(255, 255, 255) : CLR_AI, g_font_st);
    }
    // Readout block
    int ry = TAB_H + SIDE_TOP + 8 + (n + 1) * 30 + 6;
    SetTextColor(hdc, CLR_DIM);
    char ro[4][96];
    snprintf(ro[0], 96, "hands: %s", g_game_gesture_txt);
    snprintf(ro[1], 96, "aim: %.2f %.2f %s", g_game_aimx, g_game_aimy, g_game_has_aim ? "" : "(none)");
    snprintf(ro[2], 96, "fg: %s", g_game_fg_txt);
    snprintf(ro[3], 96, "handfps: %.0f f12=kill", g_game_hand_fps);
    for (int i = 0; i < 4; i++) {
        RECT rr = {14, ry + i * 16, SIDE_W - 14, ry + i * 16 + 16};
        DrawTextA(hdc, ro[i], -1, &rr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    }
    RECT hn = {14, ry + 4 * 16 + 4, SIDE_W - 14, ry + 4 * 16 + 20};
    SetTextColor(hdc, RGB(255, 150, 80));
    const char* warn = "";
    if (g_src_mode != 0) warn = "GAME needs CAM feed (auto on START)";
    else if (!g_ai_hands_ready.load()) warn = "hands model missing";
    else if (!g_game_calibrated && g_game_on) warn = "not calibrated: CALIB";
    if (warn[0]) DrawTextA(hdc, warn, -1, &hn, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    // Gesture hint: emoji legend (same pattern as the record panel).
    {
        struct { const wchar_t* e; const char* t; } lg[] = {
            {L"\x261D", "aim"},
            {L"\x270C", "fire hold"},
            {L"\xD83E\xDD0F", "pinch fire"},
            {L"\x2B06", "flick jump"},
            {L"\xD83D\xDC4A", "reload"},
            {L"\xD83D\xDC4D", "use/start"},
        };
        int ly = ry + 4 * 16 + 22;
        int colw = (SIDE_W - 28) / 2;
        for (int i = 0; i < 6; i++) {
            int cx = 14 + (i % 2) * colw, cy = ly + (i / 2) * 20;
            if (cy + 18 >= H - STATUS_H - 8) break;
            SelectObject(hdc, g_font_emoji);
            SetTextColor(hdc, RGB(255, 255, 255));
            RECT er = {cx, cy, cx + 22, cy + 18};
            DrawTextW(hdc, lg[i].e, -1, &er, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
            SelectObject(hdc, g_font_st);
            SetTextColor(hdc, CLR_DIM);
            RECT tr = {cx + 24, cy, cx + colw, cy + 18};
            DrawTextA(hdc, lg[i].t, -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        }
        // Full-width rows: palm pause/kill, sit crouch, left-hand move.
        struct { const wchar_t* e; const char* t; } lg2[] = {
            {L"\x270B", "tap pause - 2s kill"},
            {L"\x2B07", "sit = crouch"},
        };
        int ly2 = ly + 3 * 20 + 2;
        for (int i = 0; i < 2; i++) {
            int cy = ly2 + i * 18;
            if (cy + 18 >= H - STATUS_H - 8) break;
            SelectObject(hdc, g_font_emoji);
            SetTextColor(hdc, RGB(255, 255, 255));
            RECT er = {14, cy, 14 + 22, cy + 18};
            DrawTextW(hdc, lg2[i].e, -1, &er, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
            SelectObject(hdc, g_font_st);
            SetTextColor(hdc, CLR_DIM);
            RECT tr = {14 + 24, cy, SIDE_W - 14, cy + 18};
            DrawTextA(hdc, lg2[i].t, -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        }
        int my = ly2 + 2 * 18 + 2;
        if (my + 16 < H - STATUS_H - 8) {
            SelectObject(hdc, g_font_st);
            SetTextColor(hdc, CLR_DIM);
            RECT mr = {14, my, SIDE_W - 14, my + 16};
            DrawTextA(hdc, "L-hand zones = move", -1, &mr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        }
    }
}
static int game_hittest(int mx, int my, int W, int H) {
    (void)W; (void)H;
    for (int i = 0; i < 16; i++) {
        RECT r = game_row_rect(i + 1);
        if (mx >= r.left && mx < r.right && my >= r.top && my < r.bottom) return i + 1;
    }
    return 0;
}
static void game_cycle_u8(uint8_t& b, const uint8_t* opts, int n) {
    for (int i = 0; i < n; i++)
        if (b == opts[i]) { b = opts[(i + 1) % n]; game_save_ini(); return; }
    b = opts[0]; game_save_ini();
}
static void game_fire(int id) {
    game_load_ini();
    switch (id) {
    case 1: g_game_mode = (g_game_mode + 1) % 3; if (g_game_on) game_set_on(false); game_save_ini(); break;
    case 2: game_set_on(!g_game_on); game_snd(g_snd_click); break;
    case 3: g_game_target[0] = 0; game_save_ini(); break;
    case 4: {
        // Lock the last external foreground window (the game you alt-tabbed
        // from — Kagerou itself is focused at click time, so live fg is us).
        if (g_game_last_fg[0]) {
            strncpy(g_game_target, g_game_last_fg, sizeof(g_game_target) - 1);
            game_save_ini();
            game_snd(g_snd_click);
        }
        break;
    }
    case 5: g_game_aim_mode ^= 1; game_save_ini(); break;
    case 6: g_game_sens_idx = (g_game_sens_idx + 1) % 5; game_save_ini(); break;
    case 7: g_game_dz_idx = (g_game_dz_idx + 1) % 3; game_save_ini(); break;
    case 8: g_game_sm_idx = (g_game_sm_idx + 1) % 3; game_save_ini(); break;
    case 9: g_game_move_src ^= 1; game_save_ini(); break;
    case 10: game_calibrate(); game_snd(g_snd_click); break;
    case 11: g_game_vcam_out = !g_game_vcam_out; game_save_ini(); break;
    case 12: game_cycle_u8(g_bind_fire, kFireOpts, 3); break;
    case 13: game_cycle_u8(g_bind_jump, kJumpOpts, 2); break;
    case 14: game_cycle_u8(g_bind_reload, kReloadOpts, 2); break;
    case 15: game_cycle_u8(g_bind_pause, kPauseOpts, 2); break;
    case 16: game_cycle_u8(g_bind_interact, kInteractOpts, 2); break;
    }
}

// ---- headless injector self-check (--gametest): scripted inputs, no camera ----
static bool g_gametest = false;
static void game_selftest_tick(double el) {
    static bool s_on = false, s_k = false, s_done = false;
    if (s_done) return;
    if (el >= 1.0 && !s_on) {
        s_on = true;
        g_game_mode = GM_EXTERNAL; g_game_target[0] = 0;
        g_game_on = true; // bypass camera: drive injector directly
    }
    if (!s_on) return;
    if (el >= 2.0 && el < 4.0) {
        // Mouse square (absolute): proves SendInput mouse path E2E.
        float t = (float)(el - 2.0) / 2.0f;
        float px = (t < 0.5f) ? t * 2.0f : 2.0f - t * 2.0f;
        float py = (t < 0.5f) ? 0.3f : 0.7f;
        inj_mouse_abs(px, py);
    }
    if (el >= 4.5 && !s_k) {
        s_k = true;
        // Type "HI" via real key events; 400ms holds so a 50ms
        // poll loop (or a human watching a key-state viewer) can see them.
        inj_key((uint8_t)'H', true); inj_key((uint8_t)'I', true);
    }
    if (el >= 4.9 && s_k) {
        inj_key((uint8_t)'H', false); inj_key((uint8_t)'I', false);
    }
    if (el >= 6.0) { s_done = true; game_kill(); }
}
