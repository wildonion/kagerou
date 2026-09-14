// Kagerou screen capture helper — DXGI desktop duplication WITHOUT any
// CUDA/NVENC/ORT linkage. Must stay that way: in the main app process those
// libraries shift DXGI output ownership to a phantom NVIDIA output whose
// duplication fails with DXGI_ERROR_UNSUPPORTED. A bare helper process
// enumerates Intel-first and duplicates fine (see tests/test_dxgi.cpp).
//
// Protocol: named file mapping "Local\\KagerouScreen" (header + max-frame
// payload, seqlock) + stop event "Local\\KagerouScreenStop".
// Parent PID passed as argv[1]; helper exits if the parent dies.
// Build: build.bat screen_cap  (MSVC cl, d3d11.lib dxgi.lib)

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <cstdio>
#include <cstdint>

#include "screen_shm.h"

static FILE* g_log = nullptr;
#define HLOG(...) do { if (g_log) { fprintf(g_log, __VA_ARGS__); fflush(g_log); } } while (0)

int main(int argc, char** argv) {
    g_log = fopen("C:\\Users\\teknotek2025\\Desktop\\wildonion\\hanzo\\src\\HardLab\\Kagerou\\bin\\scrhelp.log", "a");
    HLOG("helper start pid=%lu parent=%s\n", GetCurrentProcessId(), argc > 1 ? argv[1] : "?");
    DWORD parent = (argc > 1) ? (DWORD)atoi(argv[1]) : 0;
    char mapname[128] = "", evname[128] = "";
    kscr_names(argc > 1 ? argv[1] : "0", mapname, sizeof(mapname), evname, sizeof(evname));
    HANDLE hstop = CreateEventA(NULL, TRUE, FALSE, evname);
    HANDLE hmap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                     0, (DWORD)(sizeof(ScreenShm) + KSCR_MAX_BYTES),
                                     mapname);
    if (!hmap) return 2;
    ScreenShm* shm = (ScreenShm*)MapViewOfFile(hmap, FILE_MAP_WRITE, 0, 0,
                                               sizeof(ScreenShm) + KSCR_MAX_BYTES);
    if (!shm) { CloseHandle(hmap); if (hstop) CloseHandle(hstop); return 3; }
    // Always (re)stamp: a stale mapping from a crashed run would otherwise
    // fail the reader's magic check forever. Single-writer in practice
    // (old instance is stopped before a new one starts).
    memset(shm, 0, sizeof(ScreenShm));
    shm->magic = KSCR_MAGIC;
    shm->version = KSCR_VERSION;
    shm->state = 0;
    uint8_t* payload = (uint8_t*)(shm + 1);

    HANDLE hparent = parent ? OpenProcess(SYNCHRONIZE, FALSE, parent) : NULL;

    IDXGIFactory1* fac = nullptr;
    IDXGIAdapter1* ad = nullptr;
    IDXGIOutput* out = nullptr;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    IDXGIOutputDuplication* dup = nullptr;
    ID3D11Texture2D* staging = nullptr;
    uint32_t sw = 0, sh = 0;
    bool ok = false;

    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac)) && fac) {
        for (UINT ai = 0; !ok; ai++) {
            if (fac->EnumAdapters1(ai, &ad) != S_OK || !ad) break;
            for (UINT oi = 0; !ok; oi++) {
                if (ad->EnumOutputs(oi, &out) != S_OK || !out) break;
                DXGI_OUTPUT_DESC od = {};
                out->GetDesc(&od);
                bool primary = (od.DesktopCoordinates.left == 0 &&
                                od.DesktopCoordinates.top == 0);
                uint32_t w = (uint32_t)(od.DesktopCoordinates.right - od.DesktopCoordinates.left);
                uint32_t h = (uint32_t)(od.DesktopCoordinates.bottom - od.DesktopCoordinates.top);
                static const D3D_FEATURE_LEVEL fls[] = {
                    D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
                    D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
                D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_1_0_CORE;
                HLOG("ad out %ux%u @(%d,%d) primary=%d\n",
                     w, h, od.DesktopCoordinates.left, od.DesktopCoordinates.top, primary ? 1 : 0);
                if (primary && w >= 160 && h >= 90 && w <= KSCR_MAX_W && h <= KSCR_MAX_H &&
                    SUCCEEDED(D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                                fls, 4, D3D11_SDK_VERSION, &dev, &got, &ctx)) && dev) {
                    IDXGIOutput1* out1 = nullptr;
                    HRESULT qhr = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
                    HLOG("  qhr=0x%lX\n", qhr);
                    if (SUCCEEDED(qhr) && out1) {
                        HRESULT uhr = out1->DuplicateOutput(dev, &dup);
                        HLOG("  uhr=0x%lX\n", uhr);
                        if (SUCCEEDED(uhr) && dup) {
                            sw = w; sh = h; ok = true;
                        }
                        out1->Release();
                    }
                }
                if (!ok) {
                    if (ctx) { ctx->Release(); ctx = nullptr; }
                    if (dev) { dev->Release(); dev = nullptr; }
                }
                out->Release();
                out = nullptr;
            }
            ad->Release();
            ad = nullptr;
        }
        fac->Release();
        fac = nullptr;
    }

    if (!ok) {
        HLOG("open FAILED\n");
        if (g_log) fclose(g_log);
        shm->state = 2; // failed: main app falls back to GDI
        if (hparent) CloseHandle(hparent);
        UnmapViewOfFile(shm);
        CloseHandle(hmap);
        if (hstop) CloseHandle(hstop);
        return 4;
    }

    // Staging sized to the probed desktop.
    {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sw; td.Height = sh;
        td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        HRESULT thr = dev->CreateTexture2D(&td, nullptr, &staging);
        HLOG("staging %ux%u hr=0x%lX\n", sw, sh, thr);
        if (FAILED(thr) || !staging)
            ok = false;
    }
    if (!ok) {
        shm->state = 2;
    } else {
        HLOG("entering loop %ux%u\n", sw, sh);
        int nframes = 0;
        // Main loop: newest frame -> SHM (seqlock), 30fps republish cadence.
        while (true) {
            if (hstop && WaitForSingleObject(hstop, 0) == WAIT_OBJECT_0) {
                HLOG("loop: stop event\n"); break;
            }
            if (hparent && WaitForSingleObject(hparent, 0) == WAIT_OBJECT_0) {
                HLOG("loop: parent gone\n"); break; // parent gone
            }
            bool have = false;
            IDXGIResource* res = nullptr;
            DXGI_OUTDUPL_FRAME_INFO fi = {};
            // Short timeout: static screens republish fast (takes stay
            // exactly 30fps CFR), motion is picked up within ~8ms.
            HRESULT hr = dup->AcquireNextFrame(8, &fi, &res);
            if (SUCCEEDED(hr) && res) {
                if (fi.LastPresentTime.QuadPart != 0) {
                    ID3D11Texture2D* src = nullptr;
                    if (SUCCEEDED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) && src) {
                        D3D11_TEXTURE2D_DESC sd = {};
                        src->GetDesc(&sd);
                        if (sd.Width == sw && sd.Height == sh &&
                            sd.Format == DXGI_FORMAT_B8G8R8A8_UNORM) {
                            ctx->CopyResource(staging, src);
                            D3D11_MAPPED_SUBRESOURCE mp = {};
                            if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mp)) && mp.pData) {
                                const uint8_t* s = (const uint8_t*)mp.pData;
                                // Downscale to the FEED box here (helper thread
                                // budget): the reader just memcpys take-size RGB.
                                double sc = (double)FEED_CAP_W / sw < (double)FEED_CAP_H / sh
                                            ? (double)FEED_CAP_W / sw : (double)FEED_CAP_H / sh;
                                if (sc > 1.0) sc = 1.0;
                                uint32_t tw = ((uint32_t)(sw * sc)) & ~1u;
                                uint32_t th = ((uint32_t)(sh * sc)) & ~1u;
                                if (tw < 160) tw = 160;
                                if (th < 90) th = 90;
                                float fx = (float)sw / tw, fy = (float)sh / th;
                                InterlockedIncrement(&shm->seq); // odd: writing
                                for (uint32_t y = 0; y < th; y++) {
                                    float syf = (y + 0.5f) * fy - 0.5f;
                                    int sy0 = (int)syf;
                                    float wy = syf - sy0;
                                    if (sy0 < 0) { sy0 = 0; wy = 0; }
                                    if ((uint32_t)sy0 >= sh - 1) { sy0 = (int)sh - 2; wy = 1; }
                                    const uint8_t* r0 = s + (size_t)sy0 * mp.RowPitch;
                                    const uint8_t* r1 = s + (size_t)(sy0 + 1) * mp.RowPitch;
                                    uint8_t* d = payload + (size_t)y * tw * 3;
                                    for (uint32_t x = 0; x < tw; x++) {
                                        float sxf = (x + 0.5f) * fx - 0.5f;
                                        int sx0 = (int)sxf;
                                        float wx = sxf - sx0;
                                        if (sx0 < 0) { sx0 = 0; wx = 0; }
                                        if ((uint32_t)sx0 >= sw - 1) { sx0 = (int)sw - 2; wx = 1; }
                                        for (int c = 0; c < 3; c++) {
                                            // BGRA source: channel 2,1,0 -> RGB
                                            int cc = 2 - c;
                                            float a = r0[sx0*4+cc], b2 = r0[(sx0+1)*4+cc];
                                            float c2 = r1[sx0*4+cc], d2 = r1[(sx0+1)*4+cc];
                                            d[x*3+c] = (uint8_t)(a*(1-wx)*(1-wy) + b2*wx*(1-wy) +
                                                                 c2*(1-wx)*wy + d2*wx*wy + 0.5f);
                                        }
                                    }
                                }
                                shm->w = tw; shm->h = th;
                                shm->state = 1;
                                InterlockedIncrement(&shm->seq); // even: done
                                have = true;
                                if (nframes == 0)
                                    HLOG("first publish %ux%u\n", tw, th);
                                nframes++;
                            }
                            ctx->Unmap(staging, 0);
                        }
                        src->Release();
                    }
                } else have = true; // static screen: republish already-held frame
                res->Release();
                dup->ReleaseFrame();
                if (!have && shm->state != 1) {
                    // acquired but unusable and nothing published yet: keep waiting
                }
            } else if (hr != DXGI_ERROR_WAIT_TIMEOUT) {
                HLOG("loop: acquire break hr=0x%lX\n", hr);
                break; // access lost etc: exit, main app restarts us
            } else if (shm->state == 1) {
                // timeout on static screen: re-stamp seq so takes stay realtime
                InterlockedIncrement(&shm->seq);
                InterlockedIncrement(&shm->seq);
            }
        }
    }

    if (dup) dup->Release();
    if (staging) staging->Release();
    if (ctx) ctx->Release();
    if (dev) dev->Release();
    shm->state = 3;
    if (hparent) CloseHandle(hparent);
    UnmapViewOfFile(shm);
    CloseHandle(hmap);
    if (hstop) CloseHandle(hstop);
    return 0;
}
