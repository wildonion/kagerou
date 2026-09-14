// DXGI desktop-duplication probe: isolates the capture path from the app.
// Reports open stages, per-frame results, and first-frame bytes.
// Build: build.bat dxgi_test
#include <cstdio>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

int main() {
    printf("--- DXGI probe ---\n");
    IDXGIFactory1* fac = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac);
    printf("factory: 0x%08X\n", hr);
    if (FAILED(hr) || !fac) return 1;
    static const D3D_FEATURE_LEVEL fls[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    for (UINT ai = 0; ai < 4; ai++) {
        IDXGIAdapter1* ad = nullptr;
        if (fac->EnumAdapters1(ai, &ad) != S_OK || !ad) break;
        DXGI_ADAPTER_DESC1 dd = {};
        ad->GetDesc1(&dd);
        printf("adapter%u: %S ven=%04X\n", ai, dd.Description, dd.VendorId);
        for (UINT oi = 0; oi < 4; oi++) {
            IDXGIOutput* out = nullptr;
            if (ad->EnumOutputs(oi, &out) != S_OK || !out) break;
            DXGI_OUTPUT_DESC od = {};
            out->GetDesc(&od);
            printf("  output%u: %dx%d @(%d,%d) attached=%d\n", oi,
                   od.DesktopCoordinates.right - od.DesktopCoordinates.left,
                   od.DesktopCoordinates.bottom - od.DesktopCoordinates.top,
                   od.DesktopCoordinates.left, od.DesktopCoordinates.top,
                   (int)od.AttachedToDesktop);
            ID3D11Device* dev = nullptr;
            ID3D11DeviceContext* ctx = nullptr;
            D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_1_0_CORE;
            hr = D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                   fls, 4, D3D11_SDK_VERSION, &dev, &got, &ctx);
            printf("  device: 0x%08X fl=0x%X\n", hr, got);
            if (SUCCEEDED(hr) && dev) {
                IDXGIOutput1* out1 = nullptr;
                hr = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
                printf("  out1 QI: 0x%08X\n", hr);
                if (SUCCEEDED(hr) && out1) {
                    IDXGIOutputDuplication* dup = nullptr;
                    hr = out1->DuplicateOutput(dev, &dup);
                    printf("  duplicate: 0x%08X\n", hr);
                    if (SUCCEEDED(hr) && dup) {
                        int timeouts = 0, presents = 0, empty = 0;
                        for (int i = 0; i < 20; i++) {
                            IDXGIResource* res = nullptr;
                            DXGI_OUTDUPL_FRAME_INFO fi = {};
                            hr = dup->AcquireNextFrame(50, &fi, &res);
                            if (hr == DXGI_ERROR_WAIT_TIMEOUT) { timeouts++; continue; }
                            if (FAILED(hr)) { printf("  acquire #%d: 0x%08X\n", i, hr); break; }
                            if (fi.LastPresentTime.QuadPart == 0) empty++;
                            else presents++;
                            res->Release();
                            dup->ReleaseFrame();
                        }
                        printf("  20 polls: timeouts=%d empty=%d presents=%d\n",
                               timeouts, presents, empty);
                        dup->Release();
                    }
                    out1->Release();
                }
                ctx->Release();
                dev->Release();
            }
            out->Release();
        }
        ad->Release();
    }
    fac->Release();
    printf("--- done ---\n");
    return 0;
}
