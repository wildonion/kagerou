// ============================================================================
// Kagerou Virtual Camera — COM DLL
// Registers DirectShow filter, reads frames from shared memory
// ============================================================================

#include "kagerou_vcam_filter.h"
#include "kagerou_vcam_shm.h"
#include <cstdio>
#include <thread>
#include <atomic>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")

static HINSTANCE g_hDll = nullptr;
static KagerouVCamClassFactory* g_pFactory = nullptr;
KagerouVirtualCamFilter* g_pFilter = nullptr;
std::atomic<bool> g_reader_running{false};
std::thread g_reader_thread;
VirtualCamReader g_reader;
static LONG g_cServerLocks = 0;



static const CLSID CLSID_KagerouVCam =
    {0x7A3B3C4D, 0x5E6F, 0x4A7B, {0x8C, 0x9D, 0x0E, 0x1F, 0x2A, 0x3B, 0x4C, 0x5D}};

// Background thread: reads NV12 frames from shared memory, delivers to filter graph.
// If the EXE isn't running (no writer), emits a moving test pattern so apps
// still get a live stream instead of "camera unavailable".
static void reader_thread_func() {
    uint8_t* buf = (uint8_t*)malloc(3840 * 2160 * 3 / 2);
    if (!buf) return;

    ULONGLONG last_ok = 0;
    int pattern_frame = 0;

    while (g_reader_running) {
        if (!g_pFilter || !g_pFilter->IsStreaming()) {
            Sleep(10);
            continue;
        }

        OutputPin* pin = g_pFilter->GetOutputPin();
        if (!pin || !pin->IsConnected()) { Sleep(10); continue; }

        uint32_t w = 0, h = 0;
        uint64_t ts = 0;
        if (g_reader.ReadFrame(buf, w, h, ts)) {
            pin->DeliverFrame(buf, w, h, ts);
            last_ok = GetTickCount64();
        } else {
            // No writer data for >500ms: synthesize 640x480@30 test pattern.
            ULONGLONG now = GetTickCount64();
            if (last_ok == 0) last_ok = now - 1000;
            if (now - last_ok > 500) {
                ULONGLONG now_us = now * 1000;
                // fill pattern directly (same bars as EXE test pattern, NV12)
                fill_nv12_bars_cpu(buf, 640, 480, pattern_frame++);
                pin->DeliverFrame(buf, 640, 480, now_us);
                Sleep(33);
            } else {
                Sleep(1);
            }
        }
    }

    free(buf);
}

void EnsureReaderStarted() {
    if (g_reader_running) return;
    g_reader.Open();
    g_reader_running = true;
    if (!g_reader_thread.joinable())
        g_reader_thread = std::thread(reader_thread_func);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hDll = hModule;
        DisableThreadLibraryCalls(hModule);
    }
    if (reason == DLL_PROCESS_DETACH) {
        // Do NOT join/release here: joining a thread under loader lock can
        // hang regsvr32/host unload. Just signal stop; OS reclaims on exit.
        g_reader_running = false;
    }
    return TRUE;
}

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (!IsEqualCLSID(rclsid, CLSID_KagerouVCam)) return CLASS_E_CLASSNOTAVAILABLE;

    if (!g_pFactory) {
        g_pFactory = new KagerouVCamClassFactory();
    }

    return g_pFactory->QueryInterface(riid, ppv);
}

STDAPI DllCanUnloadNow() {
    return (g_cServerLocks == 0 && !g_pFilter && !g_pFactory) ? S_OK : S_FALSE;
}

static HRESULT RegisterServer() {
    wchar_t szPath[MAX_PATH];
    GetModuleFileNameW(g_hDll, szPath, MAX_PATH);

    wchar_t logPath[MAX_PATH];
    GetTempPathW(MAX_PATH, logPath);
    wcscat_s(logPath, L"kagerou_vcam_register.log");
    FILE* logf = nullptr;
    _wfopen_s(&logf, logPath, L"w");
    auto log = [&](const char* msg) { if (logf) fprintf(logf, "%s\n", msg); fflush(logf); };

    char tmp[512];
    sprintf(tmp, "DLL path: %ls", szPath); log(tmp);

    // Register under HKLM (system-wide) for Chrome/sandbox compatibility
    HKEY hRoot = HKEY_LOCAL_MACHINE;
    const wchar_t* subkey = L"SOFTWARE\\Classes\\CLSID\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}";

    HKEY hKey = nullptr;
    LONG rc = RegCreateKeyExW(hRoot, subkey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                              KEY_ALL_ACCESS, nullptr, &hKey, nullptr);
    sprintf(tmp, "RegCreateKey CLSID (HKLM): rc=%ld", rc); log(tmp);
    if (rc != ERROR_SUCCESS) {
        log("HKLM failed, trying HKCU...");
        hRoot = HKEY_CURRENT_USER;
        subkey = L"Software\\Classes\\CLSID\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}";
        rc = RegCreateKeyExW(hRoot, subkey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                             KEY_ALL_ACCESS, nullptr, &hKey, nullptr);
        sprintf(tmp, "RegCreateKey CLSID (HKCU): rc=%ld", rc); log(tmp);
        if (rc != ERROR_SUCCESS) { if (logf) fclose(logf); return E_FAIL; }
    }

    RegSetValueExW(hKey, nullptr, 0, REG_SZ, (BYTE*)L"Kagerou Virtual Camera",
        (DWORD)(wcslen(L"Kagerou Virtual Camera") + 1) * sizeof(wchar_t));

    HKEY hFriendly;
    if (RegCreateKeyExW(hKey, L"FriendlyName", 0, nullptr, REG_OPTION_NON_VOLATILE,
                        KEY_ALL_ACCESS, nullptr, &hFriendly, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(hFriendly, nullptr, 0, REG_SZ,
            (BYTE*)L"Kagerou Virtual Camera",
            (DWORD)(wcslen(L"Kagerou Virtual Camera") + 1) * sizeof(wchar_t));
        RegCloseKey(hFriendly);
    }

    HKEY hInproc;
    if (RegCreateKeyExW(hKey, L"InprocServer32", 0, nullptr, REG_OPTION_NON_VOLATILE,
                        KEY_ALL_ACCESS, nullptr, &hInproc, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(hInproc, nullptr, 0, REG_SZ, (BYTE*)szPath,
            (DWORD)(wcslen(szPath) + 1) * sizeof(wchar_t));
        RegSetValueExW(hInproc, L"ThreadingModel", 0, REG_SZ,
            (BYTE*)L"Both", (DWORD)(wcslen(L"Both") + 1) * sizeof(wchar_t));
        RegCloseKey(hInproc);
        log("InprocServer32 OK");
    }

    RegCloseKey(hKey);

    // Register under Video Input Device Category
    wchar_t catSubkey[256];
    if (hRoot == HKEY_LOCAL_MACHINE)
        wcscpy(catSubkey, L"SOFTWARE\\Classes\\CLSID\\{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\Instance\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");
    else
        wcscpy(catSubkey, L"Software\\Classes\\CLSID\\{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\Instance\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");

    rc = RegCreateKeyExW(hRoot, catSubkey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                         KEY_ALL_ACCESS, nullptr, &hKey, nullptr);
    sprintf(tmp, "RegCreateKey Category: rc=%ld", rc); log(tmp);
    if (rc == ERROR_SUCCESS) {
        RegSetValueExW(hKey, L"FriendlyName", 0, REG_SZ,
            (BYTE*)L"Kagerou Virtual Camera",
            (DWORD)(wcslen(L"Kagerou Virtual Camera") + 1) * sizeof(wchar_t));
        RegSetValueExW(hKey, L"CLSID", 0, REG_SZ,
            (BYTE*)L"{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}",
            (DWORD)(wcslen(L"{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}") + 1) * sizeof(wchar_t));
        // DevicePath: picky consumers (OBS device list, Chrome matching)
        // skip video devices without one. Synthetic but well-formed.
        const wchar_t* devPath =
            L"\\\\?\\root#kagerou#0000#{7a3b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d}";
        RegSetValueExW(hKey, L"DevicePath", 0, REG_SZ,
            (BYTE*)devPath,
            (DWORD)(wcslen(devPath) + 1) * sizeof(wchar_t));
        // Description: some enumerators prefer it over FriendlyName.
        RegSetValueExW(hKey, L"Description", 0, REG_SZ,
            (BYTE*)L"Kagerou Virtual Camera",
            (DWORD)(wcslen(L"Kagerou Virtual Camera") + 1) * sizeof(wchar_t));
        RegCloseKey(hKey);
        log("Category OK");
    }

    log("RegisterServer DONE");
    if (logf) fclose(logf);
    return S_OK;
}

static HRESULT UnregisterServer() {
    RegDeleteTreeW(HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Classes\\CLSID\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");
    RegDeleteTreeW(HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Classes\\CLSID\\{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
        L"\\Instance\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");
    RegDeleteTreeW(HKEY_CURRENT_USER,
        L"Software\\Classes\\CLSID\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");
    RegDeleteTreeW(HKEY_CURRENT_USER,
        L"Software\\Classes\\CLSID\\{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
        L"\\Instance\\{7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}");
    return S_OK;
}

STDAPI DllRegisterServer() { return RegisterServer(); }
STDAPI DllUnregisterServer() { return UnregisterServer(); }
