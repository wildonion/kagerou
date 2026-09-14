#include <windows.h>
#include <dshow.h>
#include <stdio.h>

#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ole32.lib")

int main() {
    CoInitialize(nullptr);
    ICreateDevEnum* pDevEnum = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&pDevEnum);
    if (FAILED(hr)) { printf("CreateDevEnum FAILED 0x%08lX\n", hr); return 1; }
    IEnumMoniker* pEnum = nullptr;
    hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnum, 0);
    if (hr != S_OK || !pEnum) { printf("No video devices enumerator (hr=0x%08lX)\n", hr); return 1; }
    printf("Video input devices:\n");
    IMoniker* pMon = nullptr;
    int n = 0;
    while (pEnum->Next(1, &pMon, nullptr) == S_OK) {
        IPropertyBag* pBag = nullptr;
        char name[256] = "?";
        char clsid[128] = "?";
        char devpath[256] = "(none)";
        if (SUCCEEDED(pMon->BindToStorage(nullptr, nullptr, IID_IPropertyBag, (void**)&pBag))) {
            VARIANT v; VariantInit(&v);
            if (SUCCEEDED(pBag->Read(L"FriendlyName", &v, nullptr)) && v.vt == VT_BSTR) {
                WideCharToMultiByte(CP_ACP, 0, v.bstrVal, -1, name, sizeof(name), nullptr, nullptr);
            }
            VariantClear(&v);
            if (SUCCEEDED(pBag->Read(L"CLSID", &v, nullptr)) && v.vt == VT_BSTR) {
                WideCharToMultiByte(CP_ACP, 0, v.bstrVal, -1, clsid, sizeof(clsid), nullptr, nullptr);
            }
            VariantClear(&v);
            if (SUCCEEDED(pBag->Read(L"DevicePath", &v, nullptr)) && v.vt == VT_BSTR) {
                WideCharToMultiByte(CP_ACP, 0, v.bstrVal, -1, devpath, sizeof(devpath), nullptr, nullptr);
            }
            VariantClear(&v);
            pBag->Release();
        }
        printf("  [%d] %s  (CLSID=%s)\n       DevicePath=%s\n", n, name, clsid, devpath);
        // Try to instantiate it
        IBaseFilter* pF = nullptr;
        HRESULT hr2 = pMon->BindToObject(nullptr, nullptr, IID_IBaseFilter, (void**)&pF);
        printf("       BindToObject: 0x%08lX\n", hr2);
        if (SUCCEEDED(hr2) && pF) {
            // OBS check: output pin must report PIN_CATEGORY_CAPTURE
            IPin* pPin = nullptr;
            if (SUCCEEDED(pF->FindPin(L"Output", &pPin)) && pPin) {
                IKsPropertySet* pKs = nullptr;
                if (SUCCEEDED(pPin->QueryInterface(IID_IKsPropertySet, (void**)&pKs)) && pKs) {
                    GUID cat = GUID_NULL;
                    DWORD ret = 0;
                    HRESULT hr3 = pKs->Get(AMPROPSETID_Pin, AMPROPERTY_PIN_CATEGORY,
                                           nullptr, 0, &cat, sizeof(cat), &ret);
                    printf("       PinCategory: 0x%08lX %s\n", hr3,
                           (SUCCEEDED(hr3) && cat == PIN_CATEGORY_CAPTURE) ? "CAPTURE-OK" : "MISMATCH!");
                    pKs->Release();
                } else printf("       PinCategory: no IKsPropertySet\n");
                pPin->Release();
            }
            pF->Release();
        }
        // Remember our moniker for the streaming test below
        if (strstr(clsid, "7A3B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D") != nullptr) {
            pMon->AddRef();
            // store via global below (leak one ref on purpose, test tool)
            extern IMoniker* g_kagerouMoniker;
            g_kagerouMoniker = pMon;
        }
        pMon->Release();
        n++;
    }
    printf("Total: %d\n", n);
    pEnum->Release();
    pDevEnum->Release();

    // ---- Live streaming test: build a real graph with our source ----
    extern IMoniker* g_kagerouMoniker;
    if (!g_kagerouMoniker) {
        printf("STREAM TEST: SKIPPED (Kagerou device not enumerated)\n");
        CoUninitialize();
        return 1;
    }
    printf("STREAM TEST: building graph...\n");
    IGraphBuilder* pGB = nullptr;
    hr = CoCreateInstance(CLSID_FilterGraph, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IGraphBuilder, (void**)&pGB);
    if (FAILED(hr)) { printf("STREAM TEST: FilterGraph FAILED 0x%08lX\n", hr); CoUninitialize(); return 1; }
    IBaseFilter* pSrc = nullptr;
    hr = g_kagerouMoniker->BindToObject(nullptr, nullptr, IID_IBaseFilter, (void**)&pSrc);
    if (FAILED(hr)) { printf("STREAM TEST: Bind FAILED 0x%08lX\n", hr); pGB->Release(); CoUninitialize(); return 1; }
    hr = pGB->AddFilter(pSrc, L"Kagerou Virtual Camera");
    if (FAILED(hr)) { printf("STREAM TEST: AddFilter FAILED 0x%08lX\n", hr); pSrc->Release(); pGB->Release(); CoUninitialize(); return 1; }
    // Check advertised stream caps
    IPin* pPin = nullptr;
    if (SUCCEEDED(pSrc->FindPin(L"Output", &pPin)) && pPin) {
        IAMStreamConfig* pCfg = nullptr;
        if (SUCCEEDED(pPin->QueryInterface(IID_IAMStreamConfig, (void**)&pCfg)) && pCfg) {
            int count = 0, size = 0;
            pCfg->GetNumberOfCapabilities(&count, &size);
            printf("STREAM TEST: caps=%d\n", count);
            pCfg->Release();
        } else {
            printf("STREAM TEST: no IAMStreamConfig\n");
        }
        hr = pGB->Render(pPin);
        printf("STREAM TEST: Render: 0x%08lX\n", hr);
        pPin->Release();
    }
    int rc = 0;
    if (SUCCEEDED(hr)) {
        IMediaControl* pMC = nullptr;
        pGB->QueryInterface(IID_IMediaControl, (void**)&pMC);
        hr = pMC->Run();
        printf("STREAM TEST: Run: 0x%08lX\n", hr);
        if (SUCCEEDED(hr)) {
            Sleep(3000);
            OAFilterState st = State_Stopped;
            pMC->GetState(1000, &st);
            printf("STREAM TEST: state=%d (2=running), frames flowed if no error\n", (int)st);
            if (st != State_Running) { printf("STREAM TEST: NOT RUNNING\n"); rc = 1; }
        } else rc = 1;
        pMC->Stop();
        pMC->Release();
    } else rc = 1;
    pSrc->Release();
    pGB->Release();
    g_kagerouMoniker->Release();
    CoUninitialize();
    printf(rc == 0 ? "STREAM TEST: PASS\n" : "STREAM TEST: FAIL\n");
    return rc;
}

IMoniker* g_kagerouMoniker = nullptr;
