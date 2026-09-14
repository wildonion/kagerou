// ============================================================================
// Kagerou Virtual Camera — Filter Implementation
// ============================================================================

#include "kagerou_vcam_filter.h"
#include "kagerou_vcam_shm.h"
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>

// Shared singleton owned by the DLL module (defined in kagerou_vcam_dll.cpp).
// The reader thread pushes frames into this exact instance, so CreateInstance
// must return it instead of a fresh object — otherwise apps get no frames.
extern KagerouVirtualCamFilter* g_pFilter;
extern std::atomic<bool> g_reader_running;
void EnsureReaderStarted();

#ifndef E_PROPSET_UNSUPPORTED
#define E_PROPSET_UNSUPPORTED ((HRESULT)0x80070492L)
#endif

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")

// ============================================================================
// SimpleMediaSample
// ============================================================================

SimpleMediaSample::SimpleMediaSample(BYTE* data, LONG size)
    : m_ref(1), m_pData(data), m_dataSize(size), m_actualLen(size),
      m_rtStart(0), m_rtStop(0) {}

SimpleMediaSample::~SimpleMediaSample() {
    if (m_pData) CoTaskMemFree(m_pData);
}

HRESULT STDMETHODCALLTYPE SimpleMediaSample::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IMediaSample) {
        *ppv = static_cast<IMediaSample*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE SimpleMediaSample::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE SimpleMediaSample::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::GetPointer(BYTE** pp) { if (!pp) return E_POINTER; *pp = m_pData; return S_OK; }
LONG STDMETHODCALLTYPE SimpleMediaSample::GetSize() { return m_dataSize; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::GetTime(REFERENCE_TIME* s, REFERENCE_TIME* e) { if (s) *s = m_rtStart; if (e) *e = m_rtStop; return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetTime(REFERENCE_TIME* s, REFERENCE_TIME* e) { if (s) m_rtStart = *s; if (e) m_rtStop = *e; return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::IsSyncPoint() { return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetSyncPoint(BOOL) { return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::IsPreroll() { return S_FALSE; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetPreroll(BOOL) { return S_OK; }
LONG STDMETHODCALLTYPE SimpleMediaSample::GetActualDataLength() { return m_actualLen; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetActualDataLength(long l) { m_actualLen = l; return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::GetMediaType(AM_MEDIA_TYPE**) { return E_FAIL; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetMediaType(AM_MEDIA_TYPE*) { return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::IsDiscontinuity() { return S_FALSE; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetDiscontinuity(BOOL) { return S_OK; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::GetMediaTime(LONGLONG*, LONGLONG*) { return E_NOTIMPL; }
HRESULT STDMETHODCALLTYPE SimpleMediaSample::SetMediaTime(LONGLONG*, LONGLONG*) { return S_OK; }

// ============================================================================
// CEnumMediaTypes
// ============================================================================

CEnumMediaTypes::CEnumMediaTypes(const AM_MEDIA_TYPE& mt)
    : m_ref(1), m_done(false) {
    m_mt = mt;
    if (mt.cbFormat && mt.pbFormat) {
        VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)CoTaskMemAlloc(sizeof(VIDEOINFOHEADER));
        memcpy(pvi, mt.pbFormat, sizeof(VIDEOINFOHEADER));
        m_mt.pbFormat = (BYTE*)pvi;
        m_mt.cbFormat = sizeof(VIDEOINFOHEADER);
    }
}

CEnumMediaTypes::~CEnumMediaTypes() {
    if (m_mt.cbFormat && m_mt.pbFormat) CoTaskMemFree(m_mt.pbFormat);
}

HRESULT STDMETHODCALLTYPE CEnumMediaTypes::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IEnumMediaTypes) {
        *ppv = static_cast<IEnumMediaTypes*>(this); AddRef(); return S_OK;
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE CEnumMediaTypes::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE CEnumMediaTypes::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }

HRESULT STDMETHODCALLTYPE CEnumMediaTypes::Next(ULONG cMediaTypes, AM_MEDIA_TYPE** ppMediaTypes, ULONG* pcFetched) {
    if (!ppMediaTypes) return E_POINTER;
    if (pcFetched) *pcFetched = 0;
    if (m_done || cMediaTypes == 0) return S_FALSE;

    AM_MEDIA_TYPE* pmt = (AM_MEDIA_TYPE*)CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE));
    memcpy(pmt, &m_mt, sizeof(AM_MEDIA_TYPE));
    if (m_mt.cbFormat && m_mt.pbFormat) {
        pmt->pbFormat = (BYTE*)CoTaskMemAlloc(m_mt.cbFormat);
        memcpy(pmt->pbFormat, m_mt.pbFormat, m_mt.cbFormat);
        pmt->cbFormat = m_mt.cbFormat;
    }
    ppMediaTypes[0] = pmt;
    if (pcFetched) *pcFetched = 1;
    m_done = true;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE CEnumMediaTypes::Skip(ULONG) { return S_FALSE; }
HRESULT STDMETHODCALLTYPE CEnumMediaTypes::Reset() { m_done = false; return S_OK; }
HRESULT STDMETHODCALLTYPE CEnumMediaTypes::Clone(IEnumMediaTypes** pp) {
    if (!pp) return E_POINTER;
    *pp = new CEnumMediaTypes(m_mt);
    return S_OK;
}

// ============================================================================
// CEnumPins
// ============================================================================

CEnumPins::CEnumPins(IBaseFilter* pFilter)
    : m_ref(1), m_pFilter(pFilter), m_done(false) {
    if (m_pFilter) m_pFilter->AddRef();
}

CEnumPins::~CEnumPins() {
    if (m_pFilter) m_pFilter->Release();
}

HRESULT STDMETHODCALLTYPE CEnumPins::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IEnumPins) {
        *ppv = static_cast<IEnumPins*>(this); AddRef(); return S_OK;
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE CEnumPins::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE CEnumPins::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }

HRESULT STDMETHODCALLTYPE CEnumPins::Next(ULONG cPins, IPin** ppPins, ULONG* pcFetched) {
    if (!ppPins) return E_POINTER;
    if (pcFetched) *pcFetched = 0;
    if (m_done || cPins == 0) return S_FALSE;

    FILTER_INFO fi;
    m_pFilter->QueryFilterInfo(&fi);
    IPin* pPin = nullptr;
    m_pFilter->FindPin(L"Output", &pPin);
    if (fi.pGraph) fi.pGraph->Release();

    if (pPin) {
        ppPins[0] = pPin;
        if (pcFetched) *pcFetched = 1;
        m_done = true;
        return S_OK;
    }
    return S_FALSE;
}

HRESULT STDMETHODCALLTYPE CEnumPins::Skip(ULONG) { return S_FALSE; }
HRESULT STDMETHODCALLTYPE CEnumPins::Reset() { m_done = false; return S_OK; }
HRESULT STDMETHODCALLTYPE CEnumPins::Clone(IEnumPins** pp) {
    if (!pp) return E_POINTER;
    *pp = new CEnumPins(m_pFilter);
    return S_OK;
}

// ============================================================================
// OutputPin
// ============================================================================

OutputPin::OutputPin(IBaseFilter* pFilter)
    : m_ref(1), m_pFilter(pFilter), m_pConnectedTo(nullptr), m_connected(false),
      m_pPeerAllocator(nullptr), m_committedSize(0) {
    memset(&m_mt, 0, sizeof(m_mt));
    memset(&m_allocProps, 0, sizeof(m_allocProps));
    m_mt.majortype = MEDIATYPE_Video;
    m_mt.subtype = MEDIASUBTYPE_NV12;
    m_mt.formattype = FORMAT_VideoInfo;
    m_mt.bFixedSizeSamples = TRUE;
    m_mt.bTemporalCompression = FALSE;
    m_mt.lSampleSize = 640 * 480 * 3 / 2;
    m_mt.cbFormat = sizeof(VIDEOINFOHEADER);
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)CoTaskMemAlloc(sizeof(VIDEOINFOHEADER));
    memset(pvi, 0, sizeof(VIDEOINFOHEADER));
    pvi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    pvi->bmiHeader.biWidth = 640;
    pvi->bmiHeader.biHeight = 480;
    pvi->bmiHeader.biPlanes = 1;
    pvi->bmiHeader.biBitCount = 12;
    pvi->bmiHeader.biCompression = mmioFOURCC('N','V','1','2');
    pvi->bmiHeader.biSizeImage = 640 * 480 * 3 / 2;
    pvi->AvgTimePerFrame = (REFERENCE_TIME)(10000000.0 / 30.0);
    m_mt.pbFormat = (BYTE*)pvi;
    m_allocProps.cbBuffer = 640 * 480 * 3 / 2;
    m_allocProps.cBuffers = 1;
    m_allocProps.cbAlign = 1;
}

OutputPin::~OutputPin() {
    if (m_mt.pbFormat) CoTaskMemFree(m_mt.pbFormat);
    if (m_pConnectedTo) m_pConnectedTo->Release();
    if (m_pPeerAllocator) m_pPeerAllocator->Release();
}

HRESULT STDMETHODCALLTYPE OutputPin::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IPin) {
        *ppv = static_cast<IPin*>(this); AddRef(); return S_OK;
    }
    if (riid == IID_IMemInputPin) {
        *ppv = static_cast<IMemInputPin*>(this); AddRef(); return S_OK;
    }
    if (riid == IID_IMemAllocator) {
        *ppv = static_cast<IMemAllocator*>(this); AddRef(); return S_OK;
    }
    if (riid == IID_IKsPropertySet) {
        *ppv = static_cast<IKsPropertySet*>(this); AddRef(); return S_OK;
    }
    if (riid == IID_IAMStreamConfig) {
        *ppv = static_cast<IAMStreamConfig*>(this); AddRef(); return S_OK;
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE OutputPin::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE OutputPin::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }

// IPin
void OutputPin::AdoptMediaType(const AM_MEDIA_TYPE* pmt) {
    if (!pmt || pmt->majortype != MEDIATYPE_Video)
        return;
    if (pmt->subtype != MEDIASUBTYPE_NV12 && pmt->subtype != MEDIASUBTYPE_YUY2)
        return;
    if (m_mt.pbFormat) { CoTaskMemFree(m_mt.pbFormat); m_mt.pbFormat = nullptr; m_mt.cbFormat = 0; }
    m_mt = *pmt;
    m_mt.pbFormat = nullptr; m_mt.cbFormat = 0;
    if (pmt->cbFormat && pmt->pbFormat) {
        m_mt.pbFormat = (BYTE*)CoTaskMemAlloc(pmt->cbFormat);
        if (m_mt.pbFormat) {
            memcpy(m_mt.pbFormat, pmt->pbFormat, pmt->cbFormat);
            m_mt.cbFormat = pmt->cbFormat;
        }
    }
    const VIDEOINFOHEADER* pvi = (const VIDEOINFOHEADER*)m_mt.pbFormat;
    if (pvi && m_mt.formattype == FORMAT_VideoInfo && pvi->bmiHeader.biWidth > 0 && pvi->bmiHeader.biHeight > 0) {
        SetDimensions((uint32_t)pvi->bmiHeader.biWidth, (uint32_t)pvi->bmiHeader.biHeight);
    }
}

HRESULT STDMETHODCALLTYPE OutputPin::Connect(IPin* pRecv, const AM_MEDIA_TYPE* pmt) {
    if (m_connected) return VFW_E_ALREADY_CONNECTED;
    if (!pRecv) return E_POINTER;
    if (pmt) {
        if (QueryAccept(pmt) != S_OK) return VFW_E_TYPE_NOT_ACCEPTED;
        AdoptMediaType(pmt);
    }
    HRESULT hr = pRecv->ReceiveConnection(this, &m_mt);
    if (FAILED(hr)) return hr;
    m_pConnectedTo = pRecv; m_pConnectedTo->AddRef();
    m_connected = true;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::ReceiveConnection(IPin* pConn, const AM_MEDIA_TYPE* pmt) {
    if (m_connected) return VFW_E_ALREADY_CONNECTED;
    if (!pConn) return E_POINTER;
    if (pmt) {
        if (QueryAccept(pmt) != S_OK) return VFW_E_TYPE_NOT_ACCEPTED;
        AdoptMediaType(pmt);
    }
    m_pConnectedTo = pConn; m_pConnectedTo->AddRef();
    m_connected = true;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::Disconnect() {
    if (!m_connected) return S_FALSE;
    if (m_pConnectedTo) { m_pConnectedTo->Release(); m_pConnectedTo = nullptr; }
    if (m_pPeerAllocator) { m_pPeerAllocator->Release(); m_pPeerAllocator = nullptr; }
    m_connected = false;
    m_committedSize = 0;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::ConnectedTo(IPin** pp) {
    if (!pp) return E_POINTER;
    *pp = nullptr;
    if (!m_connected || !m_pConnectedTo) return VFW_E_NOT_CONNECTED;
    *pp = m_pConnectedTo; (*pp)->AddRef();
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::ConnectionMediaType(AM_MEDIA_TYPE* p) {
    if (!p) return E_POINTER;
    if (!m_connected) return VFW_E_NOT_CONNECTED;
    *p = m_mt;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::QueryPinInfo(PIN_INFO* pInfo) {
    if (!pInfo) return E_POINTER;
    pInfo->pFilter = m_pFilter;
    if (m_pFilter) m_pFilter->AddRef();
    wcscpy(pInfo->achName, L"Output");
    pInfo->dir = PINDIR_OUTPUT;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::QueryId(LPWSTR* Id) {
    if (!Id) return E_POINTER;
    *Id = (LPWSTR)CoTaskMemAlloc(32);
    if (*Id) wcscpy(*Id, L"Output");
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::QueryAccept(const AM_MEDIA_TYPE* p) {
    if (!p) return E_POINTER;
    if (p->majortype != MEDIATYPE_Video) return S_FALSE;
    if (p->formattype != FORMAT_VideoInfo && p->formattype != GUID_NULL) return S_FALSE;
    if (p->subtype == MEDIASUBTYPE_NV12 || p->subtype == MEDIASUBTYPE_YUY2) return S_OK;
    return S_FALSE;
}

// ---- CPU pixel helpers (DLL has no CUDA; 640x480 convert is <1ms) ----
void nv12_to_yuy2_cpu(const uint8_t* nv12, uint8_t* yuy2, uint32_t w, uint32_t h) {
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + (size_t)w * h;
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t* uv_row = uv_plane + (size_t)(y / 2) * w;
        for (uint32_t x = 0; x < w; x += 2) {
            size_t o = ((size_t)y * w + x) * 2;
            yuy2[o + 0] = y_plane[(size_t)y * w + x];
            yuy2[o + 1] = uv_row[x];
            yuy2[o + 2] = y_plane[(size_t)y * w + x + 1];
            yuy2[o + 3] = uv_row[x + 1];
        }
    }
}

void scale_nv12_bilinear_cpu(const uint8_t* src, uint32_t sw, uint32_t sh,
                              uint8_t* dst, uint32_t dw, uint32_t dh) {
    const uint8_t* src_y = src;
    const uint8_t* src_uv = src + (size_t)sw * sh;
    uint8_t* dst_y = dst;
    uint8_t* dst_uv = dst + (size_t)dw * dh;
    for (uint32_t y = 0; y < dh; y++) {
        uint32_t sy = (y * sh) / dh;
        for (uint32_t x = 0; x < dw; x++) {
            uint32_t sx = (x * sw) / dw;
            dst_y[(size_t)y * dw + x] = src_y[(size_t)sy * sw + sx];
        }
    }
    uint32_t sh_uv = sh / 2, dh_uv = dh / 2, sw_uv = sw / 2, dw_uv = dw / 2;
    for (uint32_t y = 0; y < dh_uv; y++) {
        uint32_t sy = (y * sh_uv) / dh_uv;
        for (uint32_t x = 0; x < dw_uv; x++) {
            uint32_t sx = (x * sw_uv) / dw_uv;
            dst_uv[((size_t)y * dw_uv + x) * 2 + 0] = src_uv[((size_t)sy * sw_uv + sx) * 2 + 0];
            dst_uv[((size_t)y * dw_uv + x) * 2 + 1] = src_uv[((size_t)sy * sw_uv + sx) * 2 + 1];
        }
    }
}

void fill_nv12_bars_cpu(uint8_t* nv12, uint32_t w, uint32_t h, int frame) {
    static const uint8_t bars[8][3] = {
        {180,128,128},{168,44,136},{133,29,177},{87,217,146},
        {105,120,176},{63,166,88},{29,154,29},{16,128,128}
    };
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            int b = (int)(((x + frame * 4) * 8) / w) % 8;
            nv12[(size_t)y * w + x] = bars[b][0];
            if ((y & 1) == 0 && (x & 1) == 0) {
                size_t o = (size_t)w * h + (size_t)(y / 2) * w + x;
                nv12[o + 0] = bars[b][1];
                nv12[o + 1] = bars[b][2];
            }
        }
    }
}

HRESULT STDMETHODCALLTYPE OutputPin::EnumMediaTypes(IEnumMediaTypes** pp) {
    if (!pp) return E_POINTER;
    *pp = new CEnumMediaTypes(m_mt);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::QueryInternalConnections(IPin**, ULONG*) { return E_NOTIMPL; }
HRESULT STDMETHODCALLTYPE OutputPin::EndOfStream() { return S_OK; }
HRESULT STDMETHODCALLTYPE OutputPin::BeginFlush() { return S_OK; }
HRESULT STDMETHODCALLTYPE OutputPin::EndFlush() { return S_OK; }
HRESULT STDMETHODCALLTYPE OutputPin::NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) { return S_OK; }

HRESULT STDMETHODCALLTYPE OutputPin::QueryDirection(PIN_DIRECTION* pd) {
    if (!pd) return E_POINTER;
    *pd = PINDIR_OUTPUT;
    return S_OK;
}

// IMemInputPin
HRESULT STDMETHODCALLTYPE OutputPin::GetAllocator(IMemAllocator** pp) {
    if (!pp) return E_POINTER;
    if (m_pPeerAllocator) {
        *pp = m_pPeerAllocator;
        (*pp)->AddRef();
    } else {
        *pp = static_cast<IMemAllocator*>(this);
        (*pp)->AddRef();
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::NotifyAllocator(IMemAllocator* pAlloc, BOOL) {
    if (m_pPeerAllocator) { m_pPeerAllocator->Release(); m_pPeerAllocator = nullptr; }
    if (pAlloc) {
        m_pPeerAllocator = pAlloc;
        m_pPeerAllocator->AddRef();
    }
    m_committedSize = 0;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::Receive(IMediaSample*) { return S_OK; }

HRESULT STDMETHODCALLTYPE OutputPin::ReceiveMultiple(IMediaSample** pS, long n, long* nR) {
    if (nR) *nR = 0;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::ReceiveCanBlock() { return S_FALSE; }

HRESULT STDMETHODCALLTYPE OutputPin::GetAllocatorRequirements(ALLOCATOR_PROPERTIES* pProps) {
    if (pProps) *pProps = m_allocProps;
    return S_OK;
}

// IMemAllocator
HRESULT STDMETHODCALLTYPE OutputPin::SetProperties(ALLOCATOR_PROPERTIES* pReq, ALLOCATOR_PROPERTIES* pAct) {
    if (pReq) m_allocProps = *pReq;
    if (pAct) *pAct = m_allocProps;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::GetProperties(ALLOCATOR_PROPERTIES* p) {
    if (!p) return E_POINTER;
    *p = m_allocProps;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::Commit() { return S_OK; }
HRESULT STDMETHODCALLTYPE OutputPin::Decommit() { return S_OK; }

HRESULT STDMETHODCALLTYPE OutputPin::GetBuffer(IMediaSample** pp, REFERENCE_TIME*, REFERENCE_TIME*, DWORD) {
    if (!pp) return E_POINTER;
    LONG sz = m_allocProps.cbBuffer;
    if (sz <= 0) sz = 640 * 480 * 3 / 2;
    BYTE* data = (BYTE*)CoTaskMemAlloc(sz);
    if (!data) return E_OUTOFMEMORY;
    memset(data, 0, sz);
    *pp = new SimpleMediaSample(data, sz);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::ReleaseBuffer(IMediaSample* pS) {
    if (pS) pS->Release();
    return S_OK;
}

// IKsPropertySet
HRESULT STDMETHODCALLTYPE OutputPin::Set(REFGUID guidPropSet, DWORD dwPropID, LPVOID,
                                          DWORD, LPVOID, DWORD) {
    if (guidPropSet == AMPROPSETID_Pin && dwPropID == AMPROPERTY_PIN_CATEGORY)
        return S_OK;
    return E_PROPSET_UNSUPPORTED;
}

HRESULT STDMETHODCALLTYPE OutputPin::Get(REFGUID guidPropSet, DWORD dwPropID, LPVOID,
                                          DWORD, LPVOID pPropData, DWORD cbPropData, DWORD* pcbReturned) {
    if (guidPropSet == AMPROPSETID_Pin && dwPropID == AMPROPERTY_PIN_CATEGORY) {
        if (pcbReturned) *pcbReturned = sizeof(GUID);
        // MUST write the category GUID: enumerators (OBS/Chrome) compare it
        // against PIN_CATEGORY_CAPTURE and drop the device on mismatch.
        if (!pPropData || cbPropData < sizeof(GUID)) return E_OUTOFMEMORY;
        memcpy(pPropData, &PIN_CATEGORY_CAPTURE, sizeof(GUID));
        return S_OK;
    }
    return E_PROPSET_UNSUPPORTED;
}

HRESULT STDMETHODCALLTYPE OutputPin::QuerySupported(REFGUID guidPropSet, DWORD dwPropID,
                                                     DWORD* pTypeSupport) {
    if (guidPropSet == AMPROPSETID_Pin && dwPropID == AMPROPERTY_PIN_CATEGORY) {
        if (pTypeSupport) *pTypeSupport = KSPROPERTY_SUPPORT_GET;
        return S_OK;
    }
    return E_PROPSET_UNSUPPORTED;
}

// IAMStreamConfig — minimal NV12 capability reporting for Chrome/graph builders
HRESULT STDMETHODCALLTYPE OutputPin::SetFormat(AM_MEDIA_TYPE* pmt) {
    if (!pmt) return E_POINTER;
    if (QueryAccept(pmt) != S_OK) return VFW_E_TYPE_NOT_ACCEPTED;
    AdoptMediaType(pmt);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::GetFormat(AM_MEDIA_TYPE** ppmt) {
    if (!ppmt) return E_POINTER;
    AM_MEDIA_TYPE* pmt = (AM_MEDIA_TYPE*)CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE));
    if (!pmt) return E_OUTOFMEMORY;
    memcpy(pmt, &m_mt, sizeof(AM_MEDIA_TYPE));
    pmt->pbFormat = nullptr; pmt->cbFormat = 0;
    if (m_mt.cbFormat && m_mt.pbFormat) {
        pmt->pbFormat = (BYTE*)CoTaskMemAlloc(m_mt.cbFormat);
        if (!pmt->pbFormat) { CoTaskMemFree(pmt); return E_OUTOFMEMORY; }
        memcpy(pmt->pbFormat, m_mt.pbFormat, m_mt.cbFormat);
        pmt->cbFormat = m_mt.cbFormat;
    }
    *ppmt = pmt;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::GetNumberOfCapabilities(int* piCount, int* piSize) {
    if (!piCount || !piSize) return E_POINTER;
    *piCount = 2; // 0 = NV12, 1 = YUY2 (both 640x480@30)
    *piSize = sizeof(VIDEO_STREAM_CONFIG_CAPS);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::GetStreamCaps(int iIndex, AM_MEDIA_TYPE** ppmt, BYTE* pSCC) {
    if (!ppmt || !pSCC) return E_POINTER;
    if (iIndex < 0 || iIndex > 1) return S_FALSE;
    if (FAILED(GetFormat(ppmt))) return E_FAIL;
    if (iIndex == 1) {
        (*ppmt)->subtype = MEDIASUBTYPE_YUY2;
        VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)(*ppmt)->pbFormat;
        if (pvi) {
            pvi->bmiHeader.biBitCount = 16;
            pvi->bmiHeader.biCompression = mmioFOURCC('Y','U','Y','2');
            pvi->bmiHeader.biSizeImage = 640 * 480 * 2;
        }
        (*ppmt)->lSampleSize = 640 * 480 * 2;
    }
    VIDEO_STREAM_CONFIG_CAPS* pCaps = (VIDEO_STREAM_CONFIG_CAPS*)pSCC;
    memset(pCaps, 0, sizeof(*pCaps));
    pCaps->guid = FORMAT_VideoInfo;
    pCaps->VideoStandard = AnalogVideo_None;
    pCaps->InputSize.cx = 640;
    pCaps->InputSize.cy = 480;
    pCaps->MinCroppingSize.cx = 640;
    pCaps->MinCroppingSize.cy = 480;
    pCaps->MaxCroppingSize.cx = 640;
    pCaps->MaxCroppingSize.cy = 480;
    pCaps->CropGranularityX = 1;
    pCaps->CropGranularityY = 1;
    pCaps->CropAlignX = 1;
    pCaps->CropAlignY = 1;
    pCaps->MinOutputSize.cx = 640;
    pCaps->MinOutputSize.cy = 480;
    pCaps->MaxOutputSize.cx = 640;
    pCaps->MaxOutputSize.cy = 480;
    pCaps->OutputGranularityX = 1;
    pCaps->OutputGranularityY = 1;
    pCaps->StretchTapsX = 0;
    pCaps->StretchTapsY = 0;
    pCaps->ShrinkTapsX = 0;
    pCaps->ShrinkTapsY = 0;
    pCaps->MinFrameInterval = (REFERENCE_TIME)(10000000.0 / 30.0);
    pCaps->MaxFrameInterval = (REFERENCE_TIME)(10000000.0 / 30.0);
    pCaps->MinBitsPerSecond = 640 * 480 * 12 * 30;
    pCaps->MaxBitsPerSecond = 640 * 480 * 12 * 30;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE OutputPin::DeliverFrame(const uint8_t* nv12, uint32_t w, uint32_t h, uint64_t timestamp_us) {
    if (!m_connected || !m_pConnectedTo) return VFW_E_NOT_CONNECTED;

    // Match what the downstream pin negotiated (size + NV12/YUY2).
    // The EXE always sends 640x480 NV12; scale/convert here if needed.
    uint32_t tw = 640, th = 480;
    GUID tsub = MEDIASUBTYPE_NV12;
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)m_mt.pbFormat;
    if (pvi && m_mt.formattype == FORMAT_VideoInfo && pvi->bmiHeader.biWidth > 0 && pvi->bmiHeader.biHeight > 0) {
        tw = (uint32_t)pvi->bmiHeader.biWidth;
        th = (uint32_t)pvi->bmiHeader.biHeight;
        if (tw > 1920) tw = 1920;
        if (th > 1080) th = 1080;
    }
    tsub = m_mt.subtype;

    const uint8_t* payload = nv12;
    uint32_t payload_size = w * h * 3 / 2;
    static uint8_t* s_scaled = nullptr; // NV12 at target size
    static uint8_t* s_packed = nullptr; // final packed frame (NV12 or YUY2)
    if (tw != w || th != h || (tsub != MEDIASUBTYPE_NV12)) {
        if (!s_scaled) s_scaled = (uint8_t*)malloc(1920 * 1080 * 3 / 2);
        if (!s_packed) s_packed = (uint8_t*)malloc(1920 * 1080 * 2);
        if (!s_scaled || !s_packed) return E_OUTOFMEMORY;
        const uint8_t* nv = nv12;
        if (tw != w || th != h) {
            scale_nv12_bilinear_cpu(nv12, w, h, s_scaled, tw, th);
            nv = s_scaled;
        }
        if (tsub == MEDIASUBTYPE_YUY2) {
            nv12_to_yuy2_cpu(nv, s_packed, tw, th);
            payload = s_packed;
            payload_size = tw * th * 2;
        } else {
            if (nv != nv12) memcpy(s_packed, nv, tw * th * 3 / 2);
            else memcpy(s_packed, nv12, tw * th * 3 / 2);
            payload = s_packed;
            payload_size = tw * th * 3 / 2;
        }
    }

    IMemInputPin* pInput = nullptr;
    HRESULT hr = m_pConnectedTo->QueryInterface(IID_IMemInputPin, (void**)&pInput);
    if (FAILED(hr) || !pInput) return hr;

    // Prefer the allocator negotiated at connect time; set it up ONCE per
    // payload size. Decommitting/committing on every frame stalls graphs
    // (this broke Chrome streaming).
    IMemAllocator* pAlloc = nullptr;
    if (m_pPeerAllocator) {
        pAlloc = m_pPeerAllocator;
        pAlloc->AddRef();
        hr = S_OK;
    } else {
        hr = pInput->GetAllocator(&pAlloc);
    }
    if (SUCCEEDED(hr) && pAlloc) {
        if (m_committedSize != (LONG)payload_size) {
            ALLOCATOR_PROPERTIES act = {};
            ALLOCATOR_PROPERTIES req = {};
            req.cbBuffer = payload_size;
            req.cBuffers = 4;
            req.cbAlign = 1;
            pAlloc->Decommit();
            if (SUCCEEDED(pAlloc->SetProperties(&req, &act)) && act.cbBuffer >= (LONG)payload_size) {
                if (SUCCEEDED(pAlloc->Commit()))
                    m_committedSize = (LONG)payload_size;
            } else {
                // Peer rejected our request (uses its own sizing); commit as-is.
                pAlloc->Commit();
                m_committedSize = (LONG)payload_size;
            }
        }

        IMediaSample* pSample = nullptr;
        hr = pAlloc->GetBuffer(&pSample, nullptr, nullptr, 0);
        if (SUCCEEDED(hr) && pSample) {
            BYTE* pData = nullptr;
            pSample->GetPointer(&pData);
            LONG bufSize = pSample->GetSize();
            // Never overflow the downstream buffer: copy what fits.
            uint32_t copySize = payload_size;
            if (bufSize > 0 && (uint32_t)bufSize < copySize) copySize = (uint32_t)bufSize;
            if (pData && copySize) memcpy(pData, payload, copySize);
            pSample->SetActualDataLength((long)copySize);

            REFERENCE_TIME rtStart = (REFERENCE_TIME)timestamp_us * 10;
            REFERENCE_TIME rtStop = rtStart + (REFERENCE_TIME)(10000000.0 / 30.0);
            pSample->SetTime(&rtStart, &rtStop);

            hr = pInput->Receive(pSample);
            pSample->Release();
        }
        pAlloc->Release();
    } else {
        LONG sz = (LONG)payload_size;
        BYTE* data = (BYTE*)CoTaskMemAlloc(sz);
        if (!data) { pInput->Release(); return E_OUTOFMEMORY; }
        memcpy(data, payload, sz);
        IMediaSample* pSample = new SimpleMediaSample(data, sz);

        REFERENCE_TIME rtStart = (REFERENCE_TIME)timestamp_us * 10;
        REFERENCE_TIME rtStop = rtStart + (REFERENCE_TIME)(10000000.0 / 30.0);
        pSample->SetTime(&rtStart, &rtStop);

        hr = pInput->Receive(pSample);
        pSample->Release();
    }
    pInput->Release();
    return hr;
}

static uint32_t SampleSizeFor(const GUID& sub, uint32_t w, uint32_t h) {
    if (sub == MEDIASUBTYPE_YUY2) return w * h * 2;
    return w * h * 3 / 2; // NV12
}

void OutputPin::SetDimensions(uint32_t w, uint32_t h) {
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)m_mt.pbFormat;
    if (pvi) {
        pvi->bmiHeader.biWidth = w;
        pvi->bmiHeader.biHeight = h;
        pvi->bmiHeader.biSizeImage = SampleSizeFor(m_mt.subtype, w, h);
    }
    m_mt.lSampleSize = SampleSizeFor(m_mt.subtype, w, h);
    m_allocProps.cbBuffer = SampleSizeFor(m_mt.subtype, w, h);
    m_committedSize = 0; // force allocator re-setup for the new size
}

// ============================================================================
// KagerouVirtualCamFilter
// ============================================================================

KagerouVirtualCamFilter::KagerouVirtualCamFilter()
    : m_ref(1), m_state(State_Stopped), m_pGraph(nullptr) {
    wcscpy(m_name, L"Kagerou Virtual Camera");
    m_pOutputPin = new OutputPin(this);
}

KagerouVirtualCamFilter::~KagerouVirtualCamFilter() {
    if (m_pOutputPin) { m_pOutputPin->Release(); m_pOutputPin = nullptr; }
}

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IBaseFilter) {
        *ppv = static_cast<IBaseFilter*>(this); AddRef(); return S_OK;
    }
    if (riid == IID_IPersist) {
        *ppv = static_cast<IPersist*>(this); AddRef(); return S_OK;
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE KagerouVirtualCamFilter::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE KagerouVirtualCamFilter::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::GetClassID(CLSID* p) { if (!p) return E_POINTER; *p = CLSID_KagerouVirtualCam; return S_OK; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::GetState(DWORD, FILTER_STATE* s) { if (!s) return E_POINTER; *s = m_state; return S_OK; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::SetSyncSource(IReferenceClock*) { return S_OK; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::GetSyncSource(IReferenceClock**) { return E_NOTIMPL; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::Stop() { m_state = State_Stopped; return S_OK; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::Pause() { m_state = State_Paused; return S_OK; }
HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::Run(REFERENCE_TIME) { m_state = State_Running; return S_OK; }

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::EnumPins(IEnumPins** pp) {
    if (!pp) return E_POINTER;
    *pp = new CEnumPins(this);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::FindPin(LPCWSTR Id, IPin** pp) {
    if (!pp) return E_POINTER;
    *pp = nullptr;
    if (wcscmp(Id, L"Output") == 0) {
        *pp = m_pOutputPin;
        (*pp)->AddRef();
        return S_OK;
    }
    return VFW_E_NOT_FOUND;
}

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::QueryFilterInfo(FILTER_INFO* pInfo) {
    if (!pInfo) return E_POINTER;
    wcscpy(pInfo->achName, m_name);
    pInfo->pGraph = m_pGraph;
    if (m_pGraph) m_pGraph->AddRef();
    return S_OK;
}

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::JoinFilterGraph(IFilterGraph* pGraph, LPCWSTR) {
    m_pGraph = pGraph;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE KagerouVirtualCamFilter::QueryVendorInfo(LPWSTR*) { return E_NOTIMPL; }

// ============================================================================
// ClassFactory
// ============================================================================

HRESULT STDMETHODCALLTYPE KagerouVCamClassFactory::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_IClassFactory) { *ppv = this; AddRef(); return S_OK; }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE KagerouVCamClassFactory::AddRef() { return InterlockedIncrement(&m_ref); }
ULONG STDMETHODCALLTYPE KagerouVCamClassFactory::Release() { LONG r = InterlockedDecrement(&m_ref); if (r==0) delete this; return r; }

HRESULT STDMETHODCALLTYPE KagerouVCamClassFactory::CreateInstance(LPUNKNOWN pOuter, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (pOuter && riid != IID_IUnknown) return CLASS_E_NOAGGREGATION;
    if (!g_pFilter) {
        g_pFilter = new KagerouVirtualCamFilter();
    }
    EnsureReaderStarted();
    return g_pFilter->QueryInterface(riid, ppv);
}

HRESULT STDMETHODCALLTYPE KagerouVCamClassFactory::LockServer(BOOL fLock) {
    if (fLock) CoAddRefServerProcess(); else CoReleaseServerProcess();
    return S_OK;
}
