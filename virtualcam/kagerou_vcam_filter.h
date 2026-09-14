#pragma once
// ============================================================================
// Kagerou Virtual Camera — DirectShow Source Filter
// ============================================================================

#include <windows.h>
#include <strmif.h>
#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <initguid.h>
#include <cstdint>

DEFINE_GUID(CLSID_KagerouVirtualCam,
    0x7A3B3C4D, 0x5E6F, 0x4A7B, 0x8C, 0x9D, 0x0E, 0x1F, 0x2A, 0x3B, 0x4C, 0x5D);

// ============================================================================
// SimpleMediaSample
// ============================================================================
class SimpleMediaSample : public IMediaSample {
    LONG m_ref;
    BYTE* m_pData;
    LONG m_dataSize;
    LONG m_actualLen;
    REFERENCE_TIME m_rtStart, m_rtStop;

public:
    SimpleMediaSample(BYTE* data, LONG size);
    ~SimpleMediaSample();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE GetPointer(BYTE** pp) override;
    LONG STDMETHODCALLTYPE GetSize() override;
    HRESULT STDMETHODCALLTYPE GetTime(REFERENCE_TIME* s, REFERENCE_TIME* e) override;
    HRESULT STDMETHODCALLTYPE SetTime(REFERENCE_TIME* s, REFERENCE_TIME* e) override;
    HRESULT STDMETHODCALLTYPE IsSyncPoint() override;
    HRESULT STDMETHODCALLTYPE SetSyncPoint(BOOL) override;
    HRESULT STDMETHODCALLTYPE IsPreroll() override;
    HRESULT STDMETHODCALLTYPE SetPreroll(BOOL) override;
    LONG STDMETHODCALLTYPE GetActualDataLength() override;
    HRESULT STDMETHODCALLTYPE SetActualDataLength(long) override;
    HRESULT STDMETHODCALLTYPE GetMediaType(AM_MEDIA_TYPE**) override;
    HRESULT STDMETHODCALLTYPE SetMediaType(AM_MEDIA_TYPE*) override;
    HRESULT STDMETHODCALLTYPE IsDiscontinuity() override;
    HRESULT STDMETHODCALLTYPE SetDiscontinuity(BOOL) override;
    HRESULT STDMETHODCALLTYPE GetMediaTime(LONGLONG*, LONGLONG*) override;
    HRESULT STDMETHODCALLTYPE SetMediaTime(LONGLONG*, LONGLONG*) override;
};

// CPU pixel helpers shared with the DLL reader thread (defined in filter.cpp)
void nv12_to_yuy2_cpu(const uint8_t* nv12, uint8_t* yuy2, uint32_t w, uint32_t h);
void scale_nv12_bilinear_cpu(const uint8_t* src, uint32_t sw, uint32_t sh,
                              uint8_t* dst, uint32_t dw, uint32_t dh);
void fill_nv12_bars_cpu(uint8_t* nv12, uint32_t w, uint32_t h, int frame);

// ============================================================================
// CEnumMediaTypes
// ============================================================================
class CEnumMediaTypes : public IEnumMediaTypes {
    LONG m_ref;
    AM_MEDIA_TYPE m_mt;
    bool m_done;
public:
    CEnumMediaTypes(const AM_MEDIA_TYPE& mt);
    ~CEnumMediaTypes();
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE Next(ULONG cMediaTypes, AM_MEDIA_TYPE** ppMediaTypes, ULONG* pcFetched) override;
    HRESULT STDMETHODCALLTYPE Skip(ULONG cMediaTypes) override;
    HRESULT STDMETHODCALLTYPE Reset() override;
    HRESULT STDMETHODCALLTYPE Clone(IEnumMediaTypes** ppEnum) override;
};

// ============================================================================
// CEnumPins
// ============================================================================
class CEnumPins : public IEnumPins {
    LONG m_ref;
    IBaseFilter* m_pFilter;
    bool m_done;
public:
    CEnumPins(IBaseFilter* pFilter);
    ~CEnumPins();
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE Next(ULONG cPins, IPin** ppPins, ULONG* pcFetched) override;
    HRESULT STDMETHODCALLTYPE Skip(ULONG cPins) override;
    HRESULT STDMETHODCALLTYPE Reset() override;
    HRESULT STDMETHODCALLTYPE Clone(IEnumPins** ppEnum) override;
};

// ============================================================================
// OutputPin — IPin + IMemInputPin + IMemAllocator + IKsPropertySet
// ============================================================================
class OutputPin : public IPin, public IMemInputPin, public IMemAllocator, public IKsPropertySet, public IAMStreamConfig {
    LONG m_ref;
    IBaseFilter* m_pFilter;
    IPin* m_pConnectedTo;
    AM_MEDIA_TYPE m_mt;
    bool m_connected;
    ALLOCATOR_PROPERTIES m_allocProps;
    IMemAllocator* m_pPeerAllocator;
    LONG m_committedSize; // payload bytes the peer allocator is set up for

public:
    OutputPin(IBaseFilter* pFilter);
    ~OutputPin();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;

    // IPin
    HRESULT STDMETHODCALLTYPE Connect(IPin* pRecv, const AM_MEDIA_TYPE* pmt) override;
    HRESULT STDMETHODCALLTYPE ReceiveConnection(IPin* pConn, const AM_MEDIA_TYPE* pmt) override;
    HRESULT STDMETHODCALLTYPE Disconnect() override;
    HRESULT STDMETHODCALLTYPE ConnectedTo(IPin** pp) override;
    HRESULT STDMETHODCALLTYPE ConnectionMediaType(AM_MEDIA_TYPE* p) override;
    HRESULT STDMETHODCALLTYPE QueryPinInfo(PIN_INFO* pInfo) override;
    HRESULT STDMETHODCALLTYPE QueryId(LPWSTR* Id) override;
    HRESULT STDMETHODCALLTYPE QueryAccept(const AM_MEDIA_TYPE* p) override;
    HRESULT STDMETHODCALLTYPE EnumMediaTypes(IEnumMediaTypes** pp) override;
    HRESULT STDMETHODCALLTYPE QueryInternalConnections(IPin**, ULONG*) override;
    HRESULT STDMETHODCALLTYPE EndOfStream() override;
    HRESULT STDMETHODCALLTYPE BeginFlush() override;
    HRESULT STDMETHODCALLTYPE EndFlush() override;
    HRESULT STDMETHODCALLTYPE NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override;
    HRESULT STDMETHODCALLTYPE QueryDirection(PIN_DIRECTION* pd) override;

    // IMemInputPin
    HRESULT STDMETHODCALLTYPE GetAllocator(IMemAllocator** pp) override;
    HRESULT STDMETHODCALLTYPE NotifyAllocator(IMemAllocator* pAlloc, BOOL bReadOnly) override;
    HRESULT STDMETHODCALLTYPE Receive(IMediaSample* pS) override;
    HRESULT STDMETHODCALLTYPE ReceiveMultiple(IMediaSample** pS, long n, long* nR) override;
    HRESULT STDMETHODCALLTYPE ReceiveCanBlock() override;
    HRESULT STDMETHODCALLTYPE GetAllocatorRequirements(ALLOCATOR_PROPERTIES* pProps) override;

    // IMemAllocator
    HRESULT STDMETHODCALLTYPE SetProperties(ALLOCATOR_PROPERTIES*, ALLOCATOR_PROPERTIES*) override;
    HRESULT STDMETHODCALLTYPE GetProperties(ALLOCATOR_PROPERTIES*) override;
    HRESULT STDMETHODCALLTYPE Commit() override;
    HRESULT STDMETHODCALLTYPE Decommit() override;
    HRESULT STDMETHODCALLTYPE GetBuffer(IMediaSample**, REFERENCE_TIME*, REFERENCE_TIME*, DWORD) override;
    HRESULT STDMETHODCALLTYPE ReleaseBuffer(IMediaSample*) override;

    // IKsPropertySet
    HRESULT STDMETHODCALLTYPE Set(REFGUID guidPropSet, DWORD dwPropID, LPVOID pInstanceData,
                                   DWORD cbInstanceData, LPVOID pPropData, DWORD cbPropData) override;
    HRESULT STDMETHODCALLTYPE Get(REFGUID guidPropSet, DWORD dwPropID, LPVOID pInstanceData,
                                   DWORD cbInstanceData, LPVOID pPropData, DWORD cbPropData,
                                   DWORD* pcbReturned) override;
    HRESULT STDMETHODCALLTYPE QuerySupported(REFGUID guidPropSet, DWORD dwPropID,
                                              DWORD* pTypeSupport) override;

    // IAMStreamConfig (Chrome/MF graph builders query this for capabilities)
    HRESULT STDMETHODCALLTYPE SetFormat(AM_MEDIA_TYPE* pmt) override;
    HRESULT STDMETHODCALLTYPE GetFormat(AM_MEDIA_TYPE** ppmt) override;
    HRESULT STDMETHODCALLTYPE GetNumberOfCapabilities(int* piCount, int* piSize) override;
    HRESULT STDMETHODCALLTYPE GetStreamCaps(int iIndex, AM_MEDIA_TYPE** ppmt, BYTE* pSCC) override;

    HRESULT DeliverFrame(const uint8_t* nv12, uint32_t w, uint32_t h, uint64_t timestamp_us);
    bool IsConnected() const { return m_connected; }
    void SetDimensions(uint32_t w, uint32_t h);
    void AdoptMediaType(const AM_MEDIA_TYPE* pmt);
};

// ============================================================================
// KagerouVirtualCamFilter
// ============================================================================
class KagerouVirtualCamFilter : public IBaseFilter {
    LONG m_ref;
    FILTER_STATE m_state;
    IFilterGraph* m_pGraph;
    OutputPin* m_pOutputPin;
    wchar_t m_name[64];

public:
    KagerouVirtualCamFilter();
    ~KagerouVirtualCamFilter();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE GetClassID(CLSID* pClsID) override;
    HRESULT STDMETHODCALLTYPE GetState(DWORD, FILTER_STATE*) override;
    HRESULT STDMETHODCALLTYPE SetSyncSource(IReferenceClock*) override;
    HRESULT STDMETHODCALLTYPE GetSyncSource(IReferenceClock**) override;
    HRESULT STDMETHODCALLTYPE Stop() override;
    HRESULT STDMETHODCALLTYPE Pause() override;
    HRESULT STDMETHODCALLTYPE Run(REFERENCE_TIME) override;
    HRESULT STDMETHODCALLTYPE EnumPins(IEnumPins**) override;
    HRESULT STDMETHODCALLTYPE FindPin(LPCWSTR, IPin**) override;
    HRESULT STDMETHODCALLTYPE QueryFilterInfo(FILTER_INFO*) override;
    HRESULT STDMETHODCALLTYPE JoinFilterGraph(IFilterGraph*, LPCWSTR) override;
    HRESULT STDMETHODCALLTYPE QueryVendorInfo(LPWSTR*) override;

    OutputPin* GetOutputPin() { return m_pOutputPin; }
    bool IsStreaming() const { return m_state == State_Running; }
};

// ============================================================================
// KagerouVCamClassFactory
// ============================================================================
class KagerouVCamClassFactory : public IClassFactory {
    LONG m_ref;
public:
    KagerouVCamClassFactory() : m_ref(1) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE CreateInstance(LPUNKNOWN pOuter, REFIID riid, void** ppv) override;
    HRESULT STDMETHODCALLTYPE LockServer(BOOL fLock) override;
};
