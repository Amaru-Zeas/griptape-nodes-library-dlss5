// DLSS 5 (NGX Feature 18, Neural Rendering) D3D12 worker.
// Based on the DLSS5-for-Nuke worker (MIT); extended for the Griptape bridge with
//   * a live mode ('D5L1'): RGBA8 in/out and per-frame settings, resident process
//   * runtime folders passed on the command line (no copying of NVIDIA DLLs)
//   * F16C/AVX2 pixel conversion (the scalar loops dominated frame time)
#include "DLSSWorker.h"
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>

using InitExtFn = NGXResult(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
using InitProjectIdFn = NGXResult(__cdecl*)(const char*, int, const char*, const wchar_t*, ID3D12Device*, int, const void*);
using AllocParamsFn = NGXResult(__cdecl*)(NGXParameter**);
using DestroyParamsFn = NGXResult(__cdecl*)(NGXParameter*);
using CreateFeatureFn = NGXResult(__cdecl*)(ID3D12GraphicsCommandList*, int, NGXParameter*, NGXHandle**);
using EvaluateFeatureFn = NGXResult(__cdecl*)(ID3D12GraphicsCommandList*, const NGXHandle*, const NGXParameter*, void*);
using ReleaseFeatureFn = NGXResult(__cdecl*)(NGXHandle*);
using ShutdownFn = NGXResult(__cdecl*)();

using SnippetInitFn = NGXResult(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, const void*, int);

using ShimInitFn = NGXResult(__cdecl*)(void*, unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
using ShimCreateFn = NGXResult(__cdecl*)(void*, ID3D12GraphicsCommandList*, int, NGXParameter*, NGXHandle**);
using ShimEvaluateFn = NGXResult(__cdecl*)(void*, ID3D12GraphicsCommandList*, const NGXHandle*, const NGXParameter*, void*);
using ShimReleaseFn = NGXResult(__cdecl*)(void*, NGXHandle*);
using ShimShutdownFn = NGXResult(__cdecl*)(void*);

static constexpr unsigned long long APP_ID = 141959980ULL;
static constexpr const char* PROJECT_ID = "53f803cc-a12f-4d69-90d5-19b7599cad19";
static constexpr int NR_FEATURE_ID = 18;

struct NGXPathListInfo {
    wchar_t const* const* Path;
    unsigned int Length;
};
enum NGXLoggingLevel { NGX_LOG_OFF = 0, NGX_LOG_ON = 1, NGX_LOG_VERBOSE = 2 };
using NGXLogCallback = void(__cdecl*)(const char*, NGXLoggingLevel, int);
struct NGXLoggingInfo {
    NGXLoggingLevel LoggingLevel;
    NGXLogCallback Callback;
    void* UserData;
    bool DisableOtherLoggingSinks;
};
struct NGXFeatureCommonInfoInternal;
struct NGXFeatureCommonInfo {
    NGXPathListInfo PathListInfo;
    NGXFeatureCommonInfoInternal* InternalData;
    NGXLoggingInfo LoggingInfo;
};

static inline UINT AlignUp(UINT val, UINT alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static inline uint16_t FloatToHalf(float f) {
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint32_t s = (x >> 16) & 0x8000u;
    int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(s);
        m = (m | 0x800000u) >> (1 - e);
        return static_cast<uint16_t>(s | (m >> 13));
    }
    if (e >= 31) return static_cast<uint16_t>(s | 0x7c00u);
    return static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | (m >> 13));
}

static inline float HalfToFloat(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    if (e == 0) {
        if (m == 0) x = s << 31;
        else {
            e = 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3ff;
            x = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 0x1f) x = (s << 31) | 0x7f800000u | (m << 13);
    else x = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, sizeof(f)); return f;
}

// ---- SIMD pixel conversion (F16C + AVX2; the exe is built with /arch:AVX2) ----

// RGBA8 -> RGBA16F, `count` channel values (count % 4 == 0).
static void U8ToHalf(const uint8_t* src, uint16_t* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i));
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)), scale);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < count; ++i) dst[i] = FloatToHalf(src[i] / 255.0f);
}

// RGBA16F -> RGBA8 with clamp/round; optionally swaps R and B per pixel.
static void HalfToU8(const uint16_t* src, uint8_t* dst, size_t count, bool swapRB) {
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 lo = _mm256_setzero_ps(), hi = _mm256_set1_ps(255.0f);
    const __m128i swz = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 f = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
        f = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(f, scale), lo), hi);
        __m256i v = _mm256_cvtps_epi32(f);
        __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        __m128i p8 = _mm_packus_epi16(p16, p16);
        if (swapRB) p8 = _mm_shuffle_epi8(p8, swz);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), p8);
    }
    for (; i < count; i += 4) {
        for (int c = 0; c < 4 && i + c < count; ++c) {
            int sc = (swapRB && c == 0) ? 2 : (swapRB && c == 2) ? 0 : c;
            dst[i + c] = static_cast<uint8_t>(std::clamp(static_cast<int>(HalfToFloat(src[i + sc]) * 255.0f + 0.5f), 0, 255));
        }
    }
}

// RGBA16F -> RGBA8 (for the DIS gray downscale); no swap.
static void HalfToU8Plain(const uint16_t* src, uint8_t* dst, size_t count) { HalfToU8(src, dst, count, false); }

static inline D3D12_RESOURCE_BARRIER Barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

static ComPtr<ID3D12Resource> CreateTexture(ID3D12Device* dev, UINT w, UINT h, DXGI_FORMAT fmt, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> r;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

static ComPtr<ID3D12Resource> CreateLinearBuffer(ID3D12Device* dev, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    ComPtr<ID3D12Resource> r;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

static bool FileExists(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static HMODULE LoadCoreNGX(const std::wstring& runtime) {
    std::wstring local = runtime + L"\\_nvngx.dll";
    if (FileExists(local)) {
        if (HMODULE m = LoadLibraryW(local.c_str())) return m;
    }
    if (HMODULE m = LoadLibraryW(L"_nvngx.dll")) return m;

    wchar_t winDir[MAX_PATH] = {};
    GetWindowsDirectoryW(winDir, MAX_PATH);
    std::wstring repo = std::wstring(winDir) + L"\\System32\\DriverStore\\FileRepository";
    std::wstring pat = repo + L"\\nv*.inf_*";

    struct Cand { std::wstring path; ULARGE_INTEGER stamp; };
    std::vector<Cand> cands;

    WIN32_FIND_DATAW fd{};
    HANDLE h = FindFirstFileW(pat.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
            if (wcscmp(fd.cFileName, L".") == 0 || wcscmp(fd.cFileName, L"..") == 0) continue;
            std::wstring candidate = repo + L"\\" + fd.cFileName + L"\\_nvngx.dll";
            WIN32_FILE_ATTRIBUTE_DATA fad{};
            if (GetFileAttributesExW(candidate.c_str(), GetFileExInfoStandard, &fad) &&
                !(fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                ULARGE_INTEGER u; u.LowPart = fad.ftLastWriteTime.dwLowDateTime; u.HighPart = fad.ftLastWriteTime.dwHighDateTime;
                cands.push_back({candidate, u});
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }

    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.stamp.QuadPart > b.stamp.QuadPart;
    });

    for (const auto& c : cands) {
        if (HMODULE m = LoadLibraryW(c.path.c_str())) return m;
    }
    return nullptr;
}

static double NowMs() {
    static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return 1000.0 * static_cast<double>(t.QuadPart) / static_cast<double>(freq.QuadPart);
}

// ---------------------------------------------------------------------------
// DLSSWorker Implementation
// ---------------------------------------------------------------------------

bool DLSSWorker::setupD3D12() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) { m_lastError = "CreateDXGIFactory1 failed"; return false; }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || desc.VendorId != 0x10DE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device))))
            break;
    }
    if (!m_device) { m_lastError = "No NVIDIA D3D12 adapter found"; return false; }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(m_device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&m_queue)))) return false;

    if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdAlloc)))) return false;
    if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&m_cmdList)))) return false;

    if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;

    return true;
}

bool DLSSWorker::executeAndWait() {
    if (FAILED(m_cmdList->Close())) return false;
    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_queue->ExecuteCommandLists(1, lists);
    waitGPU();
    m_cmdAlloc->Reset();
    m_cmdList->Reset(m_cmdAlloc.Get(), nullptr);
    return true;
}

void DLSSWorker::waitGPU() {
    ++m_fenceVal;
    m_queue->Signal(m_fence.Get(), m_fenceVal);
    if (m_fence->GetCompletedValue() < m_fenceVal) {
        m_fence->SetEventOnCompletion(m_fenceVal, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

bool DLSSWorker::allocateResources(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH) {
    m_inW = inW; m_inH = inH;
    m_outW = outW; m_outH = outH;
    m_needUpscale = (outW > inW || outH > inH);

    m_colorTex = CreateTexture(m_device.Get(), inW, inH, DXGI_FORMAT_R16G16B16A16_FLOAT,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    if (m_needUpscale) {
        m_intermediateTex = CreateTexture(m_device.Get(), inW, inH, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!m_intermediateTex) return false;
    } else {
        m_intermediateTex.Reset();
    }

    m_depthTex = CreateTexture(m_device.Get(), inW, inH, DXGI_FORMAT_R32_FLOAT,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    m_controlMaskTex = CreateTexture(m_device.Get(), inW, inH, DXGI_FORMAT_R32_FLOAT,
                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_outputTex = CreateTexture(m_device.Get(), outW, outH, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_mvTex = CreateTexture(m_device.Get(), inW, inH, DXGI_FORMAT_R16G16_FLOAT,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    if (!m_colorTex || !m_outputTex || !m_mvTex || !m_depthTex || !m_controlMaskTex) return false;

    m_colorPitch    = AlignUp(inW * 8u, 256u);
    m_mvPitch       = AlignUp(inW * 4u, 256u);
    m_guidePitch    = AlignUp(inW * 4u, 256u);
    m_readbackPitch = AlignUp(outW * 8u, 256u);

    m_uploadBytes   = (UINT64)m_colorPitch * inH;
    m_mvBytes       = (UINT64)m_mvPitch * inH;
    m_guideBytes    = (UINT64)m_guidePitch * inH;
    m_readbackBytes = (UINT64)m_readbackPitch * outH;

    m_uploadBuf   = CreateLinearBuffer(m_device.Get(), m_uploadBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_mvUploadBuf = CreateLinearBuffer(m_device.Get(), m_mvBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_depthUploadBuf = CreateLinearBuffer(m_device.Get(), m_guideBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_controlMaskUploadBuf = CreateLinearBuffer(m_device.Get(), m_guideBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_readbackBuf = CreateLinearBuffer(m_device.Get(), m_readbackBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    if (!m_uploadBuf || !m_mvUploadBuf || !m_depthUploadBuf || !m_controlMaskUploadBuf || !m_readbackBuf) return false;

    return true;
}

void DLSSWorker::setCommonParams(bool reset) {
    if (!m_params) return;
    ID3D12Resource* nrOutput = m_needUpscale ? m_intermediateTex.Get() : m_outputTex.Get();

    // The NR render-preset hint takes the DLSS model preset (J..M = 10..13) when one
    // is given, otherwise the legacy NR profile preset.
    int presetHint = m_hdr.dlss_model_preset != 0 ? (int)m_hdr.dlss_model_preset : (int)m_hdr.preset;

    m_params->Set("DLSSNR.Width", m_inW);
    m_params->Set("DLSSNR.Height", m_inH);
    m_params->Set("DLSSNR.Enabled", 1);
    m_params->Set("DLSSNR.Reset", reset ? 1 : 0);
    m_params->Set("DLSSNR.Style", (int)m_hdr.style);
    m_params->Set("DLSSNR.Hint.Render.Preset", presetHint);
    m_params->Set("DLSSNR.Intensity", m_hdr.intensity);
    m_params->Set("DLSSNR.LocalToneStrength", m_hdr.local_tone);
    m_params->Set("DLSSNR.LocalStructureStrength", m_hdr.local_structure);
    m_params->Set("DLSSNR.SkinStructureStrength", m_hdr.skin_structure);
    m_params->Set("DLSSNR.UseAutoMask", (int)m_hdr.auto_mask);
    m_params->Set("DLSSNR.UICorrection", 0);
    m_params->Set("DLSSNR.DepthInverted", 1);
    m_params->Set("DLSSNR.ScalingRatio", 1.0f);
    m_params->Set("DLSSNR.MVecScaleX", 1.0f);
    m_params->Set("DLSSNR.MVecScaleY", 1.0f);
    m_params->Set("DLSSNR.Color", m_colorTex.Get());
    m_params->Set("DLSSNR.Output", nrOutput);
    m_params->Set("DLSSNR.Backbuffer", nrOutput);
    m_params->Set("DLSSNR.MVec", m_mvTex.Get());
    m_params->Set("DLSSNR.ColorSubrectBaseX", 0);
    m_params->Set("DLSSNR.ColorSubrectBaseY", 0);
    m_params->Set("DLSSNR.ColorSubrectWidth", m_inW);
    m_params->Set("DLSSNR.ColorSubrectHeight", m_inH);
    m_params->Set("DLSSNR.OutputSubrectBaseX", 0);
    m_params->Set("DLSSNR.OutputSubrectBaseY", 0);
    m_params->Set("DLSSNR.OutputSubrectWidth", m_inW);
    m_params->Set("DLSSNR.OutputSubrectHeight", m_inH);
    m_params->Set("DLSSNR.MVecSubrectBaseX", 0);
    m_params->Set("DLSSNR.MVecSubrectBaseY", 0);
    m_params->Set("DLSSNR.MVecSubrectWidth", m_inW);
    m_params->Set("DLSSNR.MVecSubrectHeight", m_inH);
}

bool DLSSWorker::createNRFeature() {
    setCommonParams(true);
    auto shimCreate = reinterpret_cast<ShimCreateFn>(m_shimCreate);
    NGXResult cr = shimCreate(m_nrCreate, m_cmdList.Get(), NR_FEATURE_ID, m_params, &m_feature);
    if (cr != 1 || !m_feature) {
        m_lastNgx = static_cast<uint32_t>(cr);
        char buf[96];
        snprintf(buf, sizeof(buf), "CreateFeature(18: DLSS-NR) failed (code=0x%08X)", static_cast<unsigned>(cr));
        m_lastError = buf;
        m_feature = nullptr;
        return false;
    }
    m_createdPreset = m_hdr.dlss_model_preset;
    return true;
}

void DLSSWorker::releaseNRFeature() {
    if (m_feature && m_nrRelease && m_shimRelease) {
        reinterpret_cast<ShimReleaseFn>(m_shimRelease)(m_nrRelease, m_feature);
    }
    m_feature = nullptr;
}

bool DLSSWorker::applyLiveSettings(const LiveSettings& s) {
    // Measured on driver 596.36: style, tone, structure, skin and mask are plain
    // evaluate-time parameters. The render-preset hint does not change Feature 18
    // output at all (it only steers Feature 1, which is created once), so no
    // re-create is needed here; the client restarts the worker for preset/upscale changes.
    m_hdr.dlss_model_preset = s.model_preset;
    m_hdr.style             = s.style;
    m_hdr.auto_mask         = s.auto_mask;
    m_hdr.preset            = s.nr_preset;
    m_hdr.intensity         = s.intensity;
    m_hdr.local_tone        = s.local_tone;
    m_hdr.local_structure   = s.local_structure;
    m_hdr.skin_structure    = s.skin_structure;
    setCommonParams(false);
    return true;
}

bool DLSSWorker::initNGX(const VideoHeader& hdr, const RuntimePaths& paths) {
    wchar_t exePath[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring runtimeDir = exePath;
    size_t lastSlash = runtimeDir.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos) runtimeDir = runtimeDir.substr(0, lastSlash);

    const std::wstring nrDir = paths.nrDir.empty() ? runtimeDir : paths.nrDir;
    const std::wstring srDir = paths.srDir.empty() ? nrDir : paths.srDir;

    m_coreMod = LoadCoreNGX(runtimeDir);
    if (!m_coreMod) {
        m_lastError = "Could not load NVIDIA NGX core _nvngx.dll (is the NVIDIA driver installed?)";
        return false;
    }

    std::wstring nrPath = nrDir + L"\\nvngx_dlssnr.dll";
    if (!FileExists(nrPath)) {
        char buf[MAX_PATH + 64];
        snprintf(buf, sizeof(buf), "nvngx_dlssnr.dll not found in %ls", nrDir.c_str());
        m_lastError = buf;
        return false;
    }
    m_nrMod = LoadLibraryW(nrPath.c_str());
    if (!m_nrMod) {
        m_lastError = "LoadLibrary(nvngx_dlssnr.dll) failed";
        return false;
    }

    std::wstring shimPath = runtimeDir + L"\\nvngx.dll";
    if (!FileExists(shimPath)) {
        m_lastError = "Caller shim nvngx.dll not found next to the worker";
        return false;
    }
    m_shimMod = LoadLibraryW(shimPath.c_str());
    if (!m_shimMod) {
        m_lastError = "LoadLibrary(nvngx.dll shim) failed";
        return false;
    }

    auto coreInitExt     = reinterpret_cast<InitExtFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_Init_Ext"));
    auto coreInitProject = reinterpret_cast<InitProjectIdFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_Init_ProjectID"));
    auto allocParams     = reinterpret_cast<AllocParamsFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_AllocateParameters"));

    auto nrInit  = reinterpret_cast<SnippetInitFn>(GetProcAddress(m_nrMod, "NVSDK_NGX_D3D12_Init_Ext"));
    m_nrCreate   = reinterpret_cast<void*>(GetProcAddress(m_nrMod, "NVSDK_NGX_D3D12_CreateFeature"));
    m_nrEval     = reinterpret_cast<void*>(GetProcAddress(m_nrMod, "NVSDK_NGX_D3D12_EvaluateFeature"));
    m_nrRelease  = reinterpret_cast<void*>(GetProcAddress(m_nrMod, "NVSDK_NGX_D3D12_ReleaseFeature"));

    auto shimInit  = reinterpret_cast<ShimInitFn>(GetProcAddress(m_shimMod, "DLSSNR_CallInit"));
    m_shimCreate   = reinterpret_cast<void*>(GetProcAddress(m_shimMod, "DLSSNR_CallCreate"));
    m_shimEval     = reinterpret_cast<void*>(GetProcAddress(m_shimMod, "DLSSNR_CallEvaluate"));
    m_shimRelease  = reinterpret_cast<void*>(GetProcAddress(m_shimMod, "DLSSNR_CallRelease"));

    if (!allocParams || !nrInit || !m_nrCreate || !m_nrEval || !m_nrRelease || !shimInit || !m_shimCreate || !m_shimEval || !m_shimRelease) {
        m_lastError = "Failed to resolve required NGX / Snippet / Shim entry points";
        return false;
    }

    // Snippet search paths for the core (nvngx_dlss.dll for Feature 1).
    const wchar_t* pathList[3] = { srDir.c_str(), nrDir.c_str(), runtimeDir.c_str() };
    NGXFeatureCommonInfo fci{};
    fci.PathListInfo.Path = pathList;
    fci.PathListInfo.Length = 3;
    fci.LoggingInfo.LoggingLevel = NGX_LOG_OFF;
    fci.LoggingInfo.DisableOtherLoggingSinks = true;

    bool coreOk = false;
    NGXResult lastCore = 0;
    if (coreInitProject) {
        for (int ver = 0x13; ver <= 0x20 && !coreOk; ++ver) {
            lastCore = coreInitProject(PROJECT_ID, 0, "1.0.0", runtimeDir.c_str(), m_device.Get(), ver, &fci);
            coreOk = (lastCore == 1);
        }
    }
    if (!coreOk && coreInitExt) {
        for (int ver = 0x13; ver <= 0x20 && !coreOk; ++ver) {
            lastCore = coreInitExt(APP_ID, runtimeDir.c_str(), m_device.Get(), ver, &fci);
            coreOk = (lastCore == 1);
        }
    }
    if (!coreOk) {
        m_lastNgx = static_cast<uint32_t>(lastCore);
        m_lastError = "NGX Core initialization failed";
        return false;
    }

    NGXResult sr = shimInit(reinterpret_cast<void*>(nrInit), APP_ID, runtimeDir.c_str(), m_device.Get(), 0x15, &fci);
    if (sr != 1) {
        m_lastNgx = static_cast<uint32_t>(sr);
        char buf[96];
        snprintf(buf, sizeof(buf), "DLSSNR snippet initialization via caller shim failed (code=0x%08X)", static_cast<unsigned>(sr));
        m_lastError = buf;
        return false;
    }

    if (allocParams(&m_params) != 1 || !m_params) {
        m_lastError = "AllocateParameters failed";
        return false;
    }

    if (!createNRFeature()) return false;

    if (m_needUpscale) {
        if (allocParams(&m_srParams) != 1 || !m_srParams) {
            m_lastError = "AllocateParameters for DLSS-SR failed";
            return false;
        }

        auto coreCreate = reinterpret_cast<CreateFeatureFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_CreateFeature"));
        if (!coreCreate) {
            m_lastError = "NVSDK_NGX_D3D12_CreateFeature not found in core";
            return false;
        }

        m_srParams->Set("Width", m_inW);
        m_srParams->Set("Height", m_inH);
        m_srParams->Set("OutWidth", m_outW);
        m_srParams->Set("OutHeight", m_outH);
        m_srParams->Set("PerfQualityValue", (int)m_hdr.perf_quality);
        m_srParams->Set("DLSS.Feature.Create.Flags", (unsigned int)(1 | 8)); // IsHDR | DepthInverted
        m_srParams->Set("DLSS.Hint.Render.Preset", (int)m_hdr.dlss_model_preset);

        NGXResult srCr = reinterpret_cast<ShimCreateFn>(m_shimCreate)(reinterpret_cast<void*>(coreCreate), m_cmdList.Get(), 1, m_srParams, &m_srFeature);
        if (srCr != 1 || !m_srFeature) {
            m_lastNgx = static_cast<uint32_t>(srCr);
            char buf[160];
            snprintf(buf, sizeof(buf), "CreateFeature(1: DLSS-SR) failed (code=0x%08X). nvngx_dlss.dll must be reachable (sr dir: %ls)",
                     static_cast<unsigned>(srCr), srDir.c_str());
            m_lastError = buf;
            return false;
        }
    }

    if (!executeAndWait()) return false;

    return true;
}

bool DLSSWorker::init(const VideoHeader& hdr, const RuntimePaths& paths) {
    m_live  = hdr.magic == MAGIC_VIDEO_LIVE;
    m_rgba8 = hdr.magic == MAGIC_VIDEO_RGBA8 || m_live;
    m_hdr = hdr;
    if (!setupD3D12()) { if (m_lastError.empty()) m_lastError = "D3D12 setup failed"; return false; }
    if (!allocateResources(hdr.input_width, hdr.input_height, hdr.output_width, hdr.output_height)) {
        m_lastError = "D3D12 resource allocation failed"; return false;
    }
    if (!initNGX(hdr, paths)) return false;

    m_setup.magic                = MAGIC_SETUP;
    m_setup.setup_ok             = 1;
    m_setup.setup_result         = 1;
    m_setup.render_width         = hdr.input_width;
    m_setup.render_height        = hdr.input_height;
    m_setup.output_width         = hdr.output_width;
    m_setup.output_height        = hdr.output_height;
    m_setup.min_width            = hdr.input_width;
    m_setup.min_height           = hdr.input_height;
    m_setup.max_width            = hdr.input_width;
    m_setup.max_height           = hdr.input_height;
    m_setup.applied_model_preset = hdr.dlss_model_preset;

    m_flow_width = (hdr.dis_flow_width > 0) ? (int)hdr.dis_flow_width : 640;
    if (m_flow_width > (int)hdr.input_width) m_flow_width = (int)hdr.input_width;
    float scale = (float)m_flow_width / (float)hdr.input_width;
    m_flow_height = std::max(16, (int)std::round(((float)hdr.input_height * scale) / 2.0f) * 2);
    m_prev_gray.clear();

    m_initialized = true;
    return true;
}

bool DLSSWorker::processFrame(
    uint32_t idx, bool reset, int64_t pts,
    const uint8_t* rgba_in, const uint8_t* motion_in,
    const float* depth_in, const float* control_mask_in,
    std::vector<uint8_t>& rgba_out)
{
    if (!m_initialized || !rgba_in) return false;
    const double t0 = NowMs();

    // 1. Upload RGBA8 (converted) or RGBA16F (copied) into the RGBA16F GPU texture.
    {
        void* mapped = nullptr;
        if (FAILED(m_uploadBuf->Map(0, nullptr, &mapped)) || !mapped) return false;
        auto* dstBase = static_cast<uint8_t*>(mapped);
        for (uint32_t y = 0; y < m_inH; ++y) {
            auto* dstRow = reinterpret_cast<uint16_t*>(dstBase + (size_t)y * m_colorPitch);
            if (m_rgba8) {
                U8ToHalf(rgba_in + (size_t)y * m_inW * 4, dstRow, (size_t)m_inW * 4);
            } else {
                memcpy(dstRow, rgba_in + (size_t)y * m_inW * 8, (size_t)m_inW * 8);
            }
        }
        m_uploadBuf->Unmap(0, nullptr);
    }

    // 2. Optical Flow / Motion Vector Processing (worker-side DIS)
    std::vector<float> flowU, flowV;
    if (m_hdr.mv_mode == 2) {
        const uint8_t* flowSrc = rgba_in;
        if (!m_rgba8) {
            m_flowScratch.resize((size_t)m_inW * m_inH * 4);
            HalfToU8Plain(reinterpret_cast<const uint16_t*>(rgba_in), m_flowScratch.data(), m_flowScratch.size());
            flowSrc = m_flowScratch.data();
        }
        std::vector<uint8_t> currGray((size_t)m_flow_width * m_flow_height);
        DISOpticalFlow::downscaleRgbaToGray(flowSrc, m_inW, m_inH, currGray.data(), m_flow_width, m_flow_height);

        if (m_prev_gray.empty()) {
            reset = true;
        } else if (!reset) {
            DISParams params;
            params.flow_width = m_flow_width;
            params.iterations = m_hdr.dis_iterations > 0 ? (int)m_hdr.dis_iterations : 25;

            switch (m_hdr.dis_preset) {
                case 0: params.finest_scale = 2; params.patch_stride = 4; break; // Fast Preview
                case 1: params.finest_scale = 1; params.patch_stride = 4; break; // Balanced
                case 2: params.finest_scale = 0; params.patch_stride = 2; break; // High Quality
                case 3: params.finest_scale = 0; params.patch_stride = 1; break; // Extreme
                default: params.finest_scale = 1; params.patch_stride = 4; break;
            }

            m_dis.compute(currGray.data(), m_prev_gray.data(),
                          m_flow_width, m_flow_height,
                          m_inW, m_inH,
                          params, flowU, flowV);
        }
        m_prev_gray = std::move(currGray);
    }

    // 3. Upload Motion Vectors (RG16F)
    {
        void* mapped = nullptr;
        if (FAILED(m_mvUploadBuf->Map(0, nullptr, &mapped)) || !mapped) return false;
        auto* dstBase = static_cast<uint8_t*>(mapped);

        if (m_hdr.mv_mode == 2 && !flowU.empty() && !reset) {
            const __m256 one = _mm256_set1_ps(1.0f);
            for (uint32_t y = 0; y < m_inH; ++y) {
                auto* dstRow = reinterpret_cast<uint16_t*>(dstBase + (size_t)y * m_mvPitch);
                const float* u = flowU.data() + (size_t)y * m_inW;
                const float* v = flowV.data() + (size_t)y * m_inW;
                uint32_t x = 0;
                for (; x + 4 <= m_inW; x += 4) {
                    // interleave u0 v0 u1 v1 u2 v2 u3 v3
                    __m128 uu = _mm_loadu_ps(u + x), vv = _mm_loadu_ps(v + x);
                    __m128 lo = _mm_unpacklo_ps(uu, vv), hi = _mm_unpackhi_ps(uu, vv);
                    __m256 f = _mm256_mul_ps(_mm256_set_m128(hi, lo), one);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstRow + x * 2), _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
                }
                for (; x < m_inW; ++x) {
                    dstRow[x * 2 + 0] = FloatToHalf(u[x]);
                    dstRow[x * 2 + 1] = FloatToHalf(v[x]);
                }
            }
        } else if (m_hdr.mv_mode == 1 && motion_in && !reset) {
            for (uint32_t y = 0; y < m_inH; ++y) {
                memcpy(dstBase + (size_t)y * m_mvPitch, motion_in + (size_t)y * m_inW * 4, m_inW * 4);
            }
        } else {
            for (uint32_t y = 0; y < m_inH; ++y) {
                memset(dstBase + (size_t)y * m_mvPitch, 0, m_inW * 4);
            }
        }
        m_mvUploadBuf->Unmap(0, nullptr);
    }

    // 3.5. Optional CG guides
    if (depth_in) {
        void* mapped = nullptr;
        if (FAILED(m_depthUploadBuf->Map(0, nullptr, &mapped)) || !mapped) return false;
        auto* dstBase = static_cast<uint8_t*>(mapped);
        for (uint32_t y = 0; y < m_inH; ++y)
            memcpy(dstBase + (size_t)y * m_guidePitch, depth_in + (size_t)y * m_inW, (size_t)m_inW * sizeof(float));
        m_depthUploadBuf->Unmap(0, nullptr);
    }
    if (control_mask_in) {
        void* mapped = nullptr;
        if (FAILED(m_controlMaskUploadBuf->Map(0, nullptr, &mapped)) || !mapped) return false;
        auto* dstBase = static_cast<uint8_t*>(mapped);
        for (uint32_t y = 0; y < m_inH; ++y)
            memcpy(dstBase + (size_t)y * m_guidePitch, control_mask_in + (size_t)y * m_inW, (size_t)m_inW * sizeof(float));
        m_controlMaskUploadBuf->Unmap(0, nullptr);
    }

    // 4. Copy staging -> textures
    std::vector<D3D12_RESOURCE_BARRIER> preBarriers;
    preBarriers.push_back(Barrier(m_colorTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
    preBarriers.push_back(Barrier(m_mvTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
    if (depth_in) preBarriers.push_back(Barrier(m_depthTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
    if (control_mask_in) preBarriers.push_back(Barrier(m_controlMaskTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
    m_cmdList->ResourceBarrier((UINT)preBarriers.size(), preBarriers.data());

    auto copyTo = [&](ID3D12Resource* tex, ID3D12Resource* staging, DXGI_FORMAT fmt, UINT pitch) {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = staging; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint.Format = fmt;
        src.PlacedFootprint.Footprint.Width = m_inW; src.PlacedFootprint.Footprint.Height = m_inH;
        src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = pitch;
        m_cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    };
    copyTo(m_colorTex.Get(), m_uploadBuf.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, m_colorPitch);
    copyTo(m_mvTex.Get(), m_mvUploadBuf.Get(), DXGI_FORMAT_R16G16_FLOAT, m_mvPitch);
    if (depth_in) copyTo(m_depthTex.Get(), m_depthUploadBuf.Get(), DXGI_FORMAT_R32_FLOAT, m_guidePitch);
    if (control_mask_in) copyTo(m_controlMaskTex.Get(), m_controlMaskUploadBuf.Get(), DXGI_FORMAT_R32_FLOAT, m_guidePitch);

    std::vector<D3D12_RESOURCE_BARRIER> postBarriers;
    postBarriers.push_back(Barrier(m_colorTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    postBarriers.push_back(Barrier(m_mvTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    if (depth_in) postBarriers.push_back(Barrier(m_depthTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    if (control_mask_in) postBarriers.push_back(Barrier(m_controlMaskTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    m_cmdList->ResourceBarrier((UINT)postBarriers.size(), postBarriers.data());

    // 5. Evaluate Feature 18 (DLSS-NR)
    m_params->Set("DLSSNR.Reset", reset ? 1 : 0);
    if (depth_in) {
        m_params->Set("DLSSNR.Depth", m_depthTex.Get());
        m_params->Set("DLSSNR.DepthSubrectBaseX", 0);
        m_params->Set("DLSSNR.DepthSubrectBaseY", 0);
        m_params->Set("DLSSNR.DepthSubrectWidth", m_inW);
        m_params->Set("DLSSNR.DepthSubrectHeight", m_inH);
    } else {
        m_params->Set("DLSSNR.Depth", (ID3D12Resource*)nullptr);
    }
    if (control_mask_in) {
        m_params->Set("DLSSNR.ControlMask", m_controlMaskTex.Get());
        m_params->Set("DLSSNR.ControlMaskSubrectBaseX", 0);
        m_params->Set("DLSSNR.ControlMaskSubrectBaseY", 0);
        m_params->Set("DLSSNR.ControlMaskSubrectWidth", m_inW);
        m_params->Set("DLSSNR.ControlMaskSubrectHeight", m_inH);
    } else {
        m_params->Set("DLSSNR.ControlMask", (ID3D12Resource*)nullptr);
    }

    auto shimEval = reinterpret_cast<ShimEvaluateFn>(m_shimEval);
    NGXResult er = shimEval(m_nrEval, m_cmdList.Get(), m_feature, m_params, nullptr);
    if (er != 1) {
        m_lastNgx = static_cast<uint32_t>(er);
        fprintf(stderr, "[DLSS5Worker] Feature 18 (NR) Evaluate failed: 0x%08X\n", static_cast<unsigned>(er));
        fflush(stderr);
        return false;
    }

    // 5.5. Feature 1 (DLSS-SR) when upscaling
    if (m_needUpscale && m_srFeature && m_srParams) {
        auto bInter1 = Barrier(m_intermediateTex.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_cmdList->ResourceBarrier(1, &bInter1);

        m_srParams->Set("Color", m_intermediateTex.Get());
        m_srParams->Set("Output", m_outputTex.Get());
        m_srParams->Set("MotionVectors", m_mvTex.Get());
        m_srParams->Set("Depth", m_depthTex.Get());
        m_srParams->Set("Reset", reset ? 1 : 0);
        m_srParams->Set("Jitter.Offset.X", 0.0f);
        m_srParams->Set("Jitter.Offset.Y", 0.0f);
        m_srParams->Set("MV.Scale.X", 1.0f);
        m_srParams->Set("MV.Scale.Y", 1.0f);
        m_srParams->Set("DLSS.Render.Subrect.Dimensions.Width", (unsigned int)m_inW);
        m_srParams->Set("DLSS.Render.Subrect.Dimensions.Height", (unsigned int)m_inH);
        m_srParams->Set("DLSS.Pre.Exposure", 1.0f);
        m_srParams->Set("DLSS.Exposure.Scale", 1.0f);

        auto coreEval = reinterpret_cast<EvaluateFeatureFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_EvaluateFeature"));
        if (!coreEval) return false;

        NGXResult srEr = shimEval(reinterpret_cast<void*>(coreEval), m_cmdList.Get(), m_srFeature, m_srParams, nullptr);
        if (srEr != 1) {
            m_lastNgx = static_cast<uint32_t>(srEr);
            fprintf(stderr, "[DLSS5Worker] Feature 1 (SR) Evaluate failed: 0x%08X\n", static_cast<unsigned>(srEr));
            fflush(stderr);
            return false;
        }

        auto bInter2 = Barrier(m_intermediateTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_cmdList->ResourceBarrier(1, &bInter2);
    }

    // 6. Readback
    auto b5 = Barrier(m_outputTex.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_cmdList->ResourceBarrier(1, &b5);

    D3D12_TEXTURE_COPY_LOCATION dstRb{}, srcOut{};
    dstRb.pResource = m_readbackBuf.Get(); dstRb.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstRb.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    dstRb.PlacedFootprint.Footprint.Width = m_outW; dstRb.PlacedFootprint.Footprint.Height = m_outH;
    dstRb.PlacedFootprint.Footprint.Depth = 1; dstRb.PlacedFootprint.Footprint.RowPitch = m_readbackPitch;
    srcOut.pResource = m_outputTex.Get(); srcOut.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    m_cmdList->CopyTextureRegion(&dstRb, 0, 0, 0, &srcOut, nullptr);

    auto b6 = Barrier(m_outputTex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_cmdList->ResourceBarrier(1, &b6);

    if (!executeAndWait()) return false;

    // 7. Map readback & convert
    void* rmap = nullptr;
    if (FAILED(m_readbackBuf->Map(0, nullptr, &rmap)) || !rmap) return false;
    const auto* base = static_cast<const uint8_t*>(rmap);

    rgba_out.resize((size_t)m_outW * m_outH * 4 * (m_rgba8 ? 1 : 2));

    // Channel-order guard: compare a 32x32 patch against the source to detect a swapped R/B readback.
    double sumR = 0, sumB = 0, srcSumR = 0, srcSumB = 0;
    for (uint32_t y = 0; y < std::min(m_inH, 32u); ++y) {
        const uint16_t* sRow16 = m_rgba8 ? nullptr : reinterpret_cast<const uint16_t*>(rgba_in + (size_t)y * m_inW * 8);
        const uint8_t* sRow8 = m_rgba8 ? rgba_in + (size_t)y * m_inW * 4 : nullptr;
        const auto* rRow = reinterpret_cast<const uint16_t*>(base + (size_t)y * m_readbackPitch);
        for (uint32_t x = 0; x < std::min(m_inW, 32u); ++x) {
            srcSumR += m_rgba8 ? sRow8[x * 4 + 0] / 255.0 : HalfToFloat(sRow16[x * 4 + 0]);
            srcSumB += m_rgba8 ? sRow8[x * 4 + 2] / 255.0 : HalfToFloat(sRow16[x * 4 + 2]);
            sumR += HalfToFloat(rRow[x * 4 + 0]);
            sumB += HalfToFloat(rRow[x * 4 + 2]);
        }
    }
    bool swapRB = (std::abs(sumR - srcSumR) + std::abs(sumB - srcSumB)) >
                  (std::abs(sumB - srcSumR) + std::abs(sumR - srcSumB));

    for (uint32_t y = 0; y < m_outH; ++y) {
        const auto* row = reinterpret_cast<const uint16_t*>(base + (size_t)y * m_readbackPitch);
        if (m_rgba8) {
            HalfToU8(row, rgba_out.data() + (size_t)y * m_outW * 4, (size_t)m_outW * 4, swapRB);
        } else {
            uint16_t* dst16 = reinterpret_cast<uint16_t*>(rgba_out.data() + (size_t)y * m_outW * 8);
            if (!swapRB) {
                memcpy(dst16, row, (size_t)m_outW * 8);
            } else {
                for (uint32_t x = 0; x < m_outW; ++x) {
                    dst16[x * 4 + 0] = row[x * 4 + 2];
                    dst16[x * 4 + 1] = row[x * 4 + 1];
                    dst16[x * 4 + 2] = row[x * 4 + 0];
                    dst16[x * 4 + 3] = row[x * 4 + 3];
                }
            }
        }
    }
    m_readbackBuf->Unmap(0, nullptr);

    m_lastNgx = static_cast<uint32_t>(std::max(0.0, NowMs() - t0)); // ms spent in the worker
    return true;
}

void DLSSWorker::shutdown() {
    if (m_srFeature) {
        auto coreRelease = reinterpret_cast<ReleaseFeatureFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_ReleaseFeature"));
        if (coreRelease && m_shimRelease) reinterpret_cast<ShimReleaseFn>(m_shimRelease)(reinterpret_cast<void*>(coreRelease), m_srFeature);
        m_srFeature = nullptr;
    }
    if (m_srParams) {
        auto destroyParams = reinterpret_cast<DestroyParamsFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_DestroyParameters"));
        if (destroyParams) destroyParams(m_srParams);
        m_srParams = nullptr;
    }
    releaseNRFeature();
    if (m_coreMod) {
        auto coreShutdown = reinterpret_cast<ShutdownFn>(GetProcAddress(m_coreMod, "NVSDK_NGX_D3D12_Shutdown"));
        if (coreShutdown) coreShutdown();
    }
    m_params = nullptr;
    m_colorTex.Reset();
    m_intermediateTex.Reset();
    m_depthTex.Reset();
    m_controlMaskTex.Reset();
    m_outputTex.Reset();
    m_mvTex.Reset();
    m_uploadBuf.Reset();
    m_mvUploadBuf.Reset();
    m_depthUploadBuf.Reset();
    m_controlMaskUploadBuf.Reset();
    m_readbackBuf.Reset();
    m_cmdList.Reset();
    m_cmdAlloc.Reset();
    m_queue.Reset();
    m_fence.Reset();
    if (m_fenceEvent) { CloseHandle(m_fenceEvent); m_fenceEvent = nullptr; }
    m_device.Reset();

    if (m_shimMod) { FreeLibrary(m_shimMod); m_shimMod = nullptr; }
    if (m_nrMod) { FreeLibrary(m_nrMod); m_nrMod = nullptr; }
    if (m_coreMod) { FreeLibrary(m_coreMod); m_coreMod = nullptr; }

    m_initialized = false;
}
