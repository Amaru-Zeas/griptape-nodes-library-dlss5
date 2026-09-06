// DLSS5Worker.exe entry point: binary stdin/stdout protocol (see Protocol.h).
//
//   DLSS5Worker.exe [--nr-dir <folder with nvngx_dlssnr.dll>] [--sr-dir <folder with nvngx_dlss.dll>]
//
// Without arguments both DLLs are expected next to the exe (Nuke layout).
#include "Protocol.h"
#include "DLSSWorker.h"
#include <windows.h>
#include <intrin.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- Binary pipe I/O ----
static bool readAll(HANDLE h, void* buf, DWORD n) {
    DWORD total = 0;
    while (total < n) {
        DWORD got = 0;
        if (!ReadFile(h, (char*)buf + total, n - total, &got, nullptr) || got == 0)
            return false;
        total += got;
    }
    return true;
}

static bool writeAll(HANDLE h, const void* buf, DWORD n) {
    DWORD total = 0;
    while (total < n) {
        DWORD wrote = 0;
        if (!WriteFile(h, (const char*)buf + total, n - total, &wrote, nullptr) || wrote == 0)
            return false;
        total += wrote;
    }
    return true;
}

static SetupResponse makeFailSetup(uint32_t resultCode = 0) {
    SetupResponse r = {};
    r.magic        = MAGIC_SETUP;
    r.setup_ok     = 0;
    r.setup_result = resultCode;
    return r;
}

// stderr as captured at startup. NGX may redirect the CRT's stderr for its own
// logger, so diagnostics go straight to the original pipe handle.
static HANDLE g_err = INVALID_HANDLE_VALUE;
static void logErr(const char* msg) {
    if (g_err == INVALID_HANDLE_VALUE) return;
    DWORD wrote = 0;
    WriteFile(g_err, msg, (DWORD)strlen(msg), &wrote, nullptr);
}

static DWORD getParentPid() {
    // PROCESS_BASIC_INFORMATION via NtQueryInformationProcess (no toolhelp needed)
    struct PBI { PVOID Reserved1; PVOID PebBaseAddress; PVOID Reserved2[2]; ULONG_PTR UniqueProcessId; ULONG_PTR InheritedFromUniqueProcessId; };
    using NtQIP = LONG(WINAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    if (!nt) return 0;
    auto fn = reinterpret_cast<NtQIP>(GetProcAddress(nt, "NtQueryInformationProcess"));
    if (!fn) return 0;
    PBI pbi{};
    ULONG len = 0;
    if (fn(GetCurrentProcess(), 0 /*ProcessBasicInformation*/, &pbi, sizeof(pbi), &len) != 0) return 0;
    return static_cast<DWORD>(pbi.InheritedFromUniqueProcessId);
}

// Exit hard as soon as the parent (the Python bridge) is gone. Without this a
// wedged NGX teardown could leave a resident worker holding GPU memory.
static void watchParent() {
    DWORD ppid = getParentPid();
    if (!ppid) return;
    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, ppid);
    if (!parent) return;
    CreateThread(nullptr, 0, [](LPVOID h) -> DWORD {
        WaitForSingleObject(static_cast<HANDLE>(h), INFINITE);
        TerminateProcess(GetCurrentProcess(), 0);
        return 0;
    }, parent, 0, nullptr);
}

static bool cpuHasAvx2F16c() {
    int info[4] = {};
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool f16c = (info[2] & (1 << 29)) != 0;
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    __cpuidex(info, 7, 0);
    const bool avx2 = (info[1] & (1 << 5)) != 0;
    if (!(f16c && avx2 && osxsave)) return false;
    unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 6) == 6; // XMM + YMM state enabled by the OS
}

int wmain(int argc, wchar_t** argv) {
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    HANDLE hIn  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    g_err = GetStdHandle(STD_ERROR_HANDLE);
    watchParent();

    if (!cpuHasAvx2F16c()) {
        logErr("[DLSS5Worker] this build needs a CPU with AVX2 and F16C\n");
        return 1;
    }

    RuntimePaths paths;
    for (int i = 1; i + 1 < argc; ++i) {
        if (wcscmp(argv[i], L"--nr-dir") == 0) paths.nrDir = argv[++i];
        else if (wcscmp(argv[i], L"--sr-dir") == 0) paths.srDir = argv[++i];
    }

    // ---- 1. VideoHeader ----
    VideoHeader hdr = {};
    if (!readAll(hIn, &hdr, sizeof(hdr))) return 1;
    const bool live = hdr.magic == MAGIC_VIDEO_LIVE;
    if (hdr.magic != MAGIC_VIDEO_RGBA8 && hdr.magic != MAGIC_VIDEO_RGBA16F && !live) {
        char buf[96];
        snprintf(buf, sizeof(buf), "[DLSS5Worker] unknown VideoHeader magic 0x%08X\n", hdr.magic);
        logErr(buf);
        return 1;
    }

    // ---- 2. Init ----
    DLSSWorker worker;
    bool ok = worker.init(hdr, paths);
    if (!ok) {
        const std::string& err = worker.getLastError();
        std::string msg = "[DLSS5Worker] init failed: " + (err.empty() ? std::string("(no detail; D3D12 device or resource allocation failed)") : err) + "\n";
        logErr(msg.c_str());
    }

    // ---- 3. SetupResponse ----
    SetupResponse setup = ok ? worker.getSetup() : makeFailSetup(worker.getLastNgxResult());
    if (!writeAll(hOut, &setup, sizeof(setup))) return 1;
    if (!ok) return 1;

    // ---- 4. Frame loop (runs until stdin closes) ----
    const uint32_t pixelBytes  = (hdr.magic == MAGIC_VIDEO_RGBA16F) ? 8u : 4u;
    const uint32_t inBytes     = hdr.input_width  * hdr.input_height  * pixelBytes;
    const uint32_t outBytes    = hdr.output_width * hdr.output_height * pixelBytes;
    const uint32_t motionBytes = hdr.input_width  * hdr.input_height  * 2 * sizeof(uint16_t);

    std::vector<uint8_t> frameIn(inBytes);
    std::vector<uint8_t> motionIn(motionBytes);
    std::vector<float> depthIn((size_t)hdr.input_width * hdr.input_height);
    std::vector<float> controlMaskIn((size_t)hdr.input_width * hdr.input_height);
    std::vector<uint8_t> frameOut;

    while (true) {
        FrameHeader fhdr = {};
        if (!readAll(hIn, &fhdr, sizeof(fhdr))) break;
        if (fhdr.magic != MAGIC_FRAME && fhdr.magic != MAGIC_FRAME_LIVE) break;

        bool frameOk = true;
        if (fhdr.magic == MAGIC_FRAME_LIVE) {
            LiveSettings ls = {};
            if (!readAll(hIn, &ls, sizeof(ls))) break;
            if (!worker.applyLiveSettings(ls)) {
                logErr(("[DLSS5Worker] applying live settings failed: " + worker.getLastError() + "\n").c_str());
                frameOk = false;
            }
        }

        if (!readAll(hIn, frameIn.data(), inBytes)) break;

        uint8_t* motionPtr = nullptr;
        if (hdr.mv_mode == 1) {
            if (!readAll(hIn, motionIn.data(), motionBytes)) break;
            motionPtr = motionIn.data();
        }

        const float* depthPtr = nullptr;
        if (fhdr.guide_flags & GUIDE_DEPTH) {
            if (!readAll(hIn, depthIn.data(), depthIn.size() * sizeof(float))) break;
            depthPtr = depthIn.data();
        }

        const float* controlMaskPtr = nullptr;
        if (fhdr.guide_flags & GUIDE_CONTROL_MASK) {
            if (!readAll(hIn, controlMaskIn.data(), controlMaskIn.size() * sizeof(float))) break;
            controlMaskPtr = controlMaskIn.data();
        }

        if (frameOk) {
            frameOk = worker.processFrame(
                fhdr.index, fhdr.reset != 0, fhdr.pts,
                frameIn.data(), motionPtr, depthPtr, controlMaskPtr, frameOut);
        }

        FrameResponse fresp = {};
        fresp.magic      = MAGIC_OUT;
        fresp.out_index  = fhdr.index;
        fresp.ok         = frameOk ? 1u : 0u;
        fresp.byte_count = frameOk ? outBytes : 0u;
        fresp.ngx_result = worker.getLastNgxResult();
        fresp.out_pts    = fhdr.pts;

        if (!writeAll(hOut, &fresp, sizeof(fresp))) break;
        if (frameOk && !writeAll(hOut, frameOut.data(), outBytes)) break;
    }

    // stdin closed (or protocol error): leave immediately. NVSDK_NGX_D3D12_Shutdown
    // hangs on current drivers (measured: never returns), and the OS reclaims the
    // D3D12 device and GPU memory on process exit anyway, so nothing is gained by
    // running the destructors here.
    logErr("[DLSS5Worker] input closed, exiting\n");
    TerminateProcess(GetCurrentProcess(), 0);
    return 0;
}
