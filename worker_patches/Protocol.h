#pragma once
#include <cstdint>

// Wire protocol between the Python bridge and DLSS5Worker.exe. Everything is
// little-endian and #pragma pack(1); the Python side mirrors it with struct.Struct.
//
// Modes (selected by VideoHeader.magic):
//   'D5V4' 0x34563544  legacy: RGBA8 in/out, settings fixed for the process lifetime
//   'D5V5' 0x35563544  Nuke:   linear RGBA16F in/out, settings fixed
//   'D5L1' 0x314C3544  live:   RGBA8 in/out, every frame carries a LiveSettings block
//                              ('FRM2' header) that is applied before the evaluate.
//                              The process stays resident until stdin closes.

#pragma pack(push, 1)

static constexpr uint32_t MAGIC_VIDEO_RGBA8   = 0x34563544u; // 'D5V4'
static constexpr uint32_t MAGIC_VIDEO_RGBA16F = 0x35563544u; // 'D5V5'
static constexpr uint32_t MAGIC_VIDEO_LIVE    = 0x314C3544u; // 'D5L1'
static constexpr uint32_t MAGIC_SETUP         = 0x34505553u; // 'SUP4'
static constexpr uint32_t MAGIC_FRAME         = 0x314D5246u; // 'FRM1'
static constexpr uint32_t MAGIC_FRAME_LIVE    = 0x324D5246u; // 'FRM2'
static constexpr uint32_t MAGIC_OUT           = 0x3154554Fu; // 'OUT1'

struct VideoHeader {
    // 'D5V4': legacy RGBA8; 'D5V5': linear RGBA16F; 'D5L1': live RGBA8. Must match src/WorkerBridge.h.
    uint32_t magic        = MAGIC_VIDEO_RGBA16F;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t warmup_frames;
    uint32_t frame_count  = 100000;
    uint32_t perf_quality;       // 0=MaxPerf, 1=Balanced, 2=MaxQuality, 3=UltraPerf, 5=DLAA
    uint32_t dlss_model_preset;  // 0=Default, 10=J, 11=K, 12=L, 13=M
    uint32_t profile      = 0;
    uint32_t preset       = 0;
    uint32_t style        = 0;
    uint32_t auto_mask    = 0;
    uint32_t ui_correction = 0;
    float    intensity;
    float    local_tone;
    float    local_structure;
    float    skin_structure;
    uint32_t mv_mode;             // 0=None, 1=External Buffer, 2=Auto DIS
    uint32_t dis_preset;          // 0=Fast, 1=Balanced, 2=High, 3=Extreme, 4=Custom
    uint32_t dis_flow_width;      // e.g. 480, 640, 960, 1280
    uint32_t dis_iterations;      // e.g. 12, 25, 32, 48
    uint32_t _reserved_scene_cut = 0; // Unused (scene cut removed)
    float    _reserved_thresh    = 0.0f;
};

struct SetupResponse {
    uint32_t magic;          // 'SUP4'
    uint32_t setup_ok;       // 1 = success
    uint32_t setup_result;   // NGX result code
    uint32_t render_width;
    uint32_t render_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t min_width;
    uint32_t min_height;
    uint32_t max_width;
    uint32_t max_height;
    uint32_t applied_model_preset;
};

struct FrameHeader {
    uint32_t magic  = MAGIC_FRAME; // 'FRM1' (fixed settings) or 'FRM2' (LiveSettings follows)
    uint32_t index;
    uint32_t reset;
    uint32_t guide_flags = 0;
    int64_t  pts;
};

// Live-mode per-frame settings (32 bytes). Follows the FrameHeader when magic == 'FRM2'.
// Everything here is applied before the evaluate of that frame as plain NGX
// parameters. Measured: style/tone/structure/skin/mask take effect on the next
// frame; the render-preset hint does not alter Feature 18 output; intensity < 1
// only darkens the image (keep it at 1.0).
struct LiveSettings {
    uint32_t model_preset;   // DLSSNR.Hint.Render.Preset (0=Default, 10..13 = J..M)
    uint32_t style;          // DLSSNR.Style
    uint32_t auto_mask;      // DLSSNR.UseAutoMask
    uint32_t nr_preset;      // DLSSNR profile preset (inert on current runtimes)
    float    intensity;      // DLSSNR.Intensity
    float    local_tone;     // DLSSNR.LocalToneStrength
    float    local_structure;// DLSSNR.LocalStructureStrength
    float    skin_structure; // DLSSNR.SkinStructureStrength
};

enum FrameGuideFlags : uint32_t {
    GUIDE_DEPTH        = 1u << 0,
    GUIDE_CONTROL_MASK = 1u << 1
};

struct FrameResponse {
    uint32_t magic;      // 'OUT1'
    uint32_t out_index;
    uint32_t ok;         // 1 = success
    uint32_t byte_count;
    uint32_t ngx_result; // NGX result of the failing call, or milliseconds spent in the worker on success
    int64_t  out_pts;
};

#pragma pack(pop)

static_assert(sizeof(VideoHeader) == 96, "VideoHeader must be 96 bytes");
static_assert(sizeof(SetupResponse) == 48, "SetupResponse must be 48 bytes");
static_assert(sizeof(FrameHeader) == 24, "FrameHeader must be 24 bytes");
static_assert(sizeof(LiveSettings) == 32, "LiveSettings must be 32 bytes");
static_assert(sizeof(FrameResponse) == 28, "FrameResponse must be 28 bytes");
