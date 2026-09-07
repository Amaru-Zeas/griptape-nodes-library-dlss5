from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import runtime_installer  # noqa: E402

RUNTIME_DIR_SETTING = "dlss5.runtime_dir"

INSTRUCTIONS = """# DLSS 5 Neural Render - Quick Start

Run a video through NVIDIA **DLSS 5 Neural Rendering**: the same AI pass that makes
game skin, hair, fabric and lighting look photoreal, applied to your generated or
filmed clips. Optional 1.5x-3x **DLSS Super Resolution** on top. Windows + RTX only
(RTX 40 / 50 / Blackwell PRO recommended).

**0. The easy way: press the button at the top of this node**
*Download & install the DLSS 5 runtime* does steps 1 and 2 for you: it fetches the
latest DLSS 5 Visual Enhancer release from Merserk's GitHub (~520 MB, one time),
verifies the SHA-256 GitHub publishes for it, unpacks it into
`%LOCALAPPDATA%\\griptape_nodes\\dlss5` and sets the library setting `runtime_dir`.
Progress shows in `runtime_status`. When it says *Done*, skip to step 3.
(NVIDIA's runtimes cannot be shipped inside this library, so the download is yours -
it only ever talks to github.com.)

**1. Get the DLSS 5 runtime (one time, by hand)**
Download the latest release zip of Merserk's *DLSS 5 Visual Enhancer*
([github.com/Merserk/dlss5-visual-enhancer/releases](https://github.com/Merserk/dlss5-visual-enhancer/releases))
and unzip it anywhere. Use the **release zip**, not a git clone - the clone has no
worker binaries. v7.0 or newer is recommended.

**2. Tell the library where it is (one time)**
Settings -> `dlss5` -> `runtime_dir` = the unzipped folder (the one containing
`start.bat`) or its `bin\\runtime` sub-folder. Restart the app once so the setting is
picked up. You can also paste the path into a node's `runtime_dir` field instead.

**3. Wire it up**
Two nodes, same controls: **DLSS 5 Live Preview** to find the look in real time (see
below), **DLSS 5 Neural Render (Video)** for straight batch renders.
Load Video -> **DLSS 5 Neural Render (Video)** -> Display Video.
The `video` row at the top has the input port on the left and a pass-through output
port on the right (the untouched original); `output_video` sits right underneath.
Feed the `video` out-port and `output_video` into a Compare Video node for a
before/after.

**4. Run**
`mode` = *Render full video* processes every frame, writes the file and puts it on
`output_video`. The file goes where your project says: `output_file` (default
`dlss5.mp4`) is expanded through the project's `save_node_output` situation, i.e. the
project's **outputs** folder as `<node name>_dlss5.mp4`, numbered instead of
overwritten. Change the situation in `griptape-nodes-project.yml`, or press the cog on
`output_file` to attach a File Output Settings node. No Save Video node needed.

Renders larger than 3840x2160 cannot be hardware-decoded by the browser, so they stutter
in Display Video. With `browser_proxy` on (default) the node also writes a UHD-fitted
`<name>_proxy.mp4` and puts that on `output_video`; `output_file` is still the
full-resolution master.
Expect ~16-18 ms per 1080p frame at 1.0x (about 55-60 frames of footage per second,
with or without optical flow) and ~40 ms per frame at 2.0x. Each run adds ~4 s of
worker start-up. GPU usage looks low while rendering - that is normal: the neural pass
is only a few milliseconds of each frame, so watch frames per second, not GPU %.

**Quality presets (first control on the node)**
- `quality` = **Ultra** (default) / High / Medium / Low / Custom. One click sets
  `model_preset`, `intensity`, `local_tone`, `local_structure`, `skin_structure` and
  `auto_mask`. Ultra (M, 1, 1, 2, 2, mask on) is the combination measured to give the
  most convincing faces on generated footage; Low is a subtle touch-up.
- Moving any of those controls by hand flips `quality` to Custom.
- All presets render at the same speed - the neural pass is a fixed cost. Only
  `upscale_mode` changes render time.
- `pipeline` Single Frame - no temporal history, no ghosting. Use *Sequence* only
  on real footage with smooth, coherent motion.
- `upscale_mode` 1.0x - neural rendering only, same output size as the input.

**Dialing it in**
- `intensity` below 1 blends back toward the source (above 1 does nothing on
  current runtimes).
- `local_structure` = fine detail: AO, reflections, materials. `local_tone` =
  lighting and colour response. `skin_structure` = pores and skin texture (needs
  `auto_mask` on).
- `nr_style` *Natural* stays closest to the source; *Cinematic* deepens shadows.
- To test slider values quickly, set `mode` to *Test one frame only*, pick
  `preview_frame` (Advanced) and connect a Display Image node to `preview_image`.
  A single frame takes a few seconds instead of a full render.

**Live Preview node - dial the look in real time**
**DLSS 5 Live Preview** is the second node. Instead of rendering and waiting, it loops
your clip through a resident DLSS 5 worker and shows the result *inside the node* while
you move the sliders. Use it to find the settings, then bake.

- *Start live preview*: first frame after ~1.5 s (D3D12 + NGX start-up), then the clip
  plays at its own frame rate while the rest of it decodes into RAM in the background.
  Only the native worker is used here and it loads NVIDIA's DLLs straight out of the
  Visual Enhancer folder from step 2 - nothing to copy.
- The picture is a **before / after wipe**: source on the left, DLSS 5 on the right.
  Drag anywhere on it to move the split. The view menu switches to DLSS-only,
  source-only or side-by-side.
- Transport: play/pause (space), step frame (arrow keys), scrubber, and a **speed**
  menu: 0.25x-2x of the clip's frame rate, or *Max* = as fast as the pipeline goes
  (~100 fps at 1080p, ~37 fps at 4K on an RTX PRO 6000).
- **Full screen**: the button at the end of the bar, or double-click the picture. Wipe,
  scrubber and all controls work there too; Esc / Close to leave.
- The HUD shows `fps · ms GPU · ms/frame · ms preview`. GPU % in Task Manager stays
  low by design - the neural pass is a few milliseconds per frame - so judge speed by
  the fps here, not by utilisation.
- Sliders: `look` presets (Ultra default) set `local_tone`, `local_structure`,
  `skin_structure`, `auto_mask`; those plus `nr_style` take effect on the very next
  frame. `upscale_mode`, `temporal`, `model_preset` and `dis_preset` swap the worker
  in ~1.5 s while the old one keeps playing.
- Advanced: `max_cache_gb` (default 8 GB = ~55 s of 1080p or ~13 s of 4K; longer clips
  loop their first part - raise it if you have the RAM), `preview_width` (the stream is
  downscaled to this for the widget only; the worker and the bake stay full-res),
  `preview_quality`, `keep_audio`, `runtime_dir`.
- *Bake full clip with these settings* (or run the node in a flow) renders every frame
  with the current settings, saves it through the project's outputs as `output_file`
  (default `dlss5_live.mp4`) and puts it on `output_video`. Playback pauses during
  the bake and resumes after.
- *Stop live preview* frees the GPU. The loop also stops itself when the node is
  deleted or the engine exits, and pauses after 20 s with nobody watching.

**Verification**
`verify_neural_rendering` (on by default) checks the worker's `ReShade.log` after
the run and fails if it cannot prove that signed Feature 18 (Neural Rendering)
executed - so plain upscaling can never silently pass as neural rendering.
The `report` ends with `NR verified` when this check passed. Open the **Logs**
group for the full worker output.

**Troubleshooting**
- *No usable DLSS 5 runtime*: `runtime_dir` is empty, points at a git clone, or the
  app was not restarted after changing the library setting. Paste the release
  folder path straight into the node's `runtime_dir` field to check.
- *NR upscaling fell back to native* in the logs at 2x/3x is fine: DLSS upscaled
  first, then the neural pass ran at output resolution.
- Ghosting or smearing in *Sequence* mode: switch back to *Single Frame* or raise
  `scene_change_threshold` so cuts reset the temporal history.
"""


class DLSS5InstructionsNode(BaseNode):
    """Read-me note: how to set up and use the DLSS 5 Neural Rendering library."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self._install_thread: threading.Thread | None = None
        # Default to a size that shows the whole guide without scrolling.
        if "size" not in self.metadata:
            self.metadata["size"] = {"width": 680, "height": 1500}
        self.add_node_element(
            ParameterButton(
                name="install_runtime",
                label="Download & install the DLSS 5 runtime (~520 MB, one time)",
                icon="download",
                variant="default",
                full_width=True,
                tooltip=(
                    "Does steps 1 and 2 for you: downloads the latest DLSS 5 Visual Enhancer release from Merserk's "
                    "GitHub, verifies its SHA-256, unpacks it under %LOCALAPPDATA%\\griptape_nodes\\dlss5 and sets the "
                    "library setting runtime_dir. Nothing is downloaded from anywhere but github.com."
                ),
                on_click=self._on_install_clicked,
            )
        )
        self.add_parameter(
            Parameter(
                name="runtime_status",
                output_type="str",
                default_value=self._current_status(),
                allowed_modes={ParameterMode.PROPERTY},
                settable=False,
                tooltip="Where the DLSS 5 runtime currently comes from, and the install log.",
                ui_options={"multiline": True, "placeholder_text": ""},
            )
        )
        self.add_parameter(
            ParameterString(
                name="note",
                default_value=INSTRUCTIONS,
                allow_input=False,
                allow_property=True,
                allow_output=False,
                multiline=True,
                markdown=True,
                is_full_width=True,
                tooltip="How to set up and use the DLSS 5 Neural Rendering library.",
            )
        )

    # -- runtime install -------------------------------------------------------------

    @staticmethod
    def _configured_runtime_dir() -> str:
        try:
            return str(GriptapeNodes.ConfigManager().get_config_value(RUNTIME_DIR_SETTING, default="") or "").strip()
        except Exception:  # noqa: BLE001 - no config manager outside the engine
            return ""

    def _current_status(self) -> str:
        configured = self._configured_runtime_dir()
        if configured:
            return f"runtime_dir setting: {configured}"
        installed = runtime_installer.find_installed()
        if installed is not None:
            return f"Runtime found at {installed} but the runtime_dir setting is empty - press the button to link it."
        return "No DLSS 5 runtime configured yet. Press the button above, or do steps 1-2 below by hand."

    def _set_status(self, text: str) -> None:
        self.set_parameter_value("runtime_status", text)
        with contextlib.suppress(Exception):
            self.publish_update_to_parameter("runtime_status", text)

    def _on_install_clicked(self, *_args: Any) -> None:
        if self._install_thread is not None and self._install_thread.is_alive():
            return
        lines: list[str] = []

        def log(message: str) -> None:
            lines.append(message)
            self._set_status("".join(lines))

        def run() -> None:
            try:
                folder = runtime_installer.install(log)
                GriptapeNodes.ConfigManager().set_config_value(RUNTIME_DIR_SETTING, str(folder))
                log(f"\nLibrary setting {RUNTIME_DIR_SETTING} = {folder}\nDone - both DLSS 5 nodes are ready to use.\n")
            except runtime_installer.RuntimeInstallError as exc:
                log(f"\nFAILED: {exc}\n")
            except Exception as exc:  # noqa: BLE001
                log(f"\nFAILED: {type(exc).__name__}: {exc}\n")

        self._install_thread = threading.Thread(target=run, name="dlss5-runtime-install", daemon=True)
        self._install_thread.start()

    def process(self) -> None:
        pass
