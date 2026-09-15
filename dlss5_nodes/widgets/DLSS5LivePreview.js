// DLSS 5 Live Video widget — landscape layout matching Live Image.
//
// Visual language: Shot Planner / Seedance chrome (#0c0e11, amber #d9c6a4 / #a58050, ui-monospace).
// Left: MJPEG preview + transport. Right: look controls.
// Sliders hit /cmd while dragging; onChange syncs the node on release.

const STATE_POLL_MS = 500;
const CMD_MS = 16;
const FULL_SCALE = 1.2;

const FONT = "ui-monospace,'Cascadia Code','JetBrains Mono',Consolas,monospace";
const C = {
  rootBg: "#0c0e11",
  rootBorder: "#21252c",
  text: "#c6cad1",
  label: "#8a929e",
  muted: "#6c7481",
  inputBg: "#0f1116",
  inputBorder: "#333a44",
  chipBg: "#161a20",
  panelBg: "#12151a",
  accent: "#d9c6a4",
  accentBorder: "#6b5836",
  accentBg: "#241d13",
  accentBar: "#a58050",
  stageBg: "#0a0c0f",
};

const LOOKS = [
  ["Ultra (max realism)", { local_tone: 1.0, local_structure: 2.0, skin_structure: 2.0, auto_mask: true }],
  ["High", { local_tone: 1.0, local_structure: 1.5, skin_structure: 1.5, auto_mask: true }],
  ["Medium", { local_tone: 0.8, local_structure: 1.0, skin_structure: 1.0, auto_mask: true }],
  ["Low (subtle)", { local_tone: 0.5, local_structure: 0.5, skin_structure: 0.5, auto_mask: true }],
  ["Custom (use the sliders)", null],
];
const STYLES = ["Default", "Natural", "Cinematic"];
const UPSCALE = [
  "1.0x (DLAA / native)",
  "1.5x (Quality)",
  "1.724x (Balanced)",
  "2.0x (Performance)",
  "3.0x (Ultra Performance)",
];
const PRESETS = ["Default", "J", "K", "L", "M"];
const TEMPORAL = [
  ["Single Frame (no temporal history)", "Single Frame (no temporal history)"],
  ["Sequence (temporal, optical flow in worker)", "Sequence (temporal)"],
];

const IMG_FULL = "display:block;width:100%;height:100%;object-fit:contain;pointer-events:none;";

function el(tag, css, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text !== undefined) e.textContent = text;
  return e;
}

function stopDrag(node) {
  ["pointerdown", "mousedown", "dblclick"].forEach((ev) =>
    node.addEventListener(ev, (e) => e.stopPropagation()),
  );
  return node;
}

function fieldLabel(text) {
  return el("div", `font:11px/1.2 ${FONT};color:${C.label};margin:0 0 4px;letter-spacing:0.01em;`, text);
}

function mkSelect(options, value, { full = true } = {}) {
  const s = stopDrag(
    el(
      "select",
      `${full ? "width:100%;" : ""}padding:5px 7px;border-radius:6px;border:1px solid ${C.inputBorder};` +
        `background:${C.inputBg};color:${C.text};font:11.5px/1.3 ${FONT};cursor:pointer;box-sizing:border-box;outline:none;`,
    ),
  );
  options.forEach((opt) => {
    const [v, t] = Array.isArray(opt) ? opt : [opt, opt];
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    s.appendChild(o);
  });
  if (value !== undefined) s.value = value;
  return s;
}

function mkSlider(min, max, step, value) {
  const row = el("div", "display:flex;align-items:center;gap:8px;");
  const input = stopDrag(el("input", `flex:1;min-width:0;cursor:pointer;accent-color:${C.accentBar};`));
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const num = el(
    "span",
    `font:11px/1 ${FONT};color:${C.muted};min-width:36px;text-align:right;`,
    Number(value).toFixed(2),
  );
  input.addEventListener("input", () => {
    num.textContent = Number(input.value).toFixed(2);
  });
  row.append(input, num);
  row._input = input;
  row._num = num;
  return row;
}

function setSlider(row, value) {
  row._input.value = String(value);
  row._num.textContent = Number(value).toFixed(2);
}

function mkCheck(label, checked) {
  const row = stopDrag(
    el(
      "label",
      `display:flex;align-items:center;gap:8px;font:11.5px/1.3 ${FONT};cursor:pointer;color:${C.text};`,
    ),
  );
  const input = el("input", `cursor:pointer;accent-color:${C.accentBar};`);
  input.type = "checkbox";
  input.checked = !!checked;
  row.append(input, el("span", "", label));
  row._input = input;
  return row;
}

function mkBtn(label, title, { accent = false, compact = false } = {}) {
  const idleBorder = accent ? C.accentBorder : C.inputBorder;
  const idleBg = accent ? C.accentBg : C.chipBg;
  const idleFg = accent ? C.accent : C.text;
  const pad = compact ? "padding:4px 9px;" : "padding:6px 10px;";
  const width = compact ? "" : "width:100%;";
  const b = stopDrag(
    el(
      "button",
      pad +
        `border-radius:6px;border:1px solid ${idleBorder};background:${idleBg};` +
        `color:${idleFg};font:11.5px/1.3 ${FONT};${accent ? "font-weight:600;" : ""}cursor:pointer;` +
        `${width}white-space:nowrap;` +
        `transition:background-color .12s ease,border-color .12s ease,color .12s ease,filter .12s ease;`,
      label,
    ),
  );
  if (title) b.title = title;
  b.addEventListener("mouseenter", () => {
    if (b.disabled) return;
    b.style.filter = "brightness(1.28)";
    b.style.borderColor = C.accentBorder;
    b.style.color = C.accent;
  });
  b.addEventListener("mouseleave", () => {
    b.style.filter = "";
    b.style.borderColor = idleBorder;
    b.style.color = idleFg;
  });
  return b;
}

export default function DLSS5LivePreview(container, props) {
  if (container._dlss5Live?.wrapper?.isConnected) {
    container._dlss5Live.update(props);
    return { cleanup: container._dlss5Live.cleanup, update: container._dlss5Live.update };
  }

  let onChange = props.onChange;
  let latestProps = props;
  let url = "";
  let pollTimer = null;
  let firstFrameTimer = null;
  let retryTimer = null;
  let wiping = false;
  let sliding = false;
  let scrubbing = false;
  let emitting = false;
  let lastEmitKey = "";
  let fullscreen = false;
  let cmdTimer = null;
  let pendingCmd = null;
  let liveStatus = "stopped";
  let lastState = null;
  let settings = {
    look: "Ultra (max realism)",
    local_tone: 1.0,
    local_structure: 2.0,
    skin_structure: 2.0,
    auto_mask: true,
    nr_style: "Default",
    upscale_mode: "1.0x (DLAA / native)",
    model_preset: "M",
    temporal: "Single Frame (no temporal history)",
    view: "wipe",
    wipe: 0.5,
  };

  const wrapper = el(
    "div",
    `display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:760px;box-sizing:border-box;padding:8px;` +
      `background:${C.rootBg};border:1px solid ${C.rootBorder};border-radius:10px;font-family:${FONT};color:${C.text};`,
  );
  wrapper.className = "nodrag nowheel dlss5-live-video-root";
  wrapper.tabIndex = 0;
  container.style.minHeight = "780px";
  container.style.height = "100%";

  const body = el(
    "div",
    "display:flex;flex-direction:row;gap:10px;flex:1 1 auto;min-height:740px;width:100%;box-sizing:border-box;",
  );
  const left = el(
    "div",
    "display:flex;flex-direction:column;gap:8px;flex:1 1 62%;min-width:220px;min-height:0;",
  );

  const stage = el(
    "div",
    `position:relative;flex:1 1 auto;min-height:280px;background:${C.stageBg};border:1px solid ${C.rootBorder};` +
      `border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:col-resize;` +
      `user-select:none;box-sizing:border-box;`,
  );
  const img = el("img", IMG_FULL);
  img.draggable = false;
  img.alt = "";
  img.style.display = "none";
  const placeholder = el(
    "div",
    `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;` +
      `padding:24px;color:${C.muted};font:12.5px/1.5 ${FONT};pointer-events:none;z-index:3;`,
    "Connect a video, then Start live video.",
  );
  const badgeL = el(
    "div",
    `position:absolute;left:8px;top:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.inputBorder};` +
      `background:${C.chipBg};color:${C.label};font:10.5px/1.4 ${FONT};pointer-events:none;z-index:2;`,
    "BEFORE",
  );
  const badgeR = el(
    "div",
    `position:absolute;right:8px;top:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.accentBorder};` +
      `background:${C.accentBg};color:${C.accent};font:10.5px/1.4 ${FONT};pointer-events:none;z-index:2;`,
    "DLSS 5",
  );
  const hud = el(
    "div",
    `position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.rootBorder};` +
      `background:rgba(12,14,17,.85);color:${C.muted};font:10.5px/1.4 ${FONT};pointer-events:none;white-space:pre;z-index:2;`,
    "",
  );
  stage.append(img, placeholder, badgeL, badgeR, hud);

  const bar = el(
    "div",
    `display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:6px 8px;flex:0 0 auto;` +
      `background:${C.panelBg};border:1px solid ${C.rootBorder};border-radius:8px;`,
  );
  const playBtn = mkBtn("Pause", "Play / pause (space)", { compact: true });
  const prevBtn = mkBtn("‹", "Previous frame", { compact: true });
  const nextBtn = mkBtn("›", "Next frame", { compact: true });
  const scrub = stopDrag(el("input", `flex:1;min-width:80px;cursor:pointer;accent-color:${C.accentBar};`));
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = "0";
  scrub.step = "1";
  scrub.value = "0";
  const frameLbl = el("span", `font:11px/1 ${FONT};color:${C.muted};min-width:70px;text-align:right;`, "0 / 0");
  const speedSel = mkSelect(
    [
      ["0.25", "0.25x"],
      ["0.5", "0.5x"],
      ["1", "1x"],
      ["2", "2x"],
      ["0", "Max"],
    ],
    "1",
    { full: false },
  );
  speedSel.title = "Playback speed relative to the clip's frame rate. Max = as fast as the pipeline goes.";
  bar.append(playBtn, prevBtn, nextBtn, scrub, frameLbl, speedSel);
  left.append(stage, bar);

  const panel = el(
    "div",
    `flex:0 0 280px;width:280px;max-width:42%;display:flex;flex-direction:column;gap:8px;` +
      `overflow:visible;padding:8px;box-sizing:border-box;background:${C.panelBg};border:1px solid ${C.rootBorder};border-radius:8px;`,
  );
  const lookSel = mkSelect(
    LOOKS.map(([n]) => n),
    settings.look,
  );
  const styleSel = mkSelect(STYLES, settings.nr_style);
  const upscaleSel = mkSelect(UPSCALE, settings.upscale_mode);
  const presetSel = mkSelect(PRESETS, settings.model_preset);
  const temporalSel = mkSelect(TEMPORAL, settings.temporal);
  const viewSel = mkSelect(
    [
      ["wipe", "Wipe (drag)"],
      ["after", "DLSS 5 only"],
      ["before", "Source only"],
      ["split", "Side by side (full)"],
    ],
    settings.view,
  );
  const tone = mkSlider(0, 2, 0.01, settings.local_tone);
  const structure = mkSlider(0, 2, 0.01, settings.local_structure);
  const skin = mkSlider(-1, 2, 0.01, settings.skin_structure);
  const mask = mkCheck("Auto mask (skin)", settings.auto_mask);
  const restartBtn = mkBtn("▶  Start live video", "Start the resident worker on the current clip");
  const stopBtn = mkBtn("■  Stop live video", "Stop the loop and free the GPU");
  const fullBtn = mkBtn("⛶  Full screen", "Full screen preview + controls (Esc to leave)");
  const bakeBtn = mkBtn("Bake full clip with these settings", "Render every frame with the current settings", {
    accent: true,
  });
  const closeBtn = mkBtn("✕  Close full screen", "Leave full screen (Esc)", { accent: true });
  closeBtn.hidden = true;

  function block(label, control) {
    const b = el("div", "");
    b.append(fieldLabel(label), control);
    return b;
  }

  panel.append(
    block("Look", lookSel),
    block("Style", styleSel),
    block("Local tone", tone),
    block("Local structure", structure),
    block("Skin structure", skin),
    mask,
    block("Upscale", upscaleSel),
    block("Model preset", presetSel),
    block("Temporal", temporalSel),
    block("View", viewSel),
    restartBtn,
    stopBtn,
    fullBtn,
    bakeBtn,
    closeBtn,
  );

  body.append(left, panel);
  wrapper.append(body);
  container.appendChild(wrapper);

  const overlay = el(
    "div",
    `position:fixed;inset:0;z-index:2147483000;background:${C.rootBg};display:none;overflow:hidden;` +
      `font-family:${FONT};color:${C.text};`,
  );
  overlay.className = "nodrag nowheel";
  overlay.tabIndex = 0;
  const scaler = el(
    "div",
    `width:${(100 / FULL_SCALE).toFixed(4)}%;height:${(100 / FULL_SCALE).toFixed(4)}%;` +
      `transform:scale(${FULL_SCALE});transform-origin:0 0;display:flex;flex-direction:column;` +
      `padding:12px;box-sizing:border-box;`,
  );
  scaler.className = "nodrag nowheel";
  overlay.append(scaler);

  function enterFull() {
    if (fullscreen) return;
    fullscreen = true;
    document.body.appendChild(overlay);
    scaler.append(body);
    overlay.style.display = "block";
    body.style.flex = "1 1 auto";
    body.style.minHeight = "0";
    body.style.height = "100%";
    panel.style.maxWidth = "380px";
    panel.style.flex = "0 0 380px";
    panel.style.width = "380px";
    fullBtn.hidden = true;
    closeBtn.hidden = false;
    if (overlay.requestFullscreen) overlay.requestFullscreen().catch(() => {});
    overlay.focus();
  }

  function leaveFull() {
    if (!fullscreen) return;
    fullscreen = false;
    if (document.fullscreenElement === overlay && document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
    wrapper.append(body);
    overlay.style.display = "none";
    overlay.remove();
    body.style.flex = "";
    body.style.minHeight = "740px";
    body.style.height = "";
    panel.style.maxWidth = "42%";
    panel.style.flex = "0 0 280px";
    panel.style.width = "280px";
    fullBtn.hidden = false;
    closeBtn.hidden = true;
  }

  const onFsChange = () => {
    if (fullscreen && document.fullscreenElement !== overlay) leaveFull();
  };
  document.addEventListener("fullscreenchange", onFsChange);
  overlay.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      leaveFull();
    } else if (e.key === " ") {
      e.preventDefault();
      cmd({ toggle: 1 });
    } else if (e.key === "ArrowLeft") cmd({ step: -1 });
    else if (e.key === "ArrowRight") cmd({ step: 1 });
  });
  fullBtn.addEventListener("click", enterFull);
  closeBtn.addEventListener("click", leaveFull);
  stage.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    if (fullscreen) leaveFull();
    else enterFull();
  });

  function cmd(params) {
    if (!url) return;
    const q = new URLSearchParams(params).toString();
    fetch(`${url}/cmd?${q}`, { cache: "no-store" }).catch(() => {});
  }

  function cmdSoon(params) {
    pendingCmd = { ...(pendingCmd || {}), ...params };
    if (cmdTimer) return;
    cmdTimer = setTimeout(() => {
      cmdTimer = null;
      const next = pendingCmd;
      pendingCmd = null;
      if (next) cmd(next);
    }, CMD_MS);
  }

  function flushCmd() {
    if (cmdTimer) {
      clearTimeout(cmdTimer);
      cmdTimer = null;
    }
    if (pendingCmd) {
      const next = pendingCmd;
      pendingCmd = null;
      cmd(next);
    }
  }

  function applyView() {
    badgeL.hidden = settings.view === "after";
    badgeR.hidden = settings.view === "before";
    stage.style.cursor = settings.view === "wipe" ? "col-resize" : "default";
  }

  function showImage() {
    img.style.display = "block";
    placeholder.style.display = "none";
    if (firstFrameTimer) {
      clearInterval(firstFrameTimer);
      firstFrameTimer = null;
    }
  }

  function connectStream() {
    if (!url) return;
    img.src = `${url}/stream.mjpg?t=${Date.now()}`;
    if (firstFrameTimer) clearInterval(firstFrameTimer);
    firstFrameTimer = setInterval(() => {
      if (img.naturalWidth > 0) showImage();
    }, 100);
  }

  function disconnectStream() {
    if (firstFrameTimer) {
      clearInterval(firstFrameTimer);
      firstFrameTimer = null;
    }
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    img.removeAttribute("src");
    img.style.display = "none";
  }

  img.addEventListener("load", showImage);
  img.addEventListener("error", () => {
    if (!url || retryTimer) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (url) connectStream();
    }, 1000);
  });

  function applyState(s) {
    lastState = s;
    playBtn.textContent = s.playing ? "Pause" : "Play";
    if (s.count > 0) {
      scrub.max = String(Math.max(0, s.count - 1));
      if (!scrubbing) scrub.value = String(s.index);
      frameLbl.textContent = `${s.index + 1} / ${s.total ?? s.count}${s.truncated ? "*" : ""}`;
    }
    if (typeof s.speed === "number") {
      const sv = String(Number(s.speed));
      if (speedSel.value !== sv && [...speedSel.options].some((o) => o.value === sv)) speedSel.value = sv;
    }
    const size = s.out_width ? `${s.out_width}×${s.out_height}` : "";
    const perf = s.playing ? `${s.play_fps} fps · ` : "";
    const enc = s.encode_ms !== undefined ? ` · ${s.encode_ms} ms preview` : "";
    const dec = s.decoding ? " · decoding…" : "";
    hud.textContent =
      `${size}  ${perf}${s.worker_ms} ms GPU · ${s.frame_ms} ms/frame${enc}${s.sequence ? " · temporal" : ""}${dec}` +
      (s.message && s.message !== "live" && s.message !== "starting worker..." ? `\n${s.message}` : "");
    if (s.error) {
      placeholder.textContent = s.error;
      placeholder.style.display = "flex";
    } else if (img.naturalWidth > 0) {
      showImage();
    }
  }

  function poll() {
    if (!url) return;
    fetch(`${url}/state`, { cache: "no-store" })
      .then((r) => r.json())
      .then(applyState)
      .catch(() => {
        hud.textContent = "preview server not reachable";
      });
  }

  function setUrl(next) {
    if (next === url) return;
    url = next || "";
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    disconnectStream();
    if (url) {
      placeholder.textContent = "Starting DLSS 5…";
      placeholder.style.display = "flex";
      connectStream();
      poll();
      pollTimer = setInterval(poll, STATE_POLL_MS);
    } else {
      placeholder.style.display = "flex";
      placeholder.textContent =
        liveStatus === "stopped"
          ? "Preview stopped — Start live video to resume."
          : "Connect a video, then Start live video.";
      hud.textContent = "";
    }
  }

  function syncTransport(status) {
    liveStatus = status || liveStatus;
    const running = liveStatus === "running";
    restartBtn.textContent = running ? "↻  Restart live video" : "▶  Start live video";
    restartBtn.title = running
      ? "Restart the resident worker on the current clip"
      : "Start the resident worker on the current clip";
    stopBtn.disabled = !running;
    stopBtn.style.opacity = running ? "1" : "0.45";
    stopBtn.style.cursor = running ? "pointer" : "default";
  }

  function payload(extra) {
    return {
      status: latestProps.value?.status || liveStatus || "running",
      url: url || latestProps.value?.url || "",
      message: latestProps.value?.message || "",
      look: settings.look,
      local_tone: settings.local_tone,
      local_structure: settings.local_structure,
      skin_structure: settings.skin_structure,
      auto_mask: settings.auto_mask,
      nr_style: settings.nr_style,
      upscale_mode: settings.upscale_mode,
      model_preset: settings.model_preset,
      temporal: settings.temporal,
      view: settings.view,
      wipe: settings.wipe,
      _fromWidget: true,
      ...(extra || {}),
    };
  }

  function lookKey() {
    return JSON.stringify([
      settings.look,
      settings.local_tone,
      settings.local_structure,
      settings.skin_structure,
      settings.auto_mask,
      settings.nr_style,
      settings.upscale_mode,
      settings.model_preset,
      settings.temporal,
      settings.view,
      Math.round(settings.wipe * 1000),
    ]);
  }

  function emitSettings() {
    if (!onChange || emitting) return;
    const key = lookKey();
    if (key === lastEmitKey) return;
    lastEmitKey = key;
    emitting = true;
    try {
      onChange(payload());
    } finally {
      emitting = false;
    }
  }

  function emitAction(action) {
    if (!onChange || emitting) return;
    emitting = true;
    try {
      onChange(payload({ _action: action }));
    } finally {
      emitting = false;
    }
  }

  function applyLookPreset(name) {
    const entry = LOOKS.find(([n]) => n === name);
    if (!entry || !entry[1]) return;
    const p = entry[1];
    settings.look = name;
    settings.local_tone = p.local_tone;
    settings.local_structure = p.local_structure;
    settings.skin_structure = p.skin_structure;
    settings.auto_mask = p.auto_mask;
    setSlider(tone, p.local_tone);
    setSlider(structure, p.local_structure);
    setSlider(skin, p.skin_structure);
    mask._input.checked = !!p.auto_mask;
  }

  function pushLive() {
    cmdSoon({
      local_tone: settings.local_tone,
      local_structure: settings.local_structure,
      skin_structure: settings.skin_structure,
      auto_mask: settings.auto_mask ? 1 : 0,
      nr_style: settings.nr_style,
    });
    emitSettings();
  }

  function markCustom() {
    if (settings.look !== "Custom (use the sliders)") {
      settings.look = "Custom (use the sliders)";
      lookSel.value = settings.look;
    }
  }

  function bindSlider(row, key) {
    const input = row._input;
    input.addEventListener("pointerdown", () => {
      sliding = true;
    });
    input.addEventListener("input", () => {
      settings[key] = Number(input.value);
      markCustom();
      const one = {};
      one[key] = settings[key];
      cmdSoon(one);
    });
    const commit = () => {
      sliding = false;
      settings[key] = Number(input.value);
      flushCmd();
      emitSettings();
    };
    input.addEventListener("change", commit);
    input.addEventListener("pointerup", commit);
    input.addEventListener("pointercancel", commit);
  }
  const onDocPointerUp = () => {
    if (!sliding) return;
    sliding = false;
    flushCmd();
    emitSettings();
  };
  document.addEventListener("pointerup", onDocPointerUp, true);

  lookSel.addEventListener("change", () => {
    applyLookPreset(lookSel.value);
    settings.look = lookSel.value;
    pushLive();
  });
  styleSel.addEventListener("change", () => {
    settings.nr_style = styleSel.value;
    pushLive();
  });
  upscaleSel.addEventListener("change", () => {
    settings.upscale_mode = upscaleSel.value;
    cmd({ upscale_mode: settings.upscale_mode });
    emitSettings();
  });
  presetSel.addEventListener("change", () => {
    settings.model_preset = presetSel.value;
    cmd({ model_preset: settings.model_preset });
    emitSettings();
  });
  temporalSel.addEventListener("change", () => {
    settings.temporal = temporalSel.value;
    cmd({ sequence: settings.temporal.startsWith("Sequence") ? 1 : 0 });
    emitSettings();
  });
  viewSel.addEventListener("change", () => {
    settings.view = viewSel.value;
    applyView();
    cmd({ view: settings.view });
    emitSettings();
  });
  bindSlider(tone, "local_tone");
  bindSlider(structure, "local_structure");
  bindSlider(skin, "skin_structure");
  mask._input.addEventListener("change", () => {
    settings.auto_mask = !!mask._input.checked;
    markCustom();
    pushLive();
  });
  restartBtn.addEventListener("click", () => emitAction("restart"));
  stopBtn.addEventListener("click", () => {
    if (stopBtn.disabled) return;
    emitAction("stop");
  });
  bakeBtn.addEventListener("click", () => emitAction("bake"));

  playBtn.addEventListener("click", () => cmd({ toggle: 1 }));
  prevBtn.addEventListener("click", () => cmd({ step: -1 }));
  nextBtn.addEventListener("click", () => cmd({ step: 1 }));
  scrub.addEventListener("input", () => {
    scrubbing = true;
    cmd({ play: 0, seek: scrub.value });
  });
  scrub.addEventListener("change", () => {
    scrubbing = false;
  });
  speedSel.addEventListener("change", () => cmd({ speed: speedSel.value }));
  wrapper.addEventListener("keydown", (e) => {
    if (e.key === " ") {
      e.preventDefault();
      cmd({ toggle: 1 });
    } else if (e.key === "ArrowLeft") cmd({ step: -1 });
    else if (e.key === "ArrowRight") cmd({ step: 1 });
  });

  function wipeAt(clientX) {
    if (settings.view !== "wipe") return;
    const r = (img.naturalWidth ? img : stage).getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)));
    settings.wipe = x;
    cmdSoon({ wipe: x.toFixed(4), view: "wipe" });
  }
  stage.addEventListener("pointerdown", (e) => {
    if (settings.view !== "wipe" || !url) return;
    e.stopPropagation();
    wiping = true;
    stage.setPointerCapture?.(e.pointerId);
    wipeAt(e.clientX);
  });
  stage.addEventListener("pointermove", (e) => {
    if (!wiping) return;
    wipeAt(e.clientX);
  });
  const endWipe = () => {
    if (!wiping) return;
    wiping = false;
    flushCmd();
    emitSettings();
  };
  stage.addEventListener("pointerup", endWipe);
  stage.addEventListener("pointercancel", endWipe);

  function syncControls(v) {
    if (!v || v._fromWidget) return;
    if (sliding || wiping) return;
    const keys = [
      "look",
      "local_tone",
      "local_structure",
      "skin_structure",
      "auto_mask",
      "nr_style",
      "upscale_mode",
      "model_preset",
      "temporal",
      "view",
      "wipe",
    ];
    let changed = false;
    let viewChanged = false;
    for (const k of keys) {
      if (v[k] !== undefined && v[k] !== settings[k]) {
        if (k === "view") viewChanged = true;
        settings[k] = v[k];
        changed = true;
      }
    }
    if (!changed) return;
    lookSel.value = settings.look;
    styleSel.value = settings.nr_style;
    upscaleSel.value = settings.upscale_mode;
    presetSel.value = settings.model_preset;
    temporalSel.value = settings.temporal;
    viewSel.value = settings.view;
    setSlider(tone, settings.local_tone);
    setSlider(structure, settings.local_structure);
    setSlider(skin, settings.skin_structure);
    mask._input.checked = !!settings.auto_mask;
    lastEmitKey = lookKey();
    if (viewChanged) applyView();
  }

  function update(nextProps) {
    latestProps = nextProps;
    onChange = nextProps.onChange;
    let v = nextProps.value || {};
    if (typeof v === "string") {
      try {
        v = JSON.parse(v);
      } catch {
        v = {};
      }
    }
    if (v.status) syncTransport(v.status);
    syncControls(v);
    const nextUrl = v.status === "running" && v.url ? String(v.url).replace(/\/$/, "") : "";
    setUrl(nextUrl);
    if (v.status === "error" && v.message) {
      placeholder.style.display = "flex";
      placeholder.textContent = v.message;
    } else if (v.status === "stopped" && !nextUrl) {
      placeholder.style.display = "flex";
      placeholder.textContent = v.message || "Preview stopped — Start live video to resume.";
    }
  }

  function cleanup() {
    leaveFull();
    document.removeEventListener("fullscreenchange", onFsChange);
    document.removeEventListener("pointerup", onDocPointerUp, true);
    if (cmdTimer) clearTimeout(cmdTimer);
    cmdTimer = null;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    disconnectStream();
    wrapper.remove();
    delete container._dlss5Live;
  }

  applyView();
  update(props);
  container._dlss5Live = { wrapper, update, cleanup };
  return { update, cleanup };
}
