// DLSS 5 Live Image widget — landscape layout.
//
// Visual language matches Shot Planner / Seedance / Omni Image widgets:
// dark gray chrome (#0c0e11), muted amber accent (#d9c6a4 / #a58050), ui-monospace.
//
// Left: before/after wipe. Right: look controls.
// Sliders hit /cmd while dragging; onChange syncs on release.
// Full screen shows only this landscape (preview + controls).

const POLL_MS = 250;

// Shot Planner V2 palette (same as SeedanceMasterWidget / ImageGenPromptWidget).
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

function mkSelect(options, value) {
  const s = stopDrag(
    el(
      "select",
      `width:100%;padding:5px 7px;border-radius:6px;border:1px solid ${C.inputBorder};` +
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
  return row;
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

function mkBtn(label, title, { accent = false } = {}) {
  const b = stopDrag(
    el(
      "button",
      accent
        ? `padding:6px 10px;border-radius:6px;border:1px solid ${C.accentBorder};background:${C.accentBg};` +
          `color:${C.accent};font:11.5px/1.3 ${FONT};font-weight:600;cursor:pointer;width:100%;`
        : `padding:6px 10px;border-radius:6px;border:1px solid ${C.inputBorder};background:${C.chipBg};` +
          `color:${C.text};font:11.5px/1.3 ${FONT};cursor:pointer;width:100%;`,
      label,
    ),
  );
  if (title) b.title = title;
  return b;
}

export default function DLSS5LiveImage(container, props) {
  if (container._dlss5LiveImg?.wrapper?.isConnected) {
    container._dlss5LiveImg.update(props);
    return { cleanup: container._dlss5LiveImg.cleanup, update: container._dlss5LiveImg.update };
  }

  let onChange = props.onChange;
  let latestProps = props;
  let url = "";
  let pollTimer = null;
  let wiping = false;
  let sliding = false;
  let emitting = false;
  let lastSeq = -1;
  let fullscreen = false;
  let cmdTimer = null;
  let pendingCmd = null;
  let settings = {
    look: "Ultra (max realism)",
    local_tone: 1.0,
    local_structure: 2.0,
    skin_structure: 2.0,
    auto_mask: true,
    nr_style: "Default",
    upscale_mode: "1.0x (DLAA / native)",
    model_preset: "M",
    view: "wipe",
    wipe: 0.5,
  };

  const wrapper = el(
    "div",
    `display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:360px;box-sizing:border-box;padding:8px;` +
      `background:${C.rootBg};border:1px solid ${C.rootBorder};border-radius:10px;font-family:${FONT};color:${C.text};`,
  );
  wrapper.className = "nodrag nowheel dlss5-live-img-root";

  const body = el(
    "div",
    "display:flex;flex-direction:row;gap:10px;flex:1 1 auto;min-height:320px;width:100%;box-sizing:border-box;",
  );

  const stage = el(
    "div",
    `position:relative;flex:1 1 62%;min-width:220px;min-height:280px;background:${C.stageBg};border:1px solid ${C.rootBorder};` +
      `border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:col-resize;user-select:none;`,
  );
  const img = el("img", "display:none;width:100%;height:100%;object-fit:contain;pointer-events:none;");
  img.draggable = false;
  const placeholder = el(
    "div",
    `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;` +
      `padding:24px;color:${C.muted};font:12.5px/1.5 ${FONT};pointer-events:none;`,
    "Connect an image — live preview starts automatically.",
  );
  const badgeL = el(
    "div",
    `position:absolute;left:8px;top:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.inputBorder};` +
      `background:${C.chipBg};color:${C.label};font:10.5px/1.4 ${FONT};pointer-events:none;`,
    "BEFORE",
  );
  const badgeR = el(
    "div",
    `position:absolute;right:8px;top:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.accentBorder};` +
      `background:${C.accentBg};color:${C.accent};font:10.5px/1.4 ${FONT};pointer-events:none;`,
    "DLSS 5",
  );
  const hud = el(
    "div",
    `position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:5px;border:1px solid ${C.rootBorder};` +
      `background:rgba(12,14,17,.85);color:${C.muted};font:10.5px/1.4 ${FONT};pointer-events:none;white-space:pre;`,
    "",
  );
  stage.append(img, placeholder, badgeL, badgeR, hud);

  const panel = el(
    "div",
    `flex:0 0 280px;width:280px;max-width:42%;display:flex;flex-direction:column;gap:10px;` +
      `overflow:auto;padding:8px;box-sizing:border-box;background:${C.panelBg};border:1px solid ${C.rootBorder};border-radius:8px;`,
  );

  const lookSel = mkSelect(
    LOOKS.map(([n]) => n),
    settings.look,
  );
  const styleSel = mkSelect(STYLES, settings.nr_style);
  const upscaleSel = mkSelect(UPSCALE, settings.upscale_mode);
  const presetSel = mkSelect(PRESETS, settings.model_preset);
  const viewSel = mkSelect(
    [
      ["wipe", "Wipe (drag)"],
      ["after", "DLSS 5 only"],
      ["before", "Source only"],
      ["split", "Side by side"],
    ],
    settings.view,
  );
  const tone = mkSlider(0, 2, 0.01, settings.local_tone);
  const structure = mkSlider(0, 2, 0.01, settings.local_structure);
  const skin = mkSlider(-1, 2, 0.01, settings.skin_structure);
  const mask = mkCheck("Auto mask (skin)", settings.auto_mask);
  const restartBtn = mkBtn("↻  Restart live preview", "Restart the resident worker on the current image");
  const stopBtn = mkBtn("■  Stop live preview", "Stop the worker and free the GPU");
  const fullBtn = mkBtn("⛶  Full screen", "Full screen preview + controls (Esc to leave)");
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
    block("View", viewSel),
    restartBtn,
    stopBtn,
    fullBtn,
    closeBtn,
  );

  body.append(stage, panel);
  wrapper.append(body);
  container.appendChild(wrapper);

  const overlay = el(
    "div",
    `position:fixed;inset:0;z-index:2147483000;background:${C.rootBg};display:none;flex-direction:column;` +
      `padding:12px;box-sizing:border-box;font-family:${FONT};color:${C.text};`,
  );
  overlay.className = "nodrag nowheel";
  overlay.tabIndex = 0;

  function enterFull() {
    if (fullscreen) return;
    fullscreen = true;
    document.body.appendChild(overlay);
    overlay.append(body);
    overlay.style.display = "flex";
    overlay.style.inset = "0 auto auto 0";
    overlay.style.width = "83.333%";
    overlay.style.height = "83.333%";
    overlay.style.zoom = "1.2";
    body.style.flex = "1 1 auto";
    body.style.minHeight = "0";
    body.style.height = "100%";
    stage.style.borderRadius = "8px";
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
    body.style.minHeight = "320px";
    body.style.height = "";
    overlay.style.zoom = "";
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
    }
  });
  fullBtn.addEventListener("click", enterFull);
  closeBtn.addEventListener("click", leaveFull);

  function cmd(params) {
    if (!url) return;
    const q = new URLSearchParams(params).toString();
    fetch(`${url}/cmd?${q}`).catch(() => {});
  }

  function cmdSoon(params) {
    pendingCmd = { ...(pendingCmd || {}), ...params };
    if (cmdTimer) return;
    cmdTimer = setTimeout(() => {
      cmdTimer = null;
      const next = pendingCmd;
      pendingCmd = null;
      if (next) cmd(next);
    }, 80);
  }

  function refreshFrame() {
    if (!url || sliding) return;
    img.src = `${url}/frame.jpg?t=${Date.now()}`;
  }

  function showImage() {
    img.style.display = "block";
    placeholder.style.display = "none";
  }

  function emitSettings() {
    if (!onChange || emitting) return;
    emitting = true;
    try {
      onChange({
        status: latestProps.value?.status || "running",
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
        view: settings.view,
        wipe: settings.wipe,
        _fromWidget: true,
      });
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
    tone._input.value = String(p.local_tone);
    tone.querySelector("span").textContent = p.local_tone.toFixed(2);
    structure._input.value = String(p.local_structure);
    structure.querySelector("span").textContent = p.local_structure.toFixed(2);
    skin._input.value = String(p.skin_structure);
    skin.querySelector("span").textContent = p.skin_structure.toFixed(2);
    mask._input.checked = !!p.auto_mask;
  }

  function emitAction(action) {
    if (!onChange || emitting) return;
    emitting = true;
    try {
      onChange({
        status: latestProps.value?.status || "running",
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
        view: settings.view,
        wipe: settings.wipe,
        _fromWidget: true,
        _action: action,
      });
    } finally {
      emitting = false;
    }
  }

  function pushLiveCmd(extra) {
    cmdSoon({
      local_tone: settings.local_tone,
      local_structure: settings.local_structure,
      skin_structure: settings.skin_structure,
      auto_mask: settings.auto_mask ? 1 : 0,
      nr_style: settings.nr_style,
      ...(extra || {}),
    });
  }

  function pushLive() {
    pushLiveCmd();
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
      cmdSoon(one); // GPU only, throttled; don't decode a new JPEG until release
    });
    const commit = () => {
      sliding = false;
      settings[key] = Number(input.value);
      if (pendingCmd) {
        cmd(pendingCmd);
        pendingCmd = null;
      }
      if (cmdTimer) {
        clearTimeout(cmdTimer);
        cmdTimer = null;
      }
      refreshFrame();
      emitSettings();
    };
    input.addEventListener("change", commit);
    input.addEventListener("pointerup", commit);
    input.addEventListener("pointercancel", commit);
    input.addEventListener("keyup", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "Home" || e.key === "End") commit();
    });
  }

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
  restartBtn.addEventListener("click", () => emitAction("restart"));
  stopBtn.addEventListener("click", () => emitAction("stop"));
  viewSel.addEventListener("change", () => {
    settings.view = viewSel.value;
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

  function wipeAt(clientX) {
    const r = stage.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)));
    settings.wipe = x;
    cmd({ wipe: x, view: "wipe" });
    if (settings.view !== "wipe") {
      settings.view = "wipe";
      viewSel.value = "wipe";
    }
  }
  stage.addEventListener("pointerdown", (e) => {
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
    emitSettings();
  };
  stage.addEventListener("pointerup", endWipe);
  stage.addEventListener("pointercancel", endWipe);

  img.addEventListener("load", showImage);

  function poll() {
    if (!url) return;
    fetch(`${url}/state`)
      .then((r) => r.json())
      .then((st) => {
        const ms = st.worker_ms != null ? `${st.worker_ms.toFixed?.(1) ?? st.worker_ms} ms` : "";
        const size = st.out_width ? `${st.out_width}×${st.out_height}` : "";
        hud.textContent = [st.message || "", size, ms].filter(Boolean).join("  ·  ");
        if (st.error) placeholder.textContent = st.error;
        if (typeof st.seq === "number" && st.seq !== lastSeq) {
          lastSeq = st.seq;
          if (!sliding) refreshFrame();
        }
      })
      .catch(() => {});
  }

  function setUrl(next) {
    if (next === url) return;
    url = next || "";
    lastSeq = -1;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (url) {
      placeholder.textContent = "Starting DLSS 5…";
      refreshFrame();
      poll();
      pollTimer = setInterval(poll, POLL_MS);
    } else {
      img.style.display = "none";
      placeholder.style.display = "flex";
      placeholder.textContent = "Connect an image — live preview starts automatically.";
      hud.textContent = "";
    }
  }

  function syncControls(v) {
    if (!v || v._fromWidget) return;
    const keys = [
      "look",
      "local_tone",
      "local_structure",
      "skin_structure",
      "auto_mask",
      "nr_style",
      "upscale_mode",
      "model_preset",
      "view",
      "wipe",
    ];
    let changed = false;
    for (const k of keys) {
      if (v[k] !== undefined && v[k] !== settings[k]) {
        settings[k] = v[k];
        changed = true;
      }
    }
    if (!changed) return;
    lookSel.value = settings.look;
    styleSel.value = settings.nr_style;
    upscaleSel.value = settings.upscale_mode;
    presetSel.value = settings.model_preset;
    viewSel.value = settings.view;
    tone._input.value = String(settings.local_tone);
    tone.querySelector("span").textContent = Number(settings.local_tone).toFixed(2);
    structure._input.value = String(settings.local_structure);
    structure.querySelector("span").textContent = Number(settings.local_structure).toFixed(2);
    skin._input.value = String(settings.skin_structure);
    skin.querySelector("span").textContent = Number(settings.skin_structure).toFixed(2);
    mask._input.checked = !!settings.auto_mask;
  }

  function update(nextProps) {
    latestProps = nextProps;
    onChange = nextProps.onChange;
    const v = nextProps.value || {};
    syncControls(v);
    if (v.status === "error" && v.message) {
      placeholder.style.display = "flex";
      placeholder.textContent = v.message;
    }
    setUrl(v.url || "");
  }

  function cleanup() {
    leaveFull();
    document.removeEventListener("fullscreenchange", onFsChange);
    if (cmdTimer) clearTimeout(cmdTimer);
    cmdTimer = null;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    img.src = "";
  }

  update(props);
  container._dlss5LiveImg = { wrapper, update, cleanup };
  return { cleanup, update };
}
