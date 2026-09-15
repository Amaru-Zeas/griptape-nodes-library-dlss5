// DLSS 5 Live Image widget — landscape layout.
//
// Left: before/after wipe of the still (JPEG from the node's LiveImageSession).
// Right: look controls. Sliders hit /cmd while dragging (GPU only); onChange to the
// node runs on release so Bake stays in sync without lagging the UI.
// Full screen shows only this landscape (preview + controls), not the node's buttons.

const POLL_MS = 250;

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
  return el(
    "div",
    "font:11px/1.2 var(--font-sans, sans-serif);color:var(--muted-foreground, #999);margin:0 0 3px;",
    text,
  );
}

function mkSelect(options, value) {
  const s = stopDrag(
    el(
      "select",
      "width:100%;padding:5px 7px;border-radius:6px;border:1px solid var(--border, #444);" +
        "background:var(--background, #1b1b1b);color:var(--foreground, #eee);font-size:12px;cursor:pointer;box-sizing:border-box;",
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
  const input = stopDrag(el("input", "flex:1;min-width:0;cursor:pointer;"));
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const num = el(
    "span",
    "font:11px/1 monospace;color:var(--muted-foreground, #aaa);min-width:36px;text-align:right;",
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
      "display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer;color:var(--foreground, #eee);",
    ),
  );
  const input = el("input", "cursor:pointer;");
  input.type = "checkbox";
  input.checked = !!checked;
  row.append(input, el("span", "", label));
  row._input = input;
  return row;
}

function mkBtn(label, title) {
  const b = stopDrag(
    el(
      "button",
      "padding:6px 10px;border-radius:6px;border:1px solid var(--border, #444);background:var(--background, #1b1b1b);" +
        "color:var(--foreground, #eee);font-size:12px;cursor:pointer;line-height:1.3;white-space:nowrap;width:100%;",
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
  let emitting = false;
  let lastSeq = -1;
  let fullscreen = false;
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
    "display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:360px;box-sizing:border-box;padding:6px;",
  );
  wrapper.className = "nodrag nowheel";

  // Landscape body (this is what goes fullscreen — not Restart/Stop/Bake on the node).
  const body = el(
    "div",
    "display:flex;flex-direction:row;gap:10px;flex:1 1 auto;min-height:320px;width:100%;box-sizing:border-box;",
  );

  const stage = el(
    "div",
    "position:relative;flex:1 1 62%;min-width:220px;min-height:280px;background:#0e0e0e;border-radius:8px;" +
      "overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:col-resize;user-select:none;",
  );
  const img = el("img", "display:none;width:100%;height:100%;object-fit:contain;pointer-events:none;");
  img.draggable = false;
  const placeholder = el(
    "div",
    "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;" +
      "padding:24px;color:var(--muted-foreground, #999);font-size:13px;line-height:1.4;pointer-events:none;",
    "Connect an image — live preview starts automatically.",
  );
  const badgeL = el(
    "div",
    "position:absolute;left:8px;top:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.55);" +
      "color:#fff;font:11px/1.4 monospace;pointer-events:none;",
    "BEFORE",
  );
  const badgeR = el(
    "div",
    "position:absolute;right:8px;top:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.55);" +
      "color:#fff;font:11px/1.4 monospace;pointer-events:none;",
    "DLSS 5",
  );
  const hud = el(
    "div",
    "position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.55);" +
      "color:#ddd;font:11px/1.4 monospace;pointer-events:none;white-space:pre;",
    "",
  );
  stage.append(img, placeholder, badgeL, badgeR, hud);

  const panel = el(
    "div",
    "flex:0 0 280px;width:280px;max-width:42%;display:flex;flex-direction:column;gap:10px;" +
      "overflow:auto;padding:2px 2px 6px;box-sizing:border-box;",
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
  const fullBtn = mkBtn("⛶  Full screen", "Full screen preview + controls (Esc to leave)");
  const closeBtn = mkBtn("✕  Close full screen", "Leave full screen (Esc)");
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
    fullBtn,
    closeBtn,
  );

  body.append(stage, panel);
  wrapper.append(body);
  container.appendChild(wrapper);

  // Fullscreen overlay: only the landscape body (image + controls).
  const overlay = el(
    "div",
    "position:fixed;inset:0;z-index:2147483000;background:#0a0a0a;display:none;flex-direction:column;" +
      "padding:12px;box-sizing:border-box;",
  );
  overlay.className = "nodrag nowheel";
  overlay.tabIndex = 0;

  function enterFull() {
    if (fullscreen) return;
    fullscreen = true;
    document.body.appendChild(overlay);
    overlay.append(body);
    overlay.style.display = "flex";
    body.style.flex = "1 1 auto";
    body.style.minHeight = "0";
    body.style.height = "100%";
    stage.style.borderRadius = "0";
    panel.style.maxWidth = "320px";
    panel.style.flex = "0 0 320px";
    panel.style.width = "320px";
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
    stage.style.borderRadius = "8px";
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

  // ── helpers ─────────────────────────────────────────────────────────────

  function cmd(params) {
    if (!url) return;
    const q = new URLSearchParams(params).toString();
    fetch(`${url}/cmd?${q}`).catch(() => {});
  }

  function refreshFrame() {
    if (!url) return;
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

  // GPU only — used while dragging sliders.
  function pushLiveCmd() {
    cmd({
      local_tone: settings.local_tone,
      local_structure: settings.local_structure,
      skin_structure: settings.skin_structure,
      auto_mask: settings.auto_mask ? 1 : 0,
      nr_style: settings.nr_style,
      upscale_mode: settings.upscale_mode,
      model_preset: settings.model_preset,
    });
  }

  // GPU + sync node (dropdowns / release).
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
    input.addEventListener("input", () => {
      settings[key] = Number(input.value);
      markCustom();
      pushLiveCmd(); // no onChange while dragging
    });
    const commit = () => {
      settings[key] = Number(input.value);
      emitSettings();
    };
    input.addEventListener("change", commit);
    input.addEventListener("pointerup", commit);
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
    pushLive();
  });
  presetSel.addEventListener("change", () => {
    settings.model_preset = presetSel.value;
    pushLive();
  });
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
          refreshFrame();
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
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    img.src = "";
  }

  update(props);
  container._dlss5LiveImg = { wrapper, update, cleanup };
  return { cleanup, update };
}
