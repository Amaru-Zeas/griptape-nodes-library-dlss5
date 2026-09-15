// DLSS 5 Live Video widget.
//
// Visual language matches Shot Planner / Seedance / Live Image:
// dark gray chrome (#0c0e11), muted amber accent (#d9c6a4 / #a58050), ui-monospace.
//
// Shows the MJPEG stream served by the node's LiveSession (127.0.0.1) with
// transport + before/after controls. Commands go straight to /cmd so a wipe
// drag does not travel through the engine.

const POLL_MS = 500;
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

function mkBtn(label, title, { accent = false, compact = false } = {}) {
  const idleBorder = accent ? C.accentBorder : C.inputBorder;
  const idleBg = accent ? C.accentBg : C.chipBg;
  const idleFg = accent ? C.accent : C.text;
  const pad = compact ? "padding:4px 9px;" : "padding:6px 10px;";
  const b = stopDrag(
    el(
      "button",
      pad +
        `border-radius:6px;border:1px solid ${idleBorder};background:${idleBg};` +
        `color:${idleFg};font:11.5px/1.3 ${FONT};${accent ? "font-weight:600;" : ""}cursor:pointer;` +
        `white-space:nowrap;` +
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

function mkSelect(options, value) {
  const s = stopDrag(
    el(
      "select",
      `padding:4px 7px;border-radius:6px;border:1px solid ${C.inputBorder};` +
        `background:${C.inputBg};color:${C.text};font:11.5px/1.3 ${FONT};cursor:pointer;outline:none;`,
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

export default function DLSS5LivePreview(container, props) {
  if (container._dlss5Live?.wrapper?.isConnected) {
    container._dlss5Live.update(props);
    return { cleanup: container._dlss5Live.cleanup, update: container._dlss5Live.update };
  }

  let url = "";
  let pollTimer = null;
  let firstFrameTimer = null;
  let retryTimer = null;
  let dragging = false;
  let lastState = null;
  let streamToken = 0;
  let fullscreen = false;

  const wrapper = el(
    "div",
    `display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:280px;box-sizing:border-box;padding:8px;` +
      `background:${C.rootBg};border:1px solid ${C.rootBorder};border-radius:10px;font-family:${FONT};color:${C.text};`,
  );
  wrapper.className = "nodrag nowheel dlss5-live-video-root";

  const stage = el(
    "div",
    `position:relative;flex:1 1 0;min-height:180px;background:${C.stageBg};border:1px solid ${C.rootBorder};` +
      `border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;` +
      `cursor:col-resize;user-select:none;`,
  );
  const img = el("img", "display:none;width:100%;height:100%;object-fit:contain;pointer-events:none;");
  img.draggable = false;
  img.alt = "";
  const placeholder = el(
    "div",
    `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;` +
      `padding:20px;color:${C.muted};font:12.5px/1.5 ${FONT};pointer-events:none;z-index:3;`,
    "Press  Start live video  on the node.",
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
    `display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:6px 8px;` +
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
  const frameLbl = el(
    "span",
    `font:11px/1 ${FONT};color:${C.muted};min-width:70px;text-align:right;`,
    "0 / 0",
  );
  const viewSel = mkSelect(
    [
      ["wipe", "Wipe (drag)"],
      ["after", "DLSS 5 only"],
      ["before", "Source only"],
      ["split", "Side by side (full)"],
    ],
    "wipe",
  );
  const speedSel = mkSelect(
    [
      ["0.25", "0.25x"],
      ["0.5", "0.5x"],
      ["1", "1x"],
      ["2", "2x"],
      ["0", "Max"],
    ],
    "1",
  );
  speedSel.title = "Playback speed relative to the clip's frame rate. Max = as fast as the pipeline goes.";
  const fullBtn = mkBtn("⛶  Full screen", "Full screen preview + transport (Esc to leave)", { compact: true });
  const closeBtn = mkBtn("✕  Close full screen", "Leave full screen (Esc)", { accent: true, compact: true });
  closeBtn.hidden = true;

  bar.append(playBtn, prevBtn, nextBtn, scrub, frameLbl, speedSel, viewSel, fullBtn, closeBtn);
  wrapper.append(stage, bar);
  container.appendChild(wrapper);

  // Overlay itself is unscaled (fullscreen UA rules force 100% / transform:none).
  // Inner scaler is laid out at 1/FULL_SCALE and transformed up so UI is 20% larger
  // with exact pointer hit-testing (CSS zoom breaks range sliders).
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
      `transform:scale(${FULL_SCALE});transform-origin:0 0;display:flex;flex-direction:column;gap:8px;` +
      `padding:12px;box-sizing:border-box;`,
  );
  scaler.className = "nodrag nowheel";
  overlay.append(scaler);

  function enterFull() {
    if (fullscreen) return;
    fullscreen = true;
    document.body.appendChild(overlay);
    scaler.append(stage, bar);
    overlay.style.display = "block";
    stage.style.flex = "1 1 auto";
    stage.style.minHeight = "0";
    bar.style.justifyContent = "center";
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
    wrapper.append(stage, bar);
    overlay.style.display = "none";
    overlay.remove();
    stage.style.flex = "";
    stage.style.minHeight = "180px";
    bar.style.justifyContent = "";
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

  function applyState(s) {
    lastState = s;
    playBtn.textContent = s.playing ? "Pause" : "Play";
    if (s.count > 0) {
      scrub.max = String(Math.max(0, s.count - 1));
      if (!dragging) scrub.value = String(s.index);
      frameLbl.textContent = `${s.index + 1} / ${s.total ?? s.count}${s.truncated ? "*" : ""}`;
    }
    if (viewSel.value !== s.view) viewSel.value = s.view;
    if (typeof s.speed === "number") {
      const sv = String(Number(s.speed));
      if (speedSel.value !== sv && [...speedSel.options].some((o) => o.value === sv)) speedSel.value = sv;
    }
    const wipeMode = s.view === "wipe";
    badgeL.hidden = !(wipeMode || s.view === "split" || s.view === "before");
    badgeR.hidden = !(wipeMode || s.view === "split" || s.view === "after");
    stage.style.cursor = wipeMode ? "col-resize" : "default";
    const size = s.out_width ? `${s.out_width}x${s.out_height}` : "";
    const perf = s.playing ? `${s.play_fps} fps · ` : "";
    const enc = s.encode_ms !== undefined ? ` · ${s.encode_ms} ms preview` : "";
    const dec = s.decoding ? " · decoding…" : "";
    hud.textContent =
      `${size}  ${perf}${s.worker_ms} ms GPU · ${s.frame_ms} ms/frame${enc}${s.sequence ? " · temporal" : ""}${dec}` +
      (s.message && s.message !== "live" ? `\n${s.message}` : "");
    if (s.error) {
      placeholder.textContent = s.error;
      placeholder.style.display = "flex";
    }
  }

  async function poll() {
    if (!url) return;
    try {
      const r = await fetch(`${url}/state`, { cache: "no-store" });
      if (r.ok) applyState(await r.json());
    } catch {
      hud.textContent = "preview server not reachable";
    }
  }

  function showImage() {
    img.style.display = "block";
    placeholder.style.display = "none";
    if (firstFrameTimer) {
      clearInterval(firstFrameTimer);
      firstFrameTimer = null;
    }
  }

  function startStream() {
    streamToken++;
    img.src = `${url}/stream.mjpg?t=${Date.now()}-${streamToken}`;
    placeholder.style.display = "flex";
    placeholder.textContent = "Starting DLSS 5…";
    if (firstFrameTimer) clearInterval(firstFrameTimer);
    firstFrameTimer = setInterval(() => {
      if (img.naturalWidth > 0) showImage();
    }, 100);
    clearInterval(pollTimer);
    pollTimer = setInterval(poll, POLL_MS);
    poll();
  }

  function stopStream(message) {
    clearInterval(pollTimer);
    pollTimer = null;
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
    placeholder.textContent = message || "Press  Start live video  on the node.";
    placeholder.style.display = "flex";
    hud.textContent = "";
    badgeL.hidden = badgeR.hidden = true;
    leaveFull();
  }

  img.addEventListener("load", showImage);
  img.addEventListener("error", () => {
    if (!url || retryTimer) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (url) startStream();
    }, 1000);
  });

  function wipeFromEvent(e) {
    const rect = img.getBoundingClientRect();
    if (!rect.width) return;
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    cmd({ wipe: x.toFixed(4) });
  }
  stage.addEventListener("pointerdown", (e) => {
    if (!url || (lastState && lastState.view !== "wipe")) return;
    e.stopPropagation();
    dragging = true;
    stage.setPointerCapture(e.pointerId);
    wipeFromEvent(e);
  });
  stage.addEventListener("pointermove", (e) => {
    if (dragging) wipeFromEvent(e);
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try {
      stage.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener("mousedown", (e) => e.stopPropagation());

  playBtn.addEventListener("click", () => cmd({ toggle: 1 }));
  prevBtn.addEventListener("click", () => cmd({ step: -1 }));
  nextBtn.addEventListener("click", () => cmd({ step: 1 }));
  scrub.addEventListener("input", () => {
    dragging = true;
    cmd({ play: 0, seek: scrub.value });
  });
  scrub.addEventListener("change", () => {
    dragging = false;
  });
  viewSel.addEventListener("change", () => cmd({ view: viewSel.value }));
  speedSel.addEventListener("change", () => cmd({ speed: speedSel.value }));
  wrapper.addEventListener("keydown", (e) => {
    if (e.key === " ") {
      e.preventDefault();
      cmd({ toggle: 1 });
    } else if (e.key === "ArrowLeft") cmd({ step: -1 });
    else if (e.key === "ArrowRight") cmd({ step: 1 });
  });
  wrapper.tabIndex = 0;

  function update(newProps) {
    let v = newProps?.value ?? {};
    if (typeof v === "string") {
      try {
        v = JSON.parse(v);
      } catch {
        v = {};
      }
    }
    const nextUrl = v.status === "running" && v.url ? String(v.url).replace(/\/$/, "") : "";
    if (nextUrl !== url) {
      url = nextUrl;
      if (url) startStream();
      else stopStream(v.message || (v.status === "error" ? "Live video failed." : undefined));
    } else if (!url && v.message) {
      placeholder.textContent = v.message;
    }
  }

  function cleanup() {
    clearInterval(pollTimer);
    if (firstFrameTimer) clearInterval(firstFrameTimer);
    if (retryTimer) clearTimeout(retryTimer);
    leaveFull();
    document.removeEventListener("fullscreenchange", onFsChange);
    img.removeAttribute("src");
    wrapper.remove();
    delete container._dlss5Live;
  }

  container._dlss5Live = { update, cleanup, wrapper };
  update(props);
  return { update, cleanup };
}
