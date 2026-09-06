// DLSS 5 Live Preview widget.
//
// Shows the MJPEG stream served by the node's LiveSession (a tiny HTTP server on
// 127.0.0.1) and offers transport + before/after controls. The parameter value
// (set by the node) looks like:
//   { url: "http://127.0.0.1:PORT", status: "running"|"stopped"|"error", message: "..." }
// Commands go straight to the server (GET /cmd?...) so a wipe drag does not travel
// through the engine; the node itself only sets the value above.

const POLL_MS = 500;

function el(tag, css, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text !== undefined) e.textContent = text;
  return e;
}

function btn(label, title) {
  const b = el(
    "button",
    "padding:3px 9px;border-radius:5px;border:1px solid var(--border, #444);background:var(--background, #1b1b1b);" +
      "color:var(--foreground, #eee);font-size:12px;cursor:pointer;line-height:1.3;white-space:nowrap;",
    label,
  );
  if (title) b.title = title;
  ["pointerdown", "mousedown", "dblclick"].forEach((ev) => b.addEventListener(ev, (e) => e.stopPropagation()));
  return b;
}

export default function DLSS5LivePreview(container, props) {
  if (container._dlss5Live?.wrapper?.isConnected) {
    container._dlss5Live.update(props);
    return { cleanup: container._dlss5Live.cleanup, update: container._dlss5Live.update };
  }

  let url = "";
  let pollTimer = null;
  let dragging = false;
  let lastState = null;
  let streamToken = 0;

  // ── DOM ─────────────────────────────────────────────────────────────────
  // The node gives the widget a fixed-height box: fill it (stage grows, bar stays at the bottom).
  const wrapper = el(
    "div",
    "display:flex;flex-direction:column;gap:6px;width:100%;height:100%;min-height:300px;box-sizing:border-box;padding:4px;",
  );
  wrapper.className = "nodrag nowheel";

  const stage = el(
    "div",
    "position:relative;flex:1 1 0;min-height:180px;background:#0e0e0e;border-radius:8px;overflow:hidden;" +
      "display:flex;align-items:center;justify-content:center;cursor:col-resize;user-select:none;",
  );
  const img = el("img", "display:none;width:100%;height:100%;object-fit:contain;pointer-events:none;");
  img.draggable = false;
  img.alt = "";
  const placeholder = el(
    "div",
    "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;" +
      "padding:20px;color:var(--muted-foreground, #999);font-size:13px;line-height:1.4;pointer-events:none;",
    "Press  Start live preview  on the node.",
  );
  const badgeL = el(
    "div",
    "position:absolute;left:8px;top:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.6);" +
      "color:#fff;font:11px/1.4 monospace;pointer-events:none;",
    "BEFORE",
  );
  const badgeR = el(
    "div",
    "position:absolute;right:8px;top:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.6);" +
      "color:#fff;font:11px/1.4 monospace;pointer-events:none;",
    "DLSS 5",
  );
  const hud = el(
    "div",
    "position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.6);" +
      "color:#ddd;font:11px/1.4 monospace;pointer-events:none;white-space:pre;",
    "",
  );
  stage.append(img, placeholder, badgeL, badgeR, hud);

  const bar = el("div", "display:flex;align-items:center;gap:6px;flex-wrap:wrap;");
  const playBtn = btn("Pause", "Play / pause (space)");
  const prevBtn = btn("‹", "Previous frame");
  const nextBtn = btn("›", "Next frame");
  const scrub = el("input", "flex:1;min-width:80px;cursor:pointer;");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = "0";
  scrub.step = "1";
  scrub.value = "0";
  ["pointerdown", "mousedown"].forEach((ev) => scrub.addEventListener(ev, (e) => e.stopPropagation()));
  const frameLbl = el("span", "font:11px monospace;color:var(--muted-foreground, #aaa);min-width:70px;text-align:right;", "0 / 0");

  const viewSel = el(
    "select",
    "padding:3px 6px;border-radius:5px;border:1px solid var(--border, #444);background:var(--background, #1b1b1b);" +
      "color:var(--foreground, #eee);font-size:12px;cursor:pointer;",
  );
  [
    ["wipe", "Wipe (drag)"],
    ["after", "DLSS 5 only"],
    ["before", "Source only"],
    ["split", "Side by side"],
  ].forEach(([v, t]) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    viewSel.appendChild(o);
  });
  ["pointerdown", "mousedown"].forEach((ev) => viewSel.addEventListener(ev, (e) => e.stopPropagation()));

  const speedSel = el(
    "select",
    "padding:3px 6px;border-radius:5px;border:1px solid var(--border, #444);background:var(--background, #1b1b1b);" +
      "color:var(--foreground, #eee);font-size:12px;cursor:pointer;",
  );
  speedSel.title = "Playback speed relative to the clip's frame rate. Max = as fast as the pipeline goes.";
  [
    ["0.25", "0.25x"],
    ["0.5", "0.5x"],
    ["1", "1x"],
    ["2", "2x"],
    ["0", "Max"],
  ].forEach(([v, t]) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    speedSel.appendChild(o);
  });
  speedSel.value = "1";
  ["pointerdown", "mousedown"].forEach((ev) => speedSel.addEventListener(ev, (e) => e.stopPropagation()));

  const fullBtn = btn("⛶", "Full screen (Esc to leave)");
  const closeBtn = btn("✕ Close", "Leave full screen (Esc)");
  closeBtn.hidden = true;

  bar.append(playBtn, prevBtn, nextBtn, scrub, frameLbl, speedSel, viewSel, fullBtn, closeBtn);
  wrapper.append(stage, bar);
  container.appendChild(wrapper);

  // ── full screen ─────────────────────────────────────────────────────────
  // The same stage + bar are moved into a fixed overlay (one MJPEG connection, all
  // listeners intact) and moved back on exit. Uses the Fullscreen API when allowed.
  const overlay = el(
    "div",
    "position:fixed;inset:0;z-index:2147483000;background:#000;display:none;flex-direction:column;gap:8px;" +
      "padding:10px;box-sizing:border-box;",
  );
  overlay.className = "nodrag nowheel";
  overlay.tabIndex = 0;
  let fullscreen = false;

  function enterFull() {
    if (fullscreen) return;
    fullscreen = true;
    document.body.appendChild(overlay);
    overlay.append(stage, bar);
    overlay.style.display = "flex";
    stage.style.borderRadius = "0";
    bar.style.justifyContent = "center";
    fullBtn.hidden = true;
    closeBtn.hidden = false;
    if (overlay.requestFullscreen) overlay.requestFullscreen().catch(() => {});
    overlay.focus();
  }

  function leaveFull() {
    if (!fullscreen) return;
    fullscreen = false;
    if (document.fullscreenElement === overlay && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    wrapper.append(stage, bar);
    overlay.style.display = "none";
    overlay.remove();
    stage.style.borderRadius = "8px";
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

  // ── server I/O ──────────────────────────────────────────────────────────
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
    hud.textContent = `${size}  ${perf}${s.worker_ms} ms GPU · ${s.frame_ms} ms/frame${enc}${s.sequence ? " · temporal" : ""}${dec}` +
      (s.message && s.message !== "live" ? `\n${s.message}` : "");
    if (s.error) {
      placeholder.textContent = s.error;
      placeholder.hidden = false;
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

  function startStream() {
    streamToken++;
    img.src = `${url}/stream.mjpg?t=${Date.now()}-${streamToken}`;
    img.style.display = "block";
    placeholder.hidden = true;
    clearInterval(pollTimer);
    pollTimer = setInterval(poll, POLL_MS);
    poll();
  }

  function stopStream(message) {
    clearInterval(pollTimer);
    pollTimer = null;
    img.removeAttribute("src");
    img.style.display = "none";
    placeholder.textContent = message || "Press  Start live preview  on the node.";
    placeholder.hidden = false;
    hud.textContent = "";
    badgeL.hidden = badgeR.hidden = true;
    leaveFull();
  }

  // ── interaction ─────────────────────────────────────────────────────────
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
    } catch {}
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

  // ── value updates from the node ─────────────────────────────────────────
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
      else stopStream(v.message || (v.status === "error" ? "Live preview failed." : undefined));
    } else if (!url && v.message) {
      placeholder.textContent = v.message;
    }
  }

  function cleanup() {
    clearInterval(pollTimer);
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
