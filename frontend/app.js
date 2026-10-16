/*
 * SIGNAL frontend for teegg009/sign-language-recognition (FastAPI backend).
 * Put index.html, app.js and style.css in the repo's frontend/ folder, then run:
 *   py -3.11 -m uvicorn backend.main:app --reload --port 8000   → open http://localhost:8000
 *
 * Backend contract (backend/main.py):
 *   GET  /api/status   → { model_loaded: bool, labels: string[] }
 *   WS   /ws/predict   send { frame: "<jpeg dataURL>" } or { type: "clear" }
 *                      recv { current, sentence, confidence, hand_detected } or { error }
 */
const $ = (s) => document.querySelector(s);
const IGNORE_LABEL = "no_sign";        // backend IGNORE_LABEL — never added to the sentence
const SERVER_CONF_THRESHOLD = 55;      // backend CONF_THRESHOLD (0.55), in %
const fmtTime = (s) => String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/* ---------- Router ---------- */
function route() {
  const v = location.hash.slice(1) || "live";
  document.querySelectorAll("[data-view]").forEach((el) => (el.hidden = el.dataset.view !== v));
  document.querySelectorAll(".nav").forEach((b) => (b.dataset.go === v ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current")));
}
document.querySelectorAll(".nav").forEach((b) => b.addEventListener("click", () => (location.hash = b.dataset.go)));
addEventListener("hashchange", route);
route();

function banner(msg) { const b = $("#banner"); b.textContent = msg || ""; b.hidden = !msg; }

/* ---------- Server status (shared) ---------- */
const S = { modelLoaded: null, labels: [] };

async function checkModel() {
  try {
    const d = await (await fetch("/api/status")).json();
    S.modelLoaded = !!d.model_loaded; S.labels = d.labels || [];
    $("#model-state").textContent = "Model: " + (S.modelLoaded ? "model.pkl loaded" : "not loaded");
    $("#model-classes").textContent = S.modelLoaded ? S.labels.length + " classes" : "—";
    $("#label-list").innerHTML = S.labels.length
      ? S.labels.map((l) => `<span class="chip${l === IGNORE_LABEL ? " muted" : ""}">${esc(l)}</span>`).join("")
      : '<span class="muted">No model loaded.</span>';
    if (!S.modelLoaded) banner("No trained model found on the server (models/model.pkl). Run: py -3.11 scripts/train_model.py — then reload this page.");
    checkSnapshot();
  } catch {
    S.modelLoaded = null;
    $("#model-state").textContent = "Model: server unreachable";
    banner("Cannot reach the backend. Start it with: py -3.11 -m uvicorn backend.main:app --port 8000 and open http://localhost:8000");
  }
}

/* ================= LIVE ================= */
const L = { ws: null, stream: null, timer: null, running: false, paused: false, fatal: false, t0: 0, n: 0, sent: [], lat: null, frames: 0, segs: [], dur: null };
const video = $("#webcam"), canvas = $("#capture"), ctx = canvas.getContext("2d");
canvas.width = 320; canvas.height = 240; // same size as the repo's original frontend

function setConn(text, cls) { $("#conn-state").textContent = text; $("#conn-dot").className = "dot " + (cls || ""); }

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support camera access. Use a current Chrome, Edge, Safari or Firefox at http://localhost:8000 (or HTTPS).");
  try {
    L.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (e) {
    const m = { NotAllowedError: "Camera access was blocked. Click the camera icon in the address bar, allow access, then press Start again.", NotFoundError: "No camera was found. Connect one and press Start again.", NotReadableError: "The camera is in use by another app (for example collect_data.py or translate_realtime.py). Close it and press Start again." };
    throw new Error(m[e.name] || "The camera could not start (" + e.name + "). Close other apps using it and try again.");
  }
  video.srcObject = L.stream;
  $("#cam").classList.add("on"); $("#cam-empty").hidden = true;
}

function stopCamera() {
  L.stream?.getTracks().forEach((t) => t.stop());
  L.stream = null; video.srcObject = null;
  $("#cam").classList.remove("on"); $("#cam-empty").hidden = false;
}

function connectWS() {
  L.ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/predict`);
  L.ws.onopen = () => { setConn("Connected", "ok"); startFrames(); };
  L.ws.onerror = () => setConn("Connection error", "bad");
  L.ws.onclose = () => {
    stopFrames();
    // Don't loop reconnecting when the server refused us (e.g. no model.pkl).
    if (L.running && !L.fatal) { setConn("Reconnecting…", "bad"); setTimeout(() => L.running && !L.fatal && connectWS(), 2000); }
  };
  L.ws.onmessage = (e) => { try { onPrediction(JSON.parse(e.data)); } catch { /* ignore malformed message */ } };
}

function startFrames() {
  if (L.timer) return;
  L.timer = setInterval(() => {
    if (L.paused || !L.ws || L.ws.readyState !== 1 || video.readyState < 2) return;
    if (L.sent.length > 5) return; // server is behind — don't pile up frames
    // Mirror the frame, like cv2.flip(image, 1) in collect_data.py, so it matches the training data.
    ctx.save(); ctx.translate(320, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, 320, 240); ctx.restore();
    L.sent.push(performance.now());
    L.ws.send(JSON.stringify({ frame: canvas.toDataURL("image/jpeg", 0.6) }));
  }, 130);
}
function stopFrames() { clearInterval(L.timer); L.timer = null; L.sent = []; }

function onPrediction(d) {
  if (d.error) {
    L.fatal = true; banner(d.error); stopSession(); setConn("Refused by server", "bad");
    return;
  }
  const t = L.sent.shift();
  if (t) L.lat = performance.now() - t;
  L.n++; L.frames++;

  const pct = (d.confidence || 0) * 100, thr = +$("#thr").value;
  const label = d.current || "", isNoSign = label === IGNORE_LABEL, low = pct < thr;
  $("#hand-state").textContent = d.hand_detected ? "Hand detected (MediaPipe)" : "No hand in view";

  const now = $("#now");
  now.textContent = !label ? "—" : isNoSign ? "no sign" : label;
  now.classList.toggle("low", !!label && !isNoSign && low);
  now.classList.toggle("idle", !label || isNoSign);

  $("#conf").textContent = pct.toFixed(1) + "%";
  $("#conf-note").textContent = !d.hand_detected
    ? "no hand — nothing to classify"
    : low ? "below your " + thr + "% display threshold — treat as uncertain"
    : "confidence of this frame (not model accuracy)";
  $("#meter-fill").style.width = pct + "%";
  $("#meter-fill").classList.toggle("low", low);
  $("#meter").setAttribute("aria-valuenow", Math.round(pct));

  $("#caption").textContent = d.sentence || "";
  commit(d.sentence || "");
  $("#s-n").textContent = L.n;
  $("#s-lat").textContent = L.lat == null ? "N/A" : Math.round(L.lat) + " ms";
}

// The server decides when a sign is stable (rolling window); we only add words it has newly committed.
// Note: the server keeps the last 8 words, so older words scroll out of its sentence — we keep them here.
// The server joins labels with spaces, and some labels contain spaces ("I LOVE YOU", "Dont know"),
// so split using the model's own label list, longest label first.
function tokenize(sentence) {
  const parts = sentence.split(" ").filter(Boolean), labels = S.labels.map((l) => l.split(" ")).sort((a, b) => b.length - a.length), out = [];
  for (let i = 0; i < parts.length; ) {
    const hit = labels.find((l) => l.every((w, k) => parts[i + k] === w));
    const len = hit ? hit.length : 1;
    out.push(parts.slice(i, i + len).join(" ")); i += len;
  }
  return out;
}
function commit(sentence) {
  const words = tokenize(sentence), now = new Date().toLocaleTimeString([], { hour12: false });
  if (!words.length) { L.serverWords = []; return; }
  const prev = L.serverWords || [];
  // Find how many words are new by matching the longest overlap between end of prev and start of words.
  let added = words.length;
  for (let k = Math.min(prev.length, words.length); k >= 0; k--) {
    if (prev.slice(prev.length - k).every((w, i) => w === words[i])) { added = words.length - k; break; }
  }
  L.serverWords = words;
  if (!added) return;
  words.slice(words.length - added).forEach((w) => L.segs.push({ w, t: now }));
  renderTranscript();
}
function renderTranscript() {
  const box = $("#transcript");
  box.innerHTML = L.segs.length ? "" : '<span class="muted">Stable signs appear here.</span>';
  L.segs.forEach((s) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = s.w; const t = document.createElement("small"); t.textContent = s.t; c.appendChild(t); box.appendChild(c); });
  box.scrollTop = box.scrollHeight;
}

$("#btn-start").addEventListener("click", async () => {
  if (L.running) return stopSession();
  banner(""); L.fatal = false;
  if (S.modelLoaded === false) return banner("The server has no trained model (models/model.pkl). Run: py -3.11 scripts/train_model.py — then reload this page.");
  try { await startCamera(); } catch (e) { return banner(e.message); }
  L.running = true; L.paused = false; L.n = 0; L.frames = 0; L.lat = null; L.t0 = Date.now(); L.serverWords = [];
  $("#btn-start").textContent = "Stop session"; $("#btn-pause").disabled = false; setConn("Connecting…");
  connectWS();
  L.dur = setInterval(() => {
    $("#s-dur").textContent = fmtTime(Math.floor((Date.now() - L.t0) / 1000));
    $("#s-fps").textContent = L.paused ? "paused" : L.frames + " /s"; L.frames = 0;
  }, 1000);
});
function stopSession() {
  L.running = false; L.paused = false; clearInterval(L.dur);
  if (L.ws) { L.ws.onclose = null; L.ws.close(); L.ws = null; }
  stopFrames(); stopCamera(); if (!L.fatal) setConn("Offline");
  $("#btn-start").textContent = "Start recognition"; $("#btn-pause").disabled = true; $("#btn-pause").textContent = "Pause";
  $("#hand-state").textContent = "Camera off"; $("#s-fps").textContent = "N/A";
}
$("#btn-pause").addEventListener("click", () => { L.paused = !L.paused; $("#btn-pause").textContent = L.paused ? "Resume" : "Pause"; });
$("#thr").addEventListener("input", (e) => ($("#thr-v").textContent = e.target.value + "%"));
$("#btn-clear").addEventListener("click", () => {
  // The server answers "clear" with an empty frame result, which isn't a frame round trip.
  if (L.ws?.readyState === 1) { L.sent.push(0); L.ws.send(JSON.stringify({ type: "clear" })); }
  L.segs = []; L.serverWords = []; renderTranscript(); $("#caption").textContent = ""; $("#now").textContent = "—";
});
$("#btn-copy").addEventListener("click", async (e) => {
  const b = e.currentTarget; try { await navigator.clipboard.writeText(L.segs.map((s) => s.w).join(" ")); b.textContent = "Copied"; } catch { b.textContent = "Copy failed"; }
  setTimeout(() => (b.textContent = "Copy"), 1200);
});
addEventListener("pagehide", stopSession);
$("#thr-server").textContent = SERVER_CONF_THRESHOLD + "%";

/* ================= TRAIN (simulated preview) ================= */
const T = { state: "idle", e: 0, E: 30, timer: null };
const curves = (E) => {
  const tr = [], va = [], tl = [], vl = [];
  for (let e = 1; e <= E; e++) {
    const k = Math.exp((-3 * e) / E), w = Math.sin(e * 1.7) * 0.012;
    tl.push(1.5 * k + 0.08); vl.push(1.5 * k * 1.1 + 0.14 + w);
    tr.push(0.97 - 0.62 * k); va.push(0.93 - 0.6 * k + w);
  }
  return { tr, va, tl, vl };
};
let C = curves(30);

function chart(svg, a, b, lo, hi, E) {
  const W = 320, H = 120, p = { l: 28, r: 6, t: 6, b: 16 };
  const X = (i) => p.l + (i / Math.max(E - 1, 1)) * (W - p.l - p.r), Y = (v) => p.t + (1 - (v - lo) / (hi - lo)) * (H - p.t - p.b);
  const pts = (s) => s.map((v, i) => X(i) + "," + Y(v)).join(" ");
  svg.innerHTML = `<line x1="${p.l}" x2="${W - p.r}" y1="${Y(lo)}" y2="${Y(lo)}" stroke="#2a323d"/><line x1="${p.l}" x2="${W - p.r}" y1="${Y(hi)}" y2="${Y(hi)}" stroke="#2a323d"/>
  <text x="2" y="${Y(hi) + 3}">${hi}</text><text x="2" y="${Y(lo) + 3}">${lo}</text><text x="${W / 2 - 10}" y="${H - 3}">epoch</text>
  <polyline fill="none" stroke="hsl(183,82%,58%)" stroke-width="2" points="${pts(a)}"/><polyline fill="none" stroke="hsl(38,90%,64%)" stroke-width="2" stroke-dasharray="5 3" points="${pts(b)}"/>`;
}
function drawTrain() {
  chart($("#ch-loss"), C.tl.slice(0, T.e), C.vl.slice(0, T.e), 0, 1.6, T.E);
  chart($("#ch-acc"), C.tr.slice(0, T.e), C.va.slice(0, T.e), 0.3, 1, T.E);
  $("#t-fill").style.width = (T.E ? (T.e / T.E) * 100 : 0) + "%";
  const best = T.e ? Math.max(...C.va.slice(0, T.e)) : 0;
  $("#t-meta").textContent = `Epoch ${T.e} / ${T.E}   Best val. accuracy ${T.e ? (best * 100).toFixed(1) + "%" : "N/A"}   (simulated)`;
}
function setT(s) {
  T.state = s;
  const names = { idle: "Idle", validating: "Validating dataset", training: "Training (simulated)", paused: "Paused", evaluating: "Evaluating", completed: "Completed (simulated)", cancelled: "Cancelled" };
  const el = $("#t-state"); el.textContent = names[s]; el.className = "badge" + (s === "completed" ? " ok" : "");
  const busy = ["validating", "training", "evaluating", "paused"].includes(s);
  $("#t-start").disabled = busy; $("#t-pause").disabled = !(s === "training" || s === "paused");
  $("#t-pause").textContent = s === "paused" ? "Resume" : "Pause"; $("#t-cancel").disabled = !busy;
}
function tick() {
  if (T.state !== "training") return;
  if (++T.e >= T.E) { T.e = T.E; clearInterval(T.timer); setT("evaluating"); drawTrain(); return setTimeout(() => T.state === "evaluating" && setT("completed"), 900); }
  drawTrain();
}
$("#cfg").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const ep = +$("#c-ep").value, lr = +$("#c-lr").value, err = $("#cfg-err");
  if (!Number.isInteger(ep) || ep < 5 || ep > 100) return (err.textContent = "Epochs must be a whole number from 5 to 100.");
  if (!(lr >= 1e-5 && lr <= 0.1)) return (err.textContent = "Learning rate must be between 0.00001 and 0.1.");
  err.textContent = ""; T.E = ep; T.e = 0; C = curves(ep); drawTrain(); setT("validating");
  setTimeout(() => { if (T.state !== "validating") return; setT("training"); T.timer = setInterval(tick, 350); }, 800);
});
$("#t-pause").addEventListener("click", () => setT(T.state === "paused" ? "training" : "paused"));
$("#t-cancel").addEventListener("click", () => { clearInterval(T.timer); setT("cancelled"); });
drawTrain(); setT("idle");
document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; } catch { b.textContent = "Copy failed"; }
  setTimeout(() => (b.textContent = "Copy"), 1200);
}));

/* ================= RESULTS ================= */
// Snapshot of the temporal split from scripts/train_model.py on data/gestures.csv (1485 samples, 14 classes):
// first 80% of each class in collection order = train, last 20% = test (296 test samples).
// Re-run train_model.py and update this block after you collect new data or retrain.
const EVAL = {
  split: "Temporal 80/20 split (scripts/train_model.py)",
  dataset: "data/gestures.csv — 1485 samples",
  labels: ["100", "Cry", "Dont know", "E", "I", "I LOVE YOU", "L", "Y", "mini heart", "no_sign", "o", "u", "v", "why"],
  matrix: [
    [20, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 1],
    [14, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39],
  ],
};

(function results() {
  const { labels: LB, matrix: M } = EVAL, n = LB.length, tot = M.flat().reduce((a, b) => a + b, 0);
  const rowS = M.map((r) => r.reduce((a, b) => a + b, 0)), colS = LB.map((_, j) => M.reduce((a, r) => a + r[j], 0));
  const div = (a, b) => (b ? a / b : 0);
  const P = LB.map((_, i) => div(M[i][i], colS[i])), R = LB.map((_, i) => div(M[i][i], rowS[i])), F = P.map((p, i) => div(2 * p * R[i], p + R[i]));
  const avg = (a) => a.reduce((x, y) => x + y, 0) / n, pc = (v) => (v * 100).toFixed(1) + "%";
  const acc = M.reduce((a, r, i) => a + r[i], 0) / tot;
  $("#eval-src").textContent = EVAL.split + " · " + EVAL.dataset;
  $("#kpis").innerHTML = [["Accuracy", pc(acc)], ["Precision (macro)", pc(avg(P))], ["Recall (macro)", pc(avg(R))], ["F1 (macro)", pc(avg(F))], ["Test samples", tot]].map(([k, v]) => `<div class="kpi"><span class="muted small">${k}</span><b>${v}</b></div>`).join("");
  let h = '<caption class="muted small" style="text-align:left;padding-bottom:6px">Shading = share of that true sign\'s test samples. Hover a cell for details.</caption><tr><th scope="col">true ↓ / predicted →</th>' + LB.map((l) => `<th scope="col">${esc(l)}</th>`).join("") + "</tr>";
  M.forEach((r, i) => (h += `<tr><th scope="row">${esc(LB[i])}</th>` + r.map((v, j) => {
    const share = div(v, rowS[i]), off = i !== j && v > 0;
    return `<td class="cell${off ? " off" : ""}" title="True ${esc(LB[i])} → predicted ${esc(LB[j])}: ${v} of ${rowS[i]} samples" style="background:hsl(${off ? "38 90% 64%" : "183 82% 58%"} / ${v ? 0.12 + share * 0.75 : 0.03})">${v}</td>`;
  }).join("") + "</tr>"));
  $("#matrix").innerHTML = h;
  $("#perclass").innerHTML = "<tr><th scope='col'>Sign</th><th scope='col'>Precision</th><th scope='col'>Recall</th><th scope='col'>F1</th><th scope='col'>Test n</th></tr>" + LB.map((l, i) => `<tr><th scope="row">${esc(l)}</th><td class="mono">${pc(P[i])}</td><td class="mono${R[i] < 0.8 ? " warn" : ""}">${pc(R[i])}</td><td class="mono">${pc(F[i])}</td><td class="mono">${rowS[i]}</td></tr>`).join("");

  // Biggest confusion, computed rather than hardcoded.
  let worst = null;
  M.forEach((r, i) => r.forEach((v, j) => { if (i !== j && v && (!worst || v > worst.v)) worst = { v, i, j }; }));
  $("#insight").textContent = worst
    ? `Largest confusion: ${worst.v} of ${rowS[worst.i]} "${LB[worst.i]}" test samples were predicted as "${LB[worst.j]}".`
    : "No off-diagonal errors in this snapshot.";
})();

// Warn when the running model's classes differ from this snapshot (model was retrained).
function checkSnapshot() {
  if (!S.modelLoaded) return;
  const same = S.labels.length === EVAL.labels.length && S.labels.every((l, i) => l === EVAL.labels[i]);
  $("#snap-warn").hidden = same;
}

checkModel();