
const $ = (id) => document.getElementById(id);

const queryParams = new URLSearchParams(window.location.search);
let apiBase = window.GESTURE_API_BASE || queryParams.get("api") || "";
const DEMO_SEQUENCE = [
  ["OPEN_PALM", "STOP", 0.96, "OPEN_PALM", "UNKNOWN"],
  ["FIST", "FORWARD", 0.91, "FIST", "UNKNOWN"],
  ["INDEX_LEFT", "TURN_LEFT", 0.88, "INDEX_LEFT", "UNKNOWN"],
  ["INDEX_RIGHT", "TURN_RIGHT", 0.90, "INDEX_RIGHT", "UNKNOWN"],
  ["THUMB_UP", "START", 0.94, "THUMB_UP", "UNKNOWN"],
  ["WAVE_LR", "MODE_TOGGLE", 0.86, "OPEN_PALM", "WAVE_LR"],
  ["CIRCLE", "ROTATE_360", 0.84, "UNKNOWN", "CIRCLE"],
  ["PULL_TOWARD", "APPROACH_OPERATOR", 0.82, "UNKNOWN", "PULL_TOWARD"],
];
const GESTURE_COMMANDS = [
  ["OPEN_PALM", "STOP"],
  ["FIST", "FORWARD"],
  ["THUMB_UP", "START"],
  ["THUMB_DOWN", "EMERGENCY_STOP"],
  ["INDEX_LEFT", "TURN_LEFT"],
  ["INDEX_RIGHT", "TURN_RIGHT"],
  ["PEACE", "INCREASE_SPEED"],
  ["THREE_FINGERS", "DECREASE_SPEED"],
  ["PINKY", "RETURN_HOME"],
  ["OK_SIGN", "CONFIRM_ACTION"],
  ["WAVE_LR", "MODE_TOGGLE"],
  ["CIRCLE", "ROTATE_360"],
  ["PULL_TOWARD", "APPROACH_OPERATOR"],
];

let demoMode = false;
let browserCameraMode = false;
let demoIndex = 0;
let commandCache = [];
let lastBackendOk = true;
let browserStream = null;
let frameTimer = null;
let frameInFlight = false;

function text(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function fmt(value, suffix = "", digits = 2) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return `${value.toFixed(digits)}${suffix}`;
  return `${value}${suffix}`;
}

function setStateBadge(state) {
  const badge = $("stateBadge");
  if (!badge) return;
  badge.textContent = state;
  badge.className = "badge";
  if (state === "running") badge.classList.add("badge-running");
  else if (state === "demo") badge.classList.add("badge-demo");
  else if (state === "error") badge.classList.add("badge-error");
  else badge.classList.add("badge-muted");
}

function buildUrl(path) {
  const prefix = apiBase.replace(/\/$/, "");
  return `${prefix}/${path.replace(/^\//, "")}`;
}

function syncApiBaseFromInput() {
  const input = $("apiBaseInput");
  if (!input) return;
  apiBase = input.value.trim();
  if (apiBase) window.localStorage.setItem("gestureApiBase", apiBase);
  else window.localStorage.removeItem("gestureApiBase");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nowTime() {
  return new Date().toLocaleTimeString("uk-UA", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function applyStatus(status) {
  setStateBadge(status.state);
  text("transportBadge", status.transport);
  text("sourceLine", status.source);
  text("lastCommand", status.last_command);
  text("commandDisplay", status.last_command);
  text(
    "lastCommandMeta",
    `${status.last_command_gesture} / ${fmt(status.last_command_confidence)}`
  );
  text("selectedGesture", status.selected_gesture);
  text("selectedConfidence", fmt(status.selected_confidence));
  text("fps", fmt(status.fps, "", 1));
  text("latency", fmt(status.latency_ms, " ms", 1));
  text("handCount", status.hand_count);
  text("frameIndex", status.frame_index);
  text("staticGesture", `${status.static_gesture} / ${fmt(status.static_confidence)}`);
  text("dynamicGesture", `${status.dynamic_gesture} / ${fmt(status.dynamic_confidence)}`);
  text("updatedAt", status.updated_at);
  renderGestureMap(status.selected_gesture);

  const errorPanel = $("errorPanel");
  if (status.error) {
    errorPanel.hidden = false;
    text("errorText", status.error);
  } else {
    errorPanel.hidden = true;
    text("errorText", "");
  }
}

function demoStatus() {
  const [gesture, command, confidence, staticGesture, dynamicGesture] =
    DEMO_SEQUENCE[demoIndex % DEMO_SEQUENCE.length];
  return {
    state: "demo",
    source: "static-demo",
    transport: "mock",
    frame_index: 1200 + demoIndex * 18,
    fps: 24.0 + (demoIndex % 3) * 0.7,
    latency_ms: 36.0 + (demoIndex % 4) * 4.5,
    hand_count: 1,
    static_gesture: staticGesture,
    static_confidence: staticGesture === "UNKNOWN" ? 0 : Math.max(0.78, confidence - 0.04),
    dynamic_gesture: dynamicGesture,
    dynamic_confidence: dynamicGesture === "UNKNOWN" ? 0 : confidence,
    selected_gesture: gesture,
    selected_confidence: confidence,
    last_command: command,
    last_command_gesture: gesture,
    last_command_confidence: confidence,
    command_count: commandCache.length,
    error: "",
    updated_at: nowTime(),
  };
}

function setDemoMode(enabled) {
  if (enabled && browserCameraMode) stopBrowserCamera();
  demoMode = enabled;
  const button = $("demoToggle");
  if (button) button.classList.toggle("active", demoMode);
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.toggle("demo-active", demoMode);
  const demoFrame = $("demoFrame");
  if (demoFrame) demoFrame.hidden = !demoMode;
  if (demoMode) {
    applyDemoStatus();
    renderCommands(commandCache);
  } else {
    reloadStream();
  }
}

function applyDemoStatus() {
  const status = demoStatus();
  applyStatus(status);
  if (!commandCache.length || commandCache[0].gesture !== status.selected_gesture) {
    commandCache.unshift({
      created_at: status.updated_at,
      command: status.last_command,
      gesture: status.selected_gesture,
      confidence: Number(status.selected_confidence.toFixed(2)),
      frame_index: status.frame_index,
    });
    commandCache = commandCache.slice(0, 10);
  }
  renderCommands(commandCache);
  demoIndex += 1;
}

function reloadStream() {
  const stream = $("stream");
  if (stream) stream.src = `${buildUrl("stream.mjpg")}?t=${Date.now()}`;
}

async function refreshStatus() {
  if (demoMode || browserCameraMode) return;
  try {
    const response = await fetch(buildUrl("api/status"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    lastBackendOk = true;
    applyStatus(status);
  } catch (error) {
    if (lastBackendOk) setDemoMode(true);
    lastBackendOk = false;
    const status = demoStatus();
    status.error = "Backend offline; demo data is active.";
    applyStatus(status);
  }
}

async function refreshCommands() {
  if (demoMode) {
    renderCommands(commandCache);
    return;
  }
  const response = await fetch(buildUrl("api/commands"), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  commandCache = await response.json();
  renderCommands(commandCache);
}

async function startBrowserCamera() {
  setDemoMode(false);
  syncApiBaseFromInput();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showRuntimeError("Browser camera API is unavailable on this page.");
    return;
  }
  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });
    const video = $("browserVideo");
    if (!video) return;
    video.srcObject = browserStream;
    await video.play();
    browserCameraMode = true;
    const button = $("browserCameraToggle");
    if (button) button.classList.add("active");
    const stage = document.querySelector(".video-stage");
    if (stage) stage.classList.add("browser-active");
    video.hidden = false;
    const demoFrame = $("demoFrame");
    if (demoFrame) demoFrame.hidden = true;
    setStateBadge("camera");
    text("sourceLine", apiBase ? `browser-camera -> ${apiBase}` : "browser-camera -> same-origin");
    frameTimer = window.setInterval(captureAndSendFrame, 220);
  } catch (error) {
    showRuntimeError(error.message);
  }
}

function stopBrowserCamera() {
  browserCameraMode = false;
  if (frameTimer !== null) {
    window.clearInterval(frameTimer);
    frameTimer = null;
  }
  if (browserStream) {
    for (const track of browserStream.getTracks()) track.stop();
    browserStream = null;
  }
  const video = $("browserVideo");
  if (video) {
    video.pause();
    video.srcObject = null;
    video.hidden = true;
  }
  const button = $("browserCameraToggle");
  if (button) button.classList.remove("active");
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.remove("browser-active");
  reloadStream();
}

async function captureAndSendFrame() {
  if (!browserCameraMode || frameInFlight) return;
  const video = $("browserVideo");
  const canvas = $("captureCanvas");
  if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;
  frameInFlight = true;
  try {
    const width = 480;
    const height = Math.max(1, Math.round(width * (video.videoHeight / video.videoWidth)));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas capture context is unavailable.");
    context.drawImage(video, 0, 0, width, height);
    const image = canvas.toDataURL("image/jpeg", 0.72);
    const response = await fetch(buildUrl("api/frame"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    if (!response.ok) throw new Error(`Frame API HTTP ${response.status}`);
    const payload = await response.json();
    if (payload.status) applyStatus(payload.status);
    if (payload.commands) {
      commandCache = payload.commands;
      renderCommands(commandCache);
    }
  } catch (error) {
    showRuntimeError(error.message);
  } finally {
    frameInFlight = false;
  }
}

function renderCommands(commands) {
  const body = $("commandRows");
  if (!body) return;
  if (!commands.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No commands emitted</td></tr>';
    return;
  }
  body.innerHTML = commands.map((entry) => `
    <tr>
      <td>${escapeHtml(entry.created_at)}</td>
      <td>${escapeHtml(entry.command)}</td>
      <td>${escapeHtml(entry.gesture)}</td>
      <td>${escapeHtml(entry.confidence)}</td>
      <td>${escapeHtml(entry.frame_index)}</td>
    </tr>
  `).join("");
}

function renderGestureMap(activeGesture = "UNKNOWN") {
  const node = $("gestureMap");
  if (!node) return;
  node.innerHTML = GESTURE_COMMANDS.map(([gesture, command]) => `
    <div class="gesture-item ${gesture === activeGesture ? "active" : ""}">
      <strong>${gesture}</strong>
      <span>${command}</span>
    </div>
  `).join("");
}

function showRuntimeError(message) {
  setStateBadge(browserCameraMode ? "camera" : "error");
  const errorPanel = $("errorPanel");
  if (errorPanel) errorPanel.hidden = false;
  text("errorText", message);
}

function bindControls() {
  const apiInput = $("apiBaseInput");
  if (apiInput) {
    apiInput.value = apiBase || window.localStorage.getItem("gestureApiBase") || "";
    apiBase = apiInput.value.trim();
    apiInput.addEventListener("change", syncApiBaseFromInput);
  }
  const cameraButton = $("browserCameraToggle");
  if (cameraButton) {
    cameraButton.addEventListener("click", () => {
      if (browserCameraMode) stopBrowserCamera();
      else startBrowserCamera();
    });
  }
  const demoButton = $("demoToggle");
  if (demoButton) demoButton.addEventListener("click", () => setDemoMode(!demoMode));
  const reloadButton = $("reloadButton");
  if (reloadButton) reloadButton.addEventListener("click", reloadStream);
}

bindControls();
renderGestureMap();
refreshStatus();
refreshCommands().catch(() => undefined);
setInterval(refreshStatus, 500);
setInterval(() => {
  if (demoMode) applyDemoStatus();
}, 1300);
setInterval(() => {
  if (!demoMode && !browserCameraMode) refreshCommands().catch(() => undefined);
}, 1000);
