
const $ = (id) => document.getElementById(id);

const queryParams = new URLSearchParams(window.location.search);
let apiBase = window.GESTURE_API_BASE || queryParams.get("api") || "";
const BROWSER_CAMERA_FRAME_INTERVAL_MS = 50;
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
const FINGER_LABELS = [
  ["thumb", "Thumb"],
  ["index", "Index"],
  ["middle", "Middle"],
  ["ring", "Ring"],
  ["pinky", "Pinky"],
];

let browserCameraMode = false;
let commandCache = [];
let browserStream = null;
let frameTimer = null;
let frameInFlight = false;
let currentSource = "";
let settingsDirty = false;

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
  else if (state === "error") badge.classList.add("badge-error");
  else badge.classList.add("badge-muted");
}

function setVideoStarted(started) {
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.toggle("video-started", Boolean(started));
  const startButton = $("startVideoButton");
  if (startButton) startButton.textContent = started ? "Video running" : "Start video";
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) sideStartButton.textContent = started ? "Running" : "Start";
  const stopButton = $("stopVideoButton");
  if (stopButton) stopButton.disabled = !browserCameraMode;
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

function recognitionLabel(gesture, confidence, minConfidence) {
  const value = `${gesture} / ${fmt(confidence)}`;
  if (gesture !== "UNKNOWN" && confidence < minConfidence) return `${value} low`;
  return value;
}

function reasonLabel(reason) {
  if (!reason) return "";
  return String(reason).replaceAll("_", " ");
}

function applyConfirmation(status) {
  const minConfidence = status.command_min_confidence ?? 0.65;
  const gesture = status.command_candidate_gesture || "UNKNOWN";
  const command = status.command_candidate_command || "UNKNOWN";
  const confidence = status.command_candidate_confidence ?? 0;
  const stableFrames = status.command_stable_frames || 0;
  const requiredFrames = status.command_required_frames || 0;
  const progress = Math.max(0, Math.min(1, status.command_progress || 0));
  const reason = status.command_blocked_reason || "";
  const panel = document.querySelector(".confirmation-panel");
  if (panel) {
    panel.classList.toggle("ready", Boolean(status.command_ready));
    panel.classList.toggle("blocked", reason === "confidence_below_threshold");
  }
  text("candidateCommand", command);
  text("candidateGesture", `${gesture} / ${fmt(confidence)}`);
  const bar = $("confirmationProgressBar");
  if (bar) bar.style.width = `${Math.round(progress * 100)}%`;
  if (reason === "confidence_below_threshold") {
    text("confirmationProgressText", `${fmt(confidence)} < ${fmt(minConfidence)} threshold`);
  } else if (reason === "repeat_suppressed") {
    text("confirmationProgressText", `${stableFrames} / ${requiredFrames} frames, already sent`);
  } else if (requiredFrames > 0) {
    text("confirmationProgressText", `${stableFrames} / ${requiredFrames} frames`);
  } else {
    text("confirmationProgressText", reasonLabel(reason) || "0 / 0 frames");
  }
}

function renderPoseDiagnostics(status) {
  const states = status.finger_states || {};
  const expected = (status.expected_pose && status.expected_pose.finger_states) || {};
  const directions = status.pose_directions || {};
  const node = $("fingerStates");
  if (!node) return;
  node.innerHTML = FINGER_LABELS.map(([key, label]) => {
    const actual = states[key];
    const target = expected[key];
    const actualText = actual === true ? "open" : actual === false ? "closed" : "--";
    const targetText = target === true ? "open" : target === false ? "closed" : "any";
    const matched = target === null || target === undefined || actual === target;
    return `
      <div class="finger-row">
        <strong>${label}</strong>
        <span>${actualText} / ${targetText}</span>
        <span class="${matched ? "match" : "mismatch"}">${matched ? "OK" : "NO"}</span>
      </div>
    `;
  }).join("");

  const parts = [];
  if (directions.thumb) parts.push(`thumb ${directions.thumb}`);
  if (directions.index) parts.push(`index ${directions.index}`);
  if (directions.ok_tip_distance !== undefined) {
    parts.push(`ok ${fmt(directions.ok_tip_distance)}`);
  }
  text("poseDirection", parts.length ? parts.join(" | ") : "--");
}

function renderLandmarks(landmarks) {
  const points = Array.isArray(landmarks) ? landmarks : [];
  text("landmarkCount", `${points.length} / 21`);
  const node = $("landmarkList");
  if (!node) return;
  if (!points.length) {
    node.innerHTML = '<div class="empty">No hand points</div>';
    return;
  }
  node.innerHTML = points.map((point) => `
    <div class="landmark-point">
      <strong>#${escapeHtml(point.id)}</strong>
      x ${escapeHtml(point.x)}<br>
      y ${escapeHtml(point.y)}<br>
      z ${escapeHtml(point.z)}
    </div>
  `).join("");
}

function applyStatus(status) {
  currentSource = status.source || "";
  setStateBadge(status.state);
  text("transportBadge", status.transport);
  text("transportText", status.transport);
  text("sourceLine", status.source);
  text("lastCommand", status.last_command);
  text("commandDisplay", status.last_command);
  text(
    "lastCommandMeta",
    `${status.last_command_gesture} / ${fmt(status.last_command_confidence)}`
  );
  text("selectedGesture", status.selected_gesture);
  text("selectedConfidence", fmt(status.selected_confidence));
  text("dynamicState", status.dynamic_state || "idle");
  text("robotCommandCount", status.command_count);
  text("robotLastGesture", status.last_command_gesture);
  text(
    "robotTransferState",
    status.last_command && status.last_command !== "UNKNOWN" ? "SENT" : "WAITING"
  );
  text("fps", fmt(status.fps, "", 1));
  text("latency", fmt(status.latency_ms, " ms", 1));
  text("handCount", status.hand_count);
  text("frameIndex", status.frame_index);
  const minConfidence = status.command_min_confidence ?? 0.65;
  text(
    "staticGesture",
    recognitionLabel(status.static_gesture, status.static_confidence, minConfidence)
  );
  text(
    "dynamicGesture",
    recognitionLabel(status.dynamic_gesture, status.dynamic_confidence, minConfidence)
  );
  text("updatedAt", status.updated_at);
  applyConfirmation(status);
  renderPoseDiagnostics(status);
  renderLandmarks(status.landmarks);
  renderGestureMap(status.selected_gesture);
  if (!settingsDirty) applySettings(settingsPayloadFromStatus(status));
  if (!browserCameraMode) setVideoStarted(status.state === "running");

  const errorPanel = $("errorPanel");
  if (status.error) {
    errorPanel.hidden = false;
    text("errorText", status.error);
  } else {
    errorPanel.hidden = true;
    text("errorText", "");
  }
}

function reloadStream() {
  const stream = $("stream");
  if (stream) stream.src = `${buildUrl("stream.mjpg")}?t=${Date.now()}`;
}

async function refreshStatus() {
  if (browserCameraMode) return;
  try {
    const response = await fetch(buildUrl("api/status"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    applyStatus(status);
  } catch (error) {
    showRuntimeError(
      cameraAccessMessage("Backend is unavailable. Start the video backend.")
    );
  }
}

function settingsPayloadFromStatus(status) {
  return {
    static_confirmation_frames: status.static_confirmation_frames,
    dynamic_confirmation_frames: status.dynamic_confirmation_frames,
    emergency_confirmation_frames: status.emergency_confirmation_frames,
  };
}

function applySettings(settings) {
  const pairs = [
    ["staticFramesInput", settings.static_confirmation_frames],
    ["dynamicFramesInput", settings.dynamic_confirmation_frames],
    ["emergencyFramesInput", settings.emergency_confirmation_frames],
  ];
  for (const [id, value] of pairs) {
    const input = $(id);
    if (input && document.activeElement !== input && value !== undefined) {
      input.value = value;
    }
  }
  if (!settingsDirty) text("settingsStatus", "saved");
}

function readSettings() {
  return {
    static_confirmation_frames: Number($("staticFramesInput")?.value || 1),
    dynamic_confirmation_frames: Number($("dynamicFramesInput")?.value || 1),
    emergency_confirmation_frames: Number($("emergencyFramesInput")?.value || 1),
  };
}

async function loadSettings() {
  try {
    const response = await fetch(buildUrl("api/settings"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applySettings(await response.json());
  } catch (error) {
    text("settingsStatus", "offline");
  }
}

async function saveSettings() {
  syncApiBaseFromInput();
  text("settingsStatus", "saving");
  try {
    const response = await fetch(buildUrl("api/settings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readSettings()),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    settingsDirty = false;
    applySettings(payload);
    text("settingsStatus", "saved");
  } catch (error) {
    text("settingsStatus", "error");
    showRuntimeError(error.message);
  }
}

async function refreshCommands() {
  const response = await fetch(buildUrl("api/commands"), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  commandCache = await response.json();
  renderCommands(commandCache);
}

async function startBrowserCamera() {
  syncApiBaseFromInput();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showRuntimeError(cameraAccessMessage("Browser camera API is unavailable on this page."));
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
    const stage = document.querySelector(".video-stage");
    if (stage) stage.classList.add("browser-active");
    video.hidden = false;
    setVideoStarted(true);
    setStateBadge("camera");
    text("sourceLine", apiBase ? `browser-camera -> ${apiBase}` : "browser-camera -> same-origin");
    frameTimer = window.setInterval(captureAndSendFrame, BROWSER_CAMERA_FRAME_INTERVAL_MS);
  } catch (error) {
    showRuntimeError(cameraAccessMessage(error));
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
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.remove("browser-active");
  setVideoStarted(false);
  reloadStream();
}

async function startVideo() {
  syncApiBaseFromInput();
  if (!currentSource || currentSource === "browser-camera" || browserCameraMode) {
    await startBrowserCamera();
    return;
  }
  setVideoStarted(true);
  reloadStream();
  refreshStatus();
  refreshCommands().catch(() => undefined);
}

function stopVideo() {
  if (browserCameraMode) {
    stopBrowserCamera();
    return;
  }
  setVideoStarted(false);
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
    if (payload.frame) {
      const stream = $("stream");
      if (stream) stream.src = payload.frame;
    }
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

function cameraAccessMessage(error) {
  const detail = typeof error === "string" ? error : error?.message || "";
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return [
      "Camera/video access is required to control the robot.",
      "Grant video permission and try again.",
    ].join(" ");
  }
  return detail
    ? `Camera/video access is required to control the robot. ${detail}`
    : "Camera/video access is required to control the robot.";
}

function bindControls() {
  const apiInput = $("apiBaseInput");
  if (apiInput) {
    apiInput.value = apiBase || window.localStorage.getItem("gestureApiBase") || "";
    apiBase = apiInput.value.trim();
    apiInput.addEventListener("change", syncApiBaseFromInput);
  }
  const startVideoButton = $("startVideoButton");
  if (startVideoButton) startVideoButton.addEventListener("click", startVideo);
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) sideStartButton.addEventListener("click", startVideo);
  const stopVideoButton = $("stopVideoButton");
  if (stopVideoButton) stopVideoButton.addEventListener("click", stopVideo);
  const reloadButton = $("reloadButton");
  if (reloadButton) reloadButton.addEventListener("click", reloadStream);
  for (const id of ["staticFramesInput", "dynamicFramesInput", "emergencyFramesInput"]) {
    const input = $(id);
    if (input) {
      input.addEventListener("input", () => {
        settingsDirty = true;
        text("settingsStatus", "unsaved");
      });
    }
  }
  const saveSettingsButton = $("saveSettingsButton");
  if (saveSettingsButton) saveSettingsButton.addEventListener("click", saveSettings);
}

bindControls();
renderGestureMap();
loadSettings();
refreshStatus();
refreshCommands().catch(() => undefined);
setInterval(refreshStatus, 500);
setInterval(() => {
  if (!browserCameraMode) refreshCommands().catch(() => undefined);
}, 1000);
