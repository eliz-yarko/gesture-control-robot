
const $ = (id) => document.getElementById(id);

const API_BASE = window.GESTURE_API_BASE || "";
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
let demoIndex = 0;
let commandCache = [];
let lastBackendOk = true;

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

function setDemoMode(enabled) {
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

function buildUrl(path) {
  const prefix = API_BASE.replace(/\/$/, "");
  return `${prefix}/${path.replace(/^\//, "")}`;
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
  if (demoMode) return;
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

function bindControls() {
  const demoButton = $("demoToggle");
  if (demoButton) demoButton.addEventListener("click", () => setDemoMode(!demoMode));
  const reloadButton = $("reloadButton");
  if (reloadButton) reloadButton.addEventListener("click", reloadStream);
}

bindControls();
renderGestureMap();
refreshStatus();
refreshCommands();
setInterval(refreshStatus, 500);
setInterval(() => {
  if (demoMode) applyDemoStatus();
}, 1300);
setInterval(() => {
  if (!demoMode) refreshCommands().catch(() => undefined);
}, 1000);
