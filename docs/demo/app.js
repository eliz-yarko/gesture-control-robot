
const $ = (id) => document.getElementById(id);

const queryParams = new URLSearchParams(window.location.search);
let apiBase = window.GESTURE_API_BASE || queryParams.get("api") || "";
const BROWSER_CAMERA_FRAME_INTERVAL_MS = 50;
const TRANSLATIONS = {
  en: {
    appTitle: "Gesture Control Console",
    cameraFeed: "camera feed",
    liveGestureRecognition: "Live gesture recognition",
    start: "Start",
    stop: "Stop",
    reload: "Reload",
    startVideo: "Start video",
    videoRunning: "Video running",
    cameraAccessNote:
      "Camera/video access is required to recognize gestures and control the robot.",
    selectedGesture: "Selected gesture",
    preparedCommand: "Prepared command",
    confidence: "Confidence",
    motionState: "Motion state",
    commandFlow: "command flow",
    confirmationToRobot: "Confirmation to robot",
    waiting: "WAITING",
    sentState: "SENT",
    candidate: "Candidate",
    sentCommand: "Sent command",
    transport: "Transport",
    sent: "Sent",
    lastGesture: "Last gesture",
    confirmationRules: "confirmation rules",
    framesBeforeSend: "Frames before send",
    saved: "saved",
    saving: "saving",
    unsaved: "unsaved",
    offline: "offline",
    error: "error",
    stillGesture: "Still gesture",
    motionGesture: "Motion gesture",
    emergencyStop: "Emergency stop",
    apply: "Apply",
    latency: "Latency",
    hands: "Hands",
    frame: "Frame",
    stillRecognition: "Still recognition",
    motionRecognition: "Motion recognition",
    handPoseDiagnostics: "Hand pose diagnostics",
    technicalHandLandmarks: "Technical hand landmarks",
    runtime: "runtime",
    commandLog: "Command Log",
    time: "Time",
    command: "Command",
    gesture: "Gesture",
    noCommands: "No commands emitted",
    gestureMap: "Gesture Map",
    gestureCount: "13 gestures",
    frames: "frames",
    threshold: "threshold",
    alreadySent: "already sent",
    lowConfidence: "low",
    open: "open",
    closed: "closed",
    any: "any",
    ok: "OK",
    no: "NO",
    noHandPoints: "No hand points",
    cameraBackendUnavailable: "Backend is unavailable. Start the video backend.",
    browserCameraUnavailable: "Browser camera API is unavailable on this page.",
    cameraAccessRequired: "Camera/video access is required to control the robot.",
    grantVideoPermission: "Grant video permission and try again.",
    canvasUnavailable: "Canvas capture context is unavailable.",
    gestureOpenPalm: "Open palm",
    gestureFist: "Closed fist",
    gestureThumbUp: "Thumb up",
    gestureThumbDown: "Thumb down",
    gestureIndexLeft: "Index left",
    gestureIndexRight: "Index right",
    gesturePeace: "Peace sign",
    gestureThreeFingers: "Three fingers",
    gesturePinky: "Pinky",
    gestureOkSign: "OK sign",
    gestureWaveLr: "Wave left/right",
    gestureCircle: "Circle motion",
    gesturePullToward: "Pull toward",
    fingerThumb: "Thumb",
    fingerIndex: "Index",
    fingerMiddle: "Middle",
    fingerRing: "Ring",
    fingerPinky: "Pinky",
  },
  uk: {
    appTitle: "Консоль керування жестами",
    cameraFeed: "камера",
    liveGestureRecognition: "Розпізнавання жестів наживо",
    start: "Старт",
    stop: "Стоп",
    reload: "Оновити",
    startVideo: "Запустити відео",
    videoRunning: "Відео запущено",
    cameraAccessNote: "Доступ до камери потрібен для розпізнавання жестів і керування роботом.",
    selectedGesture: "Обраний жест",
    preparedCommand: "Підготовлена команда",
    confidence: "Впевненість",
    motionState: "Стан руху",
    commandFlow: "передача команди",
    confirmationToRobot: "Підтвердження для робота",
    waiting: "ОЧІКУВАННЯ",
    sentState: "НАДІСЛАНО",
    candidate: "Кандидат",
    sentCommand: "Надіслана команда",
    transport: "Транспорт",
    sent: "Надіслано",
    lastGesture: "Останній жест",
    confirmationRules: "правила підтвердження",
    framesBeforeSend: "Кадри перед надсиланням",
    saved: "збережено",
    saving: "збереження",
    unsaved: "не збережено",
    offline: "офлайн",
    error: "помилка",
    stillGesture: "Статичний жест",
    motionGesture: "Динамічний жест",
    emergencyStop: "Аварійна зупинка",
    apply: "Застосувати",
    latency: "Затримка",
    hands: "Руки",
    frame: "Кадр",
    stillRecognition: "Статичне розпізнавання",
    motionRecognition: "Динамічне розпізнавання",
    handPoseDiagnostics: "Діагностика пози руки",
    technicalHandLandmarks: "Технічні точки руки",
    runtime: "виконання",
    commandLog: "Журнал команд",
    time: "Час",
    command: "Команда",
    gesture: "Жест",
    noCommands: "Команди ще не надсилались",
    gestureMap: "Карта жестів",
    gestureCount: "13 жестів",
    frames: "кадрів",
    threshold: "поріг",
    alreadySent: "уже надіслано",
    lowConfidence: "низько",
    open: "відкрито",
    closed: "закрито",
    any: "будь-який",
    ok: "ТАК",
    no: "НІ",
    noHandPoints: "Немає точок руки",
    cameraBackendUnavailable: "Backend недоступний. Запусти відеосервер.",
    browserCameraUnavailable: "API камери браузера недоступний на цій сторінці.",
    cameraAccessRequired: "Доступ до камери потрібен для керування роботом.",
    grantVideoPermission: "Надай дозвіл на відео і спробуй ще раз.",
    canvasUnavailable: "Контекст захоплення Canvas недоступний.",
    gestureOpenPalm: "Відкрита долоня",
    gestureFist: "Кулак",
    gestureThumbUp: "Великий палець вгору",
    gestureThumbDown: "Великий палець вниз",
    gestureIndexLeft: "Вказівний ліворуч",
    gestureIndexRight: "Вказівний праворуч",
    gesturePeace: "Жест миру",
    gestureThreeFingers: "Три пальці",
    gesturePinky: "Мізинець",
    gestureOkSign: "Жест OK",
    gestureWaveLr: "Помах ліворуч/праворуч",
    gestureCircle: "Рух по колу",
    gesturePullToward: "Потягнути до себе",
    fingerThumb: "Великий",
    fingerIndex: "Вказівний",
    fingerMiddle: "Середній",
    fingerRing: "Безіменний",
    fingerPinky: "Мізинець",
  },
};
const GESTURE_COMMANDS = [
  { gesture: "OPEN_PALM", command: "STOP", icon: "\u{1F590}\uFE0E", labelKey: "gestureOpenPalm" },
  { gesture: "FIST", command: "FORWARD", icon: "\u270A\uFE0E", labelKey: "gestureFist" },
  { gesture: "THUMB_UP", command: "START", icon: "\u{1F44D}\uFE0E", labelKey: "gestureThumbUp" },
  {
    gesture: "THUMB_DOWN",
    command: "EMERGENCY_STOP",
    icon: "\u{1F44E}\uFE0E",
    labelKey: "gestureThumbDown",
  },
  {
    gesture: "INDEX_LEFT",
    command: "TURN_LEFT",
    icon: "\u261D\uFE0E",
    labelKey: "gestureIndexLeft",
  },
  {
    gesture: "INDEX_RIGHT",
    command: "TURN_RIGHT",
    icon: "\u261D\uFE0E",
    labelKey: "gestureIndexRight",
  },
  { gesture: "PEACE", command: "INCREASE_SPEED", icon: "\u270C\uFE0E", labelKey: "gesturePeace" },
  {
    gesture: "THREE_FINGERS",
    command: "DECREASE_SPEED",
    icon: "3",
    labelKey: "gestureThreeFingers",
  },
  {
    gesture: "PINKY",
    command: "RETURN_HOME",
    icon: "\u{1F91F}\uFE0E",
    labelKey: "gesturePinky",
  },
  {
    gesture: "OK_SIGN",
    command: "CONFIRM_ACTION",
    icon: "\u{1F44C}\uFE0E",
    labelKey: "gestureOkSign",
  },
  {
    gesture: "WAVE_LR",
    command: "MODE_TOGGLE",
    icon: "\u{1F590}\uFE0E",
    labelKey: "gestureWaveLr",
  },
  { gesture: "CIRCLE", command: "ROTATE_360", icon: "\u25EF", labelKey: "gestureCircle" },
  {
    gesture: "PULL_TOWARD",
    command: "APPROACH_OPERATOR",
    icon: "\u21A4",
    labelKey: "gesturePullToward",
  },
];
const FINGER_LABELS = [
  ["thumb", "fingerThumb"],
  ["index", "fingerIndex"],
  ["middle", "fingerMiddle"],
  ["ring", "fingerRing"],
  ["pinky", "fingerPinky"],
];
const REASON_LABELS = {
  confidence_below_threshold: {
    en: "confidence below threshold",
    uk: "впевненість нижча за поріг",
  },
  repeat_suppressed: {
    en: "already sent",
    uk: "уже надіслано",
  },
  no_hand_detected: {
    en: "no hand detected",
    uk: "руку не виявлено",
  },
  unknown_prediction: {
    en: "unknown prediction",
    uk: "невідоме розпізнавання",
  },
};

let browserCameraMode = false;
let commandCache = [];
let browserStream = null;
let frameTimer = null;
let frameInFlight = false;
let currentSource = "";
let settingsDirty = false;
let videoManuallyStopped = false;
let currentLanguage = initialLanguage();
let lastStatus = null;

function text(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function initialLanguage() {
  const stored = window.localStorage?.getItem("gestureConsoleLanguage");
  if (stored === "uk" || stored === "en") return stored;
  const requested = queryParams.get("lang");
  if (requested === "uk" || requested === "en") return requested;
  return navigator.language && navigator.language.toLowerCase().startsWith("uk") ? "uk" : "en";
}

function t(key) {
  return TRANSLATIONS[currentLanguage]?.[key] || TRANSLATIONS.en[key] || key;
}

function applyStaticTranslations() {
  document.documentElement.lang = currentLanguage;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    if (key) node.textContent = t(key);
  });
  for (const button of document.querySelectorAll("[data-language]")) {
    const language = button.getAttribute("data-language");
    button.classList.toggle("active", language === currentLanguage);
  }
  setVideoStarted(document.querySelector(".video-stage")?.classList.contains("video-started"));
}

function setLanguage(language) {
  if (language !== "uk" && language !== "en") return;
  currentLanguage = language;
  window.localStorage?.setItem("gestureConsoleLanguage", language);
  applyStaticTranslations();
  applyConfirmation(lastStatus || {});
  renderPoseDiagnostics(lastStatus || {});
  renderCommands(commandCache);
  renderGestureMap(lastStatus?.selected_gesture || "UNKNOWN");
}

function fmt(value, suffix = "", digits = 2) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return `${value.toFixed(digits)}${suffix}`;
  return `${value}${suffix}`;
}

function setStateBadge(state) {
  const badge = $("stateBadge");
  if (!badge) return;
  badge.textContent = state === "running" ? t("sentState") : state;
  badge.className = "badge";
  if (state === "running") badge.classList.add("badge-running");
  else if (state === "error") badge.classList.add("badge-error");
  else badge.classList.add("badge-muted");
}

function setVideoStarted(started) {
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.toggle("video-started", Boolean(started));
  const startButton = $("startVideoButton");
  if (startButton) startButton.textContent = started ? t("videoRunning") : t("startVideo");
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) {
    sideStartButton.textContent = t("start");
    sideStartButton.classList.toggle("is-hidden", Boolean(started));
  }
  const stopButton = $("stopVideoButton");
  if (stopButton) stopButton.disabled = !browserCameraMode;
  text("stopVideoButton", t("stop"));
  text("reloadButton", t("reload"));
  text("saveSettingsButton", t("apply"));
}

function clearVideoFrame() {
  const stream = $("stream");
  if (!stream) return;
  stream.removeAttribute("src");
}

function buildUrl(path) {
  const prefix = apiBase.replace(/\/$/, "");
  return `${prefix}/${path.replace(/^\//, "")}`;
}

function syncApiBaseFromInput() {
  apiBase = (window.GESTURE_API_BASE || queryParams.get("api") || "").trim();
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
  const locale = currentLanguage === "uk" ? "uk-UA" : "en-US";
  return new Date().toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function recognitionLabel(gesture, confidence, minConfidence) {
  const value = `${gesture} / ${fmt(confidence)}`;
  if (gesture !== "UNKNOWN" && confidence < minConfidence) {
    return `${value} ${t("lowConfidence")}`;
  }
  return value;
}

function reasonLabel(reason) {
  if (!reason) return "";
  const key = String(reason);
  return REASON_LABELS[key]?.[currentLanguage] || key.replaceAll("_", " ");
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
    const thresholdText = `${fmt(confidence)} < ${fmt(minConfidence)} ${t("threshold")}`;
    text("confirmationProgressText", thresholdText);
  } else if (reason === "repeat_suppressed") {
    text(
      "confirmationProgressText",
      `${stableFrames} / ${requiredFrames} ${t("frames")}, ${t("alreadySent")}`
    );
  } else if (requiredFrames > 0) {
    text("confirmationProgressText", `${stableFrames} / ${requiredFrames} ${t("frames")}`);
  } else {
    text("confirmationProgressText", reasonLabel(reason) || `0 / 0 ${t("frames")}`);
  }
}

function renderPoseDiagnostics(status) {
  const states = status.finger_states || {};
  const expected = (status.expected_pose && status.expected_pose.finger_states) || {};
  const directions = status.pose_directions || {};
  const node = $("fingerStates");
  if (!node) return;
  node.innerHTML = FINGER_LABELS.map(([key, labelKey]) => {
    const actual = states[key];
    const target = expected[key];
    const actualText = actual === true ? t("open") : actual === false ? t("closed") : "--";
    const targetText = target === true ? t("open") : target === false ? t("closed") : t("any");
    const matched = target === null || target === undefined || actual === target;
    return `
      <div class="finger-row">
        <strong>${t(labelKey)}</strong>
        <span>${actualText} / ${targetText}</span>
        <span class="${matched ? "match" : "mismatch"}">${matched ? t("ok") : t("no")}</span>
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
    node.innerHTML = `<div class="empty">${t("noHandPoints")}</div>`;
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
  lastStatus = status;
  currentSource = status.source || "";
  setStateBadge(status.state);
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
    status.last_command && status.last_command !== "UNKNOWN" ? t("sentState") : t("waiting")
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
  if (!browserCameraMode) {
    setVideoStarted(status.state === "running" && !videoManuallyStopped);
  }

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
  videoManuallyStopped = false;
  const stream = $("stream");
  if (stream) stream.src = `${buildUrl("stream.mjpg")}?t=${Date.now()}`;
  setVideoStarted(true);
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
      cameraAccessMessage(t("cameraBackendUnavailable"))
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
  if (!settingsDirty) text("settingsStatus", t("saved"));
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
    text("settingsStatus", t("offline"));
  }
}

async function saveSettings() {
  syncApiBaseFromInput();
  text("settingsStatus", t("saving"));
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
    text("settingsStatus", t("saved"));
  } catch (error) {
    text("settingsStatus", t("error"));
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
    showRuntimeError(cameraAccessMessage(t("browserCameraUnavailable")));
    return;
  }
  try {
    videoManuallyStopped = false;
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
    text("sourceLine", "browser-camera");
    frameTimer = window.setInterval(captureAndSendFrame, BROWSER_CAMERA_FRAME_INTERVAL_MS);
  } catch (error) {
    showRuntimeError(cameraAccessMessage(error));
  }
}

function stopBrowserCamera() {
  videoManuallyStopped = true;
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
  clearVideoFrame();
  setVideoStarted(false);
}

async function startVideo() {
  syncApiBaseFromInput();
  videoManuallyStopped = false;
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
  videoManuallyStopped = true;
  clearVideoFrame();
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
    if (!context) throw new Error(t("canvasUnavailable"));
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
    body.innerHTML = `<tr><td colspan="5" class="empty">${t("noCommands")}</td></tr>`;
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
  node.innerHTML = GESTURE_COMMANDS.map(({ gesture, command, icon, labelKey }) => `
    <div class="gesture-item ${gesture === activeGesture ? "active" : ""}">
      <span class="gesture-icon" title="${escapeHtml(t(labelKey))}" aria-hidden="true">
        ${escapeHtml(icon)}
      </span>
      <div class="gesture-copy">
        <strong>${escapeHtml(gesture)}</strong>
        <span>${escapeHtml(command)}</span>
      </div>
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
      t("cameraAccessRequired"),
      t("grantVideoPermission"),
    ].join(" ");
  }
  return detail
    ? `${t("cameraAccessRequired")} ${detail}`
    : t("cameraAccessRequired");
}

function bindControls() {
  syncApiBaseFromInput();
  const startVideoButton = $("startVideoButton");
  if (startVideoButton) startVideoButton.addEventListener("click", startVideo);
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) sideStartButton.addEventListener("click", startVideo);
  const stopVideoButton = $("stopVideoButton");
  if (stopVideoButton) stopVideoButton.addEventListener("click", stopVideo);
  const reloadButton = $("reloadButton");
  if (reloadButton) reloadButton.addEventListener("click", reloadStream);
  for (const button of document.querySelectorAll("[data-language]")) {
    button.addEventListener("click", () => setLanguage(button.getAttribute("data-language")));
  }
  for (const id of ["staticFramesInput", "dynamicFramesInput", "emergencyFramesInput"]) {
    const input = $(id);
    if (input) {
      input.addEventListener("input", () => {
        settingsDirty = true;
        text("settingsStatus", t("unsaved"));
      });
    }
  }
  const saveSettingsButton = $("saveSettingsButton");
  if (saveSettingsButton) saveSettingsButton.addEventListener("click", saveSettings);
}

bindControls();
applyStaticTranslations();
renderGestureMap();
loadSettings();
refreshStatus();
refreshCommands().catch(() => undefined);
setInterval(refreshStatus, 500);
setInterval(() => {
  if (!browserCameraMode) refreshCommands().catch(() => undefined);
}, 1000);
