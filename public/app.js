import { LocalSileroVad, SILERO_VAD_VERSION } from "./silero-vad.js";

const elements = {
  stage: document.querySelector("#stage"),
  connectButton: document.querySelector("#connectButton"),
  mainButtonLabel: document.querySelector("#mainButtonLabel"),
  micButton: document.querySelector("#micButton"),
  cloudAvatarButton: document.querySelector("#cloudAvatarButton"),
  cloudButtonStatus: document.querySelector("#cloudButtonStatus"),
  fullscreenButton: document.querySelector("#fullscreenButton"),
  connectionDot: document.querySelector("#connectionDot"),
  connectionText: document.querySelector("#connectionText"),
  avatarFrame: document.querySelector("#avatarFrame"),
  avatarVideo: document.querySelector("#avatarVideo"),
  avatarImage: document.querySelector("#avatarImage"),
  avatarState: document.querySelector("#avatarState"),
  assistantTranscript: document.querySelector("#assistantTranscript"),
  userTranscript: document.querySelector("#userTranscript"),
  modeMetric: document.querySelector("#modeMetric"),
  bufferMetric: document.querySelector("#bufferMetric"),
  avatarMetric: document.querySelector("#avatarMetric"),
  playbackMetric: document.querySelector("#playbackMetric"),
  personaTitle: document.querySelector("#personaTitle"),
  privacyNote: document.querySelector("#privacyNote"),
};

const state = {
  socket: null,
  connectPromise: null,
  audioContext: null,
  audioWorkletLoaded: false,
  micStream: null,
  micNode: null,
  micSource: null,
  nextPlaybackTime: 0,
  browserPcmPrerollSeconds: 1,
  browserPcmPrimed: false,
  sources: new Set(),
  connected: false,
  micActive: false,
  speaking: false,
  sampleRate: 24000,
  playbackOwner: "browser",
  waitingForRenderer: false,
  pendingPcm: [],
  segmentQueue: [],
  segmentPlaying: false,
  segmentMode: false,
  rendererDone: false,
  mediaSource: null,
  sourceBuffer: null,
  mediaSourceUrl: null,
  mediaSourceMime: null,
  mediaSourceQueue: [],
  mediaSourceFetching: false,
  mediaSourceDisabled: false,
  mediaSourcePrerollSeconds: 1,
  preferMediaSource: false,
  hls: null,
  hlsResponseId: null,
  segmentFallbackQueue: [],
  avatarVideoGeneration: 0,
  avatarVideoFramePending: false,
  avatarVideoStallTimer: null,
  localVadSpeaking: false,
  localVadBargeIn: false,
  localVadBargeInChecking: false,
  localVadSpeechStartedAt: null,
  localVadSilenceStartedAt: null,
  localVadBlocked: false,
  localVadNoiseFloor: 0.004,
  localVadProbability: 0,
  localVadRms: 0,
  localVadEnergySpeechMs: 0,
  localVadTrigger: "none",
  localVadPreRoll: [],
  localVadResumeTimer: null,
  localSileroVad: null,
  localSileroVadPromise: null,
  localSileroVadFailed: false,
  localVadQueue: Promise.resolve(),
  localVadGeneration: 0,
  holdToTalkActive: false,
  holdToTalkGeneration: 0,
  holdToTalkChunks: 0,
  cursorHideTimer: null,
  responseStartedAt: null,
  firstAudioAt: null,
  firstAvatarAt: null,
  playbackStartedAt: null,
  cloudAvatarEnabled: localStorage.getItem("cloudAvatarEnabled") !== "false",
  cloudAvatarConnected: false,
  cloudAvatarStatus:
    localStorage.getItem("cloudAvatarEnabled") !== "false" ? "pending" : "off",
  upstreamMode: "mock",
  clientConfig: {
    model: "gpt-realtime-2.1-mini",
    voice: "marin",
    vad_eagerness: "auto",
    barge_in_enabled: false,
    persona_name: "赛博女友",
    instructions: "",
  },
};

const mediaSourceTypes = [
  'video/mp4; codecs="avc1.42E01F, mp4a.40.2"',
  'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
  'video/mp4; codecs="avc1.4D401F, mp4a.40.2"',
];

function revealCursorTemporarily() {
  elements.stage.classList.remove("cursor-hidden");
  if (state.cursorHideTimer !== null) {
    clearTimeout(state.cursorHideTimer);
    state.cursorHideTimer = null;
  }
  if (state.connected) {
    state.cursorHideTimer = window.setTimeout(() => {
      if (state.connected) elements.stage.classList.add("cursor-hidden");
    }, 1200);
  }
}

function setConnection(connected, text) {
  state.connected = connected;
  if (connected) {
    revealCursorTemporarily();
  } else {
    if (state.cursorHideTimer !== null) clearTimeout(state.cursorHideTimer);
    state.cursorHideTimer = null;
    elements.stage.classList.remove("cursor-hidden");
  }
  elements.connectionDot.classList.toggle("online", connected);
  elements.connectionText.textContent = text;
  updateControlLabels();
}

function updateControlLabels() {
  const continuousListening = state.micActive && !state.holdToTalkActive;
  const responseActive =
    state.upstreamMode === "omlx" &&
    state.localVadBlocked &&
    assistantIsActive();
  elements.connectButton.classList.toggle("hold-active", state.holdToTalkActive);
  elements.connectButton.disabled = continuousListening || responseActive;
  elements.micButton.disabled = responseActive && !state.micActive;
  elements.cloudAvatarButton.disabled =
    responseActive || state.micActive || state.holdToTalkActive;
  elements.cloudAvatarButton.classList.toggle("active", state.cloudAvatarEnabled);
  elements.cloudAvatarButton.classList.toggle(
    "connecting",
    state.cloudAvatarStatus === "connecting",
  );
  elements.cloudAvatarButton.classList.toggle(
    "connected",
    state.cloudAvatarStatus === "connected",
  );
  elements.cloudAvatarButton.classList.toggle(
    "unavailable",
    state.cloudAvatarStatus === "failed",
  );
  const cloudStatusText =
    state.cloudAvatarStatus === "connected"
      ? "云端已连"
      : state.cloudAvatarStatus === "connecting"
        ? "连接中"
        : state.cloudAvatarStatus === "failed"
          ? "连接失败"
          : state.cloudAvatarEnabled
            ? "云端待连接"
            : "云端关闭";
  elements.cloudButtonStatus.textContent = cloudStatusText;
  elements.cloudAvatarButton.setAttribute(
    "aria-pressed",
    String(state.cloudAvatarEnabled),
  );
  elements.cloudAvatarButton.setAttribute(
    "aria-label",
    state.cloudAvatarStatus === "connected"
      ? "关闭云端人物动画，当前已连接"
      : state.cloudAvatarStatus === "connecting"
        ? "云端人物动画连接中"
        : state.cloudAvatarStatus === "failed"
          ? "关闭云端人物动画，当前连接失败"
          : state.cloudAvatarEnabled
            ? "关闭云端人物动画，等待连接"
            : "开启云端人物动画",
  );
  elements.connectButton.setAttribute(
    "aria-label",
    state.holdToTalkActive
      ? "松开发送"
      : responseActive
        ? "生成回答中"
      : continuousListening
        ? "持续聆听中"
        : "按住说话",
  );
  elements.mainButtonLabel.textContent = state.holdToTalkActive
    ? "松开发送"
    : responseActive
      ? "生成回答中"
    : continuousListening
      ? "持续聆听中"
      : "按住说话";
}

function setAvatarState(nextState) {
  const labels = {
    idle: "待机",
    listening: "正在听",
    thinking: "思考中",
    rendering: "生成画面",
    speaking: "正在说话",
    closed: "已断开",
  };
  state.speaking = nextState === "speaking";
  elements.stage.dataset.state = nextState;
  elements.avatarFrame.dataset.state = nextState;
  elements.avatarState.textContent = labels[nextState] || nextState;
  if (nextState === "listening") {
    releaseHalfDuplexIfIdle();
  }
  updateControlLabels();
}

function showAvatarImage() {
  state.avatarVideoFramePending = false;
  elements.avatarFrame.classList.remove("video-ready");
  elements.avatarVideo.classList.remove("is-visible");
  elements.avatarImage.hidden = false;
}

function prepareAvatarVideo() {
  state.avatarVideoGeneration += 1;
  showAvatarImage();
  elements.avatarVideo.hidden = false;
}

function showAvatarVideoAfterFirstFrame() {
  if (state.avatarVideoFramePending) return;
  state.avatarVideoFramePending = true;
  const generation = state.avatarVideoGeneration;
  const reveal = () => {
    state.avatarVideoFramePending = false;
    if (
      generation !== state.avatarVideoGeneration ||
      elements.avatarVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      return;
    }
    elements.avatarVideo.classList.add("is-visible");
    elements.avatarFrame.classList.add("video-ready");
  };
  if ("requestVideoFrameCallback" in elements.avatarVideo) {
    elements.avatarVideo.requestVideoFrameCallback(reveal);
  } else {
    requestAnimationFrame(() => requestAnimationFrame(reveal));
  }
}

function scheduleAvatarImageFallback() {
  if (state.avatarVideoStallTimer !== null) return;
  state.avatarVideoStallTimer = window.setTimeout(() => {
    state.avatarVideoStallTimer = null;
    showAvatarImage();
  }, 160);
}

function clearAvatarImageFallback() {
  if (state.avatarVideoStallTimer !== null) {
    clearTimeout(state.avatarVideoStallTimer);
    state.avatarVideoStallTimer = null;
  }
}

function renderAssistantTranscript(text) {
  elements.assistantTranscript.replaceChildren();
  if (!text) return;
  const match = text.match(/^(.*?)([^，。！？、\s]{2,5})([。！？]?)$/u);
  if (!match || !match[1]) {
    elements.assistantTranscript.textContent = text;
    return;
  }
  elements.assistantTranscript.append(document.createTextNode(match[1]));
  const accent = document.createElement("span");
  accent.className = "accent-word";
  accent.textContent = match[2];
  elements.assistantTranscript.append(accent);
  if (match[3]) {
    elements.assistantTranscript.append(document.createTextNode(match[3]));
  }
}

async function loadHealth() {
  const [healthResponse, configResponse] = await Promise.all([fetch("/healthz"), fetch("/client-config")]);
  const data = await healthResponse.json();
  state.clientConfig = await configResponse.json();
  state.upstreamMode = data.upstream_mode;
  state.sampleRate = data.sample_rate;
  state.browserPcmPrerollSeconds = Math.max(
    0.06,
    Number(data.preroll_ms || 1000) / 1000,
  );
  state.playbackOwner = data.playback_owner || "browser";
  const usesArk = state.clientConfig.chat_backend === "ark";
  updateModeMetric();
  elements.avatarMetric.textContent = `${data.avatar_backend} / ${data.avatar_renderer}`;
  elements.playbackMetric.textContent = state.playbackOwner === "renderer" ? "视频内音轨" : "浏览器 PCM";
  elements.personaTitle.textContent = state.clientConfig.persona_name;
  elements.privacyNote.textContent = usesArk
    ? "语音识别与合成在本机处理 · 对话文本发送至 Ark"
    : "语音与对话在本机处理";
  updateControlLabels();
  if (state.upstreamMode === "omlx") {
    void ensureLocalSileroVad();
  }
}

function updateModeMetric() {
  const usesArk = state.clientConfig.chat_backend === "ark";
  const inferenceMode = usesArk ? "ARK · ONLINE" : state.upstreamMode;
  if (state.upstreamMode !== "omlx") {
    elements.modeMetric.textContent = inferenceMode;
    return;
  }
  const vadMode = state.localSileroVad
    ? `SILERO V${SILERO_VAD_VERSION}`
    : state.localSileroVadFailed
      ? "RMS 回退"
      : "VAD 加载中";
  elements.modeMetric.textContent = `${inferenceMode} · ${vadMode}`;
}

function markSileroVadUnavailable(error) {
  if (!state.localSileroVadFailed) {
    console.warn("Silero VAD unavailable; falling back to RMS energy detection.", error);
  }
  state.localSileroVad = null;
  state.localSileroVadFailed = true;
  updateModeMetric();
}

async function ensureLocalSileroVad() {
  if (state.localSileroVad) return state.localSileroVad;
  if (state.localSileroVadFailed) return null;
  if (!state.localSileroVadPromise) {
    state.localSileroVadPromise = LocalSileroVad.create()
      .then((vad) => {
        state.localSileroVad = vad;
        updateModeMetric();
        return vad;
      })
      .catch((error) => {
        markSileroVadUnavailable(error);
        return null;
      })
      .finally(() => {
        state.localSileroVadPromise = null;
      });
  }
  return state.localSileroVadPromise;
}

function websocketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const avatarMode = state.cloudAvatarEnabled ? "1" : "0";
  return `${protocol}//${location.host}/v1/realtime?avatar=${avatarMode}`;
}

async function ensureAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext({ sampleRate: state.sampleRate });
  }
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
}

function send(event) {
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(event));
  }
}

async function connect() {
  if (state.connected) return;
  if (state.connectPromise) return state.connectPromise;

  await ensureAudioContext();
  if (state.cloudAvatarEnabled) {
    state.cloudAvatarStatus = "connecting";
    updateControlLabels();
  }
  state.connectPromise = new Promise((resolve, reject) => {
    const socket = new WebSocket(websocketUrl());
    state.socket = socket;
    socket.addEventListener("open", () => {
      setConnection(true, "实时链路已连接");
      setAvatarState("listening");
      send({
        type: "session.update",
        session: {
          type: "realtime",
          model: state.clientConfig.model,
          output_modalities: ["audio"],
          instructions: state.clientConfig.instructions,
          audio: {
            input: {
              format: { type: "audio/pcm", rate: state.sampleRate },
              turn_detection: {
                type: "semantic_vad",
                eagerness: state.clientConfig.vad_eagerness,
                create_response: true,
                interrupt_response: true,
              },
            },
            output: {
              format: { type: "audio/pcm", rate: state.sampleRate },
              voice: state.clientConfig.voice,
            },
          },
        },
      });
      resolve();
    });
    socket.addEventListener("message", handleMessage);
    socket.addEventListener("close", () => {
      state.connectPromise = null;
      setConnection(false, "已断开");
      setAvatarState("closed");
    });
    socket.addEventListener("error", () => {
      state.connectPromise = null;
      state.cloudAvatarStatus = state.cloudAvatarEnabled ? "failed" : "off";
      setConnection(false, "连接错误");
      reject(new Error("WebSocket connection failed"));
    });
  });
  return state.connectPromise;
}

function handleMessage(message) {
  const event = JSON.parse(message.data);
  switch (event.type) {
    case "session.created":
      elements.connectionText.textContent = "会话已就绪";
      break;
    case "avatar.mode":
      state.playbackOwner = event.playback_owner || "browser";
      state.cloudAvatarConnected = state.playbackOwner === "renderer";
      state.cloudAvatarStatus = state.cloudAvatarConnected
        ? "connected"
        : state.cloudAvatarEnabled
          ? "failed"
          : "off";
      elements.avatarMetric.textContent = state.cloudAvatarConnected
        ? "云端人物动画"
        : "本地静态人物";
      elements.playbackMetric.textContent = state.cloudAvatarConnected
        ? "视频内音轨"
        : "浏览器 PCM";
      updateControlLabels();
      break;
    case "avatar.state":
      if (
        !(
          event.state === "listening" &&
          (state.waitingForRenderer || state.segmentPlaying || (state.segmentMode && !state.rendererDone))
        )
      ) {
        setAvatarState(event.state);
      }
      break;
    case "avatar.audio.timeline":
      elements.bufferMetric.textContent = `${Math.round(event.buffered_ms)} ms`;
      break;
    case "conversation.item.input_audio_transcription.delta":
      elements.userTranscript.textContent += event.delta || "";
      break;
    case "conversation.item.input_audio_transcription.completed":
      elements.userTranscript.textContent = `你：${event.transcript || ""}`;
      break;
    case "input_audio_buffer.no_speech":
      resumeLocalVadNow();
      elements.userTranscript.textContent = "刚才没听清，请再说一次…";
      break;
    case "input_audio_buffer.barge_in.echo_ignored":
      state.localVadBargeInChecking = false;
      resetLocalVad();
      state.localVadBlocked = false;
      elements.userTranscript.textContent = "正在聆听，可随时插话…";
      break;
    case "input_audio_buffer.barge_in.accepted":
      state.localVadBargeInChecking = false;
      stopAssistantPlayback();
      resetLocalVad();
      state.localVadBlocked = false;
      elements.userTranscript.textContent = `你：${event.transcript || ""}`;
      break;
    case "response.created":
      if (state.upstreamMode === "omlx") {
        // Half-duplex mode keeps the microphone capture alive but ignores it
        // until this response's cloud-rendered video has fully finished.
        state.localVadBlocked = !state.clientConfig.barge_in_enabled;
      }
      resetMediaSource();
      elements.assistantTranscript.textContent = "";
      state.pendingPcm = [];
      state.waitingForRenderer = state.playbackOwner === "renderer";
      state.segmentQueue = [];
      state.segmentFallbackQueue = [];
      state.segmentPlaying = false;
      state.segmentMode = false;
      state.rendererDone = false;
      state.preferMediaSource = false;
      state.responseStartedAt = performance.now();
      state.firstAudioAt = null;
      state.firstAvatarAt = null;
      state.playbackStartedAt = null;
      state.browserPcmPrimed = false;
      setAvatarState("thinking");
      break;
    case "response.output_audio.delta":
      if (state.firstAudioAt === null) {
        state.firstAudioAt = performance.now();
      }
      if (state.playbackOwner === "renderer") {
        state.pendingPcm.push(event.delta);
      } else {
        playPcm16(event.delta);
      }
      break;
    case "response.output_audio_transcript.delta":
      elements.assistantTranscript.textContent += event.delta || "";
      break;
    case "response.output_audio_transcript.done":
      renderAssistantTranscript(event.transcript || elements.assistantTranscript.textContent);
      break;
    case "response.done":
      if (state.upstreamMode === "omlx" && state.clientConfig.barge_in_enabled) {
        if (!state.localVadSpeaking && !state.localVadBargeInChecking) {
          resumeLocalVadAfterPlayback();
        }
      }
      if (state.sources.size > 0) {
        setAvatarState("speaking");
      } else if (state.segmentPlaying) {
        setAvatarState("speaking");
      } else if (state.waitingForRenderer || (state.segmentMode && !state.rendererDone)) {
        setAvatarState("rendering");
      } else {
        setAvatarState("listening");
      }
      break;
    case "avatar.render.accepted":
      setAvatarState("rendering");
      break;
    case "avatar.stream.ready": {
      resetMediaSource();
      state.preferMediaSource = false;
      recordFirstAvatarFrame();
      state.waitingForRenderer = false;
      state.pendingPcm = [];
      state.segmentMode = false;
      state.segmentPlaying = true;
      prepareAvatarVideo();
      setAvatarState("speaking");
      elements.avatarMetric.textContent = event.backend || "remote";
      state.hlsResponseId = event.response_id || null;
      if (elements.avatarVideo.canPlayType("application/vnd.apple.mpegurl")) {
        elements.avatarVideo.src = event.url;
        elements.avatarVideo.play().catch(() => {
          elements.assistantTranscript.textContent = "浏览器阻止了有声视频自动播放，请点击画面继续。";
        });
      } else if (window.Hls?.isSupported()) {
        state.hls = new window.Hls({
          lowLatencyMode: true,
          liveSyncDurationCount: 2,
          liveMaxLatencyDurationCount: 5,
          backBufferLength: 30,
        });
        state.hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
          elements.avatarVideo.play().catch(() => {
            elements.assistantTranscript.textContent = "浏览器阻止了有声视频自动播放，请点击画面继续。";
          });
        });
        state.hls.on(window.Hls.Events.ERROR, (_event, data) => {
          if (!data.fatal) return;
          state.hls?.destroy();
          state.hls = null;
          state.hlsResponseId = null;
          state.segmentPlaying = false;
          showAvatarImage();
          elements.assistantTranscript.textContent = "HLS 播放中断，正在等待完整视频兜底。";
        });
        state.hls.loadSource(event.url);
        state.hls.attachMedia(elements.avatarVideo);
      } else {
        state.hlsResponseId = null;
        state.segmentPlaying = false;
        showAvatarImage();
        elements.assistantTranscript.textContent = "当前浏览器不支持 HLS，正在等待完整视频兜底。";
      }
      break;
    }
    case "avatar.video.ready": {
      if (state.hlsResponseId && state.hlsResponseId === event.response_id) {
        break;
      }
      resetMediaSource();
      state.preferMediaSource = false;
      recordFirstAvatarFrame();
      state.waitingForRenderer = false;
      state.pendingPcm = [];
      state.segmentMode = false;
      state.segmentPlaying = true;
      elements.avatarVideo.src = event.url;
      prepareAvatarVideo();
      setAvatarState("speaking");
      elements.avatarVideo.play().catch(() => {
        elements.assistantTranscript.textContent = "浏览器阻止了有声视频自动播放，请点击画面继续。";
      });
      elements.avatarMetric.textContent = event.backend || "remote";
      break;
    }
    case "avatar.video.segment":
      state.waitingForRenderer = false;
      state.pendingPcm = [];
      state.segmentMode = true;
      elements.avatarMetric.textContent = event.backend || "flashhead";
      if (state.preferMediaSource) {
        state.segmentFallbackQueue.push(event);
      } else {
        recordFirstAvatarFrame();
        enqueueSegment(event);
      }
      break;
    case "avatar.media.start": {
      state.segmentMode = true;
      state.mediaSourcePrerollSeconds = Math.max(
        0,
        Number(event.preroll_ms || 1000) / 1000,
      );
      const mime = supportedMediaSourceType(event.mime_type);
      state.preferMediaSource = Boolean(mime);
      if (mime) {
        ensureMediaSource(mime);
        elements.playbackMetric.textContent = "连续音视频流";
      }
      break;
    }
    case "avatar.media.init":
    case "avatar.media.fragment":
      if (state.preferMediaSource) {
        if (event.type === "avatar.media.fragment") {
          recordFirstAvatarFrame();
        }
        state.mediaSourceQueue.push({
          data: base64ToArrayBuffer(event.data),
          isMedia: event.type === "avatar.media.fragment",
        });
        ensureMediaSource(event.mime_type || supportedMediaSourceType());
        pumpMediaSource();
      }
      break;
    case "avatar.render.done":
      state.rendererDone = true;
      maybeStartMediaSourcePlayback();
      finishMediaSourceIfReady();
      if (
        state.segmentMode &&
        !state.segmentPlaying &&
        state.segmentQueue.length === 0 &&
        (!state.preferMediaSource || state.mediaSourceQueue.length === 0) &&
        !state.mediaSourceFetching
      ) {
        setAvatarState("listening");
        if (state.upstreamMode === "omlx" && !state.clientConfig.barge_in_enabled) {
          resumeLocalVadNow();
        }
      }
      break;
    case "avatar.response.cancelled": {
      state.waitingForRenderer = false;
      state.pendingPcm = [];
      state.segmentQueue = [];
      state.segmentFallbackQueue = [];
      state.segmentPlaying = false;
      state.segmentMode = false;
      state.rendererDone = true;
      state.preferMediaSource = false;
      resetMediaSource();
      elements.avatarVideo.hidden = true;
      showAvatarImage();
      if (state.upstreamMode === "omlx") {
        resumeLocalVadNow();
      }
      break;
    }
    case "avatar.fallback": {
      state.playbackOwner = "browser";
      state.cloudAvatarConnected = false;
      state.cloudAvatarStatus = state.cloudAvatarEnabled ? "failed" : "off";
      state.waitingForRenderer = false;
      state.rendererDone = true;
      resetMediaSource();
      elements.avatarVideo.pause();
      showAvatarImage();
      for (const delta of state.pendingPcm) {
        playPcm16(delta);
      }
      state.pendingPcm = [];
      elements.avatarMetric.textContent = "本地静态人物";
      elements.playbackMetric.textContent = "浏览器 PCM";
      elements.assistantTranscript.textContent =
        event.message || "云端人物动画未连接，已切换本地语音。";
      updateControlLabels();
      break;
    }
    case "avatar.error": {
      state.playbackOwner = "browser";
      state.cloudAvatarConnected = false;
      state.cloudAvatarStatus = state.cloudAvatarEnabled ? "failed" : "off";
      state.waitingForRenderer = false;
      state.rendererDone = true;
      resetMediaSource();
      showAvatarImage();
      for (const delta of state.pendingPcm) {
        playPcm16(delta);
      }
      state.pendingPcm = [];
      elements.avatarMetric.textContent = "本地静态人物";
      elements.playbackMetric.textContent = "浏览器 PCM";
      if (state.upstreamMode === "omlx") {
        resumeLocalVadAfterPlayback();
      }
      elements.assistantTranscript.textContent = `人物渲染错误：${event.message || "未知错误"}`;
      updateControlLabels();
      break;
    }
    case "error":
      if (state.upstreamMode === "omlx") {
        state.localVadBlocked = false;
        state.localVadSpeaking = false;
        state.localVadSpeechStartedAt = null;
        state.localVadSilenceStartedAt = null;
        state.localVadPreRoll = [];
      }
      elements.assistantTranscript.textContent = event.error?.message || "实时链路发生错误";
      break;
  }
}

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
}

function playPcm16(base64) {
  if (!state.audioContext) return;
  const pcm = new Int16Array(base64ToArrayBuffer(base64));
  const floats = new Float32Array(pcm.length);
  for (let index = 0; index < pcm.length; index += 1) {
    floats[index] = pcm[index] / 32768;
  }
  const buffer = state.audioContext.createBuffer(1, floats.length, state.sampleRate);
  buffer.copyToChannel(floats, 0);
  const source = state.audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioContext.destination);
  state.sources.add(source);
  source.onended = () => {
    state.sources.delete(source);
    if (
      state.sources.size === 0 &&
      !state.waitingForRenderer &&
      !state.segmentPlaying &&
      (!state.segmentMode || state.rendererDone)
    ) {
      setAvatarState("listening");
      if (state.upstreamMode === "omlx" && !state.clientConfig.barge_in_enabled) {
        resumeLocalVadNow();
      }
    }
  };
  const now = state.audioContext.currentTime;
  const leadSeconds = state.browserPcmPrimed
    ? 0.06
    : state.browserPcmPrerollSeconds;
  state.nextPlaybackTime = Math.max(state.nextPlaybackTime, now + leadSeconds);
  state.browserPcmPrimed = true;
  source.start(state.nextPlaybackTime);
  state.nextPlaybackTime += buffer.duration;
}

function floatToPcm16(float32) {
  const pcm = new Int16Array(float32.length);
  for (let index = 0; index < float32.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32[index]));
    pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const stride = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += stride) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + stride));
  }
  return btoa(binary);
}

function sileroThresholds(assistantActive) {
  const presets = {
    high: { start: 0.35, end: 0.18 },
    auto: { start: 0.45, end: 0.25 },
    low: { start: 0.58, end: 0.35 },
  };
  const selected = presets[state.clientConfig.vad_eagerness] || presets.auto;
  if (!assistantActive) return selected;
  return {
    start: Math.min(0.86, selected.start + 0.16),
    end: Math.min(0.72, selected.end + 0.12),
  };
}

function processLocalVadChunk(samples, audio, probabilities) {
  if (state.localVadBlocked || !state.micActive) return;
  const assistantActive = assistantIsActive();
  if (assistantActive && !state.clientConfig.barge_in_enabled) return;

  let energy = 0;
  for (const sample of samples) {
    energy += sample * sample;
  }
  const rms = Math.sqrt(energy / Math.max(samples.length, 1));
  const now = performance.now();
  const chunkDurationMs =
    (samples.length / Math.max(state.audioContext?.sampleRate || state.sampleRate, 1)) *
    1000;
  state.localVadRms = rms;

  if (!state.localVadSpeaking) {
    if (!assistantActive) {
      state.localVadNoiseFloor =
        state.localVadNoiseFloor * 0.95 + Math.min(rms, 0.02) * 0.05;
    }
    state.localVadPreRoll.push(audio);
    if (state.localVadPreRoll.length > 8) state.localVadPreRoll.shift();
  }

  const thresholds = sileroThresholds(assistantActive);
  const energyStartThreshold = assistantActive
    ? Math.max(0.024, state.localVadNoiseFloor * 4.5)
    : Math.max(0.009, state.localVadNoiseFloor * 2.5);
  const energyEndThreshold = assistantActive
    ? Math.max(0.016, state.localVadNoiseFloor * 3.2)
    : Math.max(0.006, state.localVadNoiseFloor * 1.7);
  const energyDetected =
    rms >= (state.localVadSpeaking ? energyEndThreshold : energyStartThreshold);

  if (!state.localVadSpeaking) {
    state.localVadEnergySpeechMs = energyDetected
      ? state.localVadEnergySpeechMs + chunkDurationMs
      : 0;
  }

  let sileroDetected = false;
  if (probabilities?.length) {
    const latestProbability = probabilities[probabilities.length - 1];
    const peakProbability = Math.max(...probabilities);
    state.localVadProbability = latestProbability;
    sileroDetected = state.localVadSpeaking
      ? latestProbability >= thresholds.end
      : peakProbability >= thresholds.start;
  }
  const energyRescueDetected = state.localVadSpeaking
    ? energyDetected
    : state.localVadEnergySpeechMs >= 120;
  const speechDetected = sileroDetected || energyRescueDetected;
  state.localVadTrigger = sileroDetected
    ? "silero"
    : energyRescueDetected
      ? "energy-rescue"
      : "none";
  elements.stage.dataset.vadProbability = state.localVadProbability.toFixed(4);
  elements.stage.dataset.vadRms = rms.toFixed(5);
  elements.stage.dataset.vadNoiseFloor = state.localVadNoiseFloor.toFixed(5);
  elements.stage.dataset.vadTrigger = state.localVadTrigger;

  if (speechDetected) {
    if (!state.localVadSpeaking) {
      const preRoll = [...state.localVadPreRoll];
      state.localVadSpeaking = true;
      state.localVadBargeIn = assistantActive;
      state.localVadSpeechStartedAt = now;
      for (const chunk of preRoll) {
        send({
          type: assistantActive
            ? "input_audio_buffer.barge_in.append"
            : "input_audio_buffer.append",
          audio: chunk,
        });
      }
      state.localVadPreRoll = [];
      elements.userTranscript.textContent = assistantActive
        ? "Silero 听到插话了，继续说…"
        : "Silero 听到了，继续说…";
    } else {
      send({
        type: state.localVadBargeIn
          ? "input_audio_buffer.barge_in.append"
          : "input_audio_buffer.append",
        audio,
      });
    }
    state.localVadSilenceStartedAt = null;
    return;
  }

  if (!state.localVadSpeaking) return;
  send({
    type: state.localVadBargeIn
      ? "input_audio_buffer.barge_in.append"
      : "input_audio_buffer.append",
    audio,
  });
  state.localVadSilenceStartedAt ??= now;
  const speechDuration = now - (state.localVadSpeechStartedAt || now);
  if (speechDuration >= 250 && now - state.localVadSilenceStartedAt >= 800) {
    const isBargeIn = state.localVadBargeIn;
    resetLocalVad();
    state.localVadBlocked = true;
    state.localVadBargeInChecking = isBargeIn;
    elements.userTranscript.textContent = isBargeIn
      ? "正在区分你的插话与回声…"
      : "正在本机识别…";
    send({
      type: isBargeIn
        ? "input_audio_buffer.barge_in.commit"
        : "input_audio_buffer.commit",
    });
  }
}

function queueLocalVadChunk(samples, audio) {
  const generation = state.localVadGeneration;
  state.localVadQueue = state.localVadQueue
    .catch(() => {})
    .then(async () => {
      if (!state.micActive || generation !== state.localVadGeneration) return;
      let probabilities = null;
      if (state.localSileroVad) {
        try {
          probabilities = await state.localSileroVad.process(
            samples,
            state.audioContext.sampleRate,
          );
        } catch (error) {
          markSileroVadUnavailable(error);
        }
      }
      if (!state.micActive || generation !== state.localVadGeneration) return;
      processLocalVadChunk(samples, audio, probabilities);
    });
}

function resetLocalVad() {
  state.localVadGeneration += 1;
  state.localVadSpeaking = false;
  state.localVadBargeIn = false;
  state.localVadSpeechStartedAt = null;
  state.localVadSilenceStartedAt = null;
  state.localVadProbability = 0;
  state.localVadRms = 0;
  state.localVadEnergySpeechMs = 0;
  state.localVadTrigger = "none";
  state.localVadPreRoll = [];
  if (state.localSileroVad) {
    state.localVadQueue = state.localVadQueue
      .catch(() => {})
      .then(() => state.localSileroVad?.reset());
  }
}

function resumeLocalVadNow() {
  if (state.localVadResumeTimer !== null) {
    clearTimeout(state.localVadResumeTimer);
    state.localVadResumeTimer = null;
  }
  resetLocalVad();
  state.localVadBlocked = false;
  updateControlLabels();
  if (state.micActive) {
    elements.userTranscript.textContent = "正在聆听…";
  }
}

function resumeLocalVadAfterPlayback() {
  if (state.localVadResumeTimer !== null) {
    clearTimeout(state.localVadResumeTimer);
    state.localVadResumeTimer = null;
  }
  state.localVadBlocked = false;
  if (!state.localVadSpeaking && state.micActive) {
    resetLocalVad();
  }
}

function assistantIsActive() {
  return (
    state.waitingForRenderer ||
    state.segmentPlaying ||
    (state.segmentMode && !state.rendererDone) ||
    ["thinking", "rendering", "speaking"].includes(elements.stage.dataset.state)
  );
}

function releaseHalfDuplexIfIdle() {
  if (
    state.upstreamMode !== "omlx" ||
    state.clientConfig.barge_in_enabled ||
    assistantIsActive()
  ) {
    return;
  }
  if (state.localVadResumeTimer !== null) {
    clearTimeout(state.localVadResumeTimer);
    state.localVadResumeTimer = null;
  }
  resetLocalVad();
  state.localVadBlocked = false;
}

function stopMicrophone({ commit = false, forceCommit = false } = {}) {
  const shouldCommit =
    commit &&
    !state.localVadBlocked &&
    (forceCommit || state.localVadSpeaking);
  state.micStream?.getTracks().forEach((track) => track.stop());
  state.micNode?.disconnect();
  state.micSource?.disconnect();
  state.micStream = null;
  state.micNode = null;
  state.micSource = null;
  state.micActive = false;
  if (state.localVadResumeTimer !== null) {
    clearTimeout(state.localVadResumeTimer);
    state.localVadResumeTimer = null;
  }
  resetLocalVad();
  state.holdToTalkChunks = 0;
  elements.micButton.classList.remove("active");
  elements.micButton.setAttribute("aria-pressed", "false");
  elements.micButton.setAttribute("aria-label", "开启麦克风");
  updateControlLabels();
  setConnection(state.connected, state.connected ? "麦克风已关闭" : "已断开");
  if (shouldCommit) {
    state.localVadBlocked = true;
    updateControlLabels();
    send({ type: "input_audio_buffer.commit" });
  } else if (state.connected) {
    send({ type: "input_audio_buffer.clear" });
  }
}

async function startMicrophone() {
  if (state.micActive) return;
  if (!state.connected) await connect();
  await ensureAudioContext();

  if (!state.audioWorkletLoaded) {
    await state.audioContext.audioWorklet.addModule("/pcm-worklet.js");
    state.audioWorkletLoaded = true;
  }
  if (state.upstreamMode === "omlx" && !state.holdToTalkActive) {
    await ensureLocalSileroVad();
  }

  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  state.micSource = state.audioContext.createMediaStreamSource(state.micStream);
  state.micNode = new AudioWorkletNode(state.audioContext, "pcm-capture");
  const mutedGain = state.audioContext.createGain();
  mutedGain.gain.value = 0;
  state.micSource.connect(state.micNode);
  state.micNode.connect(mutedGain).connect(state.audioContext.destination);
  state.micNode.port.onmessage = (event) => {
    const pcm = floatToPcm16(event.data);
    const audio = arrayBufferToBase64(pcm.buffer);
    if (state.upstreamMode === "omlx") {
      if (state.localVadBlocked) return;
      if (state.holdToTalkActive) {
        send({ type: "input_audio_buffer.append", audio });
        state.holdToTalkChunks += 1;
        elements.userTranscript.textContent = "正在录音，松开发送…";
        return;
      }
      queueLocalVadChunk(event.data, audio);
    } else {
      send({ type: "input_audio_buffer.append", audio });
    }
  };
  state.micActive = true;
  resetLocalVad();
  state.localVadBlocked = false;
  elements.micButton.classList.add("active");
  elements.micButton.setAttribute("aria-pressed", "true");
  elements.micButton.setAttribute("aria-label", "停止聆听");
  updateControlLabels();
  setConnection(
    true,
    state.holdToTalkActive ? "按住聆听中" : "会话已就绪 · 自动聆听",
  );
  elements.userTranscript.textContent = state.localSileroVad
    ? `正在聆听（Silero VAD ${SILERO_VAD_VERSION}）…`
    : "正在聆听（RMS 回退）…";
}

async function toggleMicrophone() {
  if (state.holdToTalkActive || (state.localVadBlocked && !state.micActive)) return;
  try {
    if (state.micActive) {
      stopMicrophone({ commit: true });
    } else {
      await startMicrophone();
    }
  } catch {
    stopMicrophone();
    elements.connectionText.textContent = "请允许麦克风权限";
    elements.userTranscript.textContent = "需要麦克风权限才能自动感应说话";
  }
}

async function toggleCloudAvatar() {
  if (elements.cloudAvatarButton.disabled) return;
  state.cloudAvatarEnabled = !state.cloudAvatarEnabled;
  state.cloudAvatarConnected = false;
  state.cloudAvatarStatus = state.cloudAvatarEnabled ? "connecting" : "off";
  localStorage.setItem("cloudAvatarEnabled", String(state.cloudAvatarEnabled));
  updateControlLabels();
  stopAssistantPlayback();

  const socket = state.socket;
  if (socket && socket.readyState !== WebSocket.CLOSED) {
    await new Promise((resolve) => {
      socket.addEventListener("close", resolve, { once: true });
      socket.close(1000, "avatar mode changed");
      window.setTimeout(resolve, 1200);
    });
  }
  state.socket = null;
  state.connectPromise = null;
  setConnection(false, "正在切换人物模式…");
  try {
    await connect();
  } catch {
    state.cloudAvatarStatus = state.cloudAvatarEnabled ? "failed" : "off";
    setConnection(false, "人物模式切换失败");
  }
}

async function beginHoldToTalk(pointerId = null) {
  releaseHalfDuplexIfIdle();
  if (state.holdToTalkActive || state.micActive || state.localVadBlocked) return;
  const generation = state.holdToTalkGeneration + 1;
  state.holdToTalkGeneration = generation;
  state.holdToTalkActive = true;
  state.holdToTalkChunks = 0;
  updateControlLabels();
  try {
    await startMicrophone();
    if (!state.holdToTalkActive || state.holdToTalkGeneration !== generation) {
      stopMicrophone({
        commit: true,
        forceCommit: state.holdToTalkChunks > 0,
      });
    } else if (
      pointerId !== null &&
      elements.connectButton.setPointerCapture
    ) {
      elements.connectButton.setPointerCapture(pointerId);
    }
  } catch {
    state.holdToTalkActive = false;
    updateControlLabels();
    elements.connectionText.textContent = "请允许麦克风权限";
    elements.userTranscript.textContent = "点击浏览器地址栏旁的权限图标，允许使用麦克风";
  }
}

function endHoldToTalk() {
  if (!state.holdToTalkActive) return;
  const hasRecordedAudio = state.holdToTalkChunks > 0;
  state.holdToTalkActive = false;
  state.holdToTalkGeneration += 1;
  updateControlLabels();
  if (state.micActive) {
    elements.userTranscript.textContent = hasRecordedAudio
      ? "正在本机识别…"
      : "没有录到声音，请再按住说一次…";
    stopMicrophone({ commit: true, forceCommit: hasRecordedAudio });
  }
}

function stopAssistantPlayback() {
  state.waitingForRenderer = false;
  state.pendingPcm = [];
  state.segmentQueue = [];
  state.segmentFallbackQueue = [];
  state.segmentPlaying = false;
  state.segmentMode = false;
  state.rendererDone = true;
  state.preferMediaSource = false;
  state.localVadBlocked = false;
  state.localVadBargeInChecking = false;
  if (state.localVadResumeTimer !== null) {
    clearTimeout(state.localVadResumeTimer);
    state.localVadResumeTimer = null;
  }
  resetLocalVad();
  resetMediaSource();
  for (const source of state.sources) {
    try {
      source.stop();
    } catch {
      // A source can finish between iteration and stop().
    }
  }
  state.sources.clear();
  state.nextPlaybackTime = state.audioContext?.currentTime || 0;
  state.browserPcmPrimed = false;
  setAvatarState("listening");
}

function interrupt() {
  send({ type: "response.cancel" });
  stopAssistantPlayback();
}

function recordFirstAvatarFrame() {
  if (state.firstAvatarAt === null) {
    state.firstAvatarAt = performance.now();
    if (state.responseStartedAt !== null) {
      elements.bufferMetric.textContent = `首画面 ${Math.round(state.firstAvatarAt - state.responseStartedAt)} ms`;
    }
  }
}

function supportedMediaSourceType(preferred = null) {
  if (state.mediaSourceDisabled || typeof window.MediaSource === "undefined") {
    return null;
  }
  if (preferred && window.MediaSource.isTypeSupported(preferred)) {
    return preferred;
  }
  return mediaSourceTypes.find((mime) => window.MediaSource.isTypeSupported(mime)) || null;
}

function enqueueSegment(segment) {
  const mime = supportedMediaSourceType();
  if (!mime) {
    state.segmentQueue.push(segment);
    playNextSegment();
    return;
  }
  state.mediaSourceQueue.push(segment);
  ensureMediaSource(mime);
  pumpMediaSource();
}

function ensureMediaSource(mime) {
  if (state.mediaSource) return;
  state.mediaSourceMime = mime;
  state.mediaSource = new window.MediaSource();
  const mediaSource = state.mediaSource;
  state.mediaSourceUrl = URL.createObjectURL(mediaSource);
  elements.avatarVideo.src = state.mediaSourceUrl;
  prepareAvatarVideo();
  state.mediaSource.addEventListener(
    "sourceopen",
    () => {
      if (state.mediaSource !== mediaSource) return;
      try {
        state.sourceBuffer = mediaSource.addSourceBuffer(state.mediaSourceMime);
        state.sourceBuffer.mode = "sequence";
        state.sourceBuffer.addEventListener("updateend", () => {
          state.mediaSourceQueue.shift();
          maybeStartMediaSourcePlayback();
          finishMediaSourceIfReady();
          pumpMediaSource();
        });
        pumpMediaSource();
      } catch (error) {
        fallbackFromMediaSource(error);
      }
    },
    { once: true },
  );
}

async function pumpMediaSource() {
  if (
    !state.sourceBuffer ||
    state.sourceBuffer.updating ||
    state.mediaSourceFetching ||
    state.mediaSourceQueue.length === 0
  ) {
    return;
  }
  state.mediaSourceFetching = true;
  const segment = state.mediaSourceQueue[0];
  try {
    let data = segment.data;
    if (!data) {
      const response = await fetch(segment.url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      data = await response.arrayBuffer();
    }
    state.sourceBuffer.appendBuffer(data);
  } catch (error) {
    fallbackFromMediaSource(error);
  } finally {
    state.mediaSourceFetching = false;
  }
}

function bufferedMediaSeconds() {
  if (!state.sourceBuffer || state.sourceBuffer.buffered.length === 0) {
    return 0;
  }
  let bufferedSeconds = 0;
  for (let index = 0; index < state.sourceBuffer.buffered.length; index += 1) {
    bufferedSeconds +=
      state.sourceBuffer.buffered.end(index) - state.sourceBuffer.buffered.start(index);
  }
  return bufferedSeconds;
}

function maybeStartMediaSourcePlayback() {
  const bufferedSeconds = bufferedMediaSeconds();
  if (
    state.segmentPlaying ||
    !state.sourceBuffer ||
    bufferedSeconds <= 0 ||
    (bufferedSeconds < state.mediaSourcePrerollSeconds && !state.rendererDone)
  ) {
    return;
  }
  state.segmentPlaying = true;
  setAvatarState("speaking");
  elements.avatarVideo.play().catch(() => {
    state.segmentPlaying = false;
    elements.assistantTranscript.textContent =
      "浏览器阻止了有声视频自动播放，请点击画面继续。";
  });
}

function finishMediaSourceIfReady() {
  if (
    state.rendererDone &&
    state.mediaSource?.readyState === "open" &&
    state.sourceBuffer &&
    !state.sourceBuffer.updating &&
    !state.mediaSourceFetching &&
    state.mediaSourceQueue.length === 0
  ) {
    try {
      state.mediaSource.endOfStream();
      maybeStartMediaSourcePlayback();
    } catch {
      // The stream may already be ending after the last update.
    }
  }
}

function fallbackFromMediaSource(error) {
  const pending = [...state.segmentFallbackQueue];
  state.mediaSourceDisabled = true;
  state.preferMediaSource = false;
  resetMediaSource();
  state.segmentQueue.push(...pending);
  state.segmentFallbackQueue = [];
  elements.assistantTranscript.textContent = `连续媒体缓冲不可用，已切换兼容播放：${error.message || error}`;
  playNextSegment();
}

function resetMediaSource() {
  clearAvatarImageFallback();
  state.avatarVideoGeneration += 1;
  showAvatarImage();
  elements.avatarVideo.pause();
  if (state.hls) {
    state.hls.destroy();
    state.hls = null;
  }
  state.hlsResponseId = null;
  if (state.sourceBuffer?.updating) {
    try {
      state.sourceBuffer.abort();
    } catch {
      // Ignore teardown races.
    }
  }
  state.sourceBuffer = null;
  state.mediaSource = null;
  state.mediaSourceMime = null;
  state.mediaSourceQueue = [];
  state.mediaSourceFetching = false;
  state.mediaSourcePrerollSeconds = 1;
  if (state.mediaSourceUrl) {
    URL.revokeObjectURL(state.mediaSourceUrl);
    state.mediaSourceUrl = null;
  }
  elements.avatarVideo.removeAttribute("src");
  elements.avatarVideo.load();
}

function skipSmallMediaGap() {
  if (!state.mediaSource || elements.avatarVideo.buffered.length < 2) {
    return;
  }
  const currentTime = elements.avatarVideo.currentTime;
  for (let index = 1; index < elements.avatarVideo.buffered.length; index += 1) {
    const nextStart = elements.avatarVideo.buffered.start(index);
    if (nextStart <= currentTime) {
      continue;
    }
    // AAC packets and 25 fps video frames do not always end on the same
    // timestamp at an fMP4 fragment boundary. Chromium can wait forever on
    // the resulting sub-frame hole even though the next fragment is ready.
    if (nextStart - currentTime <= 0.2) {
      elements.avatarVideo.currentTime = nextStart + 0.001;
    }
    return;
  }
}

function playNextSegment() {
  if (state.segmentPlaying || state.segmentQueue.length === 0) {
    if (!state.segmentPlaying && state.rendererDone) {
      setAvatarState("listening");
    }
    return;
  }
  const segment = state.segmentQueue.shift();
  state.segmentPlaying = true;
  elements.avatarVideo.src = segment.url;
  prepareAvatarVideo();
  setAvatarState("speaking");
  elements.avatarVideo.play().catch(() => {
    state.segmentPlaying = false;
    elements.assistantTranscript.textContent = "浏览器阻止了有声视频自动播放，请点击画面继续。";
  });
}

elements.avatarVideo.addEventListener("ended", () => {
  if (state.mediaSource) {
    state.segmentPlaying = false;
    if (state.rendererDone) {
      setAvatarState("listening");
    }
  } else if (state.segmentMode) {
    state.segmentPlaying = false;
    playNextSegment();
  } else {
    state.segmentPlaying = false;
    setAvatarState("listening");
  }
  if (
    state.upstreamMode === "omlx" &&
    !state.segmentPlaying &&
    (!state.segmentMode || state.rendererDone)
  ) {
    resumeLocalVadNow();
  }
  if (!state.segmentPlaying) {
    showAvatarImage();
  }
});

elements.avatarVideo.addEventListener("playing", () => {
  clearAvatarImageFallback();
  showAvatarVideoAfterFirstFrame();
  if (state.playbackStartedAt === null) {
    state.playbackStartedAt = performance.now();
    if (state.responseStartedAt !== null) {
      elements.bufferMetric.textContent = `开始播放 ${Math.round(state.playbackStartedAt - state.responseStartedAt)} ms`;
    }
  }
});

elements.avatarVideo.addEventListener("waiting", () => {
  skipSmallMediaGap();
  scheduleAvatarImageFallback();
});
elements.avatarVideo.addEventListener("stalled", () => {
  skipSmallMediaGap();
  scheduleAvatarImageFallback();
});
elements.avatarVideo.addEventListener("timeupdate", skipSmallMediaGap);

elements.avatarVideo.addEventListener("click", () => {
  elements.avatarVideo
    .play()
    .then(() => {
      state.segmentPlaying = true;
      setAvatarState("speaking");
    })
    .catch(() => {});
});

elements.micButton.addEventListener("click", toggleMicrophone);
elements.cloudAvatarButton.addEventListener("click", toggleCloudAvatar);
elements.connectButton.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  beginHoldToTalk(event.pointerId);
});
elements.connectButton.addEventListener("pointerup", endHoldToTalk);
elements.connectButton.addEventListener("pointercancel", endHoldToTalk);
elements.connectButton.addEventListener("lostpointercapture", endHoldToTalk);
elements.connectButton.addEventListener("contextmenu", (event) => event.preventDefault());
elements.connectButton.addEventListener("keydown", (event) => {
  if ((event.key === " " || event.key === "Enter") && !event.repeat) {
    event.preventDefault();
    beginHoldToTalk();
  }
});
elements.connectButton.addEventListener("keyup", (event) => {
  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    endHoldToTalk();
  }
});
elements.connectButton.addEventListener("blur", endHoldToTalk);
elements.stage.addEventListener("pointermove", revealCursorTemporarily, { passive: true });
elements.stage.addEventListener("pointerdown", revealCursorTemporarily, { passive: true });
elements.fullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await elements.stage.requestFullscreen();
    }
  } catch {
    elements.connectionText.textContent = "浏览器不支持全屏";
  }
});

document.addEventListener("fullscreenchange", () => {
  const active = Boolean(document.fullscreenElement);
  elements.fullscreenButton.setAttribute("aria-label", active ? "退出全屏" : "进入全屏");
});

loadHealth().catch(() => {
  elements.modeMetric.textContent = "不可用";
  elements.avatarMetric.textContent = "不可用";
  elements.playbackMetric.textContent = "不可用";
});
