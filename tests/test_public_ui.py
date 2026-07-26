import hashlib
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "public"


def test_realtime_ui_keeps_required_dom_hooks() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "stage",
        "connectButton",
        "mainButtonLabel",
        "micButton",
        "cloudAvatarButton",
        "fullscreenButton",
        "connectionDot",
        "connectionText",
        "avatarFrame",
        "avatarVideo",
        "avatarImage",
        "avatarState",
        "assistantTranscript",
        "userTranscript",
        "modeMetric",
        "bufferMetric",
        "avatarMetric",
        "playbackMetric",
        "personaTitle",
        "privacyNote",
    ):
        assert f'id="{element_id}"' in html


def test_hls_player_is_pinned_and_served_locally() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    hls_bundle = (PUBLIC / "hls.min.js").read_bytes()

    assert 'src="/hls.min.js?v=1.6.16"' in html
    assert hashlib.sha256(hls_bundle).hexdigest() == (
        "442f599c34f103c3355b375a23bdff560592d7117d09a8c847242ea3de2d40e0"
    )
    assert (PUBLIC / "hls.min.js.LICENSE.txt").is_file()


def test_ui_uses_current_avatar_and_responsive_fullscreen_layout() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    css = (PUBLIC / "styles.css").read_text(encoding="utf-8")
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert 'src="/avatar-ai-girlfriend-v6.png"' in html
    assert 'class="voice-orb"' in html
    assert "@media (max-width: 860px)" in css
    assert ".assistant-line .accent-word" in css
    assert 'document.querySelector("#stage")' in javascript
    assert 'document.querySelector("#fullscreenButton")' in javascript
    assert "对话文本发送至 Ark" in javascript


def test_two_controls_support_toggle_and_hold_to_talk() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert 'id="interruptButton"' not in html
    assert 'elements.micButton.addEventListener("click", toggleMicrophone)' in javascript
    assert 'elements.connectButton.addEventListener("pointerdown"' in javascript
    assert 'elements.connectButton.addEventListener("pointerup", endHoldToTalk)' in javascript
    assert 'elements.connectButton.addEventListener("keydown"' in javascript
    assert "beginHoldToTalk" in javascript
    assert "endHoldToTalk" in javascript
    assert "releaseHalfDuplexIfIdle" in javascript
    assert "!state.localVadBlocked" in javascript
    assert "state.localVadSpeaking" in javascript
    assert "state.localVadBargeInChecking" in javascript
    assert 'if (nextState === "listening")' in javascript
    assert "holdToTalkChunks" in javascript
    assert 'elements.userTranscript.textContent = "正在录音，松开发送…"' in javascript
    assert "forceCommit: hasRecordedAudio" in javascript
    assert "await startMicrophone()" in javascript
    assert 'send({ type: "input_audio_buffer.commit" })' in javascript
    assert "localVadPreRoll" in javascript
    assert "resumeLocalVadAfterPlayback" in javascript
    assert "assistantIsActive" in javascript
    assert "barge_in_enabled: false" in javascript
    assert "!state.clientConfig.barge_in_enabled" in javascript
    assert '"input_audio_buffer.barge_in.append"' in javascript
    assert '"input_audio_buffer.barge_in.commit"' in javascript
    assert '"input_audio_buffer.barge_in.echo_ignored"' in javascript
    assert 'import { LocalSileroVad, SILERO_VAD_VERSION } from "./silero-vad.js"' in javascript
    assert "await state.localSileroVad.process(" in javascript
    assert "function sileroThresholds(assistantActive)" in javascript
    assert '"Silero 听到插话了，继续说…"' in javascript
    assert "state.localVadEnergySpeechMs >= 120" in javascript
    assert '"energy-rescue"' in javascript
    assert "dataset.vadProbability" in javascript
    assert "Math.max(0.008, state.localVadNoiseFloor * 2.5)" in javascript
    assert 'type: "response.create"' not in javascript


def test_right_cloud_avatar_button_persists_and_reconnects_mode() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert 'id="cloudAvatarButton"' in html
    assert 'id="cloudButtonStatus"' in html
    assert 'src="/app.js?v=24"' in html
    assert 'src="/hls.min.js?v=1.6.16"' in html
    assert 'href="/styles.css?v=10"' in html
    assert 'localStorage.getItem("cloudAvatarEnabled")' in javascript
    assert 'localStorage.setItem("cloudAvatarEnabled"' in javascript
    assert 'elements.cloudAvatarButton.addEventListener("click", toggleCloudAvatar)' in javascript
    assert 'avatar=${avatarMode}' in javascript
    assert 'case "avatar.mode":' in javascript
    assert '"连接失败"' in javascript
    assert '"云端已连"' in javascript


def test_avatar_stream_keeps_static_portrait_until_first_video_frame() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    css = (PUBLIC / "styles.css").read_text(encoding="utf-8")
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert 'class="avatar-frame"' in html
    assert 'class="avatar-frame face-fusion"' not in html
    assert "requestVideoFrameCallback" in javascript
    assert "showAvatarVideoAfterFirstFrame" in javascript
    assert "prepareAvatarVideo" in javascript
    assert 'elements.avatarImage.hidden = true' not in javascript
    assert ".avatar-video.is-visible" in css
    assert ".avatar-frame.video-ready #avatarImage" in css
    assert ".avatar-frame.video-ready #avatarImage" in css
    assert "opacity 180ms ease" in css
    assert "window.Hls?.isSupported()" in javascript
    assert "liveSyncDurationCount: 2" in javascript
    assert '"HLS 播放中断，正在等待完整视频兜底。"' in javascript


def test_cloud_renderer_failure_switches_to_browser_audio() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert 'src="/app.js?v=24"' in html
    assert 'case "avatar.fallback":' in javascript
    assert 'state.playbackOwner = "browser"' in javascript
    assert 'elements.playbackMetric.textContent = "浏览器 PCM"' in javascript
    assert 'elements.avatarMetric.textContent = "本地静态人物"' in javascript


def test_media_source_waits_for_server_preroll_before_playing() -> None:
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert "mediaSourcePrerollSeconds: 1" in javascript
    assert "function bufferedMediaSeconds()" in javascript
    assert "function maybeStartMediaSourcePlayback()" in javascript
    assert "bufferedSeconds < state.mediaSourcePrerollSeconds" in javascript
    assert "Number(event.preroll_ms || 1000) / 1000" in javascript


def test_browser_pcm_uses_preroll_and_waits_for_actual_playback_end() -> None:
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")

    assert "browserPcmPrerollSeconds: 1" in javascript
    assert "browserPcmPrimed: false" in javascript
    assert "leadSeconds = state.browserPcmPrimed" in javascript
    assert "state.sources.size === 0" in javascript
    assert "state.sources.size > 0" in javascript


def test_local_silero_vad_assets_are_vendored_and_wired() -> None:
    javascript = (PUBLIC / "app.js").read_text(encoding="utf-8")
    vad_module = (PUBLIC / "silero-vad.js").read_text(encoding="utf-8")
    model = PUBLIC / "vendor" / "silero-vad-v6.2" / "silero_vad.onnx"
    ort_module = (
        PUBLIC / "vendor" / "onnxruntime-web-1.22.0" / "ort.wasm.min.mjs"
    )
    ort_wasm = (
        PUBLIC
        / "vendor"
        / "onnxruntime-web-1.22.0"
        / "ort-wasm-simd-threaded.wasm"
    )

    assert model.is_file() and model.stat().st_size > 2_000_000
    assert ort_module.is_file() and ort_module.stat().st_size > 40_000
    assert ort_wasm.is_file() and ort_wasm.stat().st_size > 10_000_000
    assert 'from "./silero-vad.js"' in javascript
    assert "LocalSileroVad.create()" in javascript
    assert 'const MODEL_URL = "/vendor/silero-vad-v6.2/silero_vad.onnx"' in vad_module
    assert 'executionProviders: ["wasm"]' in vad_module
