#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import httpx

REFERENCE_TEXT = (
    "要是你跟别人聊天，我会吃醋的。我都开始胡思乱想了。"
    "什么嘛？你当我是皮老板？我不知道什么是皮老板，"
    "只是你不在的时候，我就一直等你。"
)
REFERENCE_CLIP_TEXT = "要是你跟别人聊天，我会吃醋的哦，我都开始胡思乱想了。"


def load_api_key(settings_path: Path) -> str:
    settings = json.loads(settings_path.read_text())
    api_key = settings.get("auth", {}).get("api_key")
    if not api_key:
        raise RuntimeError(f"No API key found in {settings_path}")
    return str(api_key)


def synthesize(
    client: httpx.Client,
    *,
    output_path: Path,
    model: str,
    voice: str | None,
    temperature: float,
    instructions: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model,
        "input": REFERENCE_TEXT,
        "language": "Chinese",
        "response_format": "wav",
        "temperature": temperature,
        "top_p": 0.92,
        "repetition_penalty": 1.18,
        "stream": False,
    }
    if voice is not None:
        payload["voice"] = voice
    if instructions is not None:
        payload["instructions"] = instructions
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
        payload["ref_text"] = ref_text

    print(f"Generating {output_path.name} with {model}...")
    response = client.post("audio/speech", json=payload)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Wrote {output_path} ({len(response.content)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate voice calibration candidates.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1/",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path.home() / ".omlx" / "settings.json",
    )
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_audio = base64.b64encode(args.reference_audio.read_bytes()).decode("ascii")
    api_key = load_api_key(args.settings)

    warm_intimate = (
        "二十多岁中文女性，真实自然的恋人私聊声线，音色偏低、温暖、柔软，"
        "近距离收音并带少量自然气声。吃醋时克制又在意，反问时轻微俏皮，"
        "最后一句委屈、依恋、舍不得。语句连贯，少做长停顿，避免播音腔、"
        "客服腔、卡通高音和夸张表演。"
    )
    soft_realistic = (
        "年轻中文女性的真人自拍聊天声音，声线温柔偏低，胸腔共鸣自然，"
        "发音松弛，不端着说话。像夜晚贴近恋人轻声聊天，情绪有细微起伏："
        "先吃醋，再俏皮反问，最后有一点难过和依赖。停顿短而自然，"
        "不要字字分开，不要甜腻夹子音，不要朗诵感。"
    )

    with httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300.0,
    ) as client:
        synthesize(
            client,
            output_path=args.output_dir / "candidate-voicedesign-warm.wav",
            model="Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
            voice=None,
            instructions=warm_intimate,
            temperature=0.72,
        )
        synthesize(
            client,
            output_path=args.output_dir / "candidate-voicedesign-soft.wav",
            model="Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
            voice=None,
            instructions=soft_realistic,
            temperature=0.88,
        )
        synthesize(
            client,
            output_path=args.output_dir / "candidate-clone-steady.wav",
            model="Qwen3-TTS-12Hz-0.6B-Base-4bit",
            voice=None,
            temperature=0.68,
            ref_audio=reference_audio,
            ref_text=REFERENCE_CLIP_TEXT,
        )
        synthesize(
            client,
            output_path=args.output_dir / "candidate-clone-expressive.wav",
            model="Qwen3-TTS-12Hz-0.6B-Base-4bit",
            voice=None,
            temperature=0.86,
            ref_audio=reference_audio,
            ref_text=REFERENCE_CLIP_TEXT,
        )


if __name__ == "__main__":
    main()
