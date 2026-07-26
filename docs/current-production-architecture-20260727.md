# 当前生产架构清单（仙宫云）

更新时间：2026-07-27  
部署分支：`codex/xiangongyun-deploy`

## 实际运行链路

```text
浏览器 HTTPS/WSS
  └─ :80 自研 FastAPI/OpenAI Realtime 兼容网关
      ├─ :8001 Qwen3-ASR-0.6B（语音识别）
      ├─ :1934 OpenViking 0.4.11（长期记忆）
      │   ├─ :8002 Qwen3-Embedding-0.6B（CPU 向量）
      │   └─ :8000 Qwen3.6-35B-A3B-AWQ（记忆抽取）
      ├─ :8000 Qwen3.6-35B-A3B-AWQ + vLLM 0.19.1（对话）
      ├─ :8010 Higgs Audio V3 4B + vLLM-Omni 0.24.0（语音合成）
      └─ :8770 SoulX-FlashHead 1.3B Lite（人物与口型）
          └─ FFmpeg fragmented MP4（H.264 + AAC）返回浏览器
```

除公网网页与 WebSocket 网关 `:80` 外，所有推理端口都只监听
`127.0.0.1`。生产环境启用 `HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，运行时不调用
OpenAI、Ark、Claude 或其他外部推理 API。

## 开源项目与模型

| 项目/模型 | 当前用途 | 运行状态 | 固定信息 |
| --- | --- | --- | --- |
| [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) | 语音到语音总体协议与事件设计基座 | 作为源码基座，不单独启动其完整 Demo | 子模块 `656099a` |
| [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) | 真人图片驱动、面部运动、嘴型视频 | **正式启用** | 子模块 `9bc03de`；`SoulX-FlashHead-1_3B` Lite |
| [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) | 高质量离线人物视频候选 | **未用于当前生产链路** | 子模块 `ac8579b`；仅保留离线评估脚本 |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 中文语音识别 | **正式启用** | 本机 `:8001` |
| Qwen3.6-35B-A3B-AWQ | 女友对话与记忆抽取 | **正式启用** | ModelScope `tclf90/Qwen3.6-35B-A3B-AWQ`；本机 `:8000` |
| [vLLM](https://github.com/vllm-project/vllm) | Qwen LLM CUDA 推理服务 | **正式启用** | `0.19.1`，AWQ/FP16，8192 上下文 |
| [Higgs Audio V3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | 锁定女友音色的 TTS | **正式启用** | 本机 `:8010` |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | Higgs CUDA 常驻推理 | **正式启用** | `0.24.0` |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) audio tokenizer | Higgs 音频 tokenizer | **正式启用** | 仅下载 `audio_tokenizer/*` |
| [OpenViking](https://github.com/volcengine/OpenViking) | 独立虚拟女友长期记忆库 | **正式启用** | `0.4.11`；独立 workspace；本机 `:1934` |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | OpenViking 向量检索 | **正式启用** | CPU；1024 维；本机 `:8002` |
| [FastAPI](https://github.com/fastapi/fastapi) + [Uvicorn](https://github.com/encode/uvicorn) | 网页、健康接口、ASR/Embedding 服务和 WebSocket 网关 | **正式启用** | 项目自研适配层 |
| [websockets](https://github.com/python-websockets/websockets) | 浏览器网关和渲染器双 WebSocket | **正式启用** | 本地 Token 保护渲染器 |
| [FFmpeg](https://ffmpeg.org/) | RGB/PCM 编码、H.264/AAC、连续 fragmented MP4 | **正式启用** | 768×768，约 25fps |
| [PyTorch](https://github.com/pytorch/pytorch) | FlashHead CUDA 推理 | **正式启用** | `2.7.1`，CUDA 12.8 环境 |
| [SyncNet](https://github.com/joonson/syncnet_python) | 离线音画同步验收 | 只用于测试，不参与在线推理 | 子模块 `907c0b5` |
| [Silero VAD](https://github.com/snakers4/silero-vad) | 浏览器端人声端点检测候选 | 已评估，**尚未接入** | 计划采用自托管 ONNX，不影响当前切换 |

## oMLX 与云端的关系

Mac 成熟本地版可以使用 Apple MLX/oMLX。仙宫云是 Linux + NVIDIA CUDA，
**没有迁移或运行 oMLX.app**。云端配置中的 `VH_UPSTREAM_MODE=omlx` 和
`VH_CHAT_BACKEND=omlx` 是项目早期保留的 OpenAI 兼容适配器名称；它实际连接的是
`127.0.0.1:8000` 的本地 vLLM，不会连接 Mac，也不会访问外部接口。

## 当前没有进入生产链路的组件

- SoulX-LiveAct：只保留高质量离线成片与 A/B 评估路径。
- Silero VAD：只完成适配评估，当前网页仍使用既有端点检测逻辑。
- Apple oMLX/MLX：仅属于 Mac 本地版。
- OpenAI Realtime、Ark、Claude、云端 Embedding/TTS：仙宫云运行时全部关闭。
