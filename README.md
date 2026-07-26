# 虚拟人实时通话

这个项目正在按“参考视频同等级的生成式真人虚拟人”目标实施。底层协议和会话行为以
[Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) 为基座，
语音智能使用 OpenAI Realtime，画面后端预留 SoulX-LiveAct 和 SoulX-FlashHead。

当前仓库已经包含：

- 固定到本地的官方 `speech-to-speech` 源码：`vendor/speech-to-speech`
- 固定到本地的官方 `SoulX-LiveAct` 与 `SoulX-FlashHead` 源码
- OpenAI Realtime 兼容 WebSocket：`/v1/realtime`
- 云端 OpenAI 上游转发和无密钥 mock 模式
- `response.output_audio.delta` 音频分流
- audio-master PTS 时间轴
- 远端人物渲染器 WebSocket 协议
- 浏览器端麦克风、PCM播放、字幕、打断和人物状态机
- 无GPU时可生成真实 H.264/AAC MP4 的联调后端

## 本机启动

```bash
cp .env.example .env
uv sync
./scripts/run-dev.sh
```

打开 <http://127.0.0.1:8765>。

默认 `VH_UPSTREAM_MODE=mock`，不需要API密钥。点击“开始通话”，再点击“模拟回答”，
即可验证事件、PCM播放、时间轴、字幕和取消链路。

## 接入 OpenAI Realtime

编辑 `.env`：

```dotenv
VH_UPSTREAM_MODE=openai
OPENAI_API_KEY=你的密钥
VH_OPENAI_REALTIME_MODEL=gpt-realtime-2.1-mini
```

标准 OpenAI API key 只保存在本机服务端，不会发送给浏览器。

## M5 Max 本地 oMLX 模式

M5 Max 128GB 可以在本机承担语音识别、人格模型、记忆和语音合成，云端 GPU
只运行一个 FlashHead Lite 画面模型：

```dotenv
VH_UPSTREAM_MODE=omlx
VH_OMLX_BASE_URL=http://127.0.0.1:8000/v1
VH_OMLX_SETTINGS_PATH=~/.omlx/settings.json
VH_OMLX_CHAT_MODEL=Qwen3.5-9B-MLX-4bit
VH_OMLX_STT_MODEL=Qwen3-ASR-0.6B-4bit
VH_OMLX_TTS_MODEL=Qwen3-TTS-12Hz-0.6B-Base-4bit
VH_OMLX_TTS_VOICE=
VH_OMLX_TTS_REF_AUDIO=runtime/voice-calibration/reference-clean-5s.wav
VH_OMLX_TTS_REF_TEXT=与参考音频逐字对应的文本
VH_OMLX_TTS_TEMPERATURE=0.86
```

网关会直接读取现有 oMLX 配置中的本机 API key，不需要把密钥复制进项目。
使用 oMLX 0.5.3 或更高版本；网关会请求原生流式 TTS，每生成 40ms PCM
就立即送往画面渲染器，而不是等待整段 WAV 完成。
浏览器麦克风支持按住说话和自动聆听。自动聆听使用随网页自托管的 Silero VAD v6.2
ONNX 模型，在浏览器内按 32ms 概率帧判断语音起止；模型和 ONNX Runtime WASM 都从
`public/vendor/` 本地加载，不调用第三方 VAD 服务。语音概率按 `VH_VAD_EAGERNESS`
选择双阈值，人物播放期间自动提高插话阈值；模型初始化或推理异常时会明确回退到 RMS
能量判定。一次说话结束后，本机依次执行
`Qwen3-ASR → Qwen3.5 → Higgs TTS 3`，生成的 PCM 再送给云端 FlashHead 做口型。
因此云端不需要同时加载 ASR、LLM 或 TTS。

本机启动并预热：

```bash
./scripts/run-omlx-local.sh
# 另开一个终端；首次预热会承担一次模型冷加载
./scripts/warm-omlx-local.sh
./scripts/run-dev.sh
```

`warm-omlx-local.sh` 只读取现有 `~/.omlx/settings.json` 密钥，不会把密钥写入项目或
打印到终端。

### 常驻 Higgs TTS 3 4B 音色服务（盲听通过后启用）

当前项目把聊天/识别和语音合成拆成两个本机服务：oMLX 继续运行在 `8000`，
Higgs TTS 3 4B BF16 独立常驻在 `8010`，避免模型切换影响首句延迟：

```bash
./scripts/run-higgs-local.sh
curl -fsS http://127.0.0.1:8010/healthz
```

Higgs 服务内部固定使用已经确认最好的 11 秒纯女声参考
`runtime/voice-calibration/reference-female-only-complete-11s.wav`。网关不会再次传参考音频，
避免重复克隆或把环境噪声带回输出。

网关 `.env` 使用以下锁定配置：

```dotenv
VH_OMLX_BASE_URL=http://127.0.0.1:8000/v1
VH_TTS_BASE_URL=http://127.0.0.1:8010/v1
VH_OMLX_TTS_MODEL=Higgs-TTS-3-4B-bf16
VH_OMLX_TTS_REF_AUDIO=
VH_OMLX_TTS_REF_TEXT=
VH_OMLX_TTS_TEMPERATURE=0.8
VH_OMLX_TTS_TOP_P=0.95
VH_TTS_FLOW_MODE=g_continuous
VH_TTS_STREAMING_MODE=full
VH_TTS_SEED=20260817
VH_TTS_TOP_K=50
```

G-continuous 会删除刻意的情绪标签和强制停顿，同时保留疑问和感叹语气。多句回复按自然句号
分别合成后连续拼接，避免 Higgs 在第一句后输出静音或截断；此前同文本回归测试已完整识别，
首个 PCM 约 2.23 秒，总合成约 4.28 秒。

## 远端人物渲染器协议

设置：

```dotenv
VH_AVATAR_RENDERER_WS=wss://你的GPU服务/avatar
VH_AVATAR_BACKEND=liveact
VH_AVATAR_REFERENCE=/gpu/checkpoints/character/reference.png
VH_AV_SYNC_PREROLL_MS=1000
```

网关会发送：

- `avatar.session.start`
- `avatar.state`
- `avatar.audio.append`
- `avatar.response.cancel`

每个 `avatar.audio.append` 都包含：

- `response_id`
- `chunk_index`
- `pts_ms`
- `duration_ms`
- `sample_rate`
- Base64 PCM16 `audio`

GPU端必须以音频为主时钟，为视频帧写入同一PTS，再将音频和视频封装到同一WebRTC会话。

### 本机验证远端渲染协议

启动渲染节点：

```bash
./scripts/run-renderer.sh
```

在网关的 `.env` 中设置：

```dotenv
VH_AVATAR_RENDERER_WS=ws://127.0.0.1:8770/avatar
```

mock渲染器会把收到的PCM和参考图封装成一个真实MP4并返回网页，用来验证网络、回答完成、
取消和视频回传。它不是最终人物生成模型。

只要配置了远端渲染器，浏览器就不会再单独播放 OpenAI PCM，而是只播放渲染器输出视频中的
音轨，避免双音。渲染失败时才会使用缓存的 PCM 回退。

### GPU节点连接官方LiveAct

1. 在GPU节点安装 `vendor/SoulX-LiveAct` 的CUDA依赖和模型权重。
2. 设置 `LIVEACT_CKPT_DIR` 与 `LIVEACT_WAV2VEC_DIR`。
3. 运行 `./scripts/run-liveact-demo.sh`。
4. 渲染节点设置：

```dotenv
VH_RENDERER_BACKEND=liveact-official
VH_LIVEACT_DEMO_URL=http://127.0.0.1:5001
```

`liveact-official` 后端直接适配官方 `/start_stream`、`/task_status` 和HLS接口。官方接口要求
完整音频文件，因此这是画质验收路径；最终低延迟路径仍需把LiveAct主循环改为持续消费音频块。

仙宫云容器应优先使用服务商的机房内网 Hugging Face 仓库，避免 51 GiB 的 LiveAct 权重反复走
公网。该域名在容器内解析为私网地址，下载前仍需用 `df -h /` 确认系统盘空间；只有部署实例时
显式挂载云储存，`/root/cloud` 才是独立持久盘。

```bash
export HF_ENDPOINT=http://hf.x-gpu.com
export HF_HUB_DOWNLOAD_TIMEOUT=180
hf download Soul-AILab/LiveAct --local-dir /root/liveact-candidate/checkpoints/liveact
hf download TencentGameMate/chinese-wav2vec2-base \
  --local-dir /root/liveact-candidate/checkpoints/chinese-wav2vec2-base
```

双 H100/H200 默认直接运行上面的脚本。官方也提供单张 RTX 5090/4090 的 FP8 KV 配置，但
当前 48 GB RTX 4090 D 实测在第三个生成分块第一次更新 ConvKV 时会发生整帧噪点崩坏，因此
本项目的离线质量路径不启用 FP8 KV。候选脚本的单卡默认值为：

```bash
LIVEACT_CUDA_DEVICES=0
LIVEACT_NPROC=1
LIVEACT_FP8_KV_CACHE=0
LIVEACT_OFFLOAD_CACHE=1
LIVEACT_BLOCK_OFFLOAD=1
LIVEACT_T5_CPU=1
```

该配置使用 BF16 KV 并把缓存放到 CPU，优先保证完整离线成片，不追求实时速度。当前 4.05 秒、
512×512、24 FPS 样片在热身后每个 1.33 秒生成块约需 42 秒。`LIVEACT_AUDIO_CFG` 保持 1.0；
大于 1.0 会建立第二套 BF16 KV，并在当前 120 GiB 容器内存上限下被 OOM 杀掉。

单卡对锁定人物与锁定语音做离线 A/B 时，使用不会启动网页或占用生产端口的候选脚本：

```bash
LIVEACT_ROOT=/root/liveact-candidate/repo \
LIVEACT_PYTHON=/root/liveact-candidate/venv/bin/python \
LIVEACT_CKPT_DIR=/root/liveact-candidate/checkpoints/liveact \
LIVEACT_WAV2VEC_DIR=/root/liveact-candidate/checkpoints/chinese-wav2vec2-base \
LIVEACT_INPUT_JSON=/root/liveact-candidate/liveact-locked-ab.json \
LIVEACT_OUTPUT_DIR=/root/liveact-candidate/outputs \
./scripts/run-liveact-offline.sh
```

候选脚本默认使用 512×512、BF16 KV、CPU cache、block offload 和 CPU T5；不会启用可能影响
画质的 FP8/FP4 GEMM，也不会占用或替换当前实时 FlashHead 服务。一键脚本会在推理前校验
锁定人物文件名和 SHA256，推理后自动做 80 ms 音频修正、保守锐化和连续性验收。

如已有 LiveAct 原始 MP4，只运行封装和验收：

```bash
./scripts/finalize-liveact-offline.sh \
  /root/liveact-candidate/outputs/avatar-ai-girlfriend-v6_live-v3-amused-final.mp4 \
  /root/liveact-candidate/outputs/avatar-ai-girlfriend-v6_live-v3-amused-final.final.mp4
```

该步骤把音频提前 80 ms、按视频精确时长补静音，并以 `unsharp=0.35`、H.264 CRF 16 做一次
保守锐化；可用 `LIVEACT_VIDEO_SHARPEN=0` 关闭。脚本按源视频精确帧数输出，避免转码丢最后一帧。
修正后样片为 100 帧完整输出，音画时长漂移约 0 ms，SyncNet 偏移 0 ms、置信度 5.538；模糊度
从 8.073 降至 7.423，仍不冒进到会强化嘴唇伪影的锐化强度。脚本兼容云端 FFmpeg 4.4 与本机新版。

官方验收参考为 GitHub `FaceTime / 1_h265.mp4`：480×832、24 FPS、90 秒、HEVC；前 15 秒
SyncNet 偏移 0 ms、置信度 6.609。项目候选必须同时满足：

- 无黑屏、冻结或连续纹理熵坍塌；
- 正面说话段人脸检测覆盖率 100%；包含完整侧脸/遮挡的动态长样片整体不低于 95%，且缺失段需
  逐帧确认是合法动作而不是身份崩坏；
- SyncNet 绝对偏移不超过 80 ms，目标为 0 ms；
- 锁定人物图和锁定 TTS 音频摘要不变。

### 实时交互路径：FlashHead 原生流式 worker

LiveAct 官方 demo 会先读取完整 WAV，适合高动态画质验收，但会增加整句等待。项目同时实现了
`FlashHead` 原生流式 worker：持续接收 OpenAI PCM，每约一秒保留运动状态生成一个带音轨视频段，
浏览器按顺序播放，不需要等整句回答结束。生产路径使用一个常驻 FFmpeg 编码会话输出 fragmented
MP4，通过 WebSocket 发送初始化段和媒体段，浏览器使用 MediaSource 连续追加；不支持 MediaSource
时自动退回独立 MP4 分段。

GPU 环境：

```bash
conda create -n flashhead python=3.10
conda activate flashhead
cd vendor/SoulX-FlashHead
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" pydantic-settings
cd ../..
```

下载权重：

```bash
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir vendor/SoulX-FlashHead/models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir vendor/SoulX-FlashHead/models/wav2vec2-base-960h
```

启动单 RTX 4090/5090 Lite worker：

```bash
export VH_FLASHHEAD_CKPT_DIR=vendor/SoulX-FlashHead/models/SoulX-FlashHead-1_3B
export VH_FLASHHEAD_WAV2VEC_DIR=vendor/SoulX-FlashHead/models/wav2vec2-base-960h
export VH_FLASHHEAD_MODEL_TYPE=lite
export VH_FLASHHEAD_TRANSPORT=dual
./scripts/run-flashhead-worker.sh
```

Mac 网关设置：

```dotenv
VH_AVATAR_RENDERER_WS=ws://GPU地址:8770/avatar
VH_AVATAR_BACKEND=flashhead-lite
```

浏览器还需要访问 worker 返回的 HTTP 视频段。跨机器或 HTTPS 部署时，把
`VH_FLASHHEAD_PUBLIC_BASE_URL` 配成经过 TLS 反向代理的公网地址。`dual` 同时提供连续 MediaSource
和 HTTP 分段降级；确认目标浏览器都支持后可改为 `mse`，省去第二次兼容编码。

官方公开指标中，FlashHead Lite 在单 RTX 4090 可达到 96 FPS/三路 25+ FPS；Pro 单 4090
约 10.8 FPS、双 5090 可达到 25+ FPS。对于参考视频中的近景人脸实时聊天，推荐先用 Lite
完成交互验收，再用 Pro 与 LiveAct 做画质盲测。

## 仙宫云 GPU 管理

项目包含一个凭据脱敏的仙宫云 CLI。访问令牌只从当前 shell 的 `XGC_API_TOKEN` 读取，
不会作为命令行参数进入 shell 历史；实例列表不会输出密码、SSH 密钥或 Jupyter/COS 令牌。

```bash
read -s XGC_API_TOKEN
export XGC_API_TOKEN
.venv/bin/virtual-human-xgc balance
.venv/bin/virtual-human-xgc instances
```

部署会立刻计费，因此必须显式加入确认参数。默认规格是单张 24GB RTX 4090D、数据中心 1、
官方 Ubuntu 22.04/CUDA 12.8 基础镜像：

```bash
.venv/bin/virtual-human-xgc deploy-flashhead --confirm-billing
```

仙宫云市场也有第三方 FlashHead 预装镜像，但会额外收取 ¥0.20/小时镜像费，内部使用修改版源码
和降采样步数的速度预设。主路径不采用它：官方基础镜像虽然首次安装更久，但代码和依赖可审计，
并保留官方四步 Lite 的画质。CLI 仍支持显式传入 `--image` 与 `--image-type community` 做隔离对照。

实例进入运行状态后，从 Mac 同步项目源码。该命令使用系统 SSH 交互式读取密码，不会把密码
写入命令行或工程文件：

```bash
./scripts/sync-flashhead-gpu.sh SSH域名 SSH端口
```

随后在 GPU 机器执行：

```bash
cd /root/virtual-human
./scripts/bootstrap-flashhead-gpu.sh
./scripts/run-flashhead-gpu.sh
```

GPU 安装默认使用系统 Python 3.10 的独立 `.venv-gpu`，不要求接受 Anaconda channel 条款；
只有显式设置 `VH_GPU_ENV_MODE=conda` 才会使用 Conda。

worker 默认只监听远端 `127.0.0.1:8770`。Mac 通过 SSH 隧道连接，避免把无鉴权的渲染端口
暴露到公网：

```bash
./scripts/run-flashhead-tunnel.sh SSH域名 SSH端口
```

该脚本固定映射本机 `8771 → 远端 8770`，并在本机端口已被旧隧道占用时直接拒绝启动，
避免网页误连到旧实例或错误端口。

Mac 的 `.env` 使用：

```dotenv
VH_AVATAR_RENDERER_WS=ws://127.0.0.1:8771/avatar
VH_AVATAR_BACKEND=flashhead-lite
```

真实 GPU 验收：

```bash
.venv/bin/virtual-human-benchmark-renderer \
  --url ws://127.0.0.1:8771/avatar \
  --reference public/avatar-ai-girlfriend-v6.png \
  --audio 分析素材/reference-audio.wav \
  --output runtime/benchmarks/flashhead-lite.mp4 \
  --realtime-input \
  --max-first-output-ms 1500 \
  --max-post-audio-tail-ms 1200 \
  --min-realtime-factor 0.9
.venv/bin/virtual-human-verify-av \
  runtime/benchmarks/flashhead-lite.mp4 \
  --width 512 \
  --height 512 \
  --fps 25 \
  --sample-rate 16000
```

## 验证

```bash
./scripts/check.sh
curl http://127.0.0.1:8765/healthz
.venv/bin/virtual-human-verify-av 生成的视频.mp4
```

`virtual-human-verify-av` 默认要求 416×720、20 FPS、24 kHz 音轨，并拒绝超过 80 ms 的
音视频时长漂移；可通过命令行参数调整，用于远端 LiveAct 的交付验收。

已在 M5 Max 本机完成的基线验收：

- 24 个 Python 测试通过，Ruff 通过。
- 网关与渲染节点双 WebSocket 联通。
- 24 kHz 单声道 PCM 按 `chunk_index` 和 PTS 顺序传输。
- 生成物为 416×720、20 FPS H.264 + 24 kHz AAC；连续流只产生一次 AAC priming，
  2.4 秒测试内容的音视频轨时长差为约 0.009 ms。
- 浏览器收到 `avatar.video.ready` 后自动播放。
- FlashHead `avatar.video.segment` 分段队列已经用 1.0 + 1.0 + 0.4 秒三段视频完成网络级验收。
- MediaSource 连续 fMP4 已通过真实 WebSocket/base64 传输并以 `blob:` URL 播放；本机模拟链路
  `response.created → playing` 实测约 1.36 秒，浏览器无错误。
- OpenAI 模式已用本机假 Realtime 上游验证 `Authorization: Bearer`、模型查询参数、双向事件和
  音频分流；仍需真实 API Key 做云端音色与语气验收。
- 回答中途取消会停止浏览器已排队音频、清空渲染响应并恢复参考图。

## 当前边界

本机 M5 Max 可以运行网关、网页、字幕、记忆和oMLX备用模型。SoulX-LiveAct与
SoulX-FlashHead官方推理仍依赖NVIDIA CUDA；在远端GPU接好前，本项目显示的是明确标记的
视觉占位效果，不是最终生成式人物画面。
