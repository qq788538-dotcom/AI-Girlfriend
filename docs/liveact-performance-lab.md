# SoulX-LiveAct RTX 5090 性能实验知识库

## 目标与原则

目标是在单张 RTX 5090 32 GB 上显著缩短“语音结束到浏览器首个可播放画面”的时间，同时保持
人物身份、画面连续性、口型同步和音质不变，或只接受可量化的轻微质量下降。所有结论必须来自
同一人物、同一音频、同一种子和同一输出规格的 A/B，不用不同输入之间的主观印象代替证据。

## 官方口径

- SoulX-LiveAct 官方模型卡报告：18B 模型在单张 RTX 5090、FP8 KV cache 与 CPU block
  offload 下约 6 FPS；20 FPS 实时口径使用两张 H100/H200。
- 官方 5090 参数为 416×720、FP8 KV cache、block offload、T5 CPU 和
  `USE_CHANNELS_LAST_3D=1`。
- 官方安装流程要求 SageAttention 2.2.0；SageAttention 官方仓库报告 RTX 5090
  注意力内核微基准显著快于 FlashAttention 2，但端到端收益必须在 LiveAct 上单独验证。
- GUI 首批 block 需要预热，性能比较必须排除首次编译和模型冷启动。

来源：

- https://huggingface.co/xmuhtt/LiveAct
- https://arxiv.org/abs/2603.11746
- https://github.com/thu-ml/SageAttention

## 基线：2026-07-27

环境：

| 项目 | 基线 |
| --- | --- |
| GPU | RTX 5090 32 GB，PCIe 5.0 x16 |
| LiveAct | `ac8579b907756f228e975922a5159765c93c6a34` |
| LightX2V | `d658c11edb77124387f275c1c080fcf9612ba8e1` |
| Python / PyTorch / CUDA | 3.12.3 / 2.8.0+cu128 / 12.8 |
| 规格 | 416×720，20 FPS |
| 精度/卸载 | FP8 KV，FP8 GEMM，block offload，T5 CPU |
| 注意力 | SageAttention 未安装；LiveAct 主 self/cross attention 回退 PyTorch SDPA |

最近 120 个生成块：

| 指标 | 基线 |
| --- | ---: |
| 稳态吞吐 | 4.732 FPS |
| 官方单 5090 口径 | 6 FPS |
| 官方口径达成率 | 78.9% |
| 最近 30 次 T5 prompt 初始化均值 | 13.523 秒 |
| 第一块 | 21 帧 / 约 4.4–4.6 秒 |
| 浏览器首个 HLS 分片 | 约 18–20 秒 |
| 429 重叠拒绝 | 7 次 |
| wrapper 渲染失败 | 8 次 |

主要结论：硬件没有温度、频率、PCIe 或内存瓶颈。固定 T5 提示在每个回答中重复编码，是首画面
延迟最大的固定成本；SageAttention 缺失是稳态吞吐低于官方口径的首要可验证假设。

## 固定实验输入与门槛

每个候选至少运行一次冷后预热和三次热态重复，记录中位数与离散度。实验顺序交替
`A-B-B-A`，降低宿主机负载变化造成的偏差。

速度指标：

- `session_ready_ms`
- 音频结束到 `avatar.stream.ready`
- 音频结束到 `avatar.render.done`
- 首块与后续块 FPS
- prompt、CLIP、VAE、音频预处理分项时间
- 峰值显存、GPU 利用率、功耗、CPU 与 PCIe 状态

质量硬门槛：

- H.264/AAC 可完整解码；
- 无黑屏、冻结、时间戳缺口或持续纹理熵坍塌；
- A/V 时长漂移不超过 80 ms；
- SyncNet 绝对偏移不超过 2 帧；
- 相对基线的 SyncNet 置信度下降不超过 0.5；
- 锁定人物的人脸覆盖与人工 contact sheet 不出现身份漂移、嘴部破坏或手部明显崩坏。

PSNR、SSIM、VMAF/LPIPS 仅作为相对诊断：量化注意力可能产生不同但同样合理的动作轨迹，
不能单独以逐像素差异判定失败。

## 实验队列

| ID | 变量 | 预期收益 | 风险 | 状态 |
| --- | --- | --- | --- | --- |
| E01 | SDPA → SageAttention 2.2.0 | 补齐单 5090 稳态吞吐差距 | 量化误差、SM120 构建 | 进行中 |
| E02 | 缓存固定 T5 embedding | 首画面减少约 13.5 秒 | prompt key 失配 | 待测 |
| E03 | 缓存 reference CLIP/VAE latent | 首画面减少约 0.7–1 秒 | 跨人物污染 | 待测 |
| E04 | renderer 单任务队列/最新任务策略 | 消除 429 和无响应 | 排队延迟 | 待测 |
| E05 | 持续会话与增量音频 | TTS 与动画重叠 | 上游 demo API 改造 | 待测 |
| E06 | 轻微降分辨率/帧率/量化 | 像素吞吐提升 | 清晰度、口型质量 | 待测 |
| E07 | 固定/填充尾块形状，复用 compile graph | 消除偶发数分钟无响应 | 多余尾帧需裁剪 | 已定位，待实现 |

## 实验记录

### E00-B0：首次固定输入基线（无效样本）

- 输入：macOS 系统 `Tingting` 合成中文，7.995 秒，不包含用户语音；
- 运行时同时存在 SageAttention 源码构建，前两块分别耗时 12.18 秒和 17.88 秒，
  不能作为 SDPA 稳态基线；
- 挂起全部 SageAttention 构建进程后，任务仍停在 2/5，GPU 利用率为 0%；
- LiveAct 子进程出现 Triton `ptxas --gpu-name sm_120a`，确认真正停顿来自
  `torch.compile` 新图编译，不是 GPU 推理；
- 同一任务前已有 `torch._dynamo hit config.recompile_limit (8)`，VAE 的 channel
  shape 在 192/384 间变化；
- 结论：这轮不进入 A/B 汇总。需要让 compile cache 落盘，随后相同输入热态重跑；
  E07 提升为高优先级，并分别记录冷编译时间与热态生成时间。

## 可重复命令

汇总 LiveAct 日志：

```bash
python scripts/analyze-liveact-log.py /root/autodl-tmp/liveact-runtime/demo.log \
  --last-tasks 30
```

生产协议端到端基准：

```bash
virtual-human-benchmark-renderer \
  --url ws://127.0.0.1:8772/avatar \
  --reference public/avatar-ai-girlfriend-v6.png \
  --audio runtime/benchmarks/liveact-ab-input.wav \
  --output runtime/benchmarks/liveact-ab-output.mp4
```

真实 LiveAct 形状的注意力 A/B：

```bash
python scripts/benchmark-liveact-attention.py --preset first
python scripts/benchmark-liveact-attention.py --preset steady
python scripts/benchmark-liveact-attention.py --preset image-cross
python scripts/benchmark-liveact-attention.py --preset text-cross
```

`first` 使用 7020×7020 token，`steady` 使用 9360×16380 token；
均为 batch 1、40 heads、head dimension 128、BF16、NHD，与 416×720
LiveAct 主干一致。内核 A/B 只能判断注意力本身的收益和数值误差，必须再通过同输入视频
端到端 A/B 才能决定是否部署。

连续性与 A/V：

```bash
python scripts/benchmark-media-continuity.py OUTPUT.mp4 \
  --min-fps 20 --max-av-drift-ms 80
```
