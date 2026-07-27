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
| E02 | 缓存固定 T5 embedding | 首画面减少约 13.5 秒 | prompt key 失配 | 热态通过 |
| E03 | 内容哈希隔离的 reference CLIP/VAE latent 缓存 | 热态首画面减少约 0.8 秒 | GPU 常驻内存、跨人物污染 | 冷/热三次 E2E 通过 |
| E04 | renderer 全局单任务队列 | 消除 429 和无响应 | 排队延迟 | 单元测试通过，待并发 E2E |
| E05 | 持续会话与增量音频 | TTS 与动画重叠 | 上游 demo API 改造 | 待测 |
| E06 | 轻微降分辨率/帧率/量化 | 像素吞吐提升 | 清晰度、口型质量 | 待测 |
| E07 | 关闭不稳定 VAE compile；精确生成并裁剪尾块 | 消除数分钟卡死并保留完整尾音 | 最终块仍有完整块计算成本 | 冷/热端到端通过 |

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

### E02-A1/A2：T5 LRU 缓存与 VAE compile-off

固定输入为同一人物、同一段 macOS 系统合成中文（7.995 秒），不含用户语音。
服务配置为 T5 cache on、LightVAE decode compile off；SageAttention 尚未安装。

| 指标 | A1 缓存冷态 | A2 缓存热态 |
| --- | ---: | ---: |
| 生产协议完成 | 67.183 s | 33.946 s |
| T5 prompt | 19.609 s | 0.110 s |
| reference VAE `init_y` | 10.419 s | 0.605 s |
| 第一块 | 21 帧 / 7.682 s | 21 帧 / 4.455 s |
| 稳态吞吐 | 4.481 FPS | 4.552 FPS |
| 输出 | 416×720 / 20 FPS / H.264 + AAC | 同左 |

结论：

- T5 key/device LRU 命中将固定 prompt 从 19.609 秒降到 0.110 秒，降幅 99.4%；
- A1 还包含模型首请求的 VAE/CUDA 热身，不可把 33.2 秒全部归因于 T5；
- A2 热态的 `prompt + init_y + first chunk` 仅 5.17 秒；基准工具此前没有把
  `avatar.stream.ready` 计作首输出，已修正。A3 相同输入复测的 HLS 首输出为
  5.937 秒、最终完成 33.729 秒；
- A2/A3 两次热态合计：T5 0.110 秒、第一块 4.67 FPS、稳态 4.579 FPS，
  重复性良好；
- VAE compile-off 消除了十分钟 `ptxas` 挂起，但当前稳态约 4.55 FPS，需与
  static/dynamic 热态继续 A/B，判断约 3–4% 的可能吞吐损失；
- 当前上游块数公式使 7.995 秒音频只输出 7.467 秒，尾部缺 529 ms，未通过
  80 ms A/V 门槛。修复必须生成并裁剪最后一个块，不能用缩短音频掩盖。
- A3 连续性检查：149 帧可完整解码，20 FPS，最大时间戳间隔 50 ms，
  A/V 内部漂移 16.7 ms，无黑屏、冻结、长静音或熵坍塌；仅因总时长截断而失败。
  七帧 contact sheet 未见身份漂移或嘴部破坏。

### E03-A1/A2/A3：生产参考图预热与内容哈希 LRU

缓存键为参考图文件内容 SHA256、输出高宽和设备；缓存值只保存在当前模型进程内，
以 LRU 限制条目数。生产头像在既有两轮模型预热中完成 CLIP/VAE conditioning，
上传后的临时文件名不影响内容哈希命中，不同图片内容不会串用 latent。

| 指标 | 重启后首请求 | 热态 1 | 热态 2 |
| --- | ---: | ---: | ---: |
| HLS 首输出 | 13.869 s | 5.216 s | 5.032 s |
| 最终完成 | 48.819 s | 39.674 s | 39.677 s |
| reference image lookup | 2.5 ms | 4.1 ms | 4.3 ms |
| CLIP / `init_y` | 0 / 0 ms | 0 / 0 ms | 0 / 0 ms |
| T5 prompt | 0.2 ms | 0.3 ms | 0.3 ms |
| 第一块 | 21 帧 / 12.350 s | 21 帧 / 4.613 s | 21 帧 / 4.636 s |

结论：

- 三次均出现 reference cache hit。相对未预热的重启后首请求，`init_y` 从
  10.565 秒降到 0，HLS 首输出从 19.031 秒降到 13.869 秒，降幅 27.1%；
- 重启后的第一块仍有一次性去噪冷抖动（12.350 秒），随后两次第一块稳定在
  4.61–4.64 秒。热态 HLS 首输出中位数 5.124 秒，相对缓存前 5.983 秒再降 14.4%；
- 三份输出都是 160 帧、8.000 秒、A/V 漂移 4.625 ms，连续性检查全部通过；
  三个 MP4 的 SHA256 完全一致，说明缓存没有改变确定性输出或画质；
- 使用真实头像的总预热时间 94.057 秒，随机图基线为 92.443 秒，只把约
  1.6 秒增加到服务启动阶段。后续要继续消除首请求的去噪冷抖动，而不是重复优化
  已归零的 CLIP/VAE conditioning。

### E07-A1/A2：精确尾块与冷/热首播

修复使用同目录运行时副本，不修改 SoulX-LiveAct vendor 源码。目标帧数使用
`ceil(audio_seconds * 20)`，按需增加最后一个推理块，再把最后解码结果裁剪到剩余帧数。
同时关闭 LightVAE decode 的不稳定 `torch.compile`，并用生产提示词完成服务预热。

| 指标 | A1 重启后首请求 | A2 相同输入热态 |
| --- | ---: | ---: |
| HLS 首输出 | 19.031 s | 5.983 s |
| 最终完成 | 57.707 s | 40.638 s |
| T5 prompt | 0.0001 s | 0.1097 s |
| reference VAE `init_y` | 10.565 s | 0.603 s |
| 第一块 | 21 帧 / 7.541 s | 21 帧 / 4.556 s |
| 完整稳态块 | 2.91–4.66 FPS | 4.55–4.74 FPS |
| 最终块 | 11 帧（按剩余帧裁剪） | 同左 |
| 视频 / 音频时长 | 8.000 / 7.995 s | 同左 |
| A/V 漂移 | 4.625 ms | 4.625 ms |

结论：

- 尾帧修复把先前 149 帧补足为 160 帧，7.995 秒输入得到 8.000 秒视频；
  最大时间戳间隔 50 ms，无黑屏、冻结、长静音、解码错误或纹理熵坍塌；
- 人工检查每秒 contact sheet，人物身份、嘴部和画面结构稳定，未见修复引入的画质退化；
- 最终生成从 5 个块增加到 6 个块，因此完整视频多付出约一个块的计算时间；
  HLS 第一块不依赖最终块，热态首播仍保持约 6 秒；
- 生产提示词预热已命中，但重启后 reference latent 仍是冷的，`init_y` 从热态
  0.603 秒升至 10.565 秒。这解释了 A1 的 19 秒首播，并将 E03 提升为下一项首播优化；
- A1 第三块偶发 10.994 秒，A2 未复现；A2 所有完整稳态块为 6.75–7.04 秒。
  暂判为冷态一次性抖动，后续以至少三次热态样本监控。

### 环境约束：内存高水位

- AutoDL cgroup：`memory.high=86 GiB`、`memory.max=90 GiB`，不是宿主机
  `free` 显示的 754 GiB；
- 8 路 CUDA 源码构建与 LiveAct 同时运行时，`memory.current` 达 88.2 GiB，
  `memory.events.high` 超过 330 万次，18B 模型加载被拖慢到 8 分钟仍未完成；
- 终止构建树并保留源码/对象后，LiveAct 约 3 分 43 秒进入预热，预热
  96.39 秒，总冷启动约 5 分 17 秒；
- 结论：SageAttention 必须在 LiveAct 停机窗口以 1–2 路低并发构建，不能和
  block-offload 在线服务并行编译。

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
