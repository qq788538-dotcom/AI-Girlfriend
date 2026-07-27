# SoulX-LiveAct 单卡性能实验知识库（RTX 5090 / RTX PRO 6000）

## 目标与原则

目标是在单张消费级或工作站级 Blackwell GPU 上显著缩短“语音结束到浏览器首个可播放画面”
的时间，同时保持人物身份、画面连续性、口型同步和音质不变，或只接受可量化的轻微质量下降。
所有结论必须来自同一人物、同一音频、同一种子和同一输出规格的 A/B，不用不同输入之间的
主观印象代替证据；不同 GPU 的结果必须分别建立基线，不能横向混用。

## 官方口径

- SoulX-LiveAct 官方项目与论文的 20 FPS 实时口径使用两张 H100/H200，并结合端到端
  FP8、序列并行与通信计算并行；不能把这个数字当作单卡 PRO 6000 的验收线。
- 官方项目报告：18B 模型在单张 RTX 5090、FP8 KV cache 与 CPU block offload
  下约 6 FPS。
- 官方 5090 参数为 416×720、FP8 KV cache、block offload、T5 CPU 和
  `USE_CHANNELS_LAST_3D=1`。
- 官方安装流程要求 SageAttention 2.2.0；SageAttention 官方仓库报告 RTX 5090
  注意力内核微基准显著快于 FlashAttention 2，但端到端收益必须在 LiveAct 上单独验证。
- GUI 首批 block 需要预热，性能比较必须排除首次编译和模型冷启动。

来源：

- https://huggingface.co/xmuhtt/LiveAct
- https://github.com/Soul-AILab/SoulX-LiveAct
- https://arxiv.org/abs/2603.11746
- https://github.com/thu-ml/SageAttention
- https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html

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
| E01 | SDPA → SageAttention 2.2.0 | 提升长序列注意力与稳态吞吐 | 量化误差、额外显存、冷启动编译 | PRO 6000 端到端通过 |
| E02 | 缓存固定 T5 embedding | 首画面减少约 13.5 秒 | prompt key 失配 | 热态通过 |
| E03 | 内容哈希隔离的 reference CLIP/VAE latent 缓存 | 热态首画面减少约 0.8 秒 | GPU 常驻内存、跨人物污染 | 冷/热三次 E2E 通过 |
| E04 | renderer 全局单任务队列 | 消除 429 和无响应 | 排队延迟 | 单元测试通过，待并发 E2E |
| E05 | 持续会话与增量音频 | TTS 与动画重叠 | 上游 demo API 改造 | 待测 |
| E06 | 降分辨率/帧率或更低精度量化 | 像素吞吐提升 | 直接降低画质或运动质量 | 质量优先策略拒绝 |
| E07 | 关闭不稳定 VAE compile；精确生成并裁剪尾块 | 消除数分钟卡死并保留完整尾音 | 最终块仍有完整块计算成本 | 冷/热端到端通过 |
| E08 | 关闭 block offload，18B 去噪器常驻显存 | 消除逐层 CPU/GPU 搬运 | 仅适合大显存 GPU | PRO 6000 端到端通过 |
| E09 | 512×512 LightVAE static compile | 减少 VAE 解码时间 | 冷编译、编译数值差异 | 热态性能与质量门通过 |
| E10 | 持久 TorchInductor/FX/Mega-Cache | 缩短进程重启预热 | Sage graph break、额外磁盘 | 未改善，停止上线 |

## 实验记录

### E09-A/B：512×512 LightVAE static compile

固定输入为同一头像、同一种子和同一 5.000 秒 PCM 音频，规格锁定
512×512、20 FPS、100 帧。A 为仅关闭 LightVAE decode compile，B 为
`VH_LIVEACT_VAE_COMPILE_MODE=static`；去噪器、SageAttention2、FP8 GEMM/KV、
提示词和 reference cache 均保持不变。

| 指标 | compile off | static 热态 | static 增益 |
| --- | ---: | ---: | ---: |
| 首个流输出 | 2887.47 ms | 2657.16 ms | 7.98% |
| 渲染完成 | 14335.24 ms | 13642.39 ms | 4.83% |
| 输出 | 512×512 / 20 FPS / 100 帧 | 同左 | 无规格变化 |

质量门：

- 两份视频和音频均为 5.000 秒，A/V 时长漂移 0 ms；
- 全部 100 帧可解码，最大时间戳间隔 50 ms，无黑帧、冻结或持续纹理熵坍塌；
- decoded-frame PSNR 39.471 dB、SSIM 0.967721；逐帧双栏检查未见身份、
  口型、牙齿、手部或清晰度退化；
- 两份 AAC 解码后的 PCM SHA256 完全一致；
- 同一口部 ROI 的最佳时移均为 3 帧，口部运动均值和 p95 差异分别约
  1.4% 和 1.0%，没有出现 compile 导致的动作停滞。

结论：static compile 不改模型输入、采样、分辨率、帧率或输出帧数，热态总耗时缩短
约 4.8%，通过当前等质量门槛。第一次静态图编译仍是冷启动成本，不能把冷态
26 秒样本计入热态吞吐。

### E10-A1/A2：持久 TorchInductor/FX graph cache（未改善）

PyTorch 官方说明 FX graph cache 可以跨进程复用相同图、相同形状和相同配置的编译产物；
`TORCHINDUCTOR_CACHE_DIR` 控制持久目录，未单独设置时 Triton cache 也放在其子目录。
部署把 LiveAct cache 固定到 `/root/.cache/torchinductor-liveact`，显式启用
`TORCHINDUCTOR_FX_GRAPH_CACHE=1`，不占只剩约 13 GB 的 AutoDL 数据盘。

- 从临时目录迁入后缓存约 745 MB；
- 第一次在新目录完整预热 255 秒，第二次相同配置重启为 285 秒，未显示缓存收益；
- 重启脚本现在自行加载并 export `deploy/autodl/runtime.env`，避免人工执行时静默退回
  SDPA、block offload 或错误尺寸；
- 进一步生成了 99 MB `torch.compiler.save_cache_artifacts()` 工件并完成加载冒烟，
  但启动过程仍进入同一组 Sage 自定义算子 graph break 和 Dynamo 编译路径；
- 按“只有第二次进程重启实际变快才算命中”的规则，Mega-Cache 不进入默认部署。
  普通 cache 目录仍用于集中管理静态 VAE/Inductor 临时产物，不宣称它缩短冷启动。

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

### E01-K1：SageAttention 2.2.0 SM120 内核 A/B

AutoDL cgroup 只有 90 GiB 上限，必须先停止 LiveAct，再用
`EXT_PARALLEL=1 MAX_JOBS=1 NVCC_APPEND_FLAGS=--threads=2` 构建。官方源码
`TORCH_CUDA_ARCH_LIST=12.0` 会为 `_qattn_sm80`、`_qattn_sm89` 和 `_fused`
算法族生成 `sm_120` 代码；扩展名是算法族，不代表编错 GPU 架构。

- 低并发构建用时约 11 分钟，内存约 13–16 GiB，`memory.events.high` 无增量；
- wheel：
  `sageattention-2.2.0-cp312-cp312-linux_x86_64.whl`；
- wheel SHA256：
  `eafb9dd39ce24908030c65c59ac5a631f197f787116e4e98784f0542a2204b56`；
- 先安装到隔离 `--target`，RTX 5090 冒烟输出形状正确且全部为有限值，随后才安装
  到生产 LiveAct venv；
- `VH_LIVEACT_FORCE_SDPA=1` 会同时屏蔽 FlashAttention 与 SageAttention，
  是安装 wheel 后的完整回滚闸门。

相同随机种子、BF16、batch 1、40 heads、head dimension 128，3 次预热 + 5 次计时：

| LiveAct 形状 | SDPA 中位数 | Sage 中位数 | 内核加速 | cosine | 相对 L2 | Sage 额外峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first 7020×7020 | 4.661 ms | 1.871 ms | 2.49× | 0.999247 | 3.88% | 239 MiB |
| steady 9360×16380 | 15.080 ms | 5.546 ms | 2.72× | 0.999228 | 3.93% | 456 MiB |
| image cross 9360×257 | 0.369 ms | 0.282 ms | 1.31× | 0.999337 | 3.64% | 142 MiB |
| text cross 9360×512 | 0.546 ms | 0.341 ms | 1.60× | 0.999303 | 3.73% | 146 MiB |

所有候选输出 `nan_count=0`、`inf_count=0`，SNR 为 28.1–28.8 dB。内核结果只证明
速度与数值范围可接受；约 3.6–3.9% 相对 L2 是量化注意力的真实差异，仍必须通过
同输入 MP4、A/V、黑屏/冻结和人工画面检查后才能保留。

### E01-K2：RTX PRO 6000 SageAttention 微基准与完整视频

新实例为 NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB，驱动
590.44.01，PyTorch 2.8.0+cu128，计算能力 12.0。复用同一份已校验 SM120 wheel，
但重新建立该硬件的 A/B；不能套用 RTX 5090 的 2.5–2.7× 内核数字。

| LiveAct 形状 | SDPA 中位数 | Sage 中位数 | 内核加速 | cosine | 相对 L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| first 7020×7020 | 3.100 ms | 2.003 ms | 1.55× | 0.999254 | 3.86% |
| steady 9360×16380 | 9.069 ms | 5.661 ms | 1.60× | 0.999233 | 3.92% |
| image cross 9360×257 | 0.281 ms | 0.291 ms | 0.97× | 0.999334 | 3.65% |
| text cross 9360×512 | 0.400 ms | 0.363 ms | 1.10× | 0.999312 | 3.71% |

所有形状仍为 3 次预热 + 5 次计时，输出无 NaN/Inf。PRO 6000 的短 cross
attention 已接近调度开销，Sage 不能全部加速，因此端到端收益明显小于主 self-attention
微基准。

在 `VH_LIVEACT_BLOCK_OFFLOAD=0`、固定种子、同一 7.995375 秒音频和同一头像下：

| 指标 | 常驻 SDPA 热态中位数 | 常驻 Sage 热态中位数 | Sage 增益 |
| --- | ---: | ---: | ---: |
| HLS 首输出 | 3.675 s | 3.462 s | 5.8% |
| 最终 8 秒视频 | 28.370 s | 26.189 s | 7.7% |
| 第一块 | 21 帧 / 3.062 s | 21 帧 / 2.913 s | 5.1% |
| 完整稳态块 | 32 帧 / 4.94–5.02 s | 32 帧 / 4.44–4.60 s | 约 8% |
| 模型预热 | 81.867 s | 104.098 s | 慢 27.2% |

质量与稳定性：

- Sage 三次 MP4 SHA256 完全一致，重复运行具有确定性；
- 160 帧、20 FPS，视频 8.000 秒、音频 7.995375 秒，A/V 漂移 4.625 ms；
- 最大帧时间戳间隔 50 ms，无黑屏、冻结、静音、解码错误或持续熵坍塌；
- 相对常驻 SDPA 为 SSIM 0.9580、PSNR 37.08 dB；每秒双栏 contact sheet
  未见身份漂移、嘴部破坏或明显清晰度下降；
- 空闲显存约 50.3 GiB，cgroup 无 `high`、OOM 或 OOM kill 事件。

结论：Sage 是这台 PRO 6000 可接受的“轻微数值差异换取约 8% 热态吞吐”的候选，
但会增加约 22 秒冷预热，而且官方 wheel 的 Torch Dynamo 算子仍产生 graph break。
精度优先时设置 `VH_LIVEACT_FORCE_SDPA=1` 即完整回滚。

### E08-A/B：RTX PRO 6000 关闭 block offload

这项只改变权重驻留位置，不改变分辨率、帧率、采样步数、随机种子或精度。安全默认值仍为
`VH_LIVEACT_BLOCK_OFFLOAD=1`，避免 32 GB 卡 OOM；96 GB PRO 6000 候选使用
`VH_LIVEACT_BLOCK_OFFLOAD=0`。

| 指标 | block offload 开 | block offload 关 | 常驻增益 |
| --- | ---: | ---: | ---: |
| 模型预热 | 128.595 s | 81.867 s | 36.3% |
| 热态 HLS 首输出 | 4.660 s | 3.675 s | 21.1% |
| 热态最终 8 秒视频 | 32.879 s | 28.370 s | 13.7% |
| 热态第一块 | 约 5.04 FPS | 约 6.86 FPS | 36.1% |
| 热态完整稳态块 | 约 5.6 FPS | 约 6.4 FPS | 约 14% |
| 空闲显存 | 33.2 GiB | 49.6–50.3 GiB | +16.4–17.1 GiB |

三次常驻 SDPA 输出 SHA256 完全一致。与 offload 基线比较为 SSIM 0.9619、
PSNR 38.05 dB；差异来自执行/舍入路径而非随机漂移。三次连续性门槛全部通过：
160 帧、20 FPS、A/V 漂移 4.625 ms、最大时间戳间隔 50 ms，无黑屏、冻结、静音或
解码错误。GPU 尚余约 47 GiB，系统 cgroup 无内存压力事件。

结论：对 96 GB 单卡，关闭 block offload 是当前收益最大且质量风险最低的优化；对
32 GB 卡必须保留默认 offload。组合“常驻 + Sage”相对初始 offload + SDPA 热态基线：

- HLS 首输出从 4.660 秒降到 3.462 秒，缩短 25.7%；
- 完整 8 秒视频从 32.879 秒降到 26.189 秒，缩短 20.3%；
- 第一块约 7.2 FPS、稳态块约 7.0 FPS。

这仍未达到官方两张 H100/H200 + 序列并行的 20 FPS 口径。当前单卡上游
full-audio adapter 必须等 TTS 完成后才开始渲染，真正的大幅首播优化需要 E05
增量音频/持久会话，而不是继续压榨已经很短的 prompt/reference 初始化。

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

### 生产全链路与浏览器验收：PRO 6000 常驻 + Sage

输入为固定 24 kHz 单声道 PCM16 中文测试语音，走真实
`ws://127.0.0.1:8765/v1/realtime`，记忆显式关闭以避免自动实验污染用户记忆。
链路包含 Qwen3 ASR、本地 Qwen3.6 LLM、Higgs Audio v3 TTS、LiveAct renderer
WebSocket、HLS 和最终 MP4。

第二次热态端到端：

| 时间点（相对开始） | 时间 |
| --- | ---: |
| ASR 完整转写 | 0.675 s |
| LLM 文本完成 | 1.159 s |
| 首段 TTS 音频 | 2.737 s |
| TTS 回答完成 | 4.810 s |
| `avatar.stream.ready` / HLS 可播 | 8.403 s |
| 最终 MP4 与 `avatar.render.done` | 21.883 s |

ASR 转写与测试原文完全一致。HLS 清单通过本机 SSH 媒体隧道返回，目标时长 1 秒并包含
独立分片；最终 MP4 可由 Chromium 原生解码并自动播放。输出为 416×720、20 FPS、
107 帧，视频 5.350 秒、音频 5.340 秒，A/V 漂移 10 ms；无黑屏、冻结、静音、
时间戳缺口或解码错误。

浏览器“大脸裁剪”来自视频继承了封面图的 `object-fit: cover`。修复后视频的计算样式为
`object-fit: contain; object-position: center`，在 742×720 播放容器内完整显示
416×720 竖幅，左右使用深色留白；静态待机图仍可保留封面构图。页面重新连接后状态为
“会话已就绪 / 云端已连”。

本地必须同时保留：

- WebSocket/媒体主隧道：本机 8771 → 云端 `127.0.0.1:8772`；
- 若 renderer 返回 8772 public base URL，还需同端口媒体隧道；当前运行配置已返回
  `http://127.0.0.1:8771/...`，基准脚本会记录实际 HLS/MP4 URL，避免靠猜测排障。

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

完整 ASR→LLM→TTS→LiveAct 网关基准（默认不写真实记忆）：

```bash
python scripts/benchmark-gateway.py \
  --url ws://127.0.0.1:8765/v1/realtime \
  --audio runtime/benchmarks/liveact-ab-input.wav \
  --output-audio runtime/benchmarks/liveact-e2e.wav \
  --output-media runtime/benchmarks/liveact-e2e.mp4 \
  --timeout 300
```

结果会分别记录 `avatar_stream_ready_ms`、`avatar_video_ready_ms` 和最终
`render_done_ms`，并保存 renderer 实际返回的 HLS/MP4 URL。只有显式添加
`--with-memory` 才允许实验读写真实记忆。

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
