# Ditto TalkingHead 隔离候选评估（2026-07-26）

## 约束

- 未替换生产渲染器。
- 人物图固定为 `public/avatar-ai-girlfriend-v6.png`。
- 人物 SHA-256：
  `c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162`。
- 三条候选使用同一段锁定女声：
  `runtime/benchmarks/live-v3-amused-final.wav`。
- 音频 SHA-256：
  `8d559314c5ae02ce3efd461bc3117bec04748369c1c3bb94104b872bf08e41b2`。
- 云端 Ditto 源码固定到 commit：
  `c3e47eee2e626500017a0556b470d6d4182f85e8`。
- 候选环境、模型与输出均位于 `/root/ditto-candidate`，未停止常驻
  FlashHead worker。

## 样本

| 样本 | 情绪/控制 |
|---|---|
| `ditto-neutral-amused.mp4` | Neutral，官方默认眼睛与动作 |
| `ditto-happy-amused.mp4` | Happy，官方默认眼睛与动作 |
| `ditto-happy-blink-exp120-amused.mp4` | Happy，45 帧眨眼间隔，表情增益 1.2 |
| `live-v3-amused-isolated-final.mp4` | 当前 FlashHead Lite 基线 |

## 量化结果

| 指标 | Ditto Neutral | Ditto Happy | Ditto Happy 增强 | FlashHead 基线 |
|---|---:|---:|---:|---:|
| 分辨率 | 1254×1254 | 1254×1254 | 1254×1254 | 512×512 |
| 音视频时长差 | 27 ms | 27 ms | 27 ms | 0.067 ms |
| SyncNet 偏移 | 0 ms | +40 ms | +40 ms | -40 ms |
| SyncNet 置信度 | 6.985 | 5.569 | 5.703 | 5.972 |
| 合理眨眼次数 | 0 | 0 | 1 | 2 |
| 眨眼频率 | 0/min | 0/min | 14.706/min | 29.670/min |
| 头部角速度 P95 | 14.236°/s | 10.030°/s | 11.872°/s | 40.470°/s |
| 非嘴部活动 RMS | 0.02510 | 0.02138 | 0.04277 | 0.06885 |
| 微笑均值 | 0.0004 | 0.0061 | 0.0174 | 0.0587 |
| 微笑 P95 | 0.0019 | 0.0397 | 0.0865 | 0.3027 |
| FFmpeg blur mean | 6.630 | 6.548 | 6.569 | 11.462 |
| ≥0.5 秒低变化段 | 2 | 3 | 4 | 0 |

说明：

- `blur mean` 越低代表越清晰；Ditto 明显优于当前 512×512 FlashHead。
- Ditto Neutral 的口型同步是四条中最优的。
- 官方默认 `delta_eye_open_n=-1` 会关闭生成式眨眼；短视频默认没有眨眼。
- 增强参数恢复了自然频率的眨眼，并提高了上半脸活动和微笑幅度，但仍明显
  低于当前 FlashHead 的开心表现，而且低变化段没有消失。
- FlashHead 的表情和头部动作更丰富，但 4 秒样本的眨眼与头部角速度偏高，
  有“动作太急、笑得像模板”的风险。

## 决策

Ditto 当前不进入生产：

1. 它可以提升清晰度和口型同步。
2. 官方 Happy 条件的语义表情增益太弱。
3. 增强后虽然眨眼正常，但仍不能达到原作者参考中的眼神、上半脸和自然笑意。
4. 仍存在多段可见运动不足，不能满足“持续真人感”的要求。

生产继续保持锁定人物与当前 FlashHead 配置。Ditto 仅保留为高分辨率、
可控眨眼和精准口型的候选技术，不自动切换。

## 下一步

- 不再用全局动作增益修复表情，否则会放大 FlashHead 已有的头部急动。
- 情绪层需要按语义分别控制笑意、眼睑、视线和头部姿态，而不是只放大口型运动。
- 下一轮采用同人物、同音频、同测量方法，优先验证可持续流式且有显式
  expression/eye/head 控制的方案；任何生产切换仍需先通过身份、连续性、
  SyncNet、MediaPipe 和真人盲听/盲看。
