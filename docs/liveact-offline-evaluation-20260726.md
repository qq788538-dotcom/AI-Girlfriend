# SoulX-LiveAct 离线部署与验收记录（2026-07-26）

## 验收目标

最终参考为 Soul-AILab/SoulX-LiveAct 官方 GitHub `FaceTime / 1_h265.mp4`。LiveAct 只承担
高质量离线成片，不替换当前 FlashHead 实时链路。所有 A/B 均使用已锁定的虚拟女友人物图和
Higgs TTS 音频。

## 官方参考基线

| 项目 | 结果 |
| --- | --- |
| 来源 | `https://github.com/user-attachments/assets/143bb565-078a-48ba-8daa-f2fb56616189` |
| SHA256 | `b0adcfd42ffc129873ae6d212939bf6fcdbf36c52b259dbf8b0089b73bc2f0c5` |
| 规格 | HEVC，480×832，24 FPS，90.000 秒视频 |
| 人脸检测 | 前 15 秒 375/375 帧，覆盖率 100% |
| SyncNet | 偏移 0 ms，置信度 6.609 |
| 连续性 | 无黑屏、冻结、时间戳缺口或纹理熵坍塌 |

## 首次 FP8 KV 失败

输入音频为 `live-v3-amused-final.wav`，4.053 秒；输出 512×512、24 FPS、100 帧。

- 第 0–52 帧仍为人物；
- 第 53 帧正好是第三个自回归生成块起点，也是 ConvKV 第一次执行 `update_cache=True`；
- 第 53 帧起画面快速转为彩色噪点；
- 最大相邻帧亮度差 26.2097；
- 连续 8 帧归一化亮度熵相对首秒基线下降 0.114173；
- 结论：当前 RTX 4090 D + PyTorch 2.7.1 + SDPA 路径不能把 FP8 KV 作为离线质量默认值。

## BF16 KV 修复结果

固定变量：

```text
LIVEACT_FP8_KV_CACHE=0
LIVEACT_OFFLOAD_CACHE=1
LIVEACT_BLOCK_OFFLOAD=1
LIVEACT_T5_CPU=1
LIVEACT_AUDIO_CFG=1.0
LIVEACT_SIZE=512*512
LIVEACT_FPS=24
LIVEACT_SEED=42
```

结果：

| 指标 | BF16 原始输出 | 80 ms 音频修正后 |
| --- | ---: | ---: |
| 视频帧 | 100/100 | 100/100 |
| 视觉崩坏 | 无 | 无 |
| 最大相邻帧亮度差 | 4.67094 | 4.67094 |
| 最大连续纹理熵下降 | 0.006742 | 0.006742 |
| 人脸检测覆盖率 | 100% | 100% |
| SyncNet 偏移 | -80 ms | 0 ms |
| SyncNet 置信度 | 5.580 | 5.538（含保守锐化） |
| 音画时长漂移 | 113.667 ms | 0 ms（云端 FFmpeg 4.4 验证） |
| 自动连续性门槛 | 原始仅时长失败 | 全部通过 |

最终封装默认用 `unsharp=5:5:0.35` 和 H.264 CRF 16 做保守锐化，输出仍保持 100/100 帧。
FFmpeg `blurdetect` 均值从 8.073 降至 7.423；官方 FaceTime 为 5.914。更强的 0.55 虽可把指标
降到约 7.204，但收益小且更容易强化嘴唇和牙齿伪影，因此不作为默认值。

BF16 证明第三分块崩坏来自 FP8 KV/ConvKV 更新组合。离线目标不要求实时，因此默认选择
BF16 KV + CPU cache；这比继续接受 FP8 的完整画面失败更符合验收目标。

## `audio_cfg=1.7` 资源边界

官方单卡 Eval 示例使用 `audio_cfg=1.7`，但该值大于 1.0 时会为无音频 CFG 分支建立第二套 KV。
在 BF16 + CPU cache 配置下，进程常驻内存达到约 115 GiB，随后被当前容器的 120 GiB
`memory.max` 杀掉（退出码 137）。宿主机显示 1 TiB 内存不代表容器可以使用全部内存。

因此当前 48 GB 单卡实例的稳定默认值保持 `audio_cfg=1.0`。在没有扩大容器内存上限或实现更省
内存的 CFG cache 前，不把 `audio_cfg>1.0` 暴露为生产离线选项。

## 可重复命令

云端推理：

```bash
LIVEACT_ROOT=/root/liveact-candidate/repo \
LIVEACT_PYTHON=/root/liveact-candidate/venv/bin/python \
LIVEACT_CKPT_DIR=/root/liveact-candidate/checkpoints/liveact \
LIVEACT_WAV2VEC_DIR=/root/liveact-candidate/checkpoints/chinese-wav2vec2-base \
LIVEACT_INPUT_JSON=/root/liveact-candidate/liveact-locked-ab.json \
LIVEACT_OUTPUT_DIR=/root/liveact-candidate/outputs \
./scripts/run-liveact-offline.sh
```

一键脚本先校验锁定人物文件名和 SHA256，再运行候选推理，最后自动完成音画修正和连续性验收。
若已有原始 MP4，也可单独执行：

```bash
./scripts/finalize-liveact-offline.sh RAW.mp4 FINAL.mp4
```

口型验收：

```bash
PYTHONPATH=runtime/face-motion/python /opt/homebrew/bin/python3.14 \
  scripts/prepare-syncnet-face.py FINAL.mp4 FINAL-syncface.avi \
  --cascade runtime/face-motion/models/haarcascade_frontalface_default.xml

PYTHONPATH=runtime/syncnet/python runtime/audio-separator/.venv/bin/python \
  scripts/benchmark-syncnet.py FINAL-syncface.avi \
  --max-abs-offset-frames 2 --min-confidence 3.0
```

## 尚未达到官方样例的部分

- 已完成 11.292 秒、9 个生成块、271 帧的 BF16 长样片；仍未等同于官方 90 秒长时验证；
- 11 秒参考音频样片 SyncNet 偏移 0 ms、置信度 3.827，低于官方前 15 秒的 6.609；4 秒当前
  Higgs TTS 样片为 0 ms、5.538，说明长样片低分也受到参考音频内容和静音段影响；
- 11 秒样片已经出现自然眨眼、笑、明显侧转、回头、手部和手机动作，运动帧差均值从短样片
  1.800 提升到 3.506，接近官方的 5.161，但仍未达到官方的动作强度；
- 11 秒样片模糊度 7.749，优于当前 FlashHead 的 12.828，但仍弱于官方 FaceTime 的 5.914；
- Haar 人脸检测覆盖率 95.7%。缺失集中在 6.54–7.00 秒的完整侧脸和 10.96–11.04 秒的手机动作，
  逐帧检查未发现身份崩坏；不能把合法侧脸简单判为坏帧；
- 当前单张 4090 D 采用 block/CPU cache offload，离线速度慢，但不影响离线部署成功判定。

## 云端最终状态

- 未使用的常驻 FlashHead worker 已正常 `SIGTERM` 停止；
- 8770/5001 均无监听进程，GPU 进程列表为空；
- LiveAct 权重、锁定人物、锁定语音和验收样片均保留在磁盘，不做删除；
- 下一次离线请求由 `run-liveact-offline.sh` 按需加载 BF16 LiveAct，任务完成后自动释放显存。
