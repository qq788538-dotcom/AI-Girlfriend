# Silero VAD 项目适配评估

结论：适合替换网页自动聆听路径中的纯 RMS 起止判断，但不应改变按住说话路径，也不能替代
ASR、TTS、动画队列或前端状态机修复。

当前 `public/app.js` 每 40ms 计算一次 RMS，通过自适应噪声底和 800ms 静音判断一轮输入结束。
它对稳定安静环境够快，但难以区分人声、背景音乐、风扇和键盘声。Silero VAD v6.2 是流式
VAD，支持 8kHz/16kHz、ONNX 与 6000 多种语言；官方公布的单线程 ONNX 推理远快于实时要求，
M5 Max 和仙宫云 CPU 均无需占用 GPU。

推荐分阶段接入：

1. 中间按住说话继续原样发送，避免短词和句首被 VAD 吞掉。
2. 左侧自动聆听使用 16kHz Silero ONNX，保留约 300ms 前滚音频。
3. 连续静音 650–800ms 才提交；短停顿继续积累同一轮。
4. RMS 保留为快速噪声门和模型加载失败时的回退。
5. 用安静、背景音乐、键盘声、短词和长句停顿五组录音做 A/B，再决定是否进入主分支。

官方资料：

- https://github.com/snakers4/silero-vad
- https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models
- https://github.com/snakers4/silero-vad/wiki/Performance-Metrics
