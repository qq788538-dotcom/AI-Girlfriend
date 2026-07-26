# 仙宫云全本地部署分支

这个目录只属于 `codex/xiangongyun-deploy`。成熟本地版固定在 `main`，不要把本目录合并回
`main`，除非明确决定统一部署架构。

## 进程布局

- 公网 80：`virtual-human` 网页和 WebSocket 网关
- 本机 8000：Qwen3.6-35B-A3B-AWQ LLM（vLLM）
- 本机 8001：Qwen3-ASR-0.6B
- 本机 8002：Qwen3-Embedding-0.6B（CPU）
- 本机 8010：Higgs TTS 3 4B（vLLM-Omni）
- 本机 8770：FlashHead Lite
- 本机 1934：虚拟女友专用 OpenViking

所有推理服务只监听 `127.0.0.1`。`VH_OFFLINE_RUNTIME=true` 会拒绝任何非回环的 LLM、
ASR、TTS、Embedding、OpenViking 或人物渲染地址。权重下载完成后，启动脚本固定设置
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`。

## 首次部署

```bash
git clone --recurse-submodules \
  https://github.com/qq788538-dotcom/AI-Girlfriend.git \
  /root/AI-Girlfriend
cd /root/AI-Girlfriend
git switch codex/xiangongyun-deploy

cp deploy/xiangongyun/env.example deploy/xiangongyun/runtime.env
cp deploy/xiangongyun/secrets.example deploy/xiangongyun/secrets.env
chmod 600 deploy/xiangongyun/runtime.env deploy/xiangongyun/secrets.env

# 通过 SSH 单独上传，不经过公开 GitHub：
# runtime/voice-calibration/reference-female-only-complete-11s.wav
# bootstrap 会校验固定 SHA256，错误文件会被拒绝。

deploy/xiangongyun/bootstrap.sh
deploy/xiangongyun/control.sh start
deploy/xiangongyun/control.sh status
```

新机器上建议先把 `runtime.env` 的 `VH_PORT` 暂时改为未占用测试端口，然后执行：

```bash
deploy/xiangongyun/control.sh models-start
deploy/xiangongyun/control.sh app-start
```

内网接口、显存和端到端测试全部通过后，再停止旧 80 服务，将 `VH_PORT=80` 并重新执行
`app-start`。这样模型安装和预热不会中断现有公网页面。

OpenViking 固定为 `0.4.11`，独立虚拟环境位于 `.venv-openviking`；其配置和记忆数据位于
`runtime/openviking-girlfriend`，不会与 Mac 或 Claude 的 OpenViking 库混用。它的 VLM
调用 8000 本地 LLM，向量调用 8002 CPU Embedding，不需要 Ark、OpenAI 或其他云端 Key。

## 更新

```bash
deploy/xiangongyun/control.sh stop
git pull --ff-only
git submodule update --init --recursive
deploy/xiangongyun/control.sh start
```

`runtime.env`、`secrets.env`、模型、Token、日志和 OpenViking 数据均被 Git 排除。锁定的
Higgs 参考音频同样不进入公开仓库，只允许通过 SSH 上传到运行目录。
