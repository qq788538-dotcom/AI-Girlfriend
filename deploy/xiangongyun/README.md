# 仙宫云部署分支

这个目录只属于 `codex/xiangongyun-deploy`。成熟本地版固定在 `main`，不要把本目录合并回
`main`，除非明确决定统一部署架构。

## 进程布局

- 公网 80：`virtual-human` 网页和 WebSocket 网关
- 本机 8770：FlashHead Lite，只有网关可访问
- 本机 1934：虚拟女友专用 OpenViking，只有网关可访问
- Mac：oMLX、Qwen ASR/Qwen LLM 与锁定的 Higgs TTS；仙宫云不安装 oMLX

仙宫云端不能把 oMLX 地址写成 `127.0.0.1`。`runtime.env` 中应填写一个用户明确批准的
Mac 私有 HTTPS 入口。部署脚本不会自动创建第三方隧道，也不会把 Mac 的 oMLX 模型上传到云端。

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

OpenViking 固定为 `0.4.11`，独立虚拟环境位于 `.venv-openviking`；其配置和记忆数据位于
`runtime/openviking-girlfriend`，不会与 Mac 或 Claude 的 OpenViking 库混用。

## 更新

```bash
deploy/xiangongyun/control.sh stop
git pull --ff-only
git submodule update --init --recursive
deploy/xiangongyun/control.sh start
```

`runtime.env`、`secrets.env`、模型、Token、日志和 OpenViking 数据均被 Git 排除。
锁定的 Higgs 参考音频同样不进入公开仓库，只允许通过 SSH 上传到运行目录。
