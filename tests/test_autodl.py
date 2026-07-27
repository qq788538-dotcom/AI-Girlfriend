from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTODL = ROOT / "deploy" / "autodl"


def test_autodl_uses_public_port_6006_and_keeps_models_loopback_only() -> None:
    env = (AUTODL / "env.example").read_text(encoding="utf-8")

    assert "VH_HOST=0.0.0.0" in env
    assert "VH_PORT=6006" in env
    for port in (8000, 8001, 8010, 1934, 8772, 8002):
        assert f"127.0.0.1:{port}" in env


def test_autodl_control_runs_the_pro6000_cloud_stack() -> None:
    bootstrap = (AUTODL / "bootstrap.sh").read_text(encoding="utf-8")
    control = (AUTODL / "control.sh").read_text(encoding="utf-8")
    autostart = (AUTODL / "autostart.sh").read_text(encoding="utf-8")
    env = (AUTODL / "env.example").read_text(encoding="utf-8")

    assert "deploy/xiangongyun/control.sh" not in control
    assert 'VH_CHAT_BACKEND:-}" != "omlx"' in control
    assert "VH_MEMORY_LOCAL_ONLY" in control
    assert "VLLM_USE_FLASHINFER_SAMPLER" in control
    assert "allow_metadata_override: true" in (
        ROOT / "scripts" / "run-openviking-memory.sh"
    ).read_text(encoding="utf-8")
    assert "liveact-control.sh" in control
    assert "Qwen3-4B-AWQ" in control
    assert "run-xgc-asr.sh" in control
    assert "run-xgc-tts.sh" in control
    assert "run-openviking-memory.sh" in control
    assert "VH_BOOTSTRAP_LLM=false" in bootstrap
    assert "VH_BOOTSTRAP_MEMORY_LLM=true" in bootstrap
    llm_runner = (ROOT / "scripts" / "run-xgc-llm.sh").read_text(encoding="utf-8")
    assert 'single = root / "model.safetensors"' in llm_runner
    assert 'index = root / "model.safetensors.index.json"' in llm_runner
    assert "VH_OMLX_CHAT_MODEL=Qwen3-4B-AWQ" in env
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in env
    assert "VH_AVATAR_BACKEND=liveact-official" in env
    assert "VH_AVATAR_MEDIA_BASE_URL=http://127.0.0.1:8772" in env
    assert "VH_LIVEACT_BLOCK_OFFLOAD=0" in env
    assert "VH_TTS_STAGE_OVERRIDES=" in env
    assert "VH_RENDERER_PUBLIC_BASE_URL=/avatar-media" in (
        AUTODL / "liveact-control.sh"
    ).read_text(encoding="utf-8")
    assert '@app.get("/avatar-media/{media_path:path}")' in (
        ROOT / "src" / "virtual_human" / "app.py"
    ).read_text(encoding="utf-8")
    assert "flock -n 9" in autostart
    assert "deploy/autodl/control.sh status" in autostart
    assert "deploy/autodl/control.sh start" in autostart
    assert "libsox-dev" in bootstrap
    assert "\n    sox" in bootstrap


def test_liveact_defaults_to_native_flash_attention_with_sdpa_fallback() -> None:
    control = (AUTODL / "liveact-control.sh").read_text(encoding="utf-8")
    launcher = (AUTODL / "liveact_demo_launcher.py").read_text(encoding="utf-8")

    assert 'VH_LIVEACT_FORCE_SDPA="${VH_LIVEACT_FORCE_SDPA:-1}"' in control
    assert 'VH_LIVEACT_CACHE_T5="${VH_LIVEACT_CACHE_T5:-1}"' in control
    assert 'VH_LIVEACT_CACHE_REFERENCE="${VH_LIVEACT_CACHE_REFERENCE:-1}"' in control
    assert "VH_LIVEACT_WARMUP_REFERENCE=" in control
    assert 'VH_LIVEACT_VAE_COMPILE_MODE="${VH_LIVEACT_VAE_COMPILE_MODE:-static}"' in control
    assert 'TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_DIR"' in control
    assert "TORCHINDUCTOR_FX_GRAPH_CACHE=1" in control
    assert 'RUNTIME_ENV="$SCRIPT_DIR/runtime.env"' in control
    assert '. "$RUNTIME_ENV"' in control
    assert 'VH_LIVEACT_FIX_TAIL_FRAMES="${VH_LIVEACT_FIX_TAIL_FRAMES:-1}"' in control
    assert 'case "${VH_LIVEACT_BLOCK_OFFLOAD:-1}" in' in control
    assert "liveact_memory_args+=(--block_offload)" in control
    assert '"${liveact_memory_args[@]}"' in control
    assert 'VH_LIVEACT_WARMUP_PROMPT="$LIVEACT_PROMPT"' in control
    assert 'VH_LIVEACT_PROMPT="$LIVEACT_PROMPT"' in control
    assert "_install_t5_cache()" in launcher
    assert "_patch_reference_cache(source)" in launcher
    assert "_install_vae_compile_policy()" in launcher
    assert "_prepare_demo_path(demo_path)" in launcher
    assert 'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in control
    assert '--size "${VH_LIVEACT_SIZE:-512*512}"' in control
    assert 'sys.modules["flash_attn"] = None' in launcher
    assert 'sys.modules["sageattention"] = None' in launcher
    assert "scaled_dot_product_attention" in launcher
    assert "attention_module.flash_attention = sdpa_attention" in launcher
    assert '"wan.modules.clip", "wan.modules.model"' in launcher
    assert "loaded_module.flash_attention = sdpa_attention" in launcher
    assert 'SELF="$SCRIPT_DIR/$(basename "$0")"' in control
    assert 'nohup "$SELF" wait-wrapper' in control
