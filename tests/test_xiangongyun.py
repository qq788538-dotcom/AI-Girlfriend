from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from virtual_human.xiangongyun import (
    FlashHeadDeployment,
    XiangongyunClient,
    XiangongyunError,
    sanitize_instance,
)


def test_client_authenticates_and_redacts_instance_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "test-token"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "success": True,
                "data": {
                    "list": [
                        {
                            "id": "gpu-1",
                            "status": "running",
                            "gpu_model": "NVIDIA GeForce RTX 4090 D",
                            "price_per_hour": 1.59,
                            "ssh_domain": "example.invalid",
                            "ssh_port": "2222",
                            "ssh_user": "root",
                            "password": "must-not-leak",
                            "ssh_key": "must-not-leak",
                            "jupyter_token": "must-not-leak",
                            "xgcos_token": "must-not-leak",
                        }
                    ]
                },
            },
        )

    with XiangongyunClient("test-token", transport=httpx.MockTransport(handler)) as client:
        instance = client.instances()[0]

    assert instance["id"] == "gpu-1"
    assert instance["price_per_hour"] == 1.59
    assert "password" not in instance
    assert "ssh_key" not in instance
    assert "jupyter_token" not in instance
    assert "xgcos_token" not in instance


def test_flashhead_deployment_uses_documented_low_cost_defaults() -> None:
    payload = FlashHeadDeployment().as_payload()

    assert payload["gpu_model"] == "NVIDIA GeForce RTX 4090 D"
    assert payload["gpu_count"] == 1
    assert payload["data_center_id"] == 1
    assert payload["image_type"] == "public"
    assert payload["storage"] is False
    assert payload["system_disk_expand"] is False


def test_api_error_does_not_include_token() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": 1000, "msg": "invalid access token", "success": False},
        )

    with XiangongyunClient("super-secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(XiangongyunError, match="API error 1000") as captured:
            client.balance()

    assert "super-secret" not in str(captured.value)


def test_sanitize_instance_ignores_unknown_sensitive_fields() -> None:
    sanitized = sanitize_instance(
        {
            "id": "gpu-2",
            "name": "renderer",
            "password": "secret",
            "future_private_token": "secret",
        }
    )

    assert sanitized == {"id": "gpu-2", "name": "renderer"}


def test_qwen36_launcher_keeps_mamba_batch_capacity_above_alignment_block() -> None:
    project_root = Path(__file__).resolve().parents[1]
    launcher = (project_root / "scripts" / "run-xgc-llm.sh").read_text()
    environment = (
        project_root / "deploy" / "xiangongyun" / "env.example"
    ).read_text()

    assert 'VH_LLM_MAX_NUM_BATCHED_TOKENS:-4096' in launcher
    assert '--max-num-batched-tokens "$max_num_batched_tokens"' in launcher
    assert "VH_LLM_MAX_NUM_BATCHED_TOKENS=4096" in environment
    assert 'VH_LLM_MAX_MODEL_LEN:-8192' in launcher
    assert "VH_LLM_MAX_MODEL_LEN=8192" in environment
    assert "--enable-auto-tool-choice" in launcher
    assert '--tool-call-parser "$tool_call_parser"' in launcher
    assert "VH_LLM_TOOL_CALL_PARSER=qwen3_coder" in environment
    assert 'VH_LLM_ATTENTION_BACKEND:-FLASH_ATTN' in launcher
    assert '--attention-backend "$attention_backend"' in launcher
    assert "VH_LLM_ATTENTION_BACKEND=FLASH_ATTN" in environment
    assert 'VH_LLM_KV_CACHE_DTYPE:-auto' in launcher
    assert '--kv-cache-dtype "$kv_cache_dtype"' in launcher
    assert "VH_LLM_KV_CACHE_DTYPE=auto" in environment
    assert 'VH_LLM_QUANTIZATION:-awq' in launcher
    assert '--quantization "$quantization"' in launcher
    assert "VH_LLM_QUANTIZATION=awq" in environment
    assert 'VH_LLM_DTYPE:-float16' in launcher
    assert '--dtype "$dtype"' in launcher
    assert "VH_LLM_DTYPE=float16" in environment
    assert 'VH_LLM_GPU_MEMORY_UTILIZATION:-0.48' in launcher
    assert "VH_LLM_GPU_MEMORY_UTILIZATION=0.48" in environment


def test_vllm_launchers_provision_and_expose_ninja_for_runtime_jit() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bootstrap = (
        project_root / "deploy" / "xiangongyun" / "bootstrap-models.sh"
    ).read_text()
    llm_launcher = (project_root / "scripts" / "run-xgc-llm.sh").read_text()
    tts_launcher = (project_root / "scripts" / "run-xgc-tts.sh").read_text()
    environment = (
        project_root / "deploy" / "xiangongyun" / "env.example"
    ).read_text()

    assert 'if [ ! -x "$vllm_venv/bin/ninja" ]' in bootstrap
    assert 'if [ ! -x "$tts_venv/bin/ninja" ]' in bootstrap
    assert 'export PATH="$venv/bin:$PATH"' in llm_launcher
    assert 'export PATH="$venv/bin:$PATH"' in tts_launcher
    assert 'VH_TTS_SERVED_MODEL:-higgs_audio_v3' in tts_launcher
    assert '--served-model-name "$served_name"' in tts_launcher
    assert 'stage_overrides="${VH_TTS_STAGE_OVERRIDES:-}"' in tts_launcher
    assert (
        """stage_overrides='{"0":{"gpu_memory_utilization":0.27,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"},"1":{"gpu_memory_utilization":0.05,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"}}'"""
        in tts_launcher
    )
    assert 'VH_TTS_STAGE_OVERRIDES:-{\\"0\\"' not in tts_launcher
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in tts_launcher
    assert "k2-fsa/OmniVoice" in bootstrap
    assert "'audio_tokenizer/*'" in bootstrap
    assert "audio_tokenizer/model.safetensors" in bootstrap
    assert "HIGGS_AUDIO_TOKENIZER_PATH" in tts_launcher
    assert (
        "HIGGS_AUDIO_TOKENIZER_PATH=/root/AI-Girlfriend/runtime/models/OmniVoice/audio_tokenizer"
        in environment
    )
    assert (
        """VH_TTS_STAGE_OVERRIDES='{"0":{"gpu_memory_utilization":0.27,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"},"1":{"gpu_memory_utilization":0.05,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"}}'"""
        in environment
    )


def test_bootstrap_uses_complete_qwen36_awq_snapshot() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bootstrap = (
        project_root / "deploy" / "xiangongyun" / "bootstrap-models.sh"
    ).read_text()
    launcher = (project_root / "scripts" / "run-xgc-llm.sh").read_text()
    environment = (
        project_root / "deploy" / "xiangongyun" / "env.example"
    ).read_text()

    assert "tclf90/Qwen3.6-35B-A3B-AWQ" in bootstrap
    assert "mattbucci/Qwen3.6-35B-A3B-AWQ" not in bootstrap
    assert "qwen36_awq_complete" in bootstrap
    assert '{"linear_attn", "self_attn", "shared_expert"}' in bootstrap
    assert "linear_qweights" in bootstrap
    assert "Qwen3.6-35B-A3B-AWQ-QuantTrio" in launcher
    assert "Qwen3.6-35B-A3B-AWQ-QuantTrio" in environment
    assert "VH_MEMORY_VLM_MAX_TOKENS=1024" in environment
    assert "VH_MEMORY_VLM_MAX_CONCURRENT=1" in environment
