from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTODL = ROOT / "deploy" / "autodl"


def test_autodl_uses_public_port_6006_and_keeps_models_loopback_only() -> None:
    env = (AUTODL / "env.example").read_text(encoding="utf-8")

    assert "VH_HOST=0.0.0.0" in env
    assert "VH_PORT=6006" in env
    for port in (8000, 8001, 8010, 1934, 8770, 8002):
        assert f"127.0.0.1:{port}" in env


def test_autodl_control_reuses_the_locked_xiangongyun_runtime() -> None:
    control = (AUTODL / "control.sh").read_text(encoding="utf-8")
    autostart = (AUTODL / "autostart.sh").read_text(encoding="utf-8")

    assert "deploy/xiangongyun/control.sh" in control
    assert "VH_XGC_RUNTIME_ENV" in control
    assert "VH_XGC_SECRETS_ENV" in control
    assert "flock -n 9" in autostart
    assert "deploy/autodl/control.sh status" in autostart
    assert "deploy/autodl/control.sh start" in autostart
