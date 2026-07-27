from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "deploy" / "autodl" / "liveact_demo_launcher.py"


def _load_launcher(monkeypatch):
    class FakeFlask:
        def run(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "flask", SimpleNamespace(Flask=FakeFlask))
    spec = importlib.util.spec_from_file_location("liveact_demo_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t5_cache_is_opt_in(monkeypatch) -> None:
    launcher = _load_launcher(monkeypatch)
    imported: list[str] = []
    monkeypatch.delenv("VH_LIVEACT_CACHE_T5", raising=False)
    monkeypatch.setattr(
        launcher, "import_module", lambda name: imported.append(name)
    )

    launcher._install_t5_cache()

    assert imported == []


def test_t5_cache_reuses_text_device_key_and_evicts_lru(monkeypatch) -> None:
    launcher = _load_launcher(monkeypatch)

    class FakeT5:
        calls = 0

        def __call__(self, texts, device):
            type(self).calls += 1
            return [f"{texts}@{device}#{type(self).calls}"]

    monkeypatch.setenv("VH_LIVEACT_CACHE_T5", "1")
    monkeypatch.setenv("VH_LIVEACT_T5_CACHE_ENTRIES", "2")
    monkeypatch.setattr(
        launcher,
        "import_module",
        lambda name: SimpleNamespace(T5EncoderModel=FakeT5),
    )

    launcher._install_t5_cache()
    encoder = FakeT5()

    first = encoder("talk", "cpu")
    assert encoder("talk", "cpu") is first
    encoder("smile", "cpu")
    encoder("turn", "cpu")
    encoder("talk", "cpu")

    assert FakeT5.calls == 4
    assert len(encoder._vh_t5_cache) == 2


def test_vae_compile_policy_only_changes_wanvae_decode(monkeypatch) -> None:
    launcher = _load_launcher(monkeypatch)
    compile_calls: list[tuple[object, dict[str, object]]] = []

    class FakeTorch:
        @staticmethod
        def compile(model, *args, **kwargs):
            compile_calls.append((model, kwargs))
            return ("compiled", model)

    def decode():
        return None

    decode.__qualname__ = "WanVAE.decode"

    def denoiser():
        return None

    denoiser.__qualname__ = "WanModel.forward"
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setenv("VH_LIVEACT_VAE_COMPILE_MODE", "dynamic")

    launcher._install_vae_compile_policy()

    assert FakeTorch.compile(decode) == ("compiled", decode)
    assert compile_calls[-1][1]["dynamic"] is True
    assert FakeTorch.compile(denoiser) == ("compiled", denoiser)
    assert compile_calls[-1][1] == {}


def test_vae_compile_policy_can_disable_only_wanvae_decode(monkeypatch) -> None:
    launcher = _load_launcher(monkeypatch)

    class FakeTorch:
        @staticmethod
        def compile(model, *args, **kwargs):
            return ("compiled", model)

    def decode():
        return None

    decode.__qualname__ = "WanVAE.decode"
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setenv("VH_LIVEACT_VAE_COMPILE_MODE", "off")

    launcher._install_vae_compile_policy()

    assert FakeTorch.compile(decode) is decode


def test_tail_frame_fix_builds_same_directory_runtime_copy(
    tmp_path, monkeypatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    demo_path = tmp_path / "demo.py"
    demo_path.write_text(
        "import argparse\n"
        "def generate(self, audio_len_sec, fps):\n"
        "            iter_total_num = int(audio_len_sec / "
        "(self.vae_stride[0] * self.blksz_lst[-1] / fps)) + 1\n"
        "            pre_latent = None\n"
        "            if self.rank == 0:\n"
        "                    chunk_bytes, num_frames_this_chunk = "
        "tensor_chunk_to_rgb_bytes(_videos)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VH_LIVEACT_FIX_TAIL_FRAMES", "1")

    patched_path = launcher._prepare_demo_path(demo_path)
    patched = patched_path.read_text(encoding="utf-8")

    assert patched_path.parent == demo_path.parent
    assert patched_path.name == ".autodl-demo-tailfix.py"
    assert "target_total_frames = math.ceil(audio_len_sec * fps)" in patched
    assert "_videos = _videos[:, :, :frames_remaining]" in patched
    assert "generated_frames += num_frames_this_chunk" in patched
    assert demo_path.read_text(encoding="utf-8").startswith("import argparse\n")
