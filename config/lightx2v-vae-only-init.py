"""Minimal LightX2V package initializer for SoulX-LiveAct's VAE-only use."""

__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

# SoulX-LiveAct imports only lightx2v.models.video_encoders.hf.wan.vae.
# Initializing the device is required, while importing the complete LightX2V
# serving pipeline pulls unrelated runtime dependencies into this isolated test.
import lightx2v_platform.set_ai_device  # noqa: F401, E402
