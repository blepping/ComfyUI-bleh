from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from comfy import latent_formats

if TYPE_CHECKING:
    from pathlib import Path


class VideoModelInfo(NamedTuple):
    latent_format: latent_formats.LatentFormat
    fps: int = 24
    temporal_compression: int = 8
    patch_size: int = 1
    tae_model: str | Path | None = None


VIDEO_FORMATS = {
    "mochi": VideoModelInfo(
        latent_formats.Mochi,
        temporal_compression=6,
        tae_model="taem1.pth",
    ),
    "hunyuanvideo": VideoModelInfo(
        latent_formats.HunyuanVideo,
        temporal_compression=4,
        tae_model="taehv.pth",
    ),
    "cosmos1cv8x8x8": VideoModelInfo(latent_formats.Cosmos1CV8x8x8),
    "wan21": VideoModelInfo(
        latent_formats.Wan21,
        fps=16,
        temporal_compression=4,
        tae_model="taew2_1.pth",
    ),
    "wan22": VideoModelInfo(
        latent_formats.Wan22,
        fps=24,
        temporal_compression=4,
        patch_size=2,
        tae_model="taew2_2.pth",
    ),
}


__all__ = ("VIDEO_FORMATS", "VideoModelInfo")
