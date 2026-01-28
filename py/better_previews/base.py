from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from comfy import latent_formats

from .tae_vid import TAEVid, TAEVidBase, TAEVidLTX2

if TYPE_CHECKING:
    from pathlib import Path


class VideoModelInfo(NamedTuple):
    name: str
    latent_format: latent_formats.LatentFormat
    fps: int | float = 24
    temporal_compression: int = 8
    temporal_layers: int = 0
    patch_size: int = 1
    nested_tensor_index: int = 0
    tae_model: str | Path | None = None
    tae_class: TAEVidBase | None = TAEVid


VIDEO_FORMATS = {
    vmi.name: vmi
    for vmi in (
        VideoModelInfo(
            "mochi",
            latent_formats.Mochi,
            temporal_compression=6,
            tae_model="taem1.pth",
        ),
        VideoModelInfo(
            "hunyuanvideo",
            latent_formats.HunyuanVideo,
            temporal_compression=4,
            tae_model="taehv.pth",
        ),
        VideoModelInfo(
            "hunyuanvideo15",
            latent_formats.HunyuanVideo15,
            temporal_compression=4,
            patch_size=2,
            tae_model="taehv1_5.pth",
        ),
        VideoModelInfo(
            "cosmos1cv8x8x8",
            latent_formats.Cosmos1CV8x8x8,
        ),
        VideoModelInfo(
            "wan21",
            latent_formats.Wan21,
            fps=16,
            temporal_compression=4,
            temporal_layers=2,
            tae_model="taew2_1.pth",
        ),
        VideoModelInfo(
            "wan22",
            latent_formats.Wan22,
            fps=24,
            temporal_compression=4,
            temporal_layers=2,
            patch_size=2,
            tae_model="taew2_2.pth",
        ),
        VideoModelInfo(
            "ltxv",
            latent_formats.LTXV,
            fps=24,
            patch_size=4,
            temporal_layers=3,
            tae_model="taeltx_2.pth",
            tae_class=TAEVidLTX2,
        ),
        VideoModelInfo(
            "ltxav",
            latent_formats.LTXV,
            fps=24,
            patch_size=4,
            temporal_layers=3,
            tae_model="taeltx_2.pth",
            tae_class=TAEVidLTX2,
        ),
    )
}


__all__ = ("VIDEO_FORMATS", "VideoModelInfo")
