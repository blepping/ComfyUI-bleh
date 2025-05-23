from __future__ import annotations

import math
from time import time
from typing import TYPE_CHECKING, NamedTuple

import folder_paths
import latent_preview
import torch
from comfy import latent_formats
from comfy.cli_args import LatentPreviewMethod
from comfy.cli_args import args as comfy_args
from comfy.model_management import device_supports_non_blocking
from comfy.taesd.taesd import TAESD
from PIL import Image
from tqdm import tqdm

from ..settings import SETTINGS  # noqa: TID252
from .tae_vid import TAEVid

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl
_ORIG_GET_PREVIEWER = latent_preview.get_previewer

LAST_LATENT_FORMAT = None


class VideoModelInfo(NamedTuple):
    latent_format: latent_formats.LatentFormat
    fps: int = 24
    temporal_compression: int = 8
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
}


class ImageWrapper:
    def __init__(self, frames: tuple, frame_duration: int):
        self._frames = frames
        self._frame_duration = frame_duration

    def save(self, fp, format: str | None, **kwargs: dict):  # noqa: A002
        if len(self._frames) == 1:
            return self._frames[0].save(fp, format, **kwargs)
        kwargs |= {
            "loop": 0,
            "save_all": True,
            "append_images": self._frames[1:],
            "duration": self._frame_duration,
        }
        return self._frames[0].save(fp, "webp", **kwargs)

    def resize(self, *args: list, **kwargs: dict) -> ImageWrapper:
        return ImageWrapper(
            tuple(frame.resize(*args, **kwargs) for frame in self._frames),
            frame_duration=self._frame_duration,
        )

    def __getattr__(self, key):
        return getattr(self._frames[0], key)


class FallbackPreviewerModel(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        latent_format: latent_formats.LatentFormat,
        *,
        dtype: torch.dtype,
        device: torch.device,
        scale_factor: float = 8.0,
        upscale_mode: str = "bilinear",
    ):
        super().__init__()

        raw_factors = latent_format.latent_rgb_factors
        raw_bias = latent_format.latent_rgb_factors_bias
        factors = torch.tensor(raw_factors, device=device, dtype=dtype).transpose(0, 1)
        bias = (
            torch.tensor(raw_bias, device=device, dtype=dtype)
            if raw_bias is not None
            else None
        )
        self.lin = torch.nn.Linear(
            factors.shape[1],
            factors.shape[0],
            device=device,
            dtype=dtype,
            bias=bias is not None,
        )
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode=upscale_mode)
        self.requires_grad_(False)  # noqa: FBT003
        self.lin.weight.copy_(factors)
        if bias is not None:
            self.lin.bias.copy_(bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x.movedim(1, -1)).movedim(-1, 1)
        x = self.upsample(x).movedim(1, -1)
        return x.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)


class BetterPreviewer(_ORIG_PREVIEWER):
    def __init__(
        self,
        *,
        taesd: torch.nn.Module | None = None,
        latent_format: latent_formats.LatentFormat,
        vid_info: VideoModelInfo | None = None,
    ):
        self.latent_format = latent_format
        self.vid_info = vid_info
        self.fallback_previewer_model = None
        self.device = (
            None
            if SETTINGS.btp_preview_device is None
            else torch.device(SETTINGS.btp_preview_device)
        )
        if taesd is not None:
            if hasattr(taesd, "taesd_encoder"):
                del taesd.taesd_encoder
            if hasattr(taesd, "encoder"):
                del taesd.encoder
            if self.device and self.device != next(taesd.parameters()).device:
                taesd = taesd.to(self.device)
        self.taesd = taesd
        self.stamp = None
        self.cached = None
        self.blank = Image.new("RGB", size=(1, 1))
        self.oom_fallback = SETTINGS.btp_oom_fallback == "latent2rgb"
        self.oom_retry = SETTINGS.btp_oom_retry
        self.oom_count = 0
        self.skip_upscale_layers = SETTINGS.btp_skip_upscale_layers
        self.preview_max_width = SETTINGS.btp_max_width
        self.preview_max_height = SETTINGS.btp_max_height
        self.throttle_secs = SETTINGS.btp_throttle_secs
        self.max_batch_preview = SETTINGS.btp_max_batch
        self.maxed_batch_step_mode = SETTINGS.btp_maxed_batch_step_mode
        self.max_batch_cols = SETTINGS.btp_max_batch_cols
        self.compile_previewer = SETTINGS.btp_compile_previewer
        self.maybe_pop_upscale_layers()
        if self.compile_previewer:
            compile_kwargs = (
                {}
                if not isinstance(self.compile_previewer, dict)
                else self.compile_previewer
            )
            self.taesd = torch.compile(self.taesd, **compile_kwargs)

    # Popping upscale layers trick from https://github.com/madebyollin/
    def maybe_pop_upscale_layers(self, *, width=None, height=None) -> None:
        skip = self.skip_upscale_layers
        if skip == 0 or not isinstance(self.taesd, TAESD):
            return
        upscale_layers = tuple(
            idx
            for idx, layer in enumerate(self.taesd.taesd_decoder)
            if isinstance(layer, torch.nn.Upsample)
        )
        num_upscale_layers = len(upscale_layers)
        if skip < 0:
            if width is None or height is None:
                return
            aggressive = skip == -2
            skip = 0
            max_width, max_height = self.preview_max_width, self.preview_max_height
            while skip < num_upscale_layers and (
                width > max_width or height > max_height
            ):
                width //= 2
                height //= 2
                if not aggressive and width < max_width and height < max_height:
                    # Popping another would overshoot the size requirement.
                    break
                skip += 1
                if not aggressive and (width <= max_width or height <= max_height):
                    # At least one dimension is within the size requirement.
                    break
        if skip > 0:
            skip = min(skip, num_upscale_layers)
            for idx in range(1, skip + 1):
                self.taesd.taesd_decoder.pop(upscale_layers[-idx])
        self.skip_upscale_layers = 0

    def decode_latent_to_preview_image(
        self,
        preview_format: str,
        x0: torch.Tensor,
    ) -> tuple[str, Image, int]:
        preview_image = self.decode_latent_to_preview(x0)
        return (
            preview_format if not isinstance(preview_image, ImageWrapper) else "WEBP",
            preview_image,
            min(
                max(*preview_image.size),
                max(self.preview_max_width, self.preview_max_height),
            ),
        )

    def check_use_cached(self) -> bool:
        now = time()
        if (
            self.cached is not None and self.stamp is not None
        ) and now - self.stamp < self.throttle_secs:
            return True
        self.stamp = now
        return False

    def calculate_indexes(self, batch_size: int, *, is_video=False) -> range:
        max_batch = (
            SETTINGS.btp_video_max_frames if is_video else self.max_batch_preview
        )
        if max_batch < 0:
            return range(batch_size)
        if not self.maxed_batch_step_mode:
            return range(min(max_batch, batch_size))
        return range(
            0,
            batch_size,
            math.ceil(batch_size / max_batch),
        )[:max_batch]

    def prepare_decode_latent(
        self,
        x0: torch.Tensor,
        *,
        frames_to_batch=True,
    ) -> tuple[torch.Tensor, int, int]:
        is_video = x0.ndim == 5
        if frames_to_batch and is_video:
            x0 = x0.transpose(2, 1).reshape(-1, x0.shape[1], *x0.shape[-2:])
        batch = x0.shape[0]
        x0 = x0[self.calculate_indexes(batch, is_video=is_video), :]
        batch = x0.shape[0]
        height, width = x0.shape[-2:]
        if self.device and x0.device != self.device:
            x0 = x0.to(
                device=self.device,
                non_blocking=device_supports_non_blocking(x0.device),
            )
        cols, rows = self.calc_cols_rows(
            min(batch, self.max_batch_preview),
            width,
            height,
        )
        return x0, cols, rows

    def _decode_latent_taevid(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x0.shape[-2:]
        if self.device and x0.device != self.device:
            x0 = x0.to(
                device=self.device,
                non_blocking=device_supports_non_blocking(x0.device),
            )
        decoded = self.taesd.decode(
            x0.transpose(1, 2),
            parallel=SETTINGS.btp_video_parallel,
        ).movedim(2, -1)
        del x0
        decoded = decoded.reshape(-1, *decoded.shape[2:])
        batch = decoded.shape[0]
        decoded = decoded[self.calculate_indexes(batch, is_video=True), :]
        cols, rows = self.calc_cols_rows(
            min(
                batch,
                SETTINGS.btp_video_max_frames
                if SETTINGS.btp_video_max_frames >= 0
                else batch,
            ),
            width,
            height,
        )
        return (
            decoded.mul_(255.0).round_().clamp_(min=0, max=255.0).detach(),
            cols,
            rows,
        )

    def _decode_latent_taesd(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x0, cols, rows = self.prepare_decode_latent(
            x0,
            frames_to_batch=not isinstance(self.taesd, TAEVid),
        )
        height, width = x0.shape[-2:]
        if self.skip_upscale_layers < 0:
            self.maybe_pop_upscale_layers(
                width=width * 8 * cols,
                height=height * 8 * rows,
            )
        return (
            (
                self.taesd.decode(x0)
                .movedim(1, -1)
                .add_(1.0)
                .mul_(127.5)
                .clamp_(min=0, max=255.0)
                .detach()
            ),
            cols,
            rows,
        )

    def calc_cols_rows(
        self,
        batch_size: int,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        max_cols = self.max_batch_cols
        ratio = height / width
        cols = max(1, min(round((batch_size * ratio) ** 0.5), max_cols, batch_size))
        rows = math.ceil(batch_size / cols)
        return cols, rows

    @classmethod
    def decoded_to_animation(cls, samples: np.ndarray) -> ImageWrapper:
        batch = samples.shape[0]
        return ImageWrapper(
            tuple(Image.fromarray(samples[idx]) for idx in range(batch)),
            frame_duration=250,
        )

    def decoded_to_image(
        self,
        samples: torch.Tensor,
        cols: int,
        rows: int,
        *,
        is_video=False,
    ) -> Image | ImageWrapper:
        batch, (height, width) = samples.shape[0], samples.shape[-3:-1]
        samples = samples.to(
            device="cpu",
            dtype=torch.uint8,
            non_blocking=device_supports_non_blocking(samples.device),
        ).numpy()
        if batch == 1:
            self.cached = Image.fromarray(samples[0])
            return self.cached
        if SETTINGS.btp_animate_preview == "both" or (
            is_video,
            SETTINGS.btp_animate_preview,
        ) in {(True, "video"), (False, "batch")}:
            return self.decoded_to_animation(samples)
        cols, rows = self.calc_cols_rows(batch, width, height)
        img_size = (width * cols, height * rows)
        if self.cached is not None and self.cached.size == img_size:
            result = self.cached
        else:
            self.cached = result = Image.new("RGB", size=(width * cols, height * rows))
        for idx in range(batch):
            result.paste(
                Image.fromarray(samples[idx]),
                box=((idx % cols) * width, ((idx // cols) % rows) * height),
            )
        return result

    @torch.no_grad()
    def init_fallback_previewer(self, device: torch.device, dtype: torch.dtype) -> bool:
        if self.fallback_previewer_model is not None:
            return True
        if self.latent_format is None:
            return False
        self.fallback_previewer_model = FallbackPreviewerModel(
            self.latent_format,
            device=device,
            dtype=dtype,
        )
        return True

    def fallback_previewer(self, x0: torch.Tensor, *, quiet=False) -> Image:
        if not quiet:
            fallback_mode = "using fallback" if self.oom_fallback else "skipping"
            tqdm.write(
                f"*** BlehBetterPreviews: Got out of memory error while decoding preview - {fallback_mode}.",
            )
        if not self.oom_fallback:
            return self.blank
        if not self.init_fallback_previewer(x0.device, x0.dtype):
            self.oom_fallback = False
            tqdm.write(
                "*** BlehBetterPreviews: Couldn't initialize fallback previewer, giving up on previews.",
            )
            return self.blank
        x0, cols, rows = self.prepare_decode_latent(x0)
        try:
            return self.decoded_to_image(
                self.fallback_previewer_model(x0),
                cols,
                rows,
            )
        except torch.OutOfMemoryError:
            return self.blank

    def decode_latent_to_preview(self, x0: torch.Tensor) -> Image:
        if self.check_use_cached():
            return self.cached
        if x0.shape[0] == 0:
            return self.blank  # Shouldn't actually be possible.
        if (self.oom_count and not self.oom_retry) or self.taesd is None:
            return self.fallback_previewer(x0, quiet=True)
        is_video = x0.ndim == 5
        used_fallback = False
        start_time = time()
        try:
            dargs = (
                self._decode_latent_taevid(x0)
                if is_video
                else self._decode_latent_taesd(x0)
            )
            result = self.decoded_to_image(*dargs, is_video=is_video)
        except torch.OutOfMemoryError:
            used_fallback = True
            result = self.fallback_previewer(x0)
        if SETTINGS.btp_verbose:
            tqdm.write(
                f"BlehPreview: used fallback: {used_fallback}, decode time: {time() - start_time:0.2f}",
            )
        return result


def bleh_get_previewer(
    device,
    latent_format: latent_formats.LatentFormat,
    *args: list,
    **kwargs: dict,
) -> object | None:
    preview_method = comfy_args.preview_method
    format_name = latent_format.__class__.__name__.lower()
    if (
        not SETTINGS.btp_enabled
        or format_name in SETTINGS.btp_blacklist
        or (SETTINGS.btp_whitelist and format_name not in SETTINGS.btp_whitelist)
    ):
        return _ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)
    tae_model = None
    if preview_method in {LatentPreviewMethod.TAESD, LatentPreviewMethod.Auto}:
        vid_info = VIDEO_FORMATS.get(format_name)
        if vid_info is not None and vid_info.tae_model is not None:
            tae_model_path = folder_paths.get_full_path(
                "vae_approx",
                vid_info.tae_model,
            )
            tupscale_limit = SETTINGS.btp_video_temporal_upscale_level
            decoder_time_upscale = tuple(
                i < tupscale_limit for i in range(TAEVid.temporal_upscale_blocks)
            )
            tae_model = (
                TAEVid(
                    checkpoint_path=tae_model_path,
                    latent_channels=latent_format.latent_channels,
                    device=device,
                    decoder_time_upscale=decoder_time_upscale,
                ).to(device)
                if tae_model_path is not None
                else None
            )
        if tae_model is None and latent_format.taesd_decoder_name is not None:
            taesd_path = folder_paths.get_full_path(
                "vae_approx",
                f"{latent_format.taesd_decoder_name}.pth",
            )
            tae_model = (
                TAESD(
                    None,
                    taesd_path,
                    latent_channels=latent_format.latent_channels,
                ).to(device)
                if taesd_path is not None
                else None
            )
        return BetterPreviewer(
            taesd=tae_model,
            latent_format=latent_format,
            vid_info=vid_info,
        )
    if (
        preview_method == LatentPreviewMethod.NoPreviews
        or latent_format.latent_rgb_factors is None
    ):
        return None
    if preview_method == LatentPreviewMethod.Latent2RGB:
        return BetterPreviewer(latent_format=latent_format)
    return _ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)


def ensure_previewer():
    if latent_preview.get_previewer != bleh_get_previewer:
        latent_preview.BLEH_ORIG_get_previewer = _ORIG_GET_PREVIEWER
        latent_preview.get_previewer = bleh_get_previewer


ensure_previewer()
