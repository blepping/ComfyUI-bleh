from __future__ import annotations

import math
from functools import lru_cache
from io import BytesIO
from time import time
from typing import TYPE_CHECKING, Any

import comfy.utils as comfy_utils
import folder_paths
import latent_preview
import torch
from comfy import latent_formats
from comfy.cli_args import LatentPreviewMethod
from comfy.cli_args import args as comfy_args
from comfy.model_management import device_supports_non_blocking, vae_dtype
from comfy.taesd.taesd import TAESD
from PIL import Image
from tqdm import tqdm

from .. import settings  # noqa: TID252
from . import last_preview
from .base import AMBIGUOUS_VIDEO_FORMATS, VIDEO_FORMATS, VideoModelInfo
from .tae_vid import TAEVid

if TYPE_CHECKING:
    import numpy as np
    from comfy import latent_formats


class BlehPreviewerState:
    last_latent_shapes: tuple | None = None
    fps_override: float | None = None
    prefer_previewer: str | None = None


PREVIEWER_STATE = BlehPreviewerState()

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl
_ORIG_GET_PREVIEWER = latent_preview.get_previewer

AUDIO_LATENT_FORMAT_NAMES = frozenset(
    (
        "aceaudio",
        "aceaudio15",
        "minimaxmusic3",
        "stableaudio1",
        "stableaudio3",
        "yue2",
    ),
)


# Referenced from https://github.com/learnables/learn2learn/blob/752200384c3ca8caeb8487b5dd1afd6568e8ec01/learn2learn/utils/__init__.py#L51
def clone_module(module, *, memo: dict | None = None) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module")
    if memo is None:
        memo = {}
    clone = module.__new__(type(module))
    for k in ("__dict__", "_parameters", "_buffers", "_modules"):
        if not hasattr(clone, k):
            continue
        setattr(clone, k, getattr(module, k).copy())
    # We don't care about the has_grad case here.
    for k in getattr(clone, "_parameters", {}):
        v = module._parameters[k]  # noqa: SLF001
        if v is None:
            continue
        ptr = v.data_ptr
        new_v = memo.get(ptr)
        if new_v is None:
            new_v = v.clone()
            memo[ptr] = new_v
        clone._parameters[k] = new_v  # noqa: SLF001
    for k in getattr(clone, "_modules", {}):
        clone._modules[k] = clone_module(module._modules[k], memo=memo)  # noqa: SLF001
    if hasattr(clone, "flatten_parameters"):
        clone = clone._apply(lambda x: x)  # noqa: SLF001
    return clone


# Simple heuristic.
def get_module_device_dtype(
    module: torch.nn.Module,
) -> tuple[torch.device, torch.dtype] | tuple[None, None]:
    p = next(module.parameters(), None)
    if p is None:
        raise RuntimeError("Couldn't get module device/dtype!")
    return p.device, p.dtype


def normalize_to_scale(
    latent: torch.Tensor,
    target_min: float,
    target_max: float,
    *,
    dim: tuple[int, ...] | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(1, latent.ndim))
    if eps is None:
        eps = torch.finfo(latent.dtype).eps * 1.25
    min_val, max_val = (
        (
            latent.amin(dim=dim, keepdim=True),
            latent.amax(dim=dim, keepdim=True),
        )
        if len(dim) != 1
        else latent.aminmax(dim=dim[0], keepdim=True)
    )
    divisor = max_val.sub_(min_val)
    divisor = divisor.abs().clamp_min_(eps).copysign_(divisor)
    normalized = (latent - min_val).div_(divisor)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


class ImageWrapper:
    def __init__(
        self,
        frames: tuple | Image,
        *,
        frame_duration: int = 250,
        pcfg: settings.PreviewSettings | None = None,
    ):
        self._frames = (frames,) if not isinstance(frames, (tuple, list)) else frames
        self._frame_duration = frame_duration
        self._pcfg = pcfg or settings.SETTINGS.previews

    def _save_image(
        self,
        frames: tuple[Image, ...],
        *,
        format: str | None,  # noqa: A002
        **kwargs: Any,
    ) -> BytesIO:
        buf = BytesIO()
        extra_kwargs = (
            {}
            if len(frames) < 2
            else {
                "loop": 0,
                "save_all": True,
                "append_images": frames[1:],
                "duration": self._frame_duration,
            }
        )
        frames[0].save(buf, format, **extra_kwargs, **kwargs)
        return buf

    def save(self, fp, format: str | None, **kwargs: Any):  # noqa: A002
        pcfg = self._pcfg
        frames = self._frames
        publishing = last_preview.LAST_PREVIEW is not None
        animated = len(frames) > 1
        split_preview = animated and publishing and pcfg.only_animate_last_preview
        result_format = "webp" if animated else (format or "png")
        result = self._save_image(frames, format=result_format, **kwargs).getvalue()
        _preview_format, preview_result = (
            (result_format, result)
            if not split_preview
            else (
                format,
                self._save_image(frames[:1], format=format, **kwargs).getvalue(),
            )
        )
        if publishing:
            duration = (
                2 + int((len(self._frames) * self._frame_duration) / 1000)
                if animated
                else None
            )
            last_preview.LAST_PREVIEW.update(
                image_bytes=result,
                content_type=f"image/{result_format}",
                duration=duration,
            )
        fp.write(preview_result)

    def resize(self, *args: Any, **kwargs: Any) -> ImageWrapper:
        return ImageWrapper(
            tuple(frame.resize(*args, **kwargs) for frame in self._frames),
            frame_duration=self._frame_duration,
            pcfg=self._pcfg,
        )

    def copy(self) -> ImageWrapper:
        return self.__class__(
            tuple(i.copy() for i in self._frames),
            frame_duration=self._frame_duration,
            pcfg=self._pcfg,
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
        self.dtype = dtype
        self.device = device
        raw_factors = latent_format.latent_rgb_factors
        raw_bias = latent_format.latent_rgb_factors_bias
        self.reshape_fun = getattr(latent_format, "latent_rgb_factors_reshape", None)
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
        if self.reshape_fun is not None:
            x = self.reshape_fun(x)
        x = self.lin(x.movedim(1, -1)).movedim(-1, 1)
        x = self.upsample(x).movedim(1, -1)
        return x.add_(1.0).clamp_(0.0, 2.0).mul_(127.5).round_()


class AudioPreviewerModel(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        height_factor: float = 4.0,
        width_factor: float = 1.0,
        upsample_mode: str = "bilinear",
        normalize_dims: tuple = (-1,),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.normalize_dims = normalize_dims
        if not (height_factor == 1 and width_factor == 1):
            self.upsample = torch.nn.Upsample(
                scale_factor=(height_factor, width_factor),
                mode=upsample_mode,
            )
        else:
            self.upsample = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_to_scale(
            x.reshape(x.shape[0], 1, -1, x.shape[-1]),
            0.0,
            255.0,
            dim=self.normalize_dims,
        )
        if self.upsample is not None:
            x = self.upsample(x).clamp_(0.0, 255.0)
        x = x.movedim(1, -1)
        return x.expand(*x.shape[:-1], 3)


class BetterPreviewer(_ORIG_PREVIEWER):
    def __init__(
        self,
        *,
        taesd: torch.nn.Module | None = None,
        latent_format: latent_formats.LatentFormat | None,
        vid_info: VideoModelInfo | None = None,
    ):
        pcfg = self.pcfg = settings.SETTINGS.previews
        # tqdm.write(f"PREVIEWER: {pcfg}")
        self.orig_latent_format = latent_format
        self.latent_format = (
            latent_format if vid_info is None else vid_info.latent_format
        )
        self.latent_format_name = (
            "unknown"
            if latent_format is None
            else latent_format.__class__.__name__.lower()
        )
        self.spatial_compression = getattr(
            latent_format,
            "spacial_downscale_ratio",  # sic
            8,
        )
        self.vid_info = vid_info
        self.fallback_previewer_model = None
        self.device = (
            None if pcfg.preview_device is None else torch.device(pcfg.preview_device)
        )
        self.orig_previewer_model = (
            None
            if taesd is None
            else clone_module(taesd).to(device="cpu", dtype=torch.float32)
        )
        if taesd is not None:
            if hasattr(taesd, "taesd_encoder"):
                del taesd.taesd_encoder
            if hasattr(taesd, "encoder"):
                del taesd.encoder
        self.previewer_model = taesd
        self.stamp = None
        self.cached = None
        self.blank = Image.new("RGB", size=(1, 1))
        self.oom_fallback = pcfg.oom_fallback == settings.OomFallback.LATENT2RGB
        self.oom_count = 0
        self.skip_upscale_layers_state: tuple[int, int] | tuple[None, None] | None = (
            None
        )
        self.preview_counter = 0

    @property
    def blank_copy(self) -> Image:
        return self.blank.copy()

    @property
    def cached_copy(self) -> Image:
        return self.cached.copy() if self.cached is not None else self.blank_copy

    def maybe_refresh_previewer(
        self,
        *,
        device=None,
        dtype=None,
        width=None,
        height=None,
    ) -> None:
        if self.orig_previewer_model is None:
            return
        pdevice, pdtype = (
            get_module_device_dtype(self.previewer_model)
            if self.previewer_model is not None
            else (None, None)
        )
        need_refresh = (
            self.previewer_model is None
            or (dtype is not None and pdtype != dtype)
            or (device is not None and pdevice != device)
        )
        is_taesd = isinstance(self.orig_previewer_model, TAESD)
        if is_taesd and not need_refresh:
            need_refresh = (
                self.pcfg.skip_upscale_layers < 0
                and self.skip_upscale_layers_state != (width, height)
            )
        if not need_refresh:
            return
        tqdm.write("[Bleh] Refreshing previewer")
        self.previewer_model = clone_module(self.orig_previewer_model).to(
            device=device,
            dtype=dtype,
        )
        if is_taesd:
            self.skip_upscale_layers_state = None
            self.maybe_pop_upscale_layers(width=width, height=height)
        if not self.pcfg.compile_previewer:
            return
        if self.pcfg.verbose:
            tqdm.write("[Bleh] Compiling previewer")
        compile_kwargs = (
            {}
            if not isinstance(self.pcfg.compile_previewer, dict)
            else self.pcfg.compile_previewer
        )
        self.previewer_model = torch.compile(self.previewer_model, **compile_kwargs)

    # Popping upscale layers trick from https://github.com/madebyollin/
    def maybe_pop_upscale_layers(self, *, width=None, height=None) -> None:
        if self.skip_upscale_layers_state:
            return
        self.skip_upscale_layers_state = (width, height)
        skip = self.pcfg.skip_upscale_layers
        if skip == 0 or not isinstance(self.previewer_model, TAESD):
            return
        upscale_layers = tuple(
            idx
            for idx, layer in enumerate(self.previewer_model.taesd_decoder)
            if isinstance(layer, torch.nn.Upsample)
        )
        num_upscale_layers = len(upscale_layers)
        if skip < 0:
            if width is None or height is None:
                return
            aggressive = skip == -2
            skip = 0
            max_width, max_height = (
                self.pcfg.max_width,
                self.pcfg.max_height,
            )
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
                self.previewer_model.taesd_decoder.pop(upscale_layers[-idx])

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
                max(self.pcfg.max_width, self.pcfg.max_height),
            ),
        )

    def check_use_cached(self, throttle: float) -> bool:
        pcfg = self.pcfg
        if self.preview_counter < pcfg.preview_offset:
            if pcfg.verbose:
                tqdm.write(
                    f"BLEH: OFFSET: counter={self.preview_counter} < {pcfg.preview_offset}",
                )
            self.preview_counter += 1
            return True
        now = time()
        can_use_cache = self.cached is not None and self.stamp is not None
        use_cache = can_use_cache and now - self.stamp < throttle
        interval = int(pcfg.preview_interval)
        ainterval = abs(interval)
        if not use_cache and can_use_cache and ainterval > 1:
            mod_counter = self.preview_counter % ainterval
            interval_skip = mod_counter != 0 if interval >= 0 else mod_counter == 0
            use_cache = use_cache or interval_skip
            if pcfg.verbose:
                tqdm.write(
                    f"BLEH: INTERVAL: interval={interval}, counter={self.preview_counter} ({mod_counter}), skip={interval_skip}",
                )

        self.preview_counter += 1
        if use_cache:
            return True
        self.stamp = now
        return False

    def calculate_indexes(self, batch_size: int, *, is_video=False) -> tuple:
        max_batch = (
            batch_size
            if is_video and self.pcfg.video_max_frames <= 0
            else min(
                batch_size,
                self.pcfg.video_max_frames if is_video else self.pcfg.max_batch,
            )
        )
        if max_batch < 0 or max_batch == batch_size:
            return tuple(range(batch_size))
        if not self.pcfg.maxed_batch_step_mode or max_batch >= batch_size:
            return tuple(range(min(max_batch, batch_size)))
        if max_batch <= 1:
            return (0,)
        step = (batch_size - 1) / (max_batch - 1)
        return tuple(round(i * step) for i in range(max_batch))

    def prepare_decode_latent(
        self,
        x0: torch.Tensor,
        *,
        frames_to_batch=True,
    ) -> tuple[torch.Tensor, int, int]:
        is_video = x0.ndim == 5
        is_multiframe_video = is_video and x0.shape[2] > 1
        if frames_to_batch and is_video:
            x0 = x0.transpose(2, 1).reshape(-1, x0.shape[1], *x0.shape[-2:])
        x0 = x0[self.calculate_indexes(x0.shape[0], is_video=is_multiframe_video), :]
        batch = x0.shape[0]
        height, width = x0.shape[-2:]
        cols, rows = self.calc_cols_rows(
            batch_size=batch,
            width=width,
            height=height,
            max_cols=self.pcfg.max_batch_cols,
        )
        return x0, cols, rows

    def prepare_previewer(
        self,
        x0: torch.Tensor,
        *,
        img_width: int | None = None,
        img_height: int | None = None,
    ) -> torch.Tensor:
        preview_dtype = self.pcfg.preview_dtype
        if preview_dtype == settings.PreviewDtype.VAE:
            dtype = vae_dtype(x0)
        elif preview_dtype is None or preview_dtype == settings.PreviewDtype.KEEP:
            dtype = x0.dtype
        else:
            dtype = preview_dtype.as_dtype
        self.maybe_refresh_previewer(
            dtype=dtype,
            device=self.device or x0.device,
            width=img_width,
            height=img_height,
        )
        pdevice, pdtype = get_module_device_dtype(self.previewer_model)
        # tqdm.write(
        #     f"\nPREVIEW: pdevice={pdevice}, pdtype={pdtype}, device={x0.device}, dtype={x0.dtype}",
        # )
        if x0.device == pdevice and x0.dtype == pdtype:
            return x0
        return x0.to(
            device=pdevice,
            dtype=pdtype,
            non_blocking=self.pcfg.preview_non_blocking
            and device_supports_non_blocking(x0.device),
        )

    def _decode_latent_taevid(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        frames = x0.shape[2]
        height, width = x0.shape[-2:]
        x0 = self.prepare_previewer(x0)
        decoded = self.previewer_model.decode(
            x0.transpose(1, 2),
            parallel=self.pcfg.video_parallel,
        ).movedim(2, -1)
        del x0
        decoded = decoded.reshape(-1, *decoded.shape[2:])
        batch = decoded.shape[0]
        decoded = decoded[self.calculate_indexes(batch, is_video=frames > 1), :]
        cols, rows = self.calc_cols_rows(
            batch_size=min(
                batch,
                self.pcfg.video_max_frames
                if frames > 1 and self.pcfg.video_max_frames >= 0
                else batch,
            ),
            width=width,
            height=height,
            max_cols=self.pcfg.max_batch_cols,
        )
        return (
            decoded.clamp_(0.0, 1.0).mul_(255.0).round_().detach(),
            cols,
            rows,
        )

    def _decode_latent_taesd(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x0, cols, rows = self.prepare_decode_latent(
            x0,
            frames_to_batch=not isinstance(self.previewer_model, TAEVid),
        )
        height, width = x0.shape[-2:]
        img_height, img_width = (
            height * self.spatial_compression * rows,
            width * self.spatial_compression * cols,
        )
        x0 = self.prepare_previewer(x0, img_width=img_width, img_height=img_height)
        return (
            (
                self.previewer_model.decode(x0)
                .movedim(1, -1)
                .add_(1.0)
                .clamp_(0.0, 2.0)
                .mul_(127.5)
                .round_()
                .detach()
            ),
            cols,
            rows,
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def calc_cols_rows(
        *,
        batch_size: int,
        width: int,
        height: int,
        max_cols: int | None = None,
    ) -> tuple[int, int]:
        if batch_size < 2:
            return 1, 1
        max_cols = max_cols or batch_size
        limit_cols = min(batch_size, max(1, max_cols))
        best_cols, best_rows = 1, batch_size
        min_max_dim = min_empty_cells = math.inf

        for cols in range(1, limit_cols + 1):
            rows = math.ceil(batch_size / cols)
            max_dim = max(cols * width, rows * height)
            empty_cells = (cols * rows) - batch_size
            if max_dim < min_max_dim or (
                max_dim == min_max_dim and empty_cells < min_empty_cells
            ):
                min_max_dim, min_empty_cells = max_dim, empty_cells
                best_cols, best_rows = cols, rows
        return best_cols, best_rows

    def decoded_to_animation(
        self,
        samples: np.ndarray,
        video_frames: int,
    ) -> ImageWrapper:
        batch = samples.shape[0]
        fps_override = PREVIEWER_STATE.fps_override
        if self.vid_info is None or not video_frames:
            frame_duration = 250 if not fps_override else 1000 / fps_override
        else:
            time_factor = self.vid_info.temporal_compression / max(
                1,
                self.previewer_model.t_upscale,
            )
            ms_frame = 1000.0 / (fps_override or self.vid_info.fps)
            frame_duration = ms_frame * time_factor
        frames = tuple(Image.fromarray(samples[idx]) for idx in range(batch))
        self.cached = frames[0]
        return ImageWrapper(
            frames,
            frame_duration=max(1, int(frame_duration)),
        )

    def decoded_to_image(
        self,
        samples: torch.Tensor,
        cols: int | None = None,
        rows: int | None = None,
        *,
        video_frames: int = 0,
    ) -> Image | ImageWrapper:
        batch, (height, width) = samples.shape[0], samples.shape[-3:-1]
        samples = samples.to(device="cpu", dtype=torch.uint8).numpy()
        if batch == 1:
            self.cached = ImageWrapper((Image.fromarray(samples[0]),))
            return self.cached_copy
        atype = settings.AnimatePreview
        animate = self.pcfg.animate_preview == atype.BOTH or (
            video_frames != 0,
            self.pcfg.animate_preview,
        ) in {(True, atype.VIDEO), (False, atype.BATCH)}
        animate = animate and (
            self.pcfg.publish_last_preview or not self.pcfg.only_animate_last_preview
        )
        if animate:
            return self.decoded_to_animation(samples, video_frames=video_frames)
        if cols is None or rows is None:
            cols, rows = self.calc_cols_rows(
                batch_size=batch,
                width=width,
                height=height,
                max_cols=self.pcfg.max_batch_cols,
            )
        img_size = (width * cols, height * rows)
        result = Image.new("RGB", size=img_size)
        for idx in range(batch):
            result.paste(
                Image.fromarray(samples[idx]),
                box=((idx % cols) * width, ((idx // cols) % rows) * height),
            )
        self.cached = ImageWrapper((result,))
        return self.cached_copy

    @torch.no_grad()
    def init_fallback_previewer(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> bool:
        if self.latent_format is None:
            return False
        if (
            self.fallback_previewer_model is not None
            and self.fallback_previewer_model.dtype == dtype
            and self.fallback_previewer_model.device == device
        ):
            return True
        if self.latent_format_name in AUDIO_LATENT_FORMAT_NAMES:
            self.fallback_previewer_model = AudioPreviewerModel(
                device=device,
                dtype=dtype,
            )
            return True
        if self.latent_format.latent_rgb_factors is None:
            return False
        self.fallback_previewer_model = FallbackPreviewerModel(
            self.latent_format,
            device=device,
            dtype=dtype,
            scale_factor=self.spatial_compression,
        )
        return True

    def fallback_previewer(self, x0: torch.Tensor, *, quiet=False) -> Image:
        if not quiet:
            fallback_mode = "using fallback" if self.oom_fallback else "skipping"
            tqdm.write(
                f"*** BlehBetterPreviews: Got out of memory error while decoding preview - {fallback_mode}.",
            )
        if not self.oom_fallback:
            return self.blank_copy
        if not self.init_fallback_previewer(x0.device, x0.dtype):
            self.oom_fallback = False
            tqdm.write(
                "*** BlehBetterPreviews: Couldn't initialize fallback previewer, giving up on previews.",
            )
            return self.blank_copy
        x0, cols, rows = self.prepare_decode_latent(x0)
        try:
            return self.decoded_to_image(
                self.fallback_previewer_model(x0),
                cols,
                rows,
                video_frames=0,
            )
        except torch.OutOfMemoryError:
            return self.blank_copy

    def ensure_x0_shape(self, x0: torch.Tensor) -> tuple[torch.Tensor, bool]:  # noqa: PLR0911
        expected_channels = self.latent_format.latent_channels
        expected_ndim = 2 + self.latent_format.latent_dimensions
        if x0.shape[0] == 0:
            return x0, False
        if (
            self.latent_format_name in AUDIO_LATENT_FORMAT_NAMES
            and x0.ndim == expected_ndim + 1
        ):
            if x0.shape[1] == 1:
                x0 = x0.squeeze(1)
            elif x0.shape[2] == 1:
                x0 = x0.squeeze(2)
            else:
                return x0, False
        if (
            x0.ndim > 1
            and x0.ndim == expected_ndim
            and x0.shape[1] == expected_channels
        ):
            return x0, True
        last_shapes = PREVIEWER_STATE.last_latent_shapes
        if not last_shapes or not hasattr(comfy_utils, "unpack_latents"):
            return x0, False
        last_numel = sum(math.prod(tshape) for tshape in last_shapes)
        if last_numel != x0.numel():
            return x0, False
        nest_idx = self.vid_info.nested_tensor_index if self.vid_info else 0
        target_shape = None if len(last_shapes) <= nest_idx else last_shapes[nest_idx]
        if (
            # Have to have a nest shape
            target_shape is None
            # with at least a channel dimension,
            or len(target_shape) < 2
            # with the expected number of dims,
            or len(target_shape) != expected_ndim
            # And the correct number of channels.
            or target_shape[1] != expected_channels
        ):
            return x0, False
        unpacked_latents = comfy_utils.unpack_latents(x0, last_shapes)
        target_latent = (
            None if len(unpacked_latents) <= nest_idx else unpacked_latents[nest_idx]
        )
        if target_latent is None or target_latent.shape != target_shape:
            return x0, False
        return target_latent.reshape(*target_shape), True

    def decode_latent_to_preview(self, x0: torch.Tensor) -> Image:
        pcfg = self.pcfg
        using_fallback = (
            self.oom_count and not self.pcfg.oom_retry
        ) or self.previewer_model is None
        if self.vid_info is None or using_fallback:
            if self.check_use_cached(pcfg.get_throttle(fallback=using_fallback)):
                return self.cached_copy
            checked_cache = True
        else:
            checked_cache = False
        x0, can_preview = self.ensure_x0_shape(x0)
        if not can_preview:
            return self.blank_copy
        is_video = x0.ndim == 5
        video_frames = x0.shape[2] if is_video else 0
        is_multiframe_video = is_video and video_frames > 1
        eff_video_frames = video_frames if is_multiframe_video else 0
        if not checked_cache and self.check_use_cached(
            pcfg.get_throttle(
                fallback=using_fallback,
                video=is_video and is_multiframe_video,
            ),
        ):
            return self.cached_copy
        if using_fallback:
            return self.fallback_previewer(x0, quiet=True)
        used_fallback = False
        start_time = time()
        try:
            dargs = (
                self._decode_latent_taevid(x0)
                if is_video
                else self._decode_latent_taesd(x0)
            )
            result = self.decoded_to_image(
                *dargs,
                video_frames=eff_video_frames,
            )
        except torch.OutOfMemoryError:
            self.oom_count += 1
            used_fallback = True
            result = self.fallback_previewer(x0)
        if pcfg.verbose:
            tqdm.write(
                f"[Bleh] used fallback: {used_fallback}, decode time: {time() - start_time:0.2f}",
            )
        return result


def find_previewer_model(basename: str | None) -> str | None:
    if basename is None:
        return None
    for ext in ("safetensors", "st", "pth"):
        maybe_filename = folder_paths.get_full_path("vae_approx", f"{basename}.{ext}")
        if maybe_filename:
            return maybe_filename
    return None


def bleh_get_previewer(
    device,
    latent_format: latent_formats.LatentFormat,
    *args: Any,
    **kwargs: Any,
) -> object | None:
    def orig_get_previewer():
        return _ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)

    pcfg = settings.SETTINGS.previews
    preview_method = comfy_args.preview_method

    if preview_method not in {
        LatentPreviewMethod.TAESD,
        LatentPreviewMethod.Auto,
        LatentPreviewMethod.Latent2RGB,
    }:
        return orig_get_previewer()

    format_name = latent_format.__class__.__name__.lower()
    if PREVIEWER_STATE.prefer_previewer in AMBIGUOUS_VIDEO_FORMATS.get(
        format_name,
        frozenset(),
    ):
        format_name = PREVIEWER_STATE.prefer_previewer
    if (
        not pcfg.enabled
        or format_name in pcfg.blacklist_formats
        or (pcfg.whitelist_formats and format_name not in pcfg.whitelist_formats)
    ):
        return orig_get_previewer()
    if format_name in AUDIO_LATENT_FORMAT_NAMES:
        return BetterPreviewer(latent_format=latent_format)
    vid_info = VIDEO_FORMATS.get(format_name)
    eff_latent_format = (
        vid_info.latent_format if vid_info is not None else latent_format
    )
    tae_model = None
    if preview_method in {LatentPreviewMethod.TAESD, LatentPreviewMethod.Auto}:
        if (
            vid_info is not None
            and vid_info.tae_model is not None
            and vid_info.tae_class is not None
        ):
            tae_model_path = find_previewer_model(str(vid_info.tae_model))
            tae_model = (
                vid_info.tae_class(
                    checkpoint_path=tae_model_path,
                    vmi=vid_info,
                    device=torch.device("cpu"),
                    decoder_time_upscale_level=pcfg.video_temporal_upscale_level,
                )
                if tae_model_path is not None
                else None
            )
        elif vid_info is None and eff_latent_format.taesd_decoder_name is not None:
            taesd_path = find_previewer_model(eff_latent_format.taesd_decoder_name)
            tae_model = (
                TAESD(
                    None,
                    taesd_path,
                    latent_channels=eff_latent_format.latent_channels,
                )
                if taesd_path is not None
                else None
            )
        if tae_model is not None:
            return BetterPreviewer(
                taesd=tae_model,
                latent_format=eff_latent_format,
                vid_info=vid_info,
            )
    # Using Latent2RGB either via setting or because no preview model.
    if eff_latent_format.latent_rgb_factors is not None:
        return BetterPreviewer(latent_format=latent_format)
    return orig_get_previewer()


def ensure_previewer():
    if latent_preview.get_previewer != bleh_get_previewer:
        latent_preview.BLEH_ORIG_get_previewer = _ORIG_GET_PREVIEWER
        latent_preview.get_previewer = bleh_get_previewer


ensure_previewer()
