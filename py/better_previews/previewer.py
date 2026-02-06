from __future__ import annotations

import math
from io import BytesIO
from time import time
from typing import TYPE_CHECKING

import comfy.utils as comfy_utils
import folder_paths
import latent_preview
import torch
from aiohttp import web
from comfy import latent_formats
from comfy.cli_args import LatentPreviewMethod
from comfy.cli_args import args as comfy_args
from comfy.model_management import device_supports_non_blocking, vae_dtype
from comfy.taesd.taesd import TAESD
from PIL import Image
from server import PromptServer
from tqdm import tqdm

from ..settings import SETTINGS  # noqa: TID252
from .base import VIDEO_FORMATS, VideoModelInfo
from .tae_vid import TAEVid

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from comfy import latent_formats


class BlehPreviewerState:
    last_latent_shapes: tuple | None = None
    fps_override: float | None = None


PREVIEWER_STATE = BlehPreviewerState()

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl
_ORIG_GET_PREVIEWER = latent_preview.get_previewer


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
        # print("RECURSE", k)
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


def normalize_to_scale(latent, target_min, target_max, *, dim=(-3, -2, -1)):
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    normalized = (latent - min_val).div_(max_val - min_val)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


class LastPreview:
    image: bytes | None
    stamp: float | None
    content_type: str | None

    dum_page = """
    <html>
      <head>
        <title>bleh preview</title>
        <meta http-equiv="refresh" content="10">
      </head>
      <body style="background-color: #303030; margin: 0">
        <a href="/bleh/last_preview" target="_blank">
          <img src="/bleh/last_preview" style="width: 100%; height: auto; max-height: 100vh; object-fit: contain;">
        </a>
      </body>
    </html>
    """

    def __init__(self):
        self.image = None
        self.stamp = None
        self.content_type = None

    def update(
        self, *, image_bytes: bytes, content_type: str, stamp: float | None = None
    ):
        self.image = image_bytes
        self.stamp = time() if stamp is None else stamp
        self.content_type = content_type

    async def __call__(self, request: web.Request):
        if request.path.endswith(".html"):
            return web.Response(body=self.dum_page, content_type="text/html")
        if self.image is None or self.content_type is None:
            raise web.HTTPNotFound(reason="OHNO")
        return web.Response(body=self.image, content_type=self.content_type)


LAST_PREVIEW = LastPreview()
PromptServer.instance.routes.get("/bleh/last_preview")(LAST_PREVIEW)
PromptServer.instance.routes.get("/bleh/last_preview.html")(LAST_PREVIEW)


class ImageWrapper:
    def __init__(self, frames: tuple | Image, frame_duration: int = 250):
        self._frames = (frames,) if not isinstance(frames, (tuple, list)) else frames
        self._frame_duration = frame_duration

    def save(self, fp, format: str | None, **kwargs: dict):  # noqa: A002
        if len(self._frames) > 1:
            kwargs |= {
                "loop": 0,
                "save_all": True,
                "append_images": self._frames[1:],
                "duration": self._frame_duration,
            }
            format = "webp"
        if not SETTINGS.btp_publish_last_preview:
            return self._frames[0].save(fp, format, **kwargs)
        buf = BytesIO()
        result = self._frames[0].save(buf, format, **kwargs)
        # FIXME
        image_bytes = buf.getvalue()
        LAST_PREVIEW.update(image_bytes=image_bytes, content_type=f"image/{format}")
        fp.write(image_bytes)
        return result

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
        self.dtype = dtype
        self.device = device
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
        return x.add_(1.0).clamp_(0.0, 2.0).mul_(127.5).round_()


class ACEStepsPreviewerModel(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        height_factor: int = 4,
        width_factor: int = 1,
        normalize_dims: tuple = (-1,),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.normalize_dims = normalize_dims
        self.height_factor = height_factor
        self.width_factor = width_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, temporal = x.shape[0], x.shape[-1]
        x = normalize_to_scale(x, 0.0, 1.0, dim=self.normalize_dims) * 255.0
        x = x.reshape(batch, -1, temporal)
        if self.height_factor > 1:
            x = x.repeat_interleave(dim=1, repeats=self.height_factor)
        if self.width_factor > 1:
            x = x.repeat_interleave(dim=1, repeats=self.width_factor)
        return x[..., None].expand(*x.shape, 3)


class BetterPreviewer(_ORIG_PREVIEWER):
    def __init__(
        self,
        *,
        taesd: torch.nn.Module | None = None,
        latent_format: latent_formats.LatentFormat,
        vid_info: VideoModelInfo | None = None,
    ):
        self.orig_latent_format = latent_format
        self.latent_format = (
            latent_format if vid_info is None else vid_info.latent_format
        )
        self.latent_format_name = (
            "unknown"
            if latent_format is None
            else latent_format.__class__.__name__.lower()
        )
        self.spatial_compression = 8
        self.vid_info = vid_info
        self.fallback_previewer_model = None
        self.device = (
            None
            if SETTINGS.btp_preview_device is None
            else torch.device(SETTINGS.btp_preview_device)
        )
        dtype = (
            SETTINGS.btp_preview_dtype.lower()
            if SETTINGS.btp_preview_dtype is not None
            else None
        )
        self.dtype: str | torch.dtype | None = None
        if dtype in {"vae", "keep"}:
            self.dtype = dtype
        elif dtype in {"float32", "float16", "bfloat16"}:
            self.dtype = getattr(torch, dtype)
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
        self.oom_fallback = SETTINGS.btp_oom_fallback == "latent2rgb"
        self.oom_retry = SETTINGS.btp_oom_retry
        self.oom_count = 0
        self.skip_upscale_layers = SETTINGS.btp_skip_upscale_layers
        self.skip_upscale_layers_state: tuple[int, int] | tuple[None, None] | None = (
            None
        )
        self.preview_max_width = SETTINGS.btp_max_width
        self.preview_max_height = SETTINGS.btp_max_height
        self.throttle_secs = SETTINGS.btp_throttle_secs
        self.throttle_secs_fallback = SETTINGS.btp_throttle_secs_fallback
        self.max_batch_preview = SETTINGS.btp_max_batch
        self.maxed_batch_step_mode = SETTINGS.btp_maxed_batch_step_mode
        self.max_batch_cols = SETTINGS.btp_max_batch_cols
        self.compile_previewer = SETTINGS.btp_compile_previewer

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
                self.skip_upscale_layers < 0
                and self.skip_upscale_layers_state != (width, height)
            )
        if not need_refresh:
            return
        tqdm.write("Refreshing previewer")
        self.previewer_model = clone_module(self.orig_previewer_model).to(
            device=device,
            dtype=dtype,
        )
        if is_taesd:
            self.skip_upscale_layers_state = None
            self.maybe_pop_upscale_layers(width=width, height=height)
        if not self.compile_previewer:
            return
        tqdm.write("Compiling previewer")
        compile_kwargs = (
            {}
            if not isinstance(self.compile_previewer, dict)
            else self.compile_previewer
        )
        self.previewer_model = torch.compile(self.previewer_model, **compile_kwargs)

    # Popping upscale layers trick from https://github.com/madebyollin/
    def maybe_pop_upscale_layers(self, *, width=None, height=None) -> None:
        if self.skip_upscale_layers_state:
            return
        self.skip_upscale_layers_state = (width, height)
        skip = self.skip_upscale_layers
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
                max(self.preview_max_width, self.preview_max_height),
            ),
        )

    def check_use_cached(self) -> bool:
        now = time()
        throttle = (
            self.throttle_secs
            if self.previewer_model is not None
            else self.throttle_secs_fallback
        )
        if (
            self.cached is not None and self.stamp is not None
        ) and now - self.stamp < throttle:
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
        x0 = x0[self.calculate_indexes(x0.shape[0], is_video=is_video), :]
        batch = x0.shape[0]
        height, width = x0.shape[-2:]
        cols, rows = self.calc_cols_rows(
            min(batch, self.max_batch_preview),
            width,
            height,
        )
        return x0, cols, rows

    def prepare_previewer(
        self,
        x0: torch.Tensor,
        *,
        img_width: int | None = None,
        img_height: int | None = None,
    ) -> torch.Tensor:
        if self.dtype == "vae":
            dtype = vae_dtype(x0)
        elif self.dtype == "keep":
            dtype = x0.dtype
        else:
            dtype = self.dtype
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
            non_blocking=SETTINGS.btp_preview_non_blocking
            and device_supports_non_blocking(x0.device),
        )

    def _decode_latent_taevid(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x0.shape[-2:]
        x0 = self.prepare_previewer(x0)
        decoded = self.previewer_model.decode(
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
        return ImageWrapper(
            tuple(Image.fromarray(samples[idx]) for idx in range(batch)),
            frame_duration=max(1, int(frame_duration)),
        )

    def decoded_to_image(
        self,
        samples: torch.Tensor,
        cols: int,
        rows: int,
        *,
        video_frames: int = 0,
    ) -> Image | ImageWrapper:
        batch, (height, width) = samples.shape[0], samples.shape[-3:-1]
        samples = samples.to(
            device="cpu",
            dtype=torch.uint8,
            non_blocking=SETTINGS.btp_preview_non_blocking
            and device_supports_non_blocking(samples.device),
        ).numpy()
        if batch == 1:
            self.cached = ImageWrapper((Image.fromarray(samples[0]),))
            return self.cached
        if SETTINGS.btp_animate_preview == "both" or (
            video_frames != 0,
            SETTINGS.btp_animate_preview,
        ) in {(True, "video"), (False, "batch")}:
            return self.decoded_to_animation(samples, video_frames=video_frames)
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
        return ImageWrapper((result,))

    @torch.no_grad()
    def init_fallback_previewer(self, device: torch.device, dtype: torch.dtype) -> bool:
        if self.latent_format is None:
            return False
        if (
            self.fallback_previewer_model is not None
            and self.fallback_previewer_model.dtype == dtype
            and self.fallback_previewer_model.device == device
        ):
            return True
        if self.latent_format_name in {"aceaudio", "aceaudio15"}:
            self.fallback_previewer_model = ACEStepsPreviewerModel(
                device=device,
                dtype=dtype,
            )
            return True
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

    def ensure_x0_shape(self, x0: torch.Tensor) -> tuple[torch.Tensor, bool]:  # noqa: PLR0911
        expected_channels = self.latent_format.latent_channels
        expected_ndim = 2 + self.latent_format.latent_dimensions
        if x0.shape[0] == 0:
            return x0, False
        if self.latent_format_name == "aceaudio15" and x0.ndim == expected_ndim + 1:
            expected_ndim += 1
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
        if self.check_use_cached():
            return self.cached
        x0, can_preview = self.ensure_x0_shape(x0)
        if not can_preview:
            return self.blank
        if (self.oom_count and not self.oom_retry) or self.previewer_model is None:
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
            result = self.decoded_to_image(
                *dargs,
                video_frames=x0.shape[2] if is_video else 0,
            )
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
    def orig_get_previewer():
        return _ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)

    preview_method = comfy_args.preview_method

    if preview_method not in {
        LatentPreviewMethod.TAESD,
        LatentPreviewMethod.Auto,
        LatentPreviewMethod.Latent2RGB,
    }:
        return orig_get_previewer()

    format_name = latent_format.__class__.__name__.lower()
    if (
        not SETTINGS.btp_enabled
        or format_name in SETTINGS.btp_blacklist
        or (SETTINGS.btp_whitelist and format_name not in SETTINGS.btp_whitelist)
    ):
        return orig_get_previewer()
    if format_name in {"aceaudio", "aceaudio15"}:
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
            tae_model_path = folder_paths.get_full_path(
                "vae_approx",
                vid_info.tae_model,
            )
            tae_model = (
                vid_info.tae_class(
                    checkpoint_path=tae_model_path,
                    vmi=vid_info,
                    device=torch.device("cpu"),
                    decoder_time_upscale_level=SETTINGS.btp_video_temporal_upscale_level,
                )
                if tae_model_path is not None
                else None
            )
        elif vid_info is None and latent_format.taesd_decoder_name is not None:
            taesd_path = folder_paths.get_full_path(
                "vae_approx",
                f"{latent_format.taesd_decoder_name}.pth",
            )
            tae_model = (
                TAESD(
                    None,
                    taesd_path,
                    latent_channels=latent_format.latent_channels,
                )
                if taesd_path is not None
                else None
            )
        if tae_model is not None:
            return BetterPreviewer(
                taesd=tae_model,
                latent_format=latent_format,
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
