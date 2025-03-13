import math
from time import time
from typing import NamedTuple

import latent_preview
import torch
from comfy.latent_formats import LatentFormat
from comfy.model_management import device_supports_non_blocking
from PIL import Image
from tqdm import tqdm

from .settings import SETTINGS

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl
_ORIG_GET_PREVIEWER = latent_preview.get_previewer

LAST_LATENT_FORMAT = None


class FallbackPreviewerModel(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        latent_format: LatentFormat,
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


class BetterTAESDPreviewer(_ORIG_PREVIEWER):
    def __init__(self, taesd):
        del taesd.taesd_encoder
        self.latent_format = LAST_LATENT_FORMAT
        self.fallback_previewer_model = None
        self.device = (
            None
            if SETTINGS.btp_preview_device is None
            else torch.device(SETTINGS.btp_preview_device)
        )
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
        if skip == 0:
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
            preview_format,
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

    def prepare_decode_latent(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        max_batch = self.max_batch_preview
        batch = x0.shape[0]
        if not self.maxed_batch_step_mode:
            indexes = range(min(max_batch, batch))
        else:
            indexes = range(
                0,
                batch,
                math.ceil(batch / max_batch),
            )[:max_batch]
        x0 = x0[indexes, :]
        batch, (height, width) = x0.shape[0], x0.shape[-2:]
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

    def _decode_latent(self, x0: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x0, cols, rows = self.prepare_decode_latent(x0)
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
        if ratio >= 1.45:
            # Very tall images - prioritize horizontal layout.
            cols = min(math.ceil(batch_size / 2), max_cols)
        elif ratio <= 0.5:
            # Very wide images - prioritize vertical layout.
            cols = min(math.ceil(batch_size / 4), max_cols)
        else:
            cols = min(math.ceil(batch_size / math.ceil(ratio) / 2.0), max_cols)
        rows = math.ceil(batch_size / cols)
        return cols, rows

    def decoded_to_image(self, samples: torch.Tensor, cols: int, rows: int) -> Image:
        batch, (height, width) = samples.shape[0], samples.shape[-3:-1]
        samples = samples.to(
            device="cpu",
            dtype=torch.uint8,
            non_blocking=device_supports_non_blocking(samples.device),
        ).numpy()
        if batch == 1:
            self.cached = Image.fromarray(samples[0])
            return self.cached
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
                f"*** BlehBetterTAESDPreviews: Got out of memory error while decoding preview - {fallback_mode}.",
            )
        if not self.oom_fallback:
            return self.blank
        if not self.init_fallback_previewer(x0.device, x0.dtype):
            self.oom_fallback = False
            tqdm.write(
                "*** BlehBetterTAESDPreviews: Couldn't initialize fallback previewer, giving up on previews.",
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
        if self.oom_count and not self.oom_retry:
            return self.fallback_previewer(x0, quiet=True)
        try:
            return self.decoded_to_image(*self._decode_latent(x0))
        except torch.OutOfMemoryError:
            return self.fallback_previewer(x0)


def bleh_get_previewer_wrapper(
    device,
    latent_format: LatentFormat,
    *args: list,
    **kwargs: dict,
):
    global LAST_LATENT_FORMAT  # noqa: PLW0603
    LAST_LATENT_FORMAT = latent_format
    return _ORIG_GET_PREVIEWER(device, latent_format, *args, **kwargs)


if not isinstance(latent_preview.TAESDPreviewerImpl, BetterTAESDPreviewer):
    latent_preview.BLEH_ORIG_TAESDPreviewerImpl = _ORIG_PREVIEWER
    latent_preview.TAESDPreviewerImpl = BetterTAESDPreviewer

if latent_preview.get_previewer != bleh_get_previewer_wrapper:
    latent_preview.BLEH_ORIG_get_previewer = _ORIG_GET_PREVIEWER
    latent_preview.get_previewer = bleh_get_previewer_wrapper
