import math
from time import time

import latent_preview
import torch
from comfy.model_management import device_supports_non_blocking
from PIL import Image

from .settings import SETTINGS

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl


class BetterTAESDPreviewer(_ORIG_PREVIEWER):
    def __init__(self, taesd):
        del taesd.taesd_encoder
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
        self.skip_upscale_layers = SETTINGS.btp_skip_upscale_layers
        self.preview_max_width = SETTINGS.btp_max_width
        self.preview_max_height = SETTINGS.btp_max_height
        self.throttle_secs = SETTINGS.btp_throttle_secs
        self.max_batch_preview = SETTINGS.btp_max_batch
        self.maxed_batch_step_mode = SETTINGS.btp_maxed_batch_step_mode
        self.max_batch_cols = SETTINGS.btp_max_batch_cols
        self.maybe_pop_upscale_layers()

    def maybe_pop_upscale_layers(self, *, width=None, height=None):
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

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return (
            preview_format,
            preview_image,
            min(
                max(*preview_image.size),
                max(self.preview_max_width, self.preview_max_height),
            ),
        )

    def check_use_cached(self):
        now = time()
        if (
            self.cached is not None and self.stamp is not None
        ) and now - self.stamp < self.throttle_secs:
            return True
        self.stamp = now
        return False

    def _decode_latent(self, x0):
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
        batch, _channels, height, width = x0.shape
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
        if self.skip_upscale_layers < 0:
            self.maybe_pop_upscale_layers(
                width=width * 8 * cols,
                height=height * 8 * rows,
            )
        return (
            (
                self.taesd.decode(x0)
                .movedim(1, -1)
                .add_(1)
                .mul_(0.5)
                .clamp_(min=0, max=1)
                .mul_(255)
                .detach()
            ),
            cols,
            rows,
        )

    def calc_cols_rows(self, batch_size, width, height):
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

    def decoded_to_image(self, samples, cols, rows):
        batch, height, width = samples.shape[:-1]
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

    def decode_latent_to_preview(self, x0):
        if self.check_use_cached():
            return self.cached
        if x0.shape[0] == 0:
            return self.blank  # Shouldn't actually be possible.
        return self.decoded_to_image(*self._decode_latent(x0))


if not isinstance(latent_preview.TAESDPreviewerImpl, BetterTAESDPreviewer):
    latent_preview.BLEH_ORIG_TAESDPreviewerImpl = _ORIG_PREVIEWER
    latent_preview.TAESDPreviewerImpl = BetterTAESDPreviewer
