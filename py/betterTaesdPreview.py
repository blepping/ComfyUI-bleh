import math
from time import time

import latent_preview
import numpy as np
import torch
from PIL import Image

from .settings import SETTINGS

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl


class BetterTAESDPreviewer(_ORIG_PREVIEWER):
    def __init__(self, taesd):
        self.taesd = taesd
        self.stamp = None
        self.cached = None
        self.blank = Image.new("RGB", size=(1, 1))
        self.stream = None
        self.prev_work = None
        self.cpudev = torch.device("cpu")

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return (
            preview_format,
            preview_image,
            min(max(*preview_image.size), SETTINGS.btp_max_size),
        )

    def check_use_cached(self):
        now = time()
        if (
            self.cached is not None and self.stamp is not None
        ) and now - self.stamp < SETTINGS.btp_throttle_secs:
            return True
        self.stamp = now
        return False

    def _decode_latent(self, x0):
        max_batch = SETTINGS.btp_max_batch
        batch_size = x0.shape[0]
        if not SETTINGS.btp_maxed_batch_step_mode:
            indexes = range(min(max_batch, batch_size))
        else:
            indexes = range(
                0,
                batch_size,
                math.ceil(batch_size / max_batch),
            )[:max_batch]
        samples = (self.taesd.decode(x0[indexes, :]) + 1.0) / 2.0
        samples = torch.clamp(samples, min=0.0, max=1.0) * 255.0
        return samples.to(dtype=torch.uint8).detach()

    def calc_cols_rows(self, batch_size, width, height):
        max_cols = SETTINGS.btp_max_batch_cols
        ratio = height / width
        if ratio >= 1.75:
            # Very tall images - prioritize horizontal layout.
            cols = min(batch_size, max_cols)
        elif ratio <= 0.5:
            # Very wide images - prioritize vertical layout.
            cols = min(math.ceil(batch_size / 4), max_cols)
        else:
            cols = min(math.ceil(batch_size / ratio / 2.0), max_cols)
        rows = math.ceil(batch_size / cols)
        return cols, rows

    def decoded_to_image(self, samples):
        samples = tuple(np.moveaxis(x, 0, 2) for x in samples.numpy())
        batch_size = len(samples)
        height, width, _ = samples[0].shape
        if batch_size < 2:
            self.cached = Image.fromarray(samples[0])
            return self.cached
        cols, rows = self.calc_cols_rows(batch_size, width, height)

        self.cached = result = Image.new("RGB", size=(width * cols, height * rows))
        for idx in range(batch_size):
            result.paste(
                Image.fromarray(samples[idx]),
                box=((idx % cols) * width, ((idx // cols) % rows) * height),
            )
        self.cached = result
        return result

    def decode_latent_to_preview(self, x0):
        if self.check_use_cached():
            return self.cached
        return self.decoded_to_image(self._decode_latent(x0).cpu())


latent_preview.TAESDPreviewerImpl = BetterTAESDPreviewer
