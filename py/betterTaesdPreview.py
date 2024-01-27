import math
from time import time

import numpy as np
import torch
from PIL import Image

import latent_preview

from .settings import SETTINGS

_ORIG_PREVIEWER = latent_preview.TAESDPreviewerImpl

class BetterTAESDPreviewer(_ORIG_PREVIEWER):
  def __init__(self, taesd):
    self.taesd = taesd
    self.stamp = None
    self.cached = None
    self.blank = Image.new("RGB", size=(1,1))
    self.stream = None
    self.prev_work = None
    self.use_cuda = SETTINGS.btp_use_cuda and hasattr(torch, "cuda") and torch.cuda.is_available()

  def decode_latent_to_preview_image(self, preview_format, x0):
    preview_image = self.decode_latent_to_preview(x0)
    return (preview_format, preview_image, min(max(*preview_image.size), SETTINGS.btp_max_size))

  def check_use_cached(self):
    now = time()
    if self.cached is not None and self.stamp is not None:
      if now - self.stamp < SETTINGS.btp_throttle_secs:
        return True
    self.stamp = now
    return False

  def _decode_latent(self, x0):
    samples = (self.taesd.decode(x0[:SETTINGS.btp_max_batch]) + 1.0) / 2.0
    samples = torch.clamp(samples, min = 0.0, max = 1.0) * 255.0
    return samples.to(dtype = torch.uint8).detach()

  def decode_latent_to_preview(self, x0, cpudev = torch.device("cpu")):
    if self.check_use_cached():
      return self.cached
    if x0.device == cpudev or not self.use_cuda:
      return self._decode_latent_to_preview_nocuda(x0)
    if self.stream is None:
      self.stream = torch.cuda.Stream()
    elif not self.stream.query():
      return self.cached or self.blank
    work = None
    if self.prev_work is not None:
      # We will only arrive here if the stream is ready. Sync just to be safe, should be instant.
      self.stream.synchronize()
      work = self.prev_work
      del self.prev_work
    # The default stream may be still processing the current step.
    self.stream.wait_stream(torch.cuda.default_stream())
    try:
      torch.cuda.set_stream(self.stream)
      self.prev_work = self._decode_latent(x0).to(device=cpudev, non_blocking=True)
    finally:
      torch.cuda.set_stream(torch.cuda.default_stream())
    return self.work_to_image(work) if work is not None else self.blank

  def work_to_image(self, samples):
    samples = tuple(np.moveaxis(x, 0, 2) for x in samples.numpy())
    batch_size = len(samples)
    height, width, _ = samples[0].shape
    if batch_size < 2:
      self.cached = Image.fromarray(samples[0])
      return self.cached
    cols = min(math.ceil(batch_size / 2), SETTINGS.btp_max_batch_cols)
    rows = math.ceil(batch_size / cols)
    self.cached = result = Image.new("RGB", size = (width * cols, height * rows))
    for idx in range(batch_size):
      result.paste(Image.fromarray(samples[idx]),
        box = ((idx % cols) * width, ((idx // cols) % rows) * height),
      )
    self.cached = result
    return result

  def _decode_latent_to_preview_nocuda(self, x0):
    return self.work_to_image(self._decode_latent(x0).cpu())

latent_preview.TAESDPreviewerImpl = BetterTAESDPreviewer