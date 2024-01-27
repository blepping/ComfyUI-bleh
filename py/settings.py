import json
import os
from pathlib import Path

class Settings:
  def __init__(self, obj = {}):
    btp = obj.get("betterTaesdPreviews", None)
    if btp is None:
      self.btp_enabled = False
    else:
      self.btp_enabled = True
      self.btp_max_size = btp.get("max_size", 768)
      self.btp_max_batch = btp.get("max_batch", 4)
      self.btp_max_batch_cols = btp.get("max_batch_cols", 2)
      self.btp_throttle_secs = btp.get("throttle_secs", 1)
      self.btp_use_cuda = btp.get("use_cuda", True)

SETTINGS = None

def load_settings():
  global SETTINGS
  my_path = Path(os.path.abspath(os.path.dirname(__file__)))
  cfg_path = my_path.parent / "blehconfig.json"
  try:
    with open(cfg_path, "r") as fp:
      SETTINGS = Settings(json.load(fp))
  except OSError:
    SETTINGS = Settings()
  return SETTINGS
