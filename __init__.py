from .py import settings
settings.load_settings()

if settings.SETTINGS.btp_enabled:
  from .py import betterTaesdPreview

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
