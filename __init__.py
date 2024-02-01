from .py import settings

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview  # noqa: F401

from .py import deepshrink, hypertile

NODE_CLASS_MAPPINGS = {
    "BlehHyperTile": hypertile.HyperTileBleh,
    "BlehDeepShrink": deepshrink.DeepShrinkBleh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
