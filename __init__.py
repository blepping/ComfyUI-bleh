from .py import settings

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview

from .py import hypertile

NODE_CLASS_MAPPINGS = {
    "BlehHyperTile": hypertile.HyperTileBleh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperTile (bleh)": "HyperTile (bleh)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
