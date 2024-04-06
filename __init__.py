from .py import settings

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview  # noqa: F401

from .py import (
    deepshrink,
    hypertile,
    modelPatchConditional,
    refinerAfter,
    samplers,
    sigmas,
)

NODE_CLASS_MAPPINGS = {
    "BlehHyperTile": hypertile.HyperTileBleh,
    "BlehDeepShrink": deepshrink.DeepShrinkBleh,
    "BlehDiscardPenultimateSigma": sigmas.DiscardPenultimateSigma,
    "BlehInsaneChainSampler": samplers.BlehInsaneChainSampler,
    "BlehForceSeedSampler": samplers.BlehForceSeedSampler,
    "BlehModelPatchConditional": modelPatchConditional.ModelPatchConditionalNode,
    "BlehRefinerAfter": refinerAfter.BlehRefinerAfter,
    "BlehLatentUpscaleBy": deepshrink.BlehLatentUpscaleBy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
