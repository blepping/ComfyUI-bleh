from .py import settings

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview  # noqa: F401

from .py.nodes import (
    blockOps,
    deepShrink,
    hyperTile,
    modelPatchConditional,
    refinerAfter,
    samplers,
    sigmas,
)

NODE_CLASS_MAPPINGS = {
    "BlehBlockOps": blockOps.BlehBlockOps,
    "BlehDeepShrink": deepShrink.DeepShrinkBleh,
    "BlehDiscardPenultimateSigma": sigmas.DiscardPenultimateSigma,
    "BlehForceSeedSampler": samplers.BlehForceSeedSampler,
    "BlehHyperTile": hyperTile.HyperTileBleh,
    "BlehInsaneChainSampler": samplers.BlehInsaneChainSampler,
    "BlehLatentUpscaleBy": blockOps.BlehLatentUpscaleBy,
    "BlehModelPatchConditional": modelPatchConditional.ModelPatchConditionalNode,
    "BlehRefinerAfter": refinerAfter.BlehRefinerAfter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
