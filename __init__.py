from .py import settings

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview  # noqa: F401

from .py.nodes import (
    deepShrink,
    hyperTile,
    modelPatchConditional,
    ops,
    refinerAfter,
    samplers,
    sigmas,
)

NODE_CLASS_MAPPINGS = {
    "BlehBlockOps": ops.BlehBlockOps,
    "BlehDeepShrink": deepShrink.DeepShrinkBleh,
    "BlehDiscardPenultimateSigma": sigmas.DiscardPenultimateSigma,
    "BlehForceSeedSampler": samplers.BlehForceSeedSampler,
    "BlehHyperTile": hyperTile.HyperTileBleh,
    "BlehInsaneChainSampler": samplers.BlehInsaneChainSampler,
    "BlehLatentScaleBy": ops.BlehLatentScaleBy,
    "BlehLatentOps": ops.BlehLatentOps,
    "BlehModelPatchConditional": modelPatchConditional.ModelPatchConditionalNode,
    "BlehRefinerAfter": refinerAfter.BlehRefinerAfter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
