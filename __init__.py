from .py import settings

BLEH_VERSION = 1

settings.load_settings()

if settings.SETTINGS.btp_enabled:
    from .py import betterTaesdPreview  # noqa: F401

from .py.nodes import (
    blockCFG,
    deepShrink,
    hyperTile,
    misc,
    modelPatchConditional,
    ops,
    refinerAfter,
    sageAttention,
    samplers,
)

samplers.add_sampler_presets()

NODE_CLASS_MAPPINGS = {
    "BlehBlockCFG": blockCFG.BlockCFGBleh,
    "BlehBlockOps": ops.BlehBlockOps,
    "BlehDeepShrink": deepShrink.DeepShrinkBleh,
    "BlehDisableNoise": misc.BlehDisableNoise,
    "BlehDiscardPenultimateSigma": misc.DiscardPenultimateSigma,
    "BlehForceSeedSampler": samplers.BlehForceSeedSampler,
    "BlehGlobalSageAttention": sageAttention.BlehGlobalSageAttention,
    "BlehHyperTile": hyperTile.HyperTileBleh,
    "BlehInsaneChainSampler": samplers.BlehInsaneChainSampler,
    "BlehLatentOps": ops.BlehLatentOps,
    "BlehLatentScaleBy": ops.BlehLatentScaleBy,
    "BlehLatentBlend": ops.BlehLatentBlend,
    "BlehModelPatchConditional": modelPatchConditional.ModelPatchConditionalNode,
    "BlehPlug": misc.BlehPlug,
    "BlehRefinerAfter": refinerAfter.BlehRefinerAfter,
    "BlehSageAttentionSampler": sageAttention.BlehSageAttentionSampler,
    "BlehSetSamplerPreset": samplers.BlehSetSamplerPreset,
    "BlehCast": misc.BlehCast,
    "BlehSetSigmas": misc.BlehSetSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ("BLEH_VERSION", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")
