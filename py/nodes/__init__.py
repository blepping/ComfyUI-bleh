from . import (
    blockCFG,
    deepShrink,
    hyperTile,
    misc,
    modelPatchConditional,
    ops,
    refinerAfter,
    sageAttention,
    samplers,
    taevid,
)

_blepping_integrations = None

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
    "BlehCFGInitSampler": samplers.BlehCFGInitSampler,
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
    "BlehEnsurePreviewer": misc.BlehEnsurePreviewer,
    "BlehTAEVideoDecode": taevid.TAEVideoDecode,
    "BlehTAEVideoEncode": taevid.TAEVideoEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")
