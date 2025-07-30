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
    "BlehCast": misc.BlehCast,
    "BlehCFGInitSampler": samplers.BlehCFGInitSampler,
    "BlehDeepShrink": deepShrink.DeepShrinkBleh,
    "BlehDisableNoise": misc.BlehDisableNoise,
    "BlehDiscardPenultimateSigma": misc.DiscardPenultimateSigma,
    "BlehEnsurePreviewer": misc.BlehEnsurePreviewer,
    "BlehForceSeedSampler": samplers.BlehForceSeedSampler,
    "BlehGlobalSageAttention": sageAttention.BlehGlobalSageAttention,
    "BlehHyperTile": hyperTile.HyperTileBleh,
    "BlehImageAsLatent": misc.BlehImageAsLatent,
    "BlehInsaneChainSampler": samplers.BlehInsaneChainSampler,
    "BlehLatentAsImage": misc.BlehLatentAsImage,
    "BlehLatentBlend": ops.BlehLatentBlend,
    "BlehLatentOps": ops.BlehLatentOps,
    "BlehLatentScaleBy": ops.BlehLatentScaleBy,
    "BlehModelPatchConditional": modelPatchConditional.ModelPatchConditionalNode,
    "BlehPlug": misc.BlehPlug,
    "BlehRefinerAfter": refinerAfter.BlehRefinerAfter,
    "BlehSageAttentionSampler": sageAttention.BlehSageAttentionSampler,
    "BlehSetSamplerPreset": samplers.BlehSetSamplerPreset,
    "BlehSetSigmas": misc.BlehSetSigmas,
    "BlehTAEVideoDecode": taevid.TAEVideoDecode,
    "BlehTAEVideoEncode": taevid.TAEVideoEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlehHyperTile": "HyperTile (bleh)",
    "BlehDeepShrink": "Kohya Deep Shrink (bleh)",
}

__all__ = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")
