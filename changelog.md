# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20250504

This is a fairly large set of changes. Please create an issue if you experience problems.

* Support for TAE video models/video previewing.
* Added `BlehTAEVideoEncode` and `BlehTAEVideoDecode` nodes for fast video latent encoding/decoding.
* Added many more blend modes.
* Added `BlehEnsurePreviewer` node (for use when other custom node packs overwrite Bleh's previewer).

## 20250313

* Added OOM fallback to the previewer.
* Added ability to compile the previewer (and a few other related options).
* Added `BlehEnsurePreviewer` node that can be used to ensure Bleh's previewer is used if some other custom node overrides it.

## 20250119

* Fixed SageAttention 1.x support and setting `tensor_layout` should work properly for SageAttention 2.x now. Please create an issue if you experience problems.

## 20250114

* Added `BlehLatentBlend` node.
* Added `BlehCast` node that lets crazy people connect things that shouldn't be connected.
* Added `BlehSetSigmas` node.
* Some BlockOps functions have expanded capabilities now.

## 20250109

* The strategy the SageAttention nodes use to patch ComfyUI's attention should work for third-party custom nodes more reliably now. Please create an issue if you experience problems.

## 20241228

* The SageAttention nodes can now take advantage of SageAttention 2.0.1 which supports head sizes up to 128.
* Fixed an issue which prevented SageAttention from applying to models like Flux.

## 20241122

* Added the `BlehGlobalSageAttention` and `BlehSageAttentionSampler` nodes. See the README for details.

## 20241103

* Added the `BlehSetSamplerPreset` node and sampler presets feature.

## 20241021

* Added `seed_offset` parameter to `BlehDisableNoise` and `BlehForceSeedSampler` nodes. This is to avoid a case where the same noise would be used during sampling as the initial noise. **Note**: Changes seeds. You can set `seed_offset` to 0 to get the same behavior as before.

## 20240830

* Added the `BlehBlockCFG` node (see README for usage and details).
* More scaling/blending types. Some of them don't work well with scaling and will be filtered, you can set the environment variable `COMFYUI_BLEH_OVERRIDE_NO_SCALE` if you want the full list to be available (but you might just get garbage if you try to use them for scaling).
* Possibly better normalization function (may change seeds). Set the environment variable `COMFYUI_BLEH_ORIG_NORMALIZE` to disable.
* TAESD previews should be faster. Also now can dynamically set the number of upscale layers to skip based on the preview size limits. Additionally it's possible to set the max preview width/height seperately - see the YAML example config.

## 20240506

* Add many new scaling types.
* Add enhancements that can be combined with scaling, also `apply_enhancement` blockops function.

## 20240423

* Added `BlehPlug` and `BlehDisableNoise` (see README for usage and description).
* Increased the available upscale/downscale types for `BlehDeepShrink`.

## 20240412

* Added `BlehBlockOps` and `BlehLatentOps` nodes.

## 20240403

* Added `BlehRefinerAfter` node.

## 20240218

* Added `BlehForceSeedSampler` node.

## 20240216

* Added `BlehModelPatchConditional` node (see README for usage and description).

## 20240208

* Added `BlehInsaneChainSampler` node.
* Added ability to run previews on a specific device and skip TAESD upscale layers for increased TAESD decoding performance (see README).

## 20240202

* Added `BlehDiscardPenultimateSigma` node.

## 20240201

* Added `BlehDeepShrink` node (see README for usage and description)
* Add more upscale/downscale methods to the Deep Shrink node, allow setting a higher downscale factor, allow enabling antialiasing for `bilinear` and `bicubic` modes.

## 20240128

* Removed CUDA-specific stuff from TAESD previewer as the performance gains were marginal and it had a major effect on VRAM usage.
* (Hopefully) improved heuristics for batch preview layout.
* Added `maxed_batch_step_mode` setting for TAESD previewer.
* Fixed reversed HyperTile default start/end step values.
* Allow only applying HyperTile at a step interval.
