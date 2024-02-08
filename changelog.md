# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20240208

* Added `BlehInsaneChainSampler` node.

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
