# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20240128

* Removed CUDA-specific stuff from TAESD previewer as the performance gains were marginal and it had a major effect on VRAM usage.
* (Hopefully) improved heuristics for batch preview layout.
* Added `maxed_batch_step_mode` setting for TAESD previewer.
* Fixed reversed HyperTile default start/end step values.
* Allow only applying HyperTile at a step interval.
