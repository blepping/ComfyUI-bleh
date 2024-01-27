# BLEH

ComfyUI nodes collection... eventually.

## Features

1. Better TAESD previews (see below)
2. Allow setting seed and timestep range for HyperTile (look for the `BlehHyperTile` node)

## Configuration

Copy `blehconfig.json.example` to `blehconfig.json` in the node repo directory and edit the copy.

Restart ComfyUI to apply new changes.

### Better TAESD previews

* Supports setting max preview size (ComfyUI default is hardcoded to 512 max).
* Supports showing previews for more than the first latent in the batch.
* Supports throttling previews. Do you really need your expensive TAESD preview to get updated 3 times a second?
* Supports using CUDA streams to avoid waiting for a synchronize. This might be faster.

Current defaults from `blehconfig.json`

|Key|Default|Description|
|-|-|-|
|`enabled`|`true`|Toggles whether enhanced TAESD previews are enabled|
|`max_size`|`768`|Max width or height for previews. Note this does not affect TAESD decoding, just the preview image|
|`max_batch`|`4`|Max number of latents in a batch to preview|
|`max_batch_cols`|`2`|Max number of columns to use when previewing batches|
|`throttle_secs`|`1`|Max frequency to decode the latents for previewing. `0.25` would be every 1/4 sec, `2` would be only once every two seconds|
|`use_cuda`|`true`|Use special logic for CUDA (and maybe pretend-CUDA like ROCM) to reduce the performance impact of preview generation|

I would recommend setting `throttle_secs` to something relatively high like 5-10 sec especially if you are generating batches at high resolution.

### BlehHyperTile

Adds the ability to set a seed and timestep range that HyperTile gets applied for. *Not* well tested, and I just assumed the Inspire version works which may or may not be the case.

**Note**: Timesteps start from 999 and count down to 0 and also are not necessarily linear. Exactly what sampling step a timestep applies
to is left as an exercise for you, dear node user.

HyperTile credits:

The node was originally taken by Comfy from taken from: https://github.com/tfernd/HyperTile/

Then the Inspire node pack took it from the base ComfyUI node: https://github.com/ltdrdata/ComfyUI-Inspire-Pack

Then I took it from the Inspire node pack. The original license was MIT so I assume yoinking it into this repo is probably okay.
