# BLEH

A ComfyUI nodes collection... eventually.

## Features

1. Better TAESD previews (see below)
2. Allow setting seed, timestep range and step interval for HyperTile (look for the `BlehHyperTile` node)
3. Allow applying Kohya Deep Shrink to multiple blocks, also allow gradually fading out the downscale factor (look for the `BlehDeepShrink` node)

## Configuration

Copy either `blehconfig.yaml.example` or `blehconfig.json.example` to `blehconfig.yaml` or `blehconfig.json` respectively and edit the copy. When loading configuration, the YAML file will be prioritized if it exists and Python has YAML support.

Restart ComfyUI to apply any new changes.

### Better TAESD previews

* Supports setting max preview size (ComfyUI default is hardcoded to 512 max).
* Supports showing previews for more than the first latent in the batch.
* Supports throttling previews. Do you really need your expensive TAESD preview to get updated 3 times a second?

Current defaults:

|Key|Default|Description|
|-|-|-|
|`enabled`|`true`|Toggles whether enhanced TAESD previews are enabled|
|`max_size`|`768`|Max width or height for previews. Note this does not affect TAESD decoding, just the preview image|
|`max_batch`|`4`|Max number of latents in a batch to preview|
|`max_batch_cols`|`2`|Max number of columns to use when previewing batches|
|`throttle_secs`|`2`|Max frequency to decode the latents for previewing. `0.25` would be every quarter second, `2` would be once every two seconds|
|`maxed_batch_step_mode`|`false`|When `false`, you will see the first `max_batch` previews, when `true` you will see previews spread across the batch|

These defaults are conservative. I would recommend setting `throttle_secs` to something relatively high (like 5-10) especially if you are generating batches at high resolution.

Slightly more detailed explanation for `maxed_batch_step_mode`: If max previews is set to `3` and the batch size is `15` you will see previews for indexes `0, 5, 10`. Or to put it a different way, it steps through the batch by `batch_size / max_previews` rounded up. This behavior may be useful for previewing generations with a high batch count like when using AnimateDiff.

### BlehHyperTile

Adds the ability to set a seed and timestep range that HyperTile gets applied for. *Not* well tested, and I just assumed the Inspire version works which may or may not be the case.

It is also possible to set an interval for HyperTile steps, this time it is just normal sampling steps that match the timestep range. The first sampling step that matches the timestep range always applies HyperTile, after that the follow behavior applies: If the interval is positive then you just get HyperTile every `interval` steps. It is also possible to set interval to a negative value, for example `-3` would mean out of every three steps, the first two have HyperTile and the third doesn't.

**Note**: Timesteps start from 999 and count down to 0 and also are not necessarily linear. Exactly what sampling step a timestep applies
to is left as an exercise for you, dear node user. As an example, Karras and exponentially samplers essentially rush to low timesteps and spend quite a bit of time there.

HyperTile credits:

The node was originally taken by Comfy from taken from: https://github.com/tfernd/HyperTile/

Then the Inspire node pack took it from the base ComfyUI node: https://github.com/ltdrdata/ComfyUI-Inspire-Pack

Then I took it from the Inspire node pack. The original license was MIT so I assume yoinking it into this repo is probably okay.

### BlehDeepShrink

AKA `PatchModelAddDownScale` AKA Kohya Deep Shrink. Compared to the built-in Deep Shrink node this version has the
following differences:

1. Instead of choosing a block to apply the downscale effect to, you can enter a comma-separated list of blocks. This may or not actually be useful but it seems like you can get interesting effects applying it to multiple blocks. Try `2,3` or `1,2,3`.
2. Adds a `start_fadeout_percent` input. When this is less than `end_percent` the downscale will be scaled to end at `end_percent`. For example, if `downscale_factor=2.0`, `start_percent=0.0`, `end_percent=0.5` and `start_fadeout_percent=0.0` then at 25% you could expect `downscale_factor` to be around `1.5`. This is because we are deep shrinking between 0 and 50% and we are halfway through the effect range. (`downscale_factor=1.0` would of course be a no-op and values below 1 don't seem to work.)

Deep Shrink credits:

Adapted from the ComfyUI source which I presume was adapted from the version Kohya initially published.
