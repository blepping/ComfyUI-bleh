# BLEH

A ComfyUI nodes collection... eventually.

## Features

1. Better TAESD previews (see below)
2. Allow setting seed and timestep range for HyperTile (look for the `BlehHyperTile` node)

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
