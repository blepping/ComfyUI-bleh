# BLEH

A ComfyUI nodes collection... eventually.

## Features

1. Better TAESD previews (see below)
2. Allow setting seed, timestep range and step interval for HyperTile (look for the [`BlehHyperTile`](#blehhypertile) node)
3. Allow applying Kohya Deep Shrink to multiple blocks, also allow gradually fading out the downscale factor (look for the [`BlehDeepShrink`](#blehdeepshrink) node)
4. Allow discarding penultimate sigma (look for the `BlehDiscardPenultimateSigma` node). This can be useful if you find certain samplers are ruining your image by spewing a bunch of noise into it at the very end (usually only an issue with `dpm2 a` or SDE samplers).
5. Allow more conveniently switching between samplers during sampling (look for the [BlehInsaneChainSampler](#blehinsanechainsampler) node).
6. Apply arbitrary model patches at an interval and/or for a percentage of sampling (look for the [BlehModelPatchConditional](#blehmodelpatchconditional) node).
7. Ensure a seed is set even when `add_noise` is turned off in a sampler. Yes, that's right: if you don't have `add_noise` enabled _no_ seed gets set for samplers like `euler_a` and it's not possible to reproduce generations. (look for the [BlehForceSeedSampler](#blehforceseedsampler) node). For `SamplerCustomAdvanced` you can use `BlehDisableNoise` to accomplish the same thing.
8. Allows swapping to a refiner model at a predefined time (look for the [BlehRefinerAfter](#blehrefinerafter) node).
9. Allow defining arbitrary model patches (look for the [BlehBlockOps](#blehblockops) node).

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
|`preview_device`|`null`|`null` (use the default device) or a string with a PyTorch device name like `"cpu"`, `"cuda:0"`, etc. Can be used to run TAESD previews on CPU or other available devices.|
|`skip_upscale_layers`|`0`|The TAESD model has three upscale layers, each doubles the size of the result. Skipping some of them will significantly speed up TAESD previews at the cost of smaller preview image results.|

These defaults are conservative. I would recommend setting `throttle_secs` to something relatively high (like 5-10) especially if you are generating batches at high resolution.

Slightly more detailed explanation for `maxed_batch_step_mode`: If max previews is set to `3` and the batch size is `15` you will see previews for indexes `0, 5, 10`. Or to put it a different way, it steps through the batch by `batch_size / max_previews` rounded up. This behavior may be useful for previewing generations with a high batch count like when using AnimateDiff.

More detailed explanation for skipping upscale layers: Latents (the thing you're running the TAESD preview on) are 8 times smaller than the image you get decoding by normal VAE or TAESD. The TAESD decoder has three upscale layers, each doubling the size: `1 * 2 * 2 * 2 = 8`. So for example if normal decoding would get you a `1280x1280` image, skipping one TAESD upscale layer will get you a `640x640` result, skipping two will get you `320x320` and so on. I did some testing running TAESD decode on CPU for a `1280x1280` image: the base speed is about `1.95` sec base, `1.15` sec with one upscale layer skipped, `0.44` sec with two upscale layers skipped and `0.16` sec with all three upscale layers popped (of course you only get a `160x160` preview at that point). The upshot is if you are using TAESD to preview large images or batches or you want to run TAESD on CPU (normally pretty slow) you would probably benefit from setting `skip_upscale_layers` to `1` or `2`. Also if your max preview size is `768` and you are decoding a `1280x1280` image, it's just going to get scaled down to `768x768` anyway.

### BlehModelPatchConditional

**Note**: Very experimental.

This node takes a `default` model and a `matched` model. When the interval or start/end percentage match, the `matched` model will apply, otherwise the `default` one will. This can be used to apply something like HyperTile, Self Attention Guidance or other arbitrary model patches conditionally.

The first sampling step that matches the timestep range always applies `matched`, after that the following behavior applies: If the interval is positive then you just get `matched` every `interval` steps. It is also possible to set interval to a negative value, for example `-3` would mean out of every three steps, the first two use `matched` and the third doesn't.

_Notes and limitations_: Not all types of model modifications/patches can be intercepted with a node like this. You also almost certainly can't use this to mix different models: both inputs should be instances of the same loaded model. It's also probably a bad idea to apply further patches on top of the `BlehModelPatchConditional` node output: it should most likely be the last thing before a sampler or something that actually uses the model.

### BlehHyperTile

Adds the ability to set a seed and timestep range that HyperTile gets applied for. *Not* well tested, and I just assumed the Inspire version works which may or may not be the case.

It is also possible to set an interval for HyperTile steps, this time it is just normal sampling steps that match the timestep range. The first sampling step that matches the timestep range always applies HyperTile, after that the following behavior applies: If the interval is positive then you just get HyperTile every `interval` steps. It is also possible to set interval to a negative value, for example `-3` would mean out of every three steps, the first two have HyperTile and the third doesn't.

**Note**: Timesteps start from 999 and count down to 0 and also are not necessarily linear. Figuring out exactly which sampling step a timestep applies
to is left as an exercise for you, dear node user. As an example, Karras and exponential samplers essentially rush to low timesteps and spend quite a bit of time there.

HyperTile credits:

The node was originally taken by Comfy from taken from: https://github.com/tfernd/HyperTile/

Then the Inspire node pack took it from the base ComfyUI node: https://github.com/ltdrdata/ComfyUI-Inspire-Pack

Then I took it from the Inspire node pack. The original license was MIT so I assume yoinking it into this repo is probably okay.

### BlehDeepShrink

AKA `PatchModelAddDownScale` AKA Kohya Deep Shrink. Compared to the built-in Deep Shrink node this version has the following differences:

1. Instead of choosing a block to apply the downscale effect to, you can enter a comma-separated list of blocks. This may or not actually be useful but it seems like you can get interesting effects applying it to multiple blocks. Try `2,3` or `1,2,3`.
2. Adds a `start_fadeout_percent` input. When this is less than `end_percent` the downscale will be scaled to end at `end_percent`. For example, if `downscale_factor=2.0`, `start_percent=0.0`, `end_percent=0.5` and `start_fadeout_percent=0.0` then at 25% you could expect `downscale_factor` to be around `1.5`. This is because we are deep shrinking between 0 and 50% and we are halfway through the effect range. (`downscale_factor=1.0` would of course be a no-op and values below 1 don't seem to work.)
3. Expands the options for upscale and downscale types, you can also turn on antialiasing for `bicubic` and `bilinear` modes.

*Notes*: It seems like when shrinking multiple blocks, blocks downstream are also affected. So if you do x2 downscaling on 3 blocks, you are going to be applying `x2 * 3` downscaling to the lowest block (and maybe downstream ones?). I am not 100% sure how it works, but the takeway is you want to reduce the downscale amount when you are downscaling multiple blocks. For example, using blocks `2,3,4` and a downscale factor of `2.0` or `2.5` generating at 3072x3072 seems to work pretty well. Another note is schedulers that move at a steady pace seem to produce better results when fading out the deep shrink effect. In other words, exponential or Karras schedulers don't work well (and may produce complete nonsense). `ddim_uniform` and `sgm_uniform` seem to work pretty well and `normal` appears to be decent.

Deep Shrink credits:

Adapted from the ComfyUI source which I presume was adapted from the version Kohya initially published.

### BlehInsaneChainSampler

A picture is worth a thousand words, so:

![Insane chain example](assets/insane_chain_example.png)

This will use `heunpp2` for the first five steps, `euler_ancestral` for the next five, and `dpmpp_2m` for however many remain.

This is basically the same as chaining a bunch of samplers together and manually setting the start/end steps.

**Note**: Even though the `dpmpp_2m` insane chain sampler node has `steps=1` it will run for five steps. This is because the requirement of fifteen total steps must be fulfilled and... you can't really sample stuff without a sampler. Also note progress might be a little weird splitting sampling up like this.

### BlehForceSeedSampler

Currently, the way ComfyUI's advanced and custom samplers work is if you turn off `add_noise` _no_ global RNG seed gets set. Samplers like `euler_a` use this (SDE samplers use a different RNG method and aren't subject to this issue). Anyway, the upshot is you will get a different generation every time regardless of what the seed is set to. This node simply wraps another sampler and ensures that the seed gets set.

### BlehDisableNoise

Basically the same idea as `BlehForceSeedSampler`, however it is usable with `SamplerCustomAdvanced`.


### BlehPlug

You can connect this node to any input and it will be the same as if the input had no connection. Why is this useful? It's mainly for [Use Everywhere](https://github.com/chrisgoringe/cg-use-everywhere) — sometimes it's desirable to leave an input unconnected, but if you have Use Everywhere broadcasting an output it can be inconvenient. Just shove a plug in those inputs.

### BlehRefinerAfter

Allows switching to a refiner model at a predefined time. There are three time modes:

* `timestep`: Note that this is not a sampling step but a value between `0` and `999` where `999` is the beginning of sampling
  and `0` is the end. It is basically equivalent to the percentage of sampling remaining - `999` = ~99.9% sampling remaining.
* `percent`: Value between `0.0` and `1.0` where `0.0` is the start of sampling and 1.0 is the end. Note that this is not
  based on sampling _steps_.
* `sigma`: Advanced option. If you don't know what this is you probably  don't need to worry about it.

**Note**: This only patches the unet apply function, most other stuff including conditioning comes from the base model so
you likely can only use this to swap between models that are closely related. For example, switching from SD 1.5 to
SDXL is not going to work at all.

### BlehBlockOps

Very experimental advanced node that allows defining model patches using YAML. This node is still under development and may be changed.

The top level YAML should consist of a list of objects with a condition `if`, a list of `ops` that run if the condition succeeds.
Objects `then` and `else` also take the same form as the top level object and apply when the `if` condition matches (or not in the case of `else`).

All object fields (`if`, `then`, `else`, `ops`) are optional. An empty object is valid, it just doesn't do anything.

```yaml
- if:
    cond1: [value1, value2]
    cond2: value # Values may be specified as a list or single item.
  ops: [[opname1, oparg1, oparg2], [opname2, oparg1, oparg2]]
  then:
    if: [[opname1, oparg1, oparg2]] # Conditions may also be specified as a list.
    ops: [] # and so on
    else:
      ops: []
    # then and else may also be nested to an arbitrary depth.
```

*Note*: Blocks match by default, conditions restrict them. So a block with no `if` matches everything.

<details>

<summary>Expand to see full node documentation</summary>

#### Conditions

**`type`**: One of `input`, `input_after_skip`, `middle`, `output` (preceding are block patches), `latent`, `post_cfg`.
**Note**: ComfyUI doesn't allow patching the middle blocks by default, this feature is only available if you have
[FreeU Advanced](https://github.com/WASasquatch/FreeU_Advanced) installed and enabled. (It patches ComfyUI to support patching
the middle blocks.)

**`block`**: The block number. Only applies when type is `input`, `input_after_skip`, `middle` or `output`.

**`stage`**: The model stage. Applies to the same types as `block`. You can think of this in terms of FreeU's `b1`, `b2` - the number is the stage.

**`percent`**: Percentage of sampling completed as a number between `0.0` and `1.0`. Note that this is sampling percentage, not percentage of steps.
Does not apply to type `latent`.

**`from_percent`**: Matches when sampling is greater or equal to the percent. Same restrictions as `percent`.

**`to_percent`**: Matches when sampling is less or equal to the percent. Same restrictions as `from_percent`.

**`step`**: Only applies when sigmas are connected to the `BlehBlockOps` node. A step will be determined as the index of the closest
matching sigma. In other words, if you don't connect sigmas that exactly match the sigmas used for sampling you won't get accurate steps.
Does not apply to type `latent`.

**`step_exact`**: Same restrictions as `step`, however will only be set if the current sigma _exactly_ matches a step. Otherwise the
value will be `-1`.

**`from_step`**: As above, but matches when the step is greater or equal to the value.

**`from_step`**: As above, but matches when the step is less or equal to the value.

**`step_interval`**: Same restrictions as the other step condition types. Matches when the step modulus interval is 0. In other words,
every other step starting from the first step you'd use an interval of `2` and the `then` branch (since `1 % 2 == 1` which is not 0).

**`cond`**: Generic condition, has two forms:

*Comparison*: Takes three arguments: comparison type (`eq`, `ne`, `gt`, `lt`, `ge`, `le`), a condition type with
a numeric value (`block`, `stage`, `percent`, `step`, `step_exact`) and a value or list of values to compare with.

Example:
```yaml
- if: [cond, [lt, percent, 0.35]]
```

*Logic*: Takes a logic operation type (`not`, `and`, `or`) and a list of condition blocks. **Note**: The logic operation is applied
to the result of the condition block and not the fields within it.

Example:
```yaml
- if:
    cond: [not,
      [cond, [or,
        [cond, [lt, step, 1]],
        [cond, [gt, step, 5]],
      ]]
    ] # A verbose way of expressing step >= 1 and step <= 5
- if:
    - [cond, [ge, step, 1]]
    - [cond, [le, step, 5]] # Same as above
- if: [[from_step, 1], [to_step, 5]] # Also same as above
```

#### Operations

Operations mostly modify a target which can be `h` or `hsp`. `hsp` is only a valid target when `type` is `output`. I think it has something
to do with skip connections but I don't know the specifics. It's important for FreeU.

Default values are show in parenthesis next to the operation argument name. You may supply an incomplete argument list,
in which case default values will be used for the remaining arguments. Ex: `[flip]` is the same as `[flip, h]`. You may
also specify the arguments as a map, keys that aren't included will use the default values. Ex: `[flip, {direction: h}]`

**`slice`**: Applies a filtering operation on a slice of the target.

1. `scale`(`1.0`): Slice scale, `1.0` would mean apply to 100% of the target, `0.5` would mean 50% of it.
2. `strength`(`1.0`): Scales the target. `1.0` would mean 100%.
3. `blend`(`1.0`): Ratio of the transformed value to blend in. `1.0` means replace it with no blending.
4. `blend_mode`(`bislerp`): See the blend mode section.
5. `use_hidden_mean`(`true`): No idea what this does really, but FreeU V2 uses it when slicing and V1 doesn't.

**`ffilter`**: Applies a Fourier filter operation to the target.

1. `scale`(`1.0`): Scales the target. `1.0` would mean 100%.
2. `filter`(`none`): May be a string with a predefined filter name or a list of lists defining filters. See the filter section.
3. `filter_strength`(`0.5`): Strength of the filter. `1.0` would mean to apply it at 100%.
4. `threshold`(`1`): Threshold for the Fourier filter. This generally should be 1.

**`scale_torch`**: Scales the target up or down, using PyTorch's `interpolate` function.

1. `type`(`bicubic`): One of `bicubic`, `nearest`, `bilinear` or `area`.
2. `scale_width`(`1.0`): Ratio to scale the width. `2.0` would mean double it, `0.5` would mean half of it.
3. `scale_height`(`1.0`): As above.
4. `antialias`(`false`): `true` to apply antialiasing after scaling or `false`.

**`unscale_torch`**: Scale the target to be the same size as `hsp`. Only can be used when the target isn't `hsp` and condition `type` is `output`.
Can be used to reverse a `scale` or `scale_torch` operation without having to worry about calculating the ratios to get the original size back.

1. `type`(`bicubic`): Same as `scale_torch`.
2. `antialias`(`false`): Same as `scale_torch`.

**`scale`**: Scales the target up or down using various functions. See the scaling functions section.

1. `type_width`(`bicubic`): Scaling function to use for width. Note if the type is one of the ones from `scale_torch` it cannot be combined with other scaling functions.
2. `type_height`(`bicubic`): As above.
3. `scale_width`(`1.0`): Ratio to scale the width. `2.0` would mean double it, `0.5` would mean half of it.
4. `scale_height`(`1.0`): As above.
5. `antialias_size`(`0`): Size of the antialias kernel. Between 1 and 7 inclusive. Higher numbers seem to increase blurriness.

**`unscale`**: Like `unscale_torch` except it supports more scale functions and can specify width/height scale function independently.
Same restriction as `scale`.

1. `type_width`(`bicubic`): Scaling function to use for width. Note if the type is one of the ones from `scale_torch` it cannot be combined with other scaling functions.
2. `type_height`(`bicubic`): As above.
3. `antialias_size`(`0`): Size of the antialias kernel. Between 1 and 7 inclusive. Higher numbers seem to increase blurriness.

**`flip`**: Flips the target.

1. `direction`(`h`): `h` for horizontal flip, `v` for vertical. Note that latents generally don't tolerate being flipped very well.

**`rot90`**: Does a 90 degree rotation of the target.

1. `count`(`1`): Number of times to rotate (can also be negative). Note that if you rotate in a way that makes the tensors not match then stuff will probably break.
   also as with `flip` it generally is pretty destructive to latents.

**`roll`**: Rotates the values in a dimension of the target.

1. `direction`(`c`): `horizontal`, `vertical`, `channels`. Note that when `type` is `input`, `input_after_skip`, `middle` or `output` you aren't actually dealing
   with a latent. The second dimension ("channels") is actually the features in the layer. Rotating them can produce some pretty weird effects.
2. `amount`(`1`): If it's a number greater than `-1.0` and less than `1.0` this will rotate forward or backward by a percentage of the size. Otherwise it is
   interpreted as the number of items to rotate forward or backward.

**`roll_channels`**: Same as `roll` but you only specify the count, it always targets channels and you can't use percentages.

1. `count`(`1`): Number of channels to rotate. May be negative.

**`target_skip`**: Changes the target.

1. `active`(`true`): If `true` will target `hsp`, otherwise will target `h`. Targeting `hsp` is only allowed when `type` is `output`, no effect otherwise.

**`multiply`**: Multiply the target by the value.

1. `factor`(`1.0`): Multiplier. `2.0` would double all values in the target.

**`antialias`**: Applies an antialias effect to the target. Works the same ase with `scale`.

1. `size`(`7`): The antialias kernel size as a number between 1 and 7.

**`noise`**: Adds noise to the target. Can only be used when sigmas are connected. Noise will be scaled by `sigma - sigma_next`.

1. `scale`(`0.5`): Additionally scale the noise by the supplied factor. `1.0` would mean no scaling, `2.0` would double it, etc.
2. `type`(`gaussian`): Only `gaussian` unless [ComfyUI-sonar](https://github.com/blepping/ComfyUI-sonar) is installed and active, otherwise
   you may use the additional noise types Sonar provides.
3. `scale_mode`(`sigdiff`): `sigdiff` scales the noise by the current sigma minus the next (requires sigmas connected),
   `sigma` scales by the current sigma, `none` or an invalid type uses no scaling (you get exactly `noise * scale`).

**`debug`**: Outputs some debug information about the state.

**`blend_op`**: Allows applying a blend function to the result of another operation.

1. `blend`(`1.0`): Ratio of the transformed value to blend in.
2. `blend_mode`(`bislerp`): See the blend mode section.
3. `ops`(empty): The operation as a list, with the name first. i.e. `[blend_op, 0.5, inject, [multiply, 0.5]]`. May also be a list of operations.

**`pad`**: Pads the target.

1. `mode`(`reflect`): One of `constant`, `reflect`, `replicate`, `circular` - see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
2. `top`(`0`): Amount of top padding. If this is a floating point value, it will be treated as a percentage of the dimension.
3. `bottom`(`0`): " " "
4. `left`(`0`): " " "
5. `right`(`0`): " " "
6. `constant`(`0`): Constant value to use, only applies when mode is `constant`.

_Note_: If you pad `input` (rather than `input_after_skip`) then you will need to crop the corresponding block in `output`
for both `h` and `hsp` (i.e. with `target_skip`).

**`crop`**: Crops the target.

1. `top`(`0`): Items to crop from the top. If this is a floating point value, it will be treated as a percentage of the dimension.
2. `bottom`(`0`): " " "
3. `left`(`0`): " " "
4. `right`(`0`): " " "

**`mask_example_op`**: Applies providing a mask by example and masks the result of an operation or list of operations.

1. `scale_mode`(`bicubic`) type: Same as with `scale`.
2. `antialias`(`7`) size: Same as with `scale`.
3. `mask`(mask targeting corners): A two dimensional list of mask values. See below.
4. `ops`(empty): Same as with `blend_op`.

Simple example of a mask:

```plaintext
[ [1.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0],
  [1.0, 0.0, 0.0, 1.0],
]
```

With this mask, the result of the mask ops will be applied at full strength to the corners. The mask is scaled up to
the size of the target tensor, so with this example the masked corners will be proportionately quite large if the
latent or tensor is much bigger than the mask. There are two convenience tricks for defining larger masks without
having to specify each value:

* If the first element in a row is `"rep"` then the second element is interpreted as a row repeat count and the
  rest of the items in the row constitute the row. Ex: `["rep", 2, 1, 0, 1]` expands to two rows of `1, 0, 1`.
* If a column item is a list, the first element is interpreted as the repeat count and the remaining items are repeated
  however many times. Ex: `[2, 1.2, 0.5]` as a column would expand to `1.2, 0.5, 1.2, 0.5`.

These two shortcuts can be combined. A mask of `[["rep", 2, 1, [3, 0], 2]]` expands to:

```plaintext
[
  [1, 0, 0, 0, 2],
  [1, 0, 0, 0, 2],
]
```

#### Blend Modes

1. bislerp: Interpolates between tensors a and b using normalized linear interpolation.
2. colorize: Supposedly transfers color. May or may not work that way.
3. cosinterp: Cosine interpolation.
4. cuberp
5. hslerp: Hybrid Spherical Linear Interporation, supposedly smooths transitions between orientations and colors.
6. inject: Inject just adds the value scaled by the ratio, so if ratio is `1.0` this simply adds it.
7. lerp: Linear interpolation.
8. lineardodge: Supposedly simulates a brightning effect.

#### Filters

1. none
2. bandpass
3. lowpass: Allows low frequencies and suppresses high frequencies.
4. highpass: Allows high frequencies and suppresses low frequencies.
5. passthrough: Maybe doesn't do anything?
6. gaussianblur: Blur.
7. edge: Edge enhance.
8. sharpen: Sharpens the target.
9. multilowpass: The multi versions apply to multiple bands.
10. multihighpass
11. multipassthrough
12. multigaussianblur
13. multiedge
14. multisharpen

Custom filters may also be defined. For example, `gaussianblur` in the YAML filter definition would be `[[10,0.5]]`,
`sharpen` would be `[[10, 1.5]]`.

#### Scaling Functions

1. bicubic: Generally the best option.
2. bilinear
3. nearest-exact
4. area
5. bislerp: Interpolates between tensors a and b using normalized linear interpolation.
6. colorize: Supposedly transfers color. May or may not work that way.
7. hslerp: Hybrid Spherical Linear Interporation, supposedly smooths transitions between orientations and colors.
8. bibislerp: Uses bislerp as the slerp function in bislerp. When slerping once just isn't enough.
9. cosinterp: Cosine interpolation.
10. cuberp: Cubic interpolation.
11. inject: Adds the value scaled by the ratio. Probably not the best for scaling.
12. lineardodge: Supposedly simulates a brightning effect.

#### Examples

**FreeU V2**

```yaml
# FreeU V2 b1=1.1, b2=1.2, s1=0.9, s2=0.2
- if:
    type: output
    stage: 1
  ops:
    - [slice, 0.75, 1.1, 1, null, true]
    - [target_skip, true]
    - [ffilter, 0.9, none, 1.0, 1]
- if:
    type: output
    stage: 2
  ops:
    - [slice, 0.75, 1.2, 1, null, true]
    - [target_skip, true]
    - [ffilter, 0.2, none, 1.0, 1]
```

**Kohya Deep Shrink**

```yaml
# Deep Shrink, downscale 2, apply up to 35%.
- if:
    type: input_after_skip
    block: 3
    to_percent: 0.35
  ops: [[scale, bicubic, bicubic, 0.5, 0.5, 0]]
- if:
    type: output
  ops: [[unscale, bicubic, bicubic, 0]]
```

</details>

### BlehLatentOps

Basically the same as BlehBlockOps, except the condition `type` will be `latent`. Obviously stuff involving steps, percentages, etc does not apply.
This node allows you to apply the blending/filtering/scaling operations to a latent.

## Credits

Latent blending and scaling and filter functions based on implementation from https://github.com/WASasquatch/FreeU_Advanced - thanks!
