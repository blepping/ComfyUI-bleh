
# BlehBlockOps

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

See [Scaling Types](#scaling-types) below.

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

**`apply_enhancement`**: Applies an [enhancement](#enhancement-types) to the target.

1. `scale`: 1.0
2. `type`: korniabilateralblur

