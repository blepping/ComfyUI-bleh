# Blend Modes

There are so many blend modes now that it's probably pretty overwhelming. Also a lot of them are junk/don't work well but can't be removed without breaking existing workflows that might use them. So here is some incomplete, low-effort documentation!

## Meta Modes

Every blend function takes at least three parameters: `a`, `b` and `t` (the ratio to blend). `lerp(a, b, 0.25)` would mean `a * 0.75 + b * 0.25`.

* `revWHATEVER` - Just flips the inputs, so if we were doing something like `lerp(a, b, 0.25)` (`a * 0.75 + b * 0.5`) the blend would be applied like `lerp(b, a, 0.25)`.
* `normWHATEVER` - Tries to scale the input to -1...1 and then uses a simple LERP to find a range for the output. Pretty much garbage.

Some modes also have suffixes. Not precisely a meta mode but this is probably the logical place to cover it.

* `_d1`, `_d2`, etc - Indicates the mode will operate on that dimension. Dimension 1 in this case, which is typically channels in most latents.
* `_copysign_a` - The blended result copies the sign from the `a` parameter.
* `_avoidsign_a` - The blended result avoids the sign of the `a` parameter. In other words, negate A and then copy the sign from it to the result.
* Digits like `_025` - Usually indicates a multiplier. `025` would stand for `0.25`.
* `_base_a` - Mostly for CFG type blend modes. This means if `a` is cond and `b` is uncond, a ratio of 0 will give you cond. Normal CFG is `lerp(uncond, cond, ratio)` so if ratio is 0 you get pure uncond.

## Custom Blend Parameters

Nodes taking a blend mode use a selection but you can define modes as a string (whitespace is ignored so I recommend a multiline string widget) and force the connection to the mode parameter with the `BlehCast` node.

Custom definitions use this syntax: `mode_name:param1=val1:param2=val2`. Integer and float values don't require any special handling. Boolean parameters use `true` and `false`. Nullable parameters can use `none`. An empty list is `()`. Lists are comma-separated and a singleton list is specified like `1,`. String literals (like mode names) should start with the caret, like `^whatever` otherwise they will be interpreted as a blend mode. Finally, some blend modes are wrappers for other blend modes. If it exists, a single trailing underscore will be stripped from the parameters when they are passed to the nested blend function. Not very convenient to use, but it allows specifying parameters where the names may clash. I.E. `some_mode:blend_mode=blah:blend_mode=other` would let `some_mode` use the `blend_mode` parameter and then pass the second to the `blah` blend mode handler.

This is clunky/inconvenient and pretty limited but it does allow specifying custom parameters in most cases.

All blend modes support some common parameters:

* `rev` - boolean. Example `lerp:rev=true`. Flips the inputs.
* `scale_multiplier` - float. Rescales the blend ratio. Example: `lerp:scale_multiplier=0.5`. `lerp(a, b, 1.0)` would result in `lerp(a, b, 0.5)`.
* `invert_scale` - float. Adjusts the blend ratio by doing `scale_value - ratio`. Example: `lerp:invert_scale=1.0`. `lerp(a, b, 0.4)` would result in `lerp(a, b, 0.6)` (1 - 0.4 == 0.6).
* `fork_rng` - boolean. Forks the random number generator state when calling the blend mode. Can be useful for probalistic blend modes like `problerp` which would perturb the RNG and affect stuff like noise used for ancestral sampling, changing your seed even if the ratio is tiny. Example: `problerp:fork_rng=true`

Realistically, the unique parameters for custom blend modes will probably never get documented. Unfortunately, you will need to read the source in `latent_utils.py` to find out what the options are.

## Generally Useful

* `lerp` and friends. Linear interpolation, the most common blend mode. CFG is also just LERP.
* `slerp` - Can sometimes be better than LERP for blending latents.
* `inject` - Simple addition. `inject(a, b, 0.3)` is just `a + b * 0.3`. Useful for adding stuff like a CFG diff.
* 

## Garbage/Redundant

* `bislerp_wrong` - This is just LERP with the useless normalize.
* `hslerp` (anything starting with HSLERP).
* `colorize` - Literally just LERP.
* `colordodge`, `difference`, `exclusion`, `glow`, `hardlight`, `linearlight`, `overlay`, `pinlight`, `reflect`, `screen`, `vividlight` - Photoshop filter type modes. They are designed for images and assume certain value ranges for the input so they are essentually useless for blending latents.
* `linear_dodge` - Same as `inject`. This is just scaled addition.
* `cosinesimilarity` - I actually like using this but it is roughly just a worse SLERP.

## Experimental/Exotic Modes

Note: Many of these are vibe-mathed, so the description explains what I was attempting to do and what I believe the mode does. I can't guarantee it is doing what it purports to because I don't always fully understand the math involved.

* `ortho` - Orthogonal addition. You may want to specify the dimension parameters to control how the normalization happens. Example: `ortho:start_dim=2:end_dim=3` - For 4D latents, this would normalize over the height/width dimensions. The ortho blend function has many parameters, you will need to look at the source to see them.
* `ortho_rescaled_lerpish` - Orthogonal blending means the parallel component gets thrown away, in other words you may end up adding less of something than you expected. This mode tries to adjust the result to target something like the result of a LERP.
* `ortho_dyn_lerp` - Similar to the previous, except it calculates how much `b` got scaled down and LERPs to compensate.
* `ortho_cfg` and `ortho_cfg_base_a` - Does ortho addition of the CFG diff.
* `contrastive_ortho_cfg` (and `_base_a`) - Mostly useful for CFG. Let's say we're in base A mode and `a` is cond and `b` is uncond. The mode takes a `a_ortho_scale` (what is unique to cond) and `b_ortho_scale` (what is unique to uncond) parameters. `b_ortho` (AKA what is unique to uncond) gets _subtracted. So `contrastive_ortho_cfg:a_ortho_scale=0.0:b_ortho_scale=1.0` would mean only subtract what is unique to uncond but _don't_ scale up what is unique to cond. The reverse is also possible, only enhance what is unique to cond but don't subtract uncond's unique features. Compare this with normal CFG: `cond + (cond - uncond) * scale` or in other words `cond + cond * scale - uncond * scale`. If `cond` is 1 and uncond is 0 then the result would effectively be `cond + cond * scale` - we scale up cond since there isn't a value on the uncond side to cancel it out.
* Modes starting with `slice` - Slices along a dimension. The ratio is the size of the dimension multiplied by the ratio. Example: `slice_d1` - slices dimension 1 (second dimension in zero-based dimension indexing). `slice_d1(a, b, 0.5)` would use the first 50% of channels from `a` and the remaining ones from `b`. Or maybe it's the other way around, I forget! These modes also have a `_flip` variant which would make it so the `a` result comes first or vice versa. The blend function supports various parameters from smoothing the result and controling the offset so if you wanted something like just the middle 25% of a dimension that is possible with custom parameters.
* Modes starting with `wavelet`. `wavelet_b_hi_100_lo_0` means 100% of the high frequency components of `b` and none of the low frequency. `wavelet_b_hi_0_lo_100` is the reverse. Can be interesting as a CFG function. The blend function supports many parameters such as setting the wavelet type and ratios. This requires wavelet support from the `pytorch_wavelets` package which is unfortunately broken with recent Python versions and hasn't been updated in years. You can use my pull with a fix (or the repo it links to): https://github.com/fbcotter/pytorch_wavelets/pull/66
* `pct_limited_025` - Limits the result to a maximum change (relative to `a` by default). This is a wrapper for other blend modes which defaults to LERP. Example: `pct_limited_025:diff_limit=0.25:blend_mode=inject` Let's say we do `blend(0.5, 100.0, 1.0)` Without limiting this is `0.5 + 100.0 * 1.0` which results in something like a 2,000% change. If we limit to adjusting `a` by 25% at most we get a limit of `0.125` (`0.5 * 0.25`) so the output is `0.5 + 0.125 == 0.625`. This blend function has various other parameters, for example to allow soft clamping, prevent sign flipping (which not limiting the actual value), calculating percentage change over a dimension rather than elementwise, etc.
* `distro_aligned` - By default matches the distribution of `b` to `a`. `distro_aligned_result` aligns the output from the blend to the distribution of `a`. The blend function supports many options controling which gets aligned to what else, how it occurs, etc. You will need to look at the source.
* `gaussian_aligned` and `gaussian_aligned_result`. The latter only aligns the result to the Gaussian distribution, the former aligns both `a` and `b` before blending and then also aligns the result. SDXL latents (and probably most latent formats?) should be in the Gaussian distribution (zero mean, std 1) so using this as a CFG function actually works pretty well.
* `rms_interpolation_lerpsign` - By default, this is roughly `lerp(a**2, b**2, ratio)**0.5` and then copies the sign from `lerp(a, b, ratio)`. There are many options to control what blend mode is used, the power, whether the input gets converted to absolute values and how the sign is restored (since `whatever**2` will always be positive). `magnitude_interpolation_lerpsign` is just a preset for this with power 1 and using absolute inputs.
* `moment_aligned` - Similar to `distro_aligned`, maybe just a worse version of it. Attempts to align the std and mean (of the input) to a reference. It does not align the result, though you could probably do that through nesting.
* `pythagorean_lerp` - Like LERP but attempts to preserve the variance of the inputs. Let's say ratio is 0.4, so the multiplier for `a` would be `0.6` and `b` would be `0.4`. We'd calculate a variance division `(0.6**2 + 0.4 ** 2)**0.5` (roughly `0.7211`) and scale the weights (`0.6 / 0.7211 == 0.832`, `0.4 / 0.7211 == 0.554`). If the ratio was `0.1` you'd get something like `a * 0.993 + b * 0.11`.
