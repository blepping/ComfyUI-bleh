# Adapted from the ComfyUI built-in node

from .. import latent_utils  # noqa: TID252


class DeepShrinkBleh:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"
    DESCRIPTION = "Model patch that enables generating at higher resolution than the model was trained for by downscaling the image near the start of generation."

    upscale_methods = (
        "bicubic",
        "nearest-exact",
        "bilinear",
        "area",
        "bislerp",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch",
                    },
                ),
                "commasep_block_numbers": (
                    "STRING",
                    {
                        "default": "3",
                        "tooltip": "A comma separated list of input block numbers, the default should work for SD 1.5 and SDXL.",
                    },
                ),
                "downscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 32.0,
                        "step": 0.1,
                        "tooltip": "Controls how much the block will get downscaled while the effect is active.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Start time as sampling percentage (not percentage of steps). Percentages are inclusive.",
                    },
                ),
                "start_fadeout_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "When enabled, the downscale_factor will fade out such that at end_percent it will be around 1.0 (no downscaling). May reduce artifacts... or cause them!",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "End time as sampling percentage (not percentage of steps). Percentages are inclusive.",
                    },
                ),
                "downscale_after_skip": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether the downscale effect occurs after the skip conection. Generally should be left enabled.",
                    },
                ),
                "downscale_method": (
                    latent_utils.UPSCALE_METHODS,
                    {
                        "default": "bicubic",
                        "tooltip": "Mode used for downscaling. Bicubic is generally a safe choice.",
                    },
                ),
                "upscale_method": (
                    latent_utils.UPSCALE_METHODS,
                    {
                        "default": "bicubic",
                        "tooltip": "Mode used for upscaling. Bicubic is generally a safe choice.",
                    },
                ),
                "antialias_downscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental option to anti-alias (smooth) the latent after downscaling.",
                    },
                ),
                "antialias_upscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental option to anti-alias (smooth) the latent after upscaling.",
                    },
                ),
            },
        }

    @classmethod
    def patch(
        cls,
        *,
        model,
        commasep_block_numbers,
        downscale_factor,
        start_percent,
        start_fadeout_percent,
        end_percent,
        downscale_after_skip,
        downscale_method,
        upscale_method,
        antialias_downscale,
        antialias_upscale,
    ):
        block_numbers = tuple(
            int(x) for x in commasep_block_numbers.split(",") if x.strip()
        )
        downscale_factor = 1.0 / downscale_factor
        if not (block_numbers and all(val > 0 and val <= 32 for val in block_numbers)):
            raise ValueError(
                "BlehDeepShrink: Bad value for block numbers: must be comma-separated list of numbers between 1-32",
            )
        antialias_downscale = antialias_downscale and downscale_method in {
            "bicubic",
            "bilinear",
        }
        antialias_upscale = antialias_upscale and upscale_method in {
            "bicubic",
            "bilinear",
        }
        if start_fadeout_percent < start_percent:
            start_fadeout_percent = start_percent
        elif start_fadeout_percent > end_percent:
            # No fadeout.
            start_fadeout_percent = 1000.0

        ms = model.get_model_object("model_sampling")
        sigma_start = ms.percent_to_sigma(start_percent)
        sigma_end = ms.percent_to_sigma(end_percent)

        def input_block_patch(h, transformer_options):
            block_num = transformer_options["block"][1]
            sigma_tensor = transformer_options["sigmas"].max()
            sigma = sigma_tensor.detach().cpu().item()
            if (
                sigma > sigma_start
                or sigma < sigma_end
                or block_num not in block_numbers
            ):
                return h
            pct = 1.0 - (ms.timestep(sigma_tensor).detach().cpu().item() / 999)
            if (
                pct < start_fadeout_percent
                or start_fadeout_percent > end_percent
                or pct > end_percent
            ):
                scaled_scale = downscale_factor
            else:
                # May or not be accurate but the idea is to scale the downscale factor by the percentage
                # of the start fade to end deep shrink we have currently traversed. It at least sort of works.
                downscale_pct = 1.0 - (
                    (pct - start_fadeout_percent)
                    / (end_percent - start_fadeout_percent)
                )
                scaled_scale = 1.0 - ((1.0 - downscale_factor) * downscale_pct)
            orig_width, orig_height = h.shape[-1], h.shape[-2]
            width, height = (
                round(orig_width * scaled_scale),
                round(orig_height * scaled_scale),
            )
            if scaled_scale >= 0.98 or width >= orig_width or height >= orig_height:
                return h
            return latent_utils.scale_samples(
                h,
                width,
                height,
                mode=downscale_method,
                antialias_size=3 if antialias_downscale else 0,
                sigma=sigma,
            )

        def output_block_patch(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"][0].cpu().item()
            if (
                h.shape[-2:] == hsp.shape[-2:]
                or sigma > sigma_start
                or sigma < sigma_end
            ):
                return h, hsp
            return latent_utils.scale_samples(
                h,
                hsp.shape[-1],
                hsp.shape[-2],
                mode=upscale_method,
                antialias_size=3 if antialias_upscale else 0,
                sigma=sigma,
            ), hsp

        m = model.clone()
        if downscale_factor == 0.0 or start_percent >= 1.0:
            return (m,)
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m,)
