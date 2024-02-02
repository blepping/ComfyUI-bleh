# Adapted from the ComfyUI built-in node

import bisect

import torch
from comfy.utils import bislerp


class DeepShrinkBleh:
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
                "model": ("MODEL",),
                "commasep_block_numbers": (
                    "STRING",
                    {
                        "default": "3",
                    },
                ),
                "downscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 32.0, "step": 0.1},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "start_fadeout_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "downscale_after_skip": ("BOOLEAN", {"default": True}),
                "downscale_method": (cls.upscale_methods,),
                "upscale_method": (cls.upscale_methods,),
                "antialias_downscale": ("BOOLEAN", {"default": False}),
                "antialias_upscale": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    def patch(
        self,
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
        antialias_downscale = antialias_downscale and downscale_method in (
            "bicubic",
            "bilinear",
        )
        antialias_upscale = antialias_upscale and upscale_method in (
            "bicubic",
            "bilinear",
        )
        if start_fadeout_percent < start_percent:
            start_fadeout_percent = start_percent
        elif start_fadeout_percent > end_percent:
            # No fadeout.
            start_fadeout_percent = 1000.0

        sigma_start = model.model.model_sampling.percent_to_sigma(start_percent)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_percent)
        # Arbitrary number that should have good enough precision
        pct_steps = 400
        pct_incr = 1.0 / pct_steps
        sig2pct = tuple(
            model.model.model_sampling.percent_to_sigma(x / pct_steps)
            for x in range(pct_steps, -1, -1)
        )

        def input_block_patch(h, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            if (
                sigma > sigma_start
                or sigma < sigma_end
                or transformer_options["block"][1] not in block_numbers
            ):
                return h
            # This is obviously terrible but I couldn't find a better way to get the percentage from the current sigma.
            idx = bisect.bisect_right(sig2pct, sigma)
            if idx >= len(sig2pct):
                # Sigma out of range somehow?
                return h
            pct = pct_incr * (pct_steps - idx)
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
            width, height = (
                round(h.shape[-1] * scaled_scale),
                round(h.shape[-2] * scaled_scale),
            )
            if downscale_method == "bislerp":
                return bislerp(h, width, height)
            return torch.nn.functional.interpolate(
                h,
                size=(height, width),
                mode=downscale_method,
                antialias=antialias_downscale,
            )

        def output_block_patch(h, hsp, _transformer_options):
            if h.shape[2] == hsp.shape[2]:
                return h, hsp
            if upscale_method == "bislerp":
                return bislerp(
                    h,
                    hsp.shape[-1],
                    hsp.shape[-2],
                ), hsp
            return torch.nn.functional.interpolate(
                h,
                size=(hsp.shape[-2], hsp.shape[-1]),
                mode=upscale_method,
                antialias=antialias_upscale,
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
