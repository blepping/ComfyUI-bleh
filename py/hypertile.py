# The chain of yoinks grows ever longer.
# Originally taken from: https://github.com/tfernd/HyperTile/
# Modified version of ComfyUI main code
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_hypertile.py
from __future__ import annotations

import math
import random

from einops import rearrange


class HyperTile:
    def __init__(
        self,
        model,
        seed,
        tile_size,
        swap_size,
        max_depth,
        scale_depth,
        start_step,
        end_step,
    ):
        self.rng = random.Random()
        self.rng.seed(seed)
        self.model = model
        self.latent_tile_size = max(32, tile_size) // 8
        self.swap_size = swap_size
        self.max_depth = max_depth
        self.scale_depth = scale_depth
        self.start_step = start_step
        self.end_step = end_step
        # Temporary storage for rearranged tensors in the output part
        self.temp = None

    def patch(self):
        model = self.model
        model.set_model_attn1_patch(self.attn1_in)
        model.set_model_attn1_output_patch(self.attn1_out)
        return model

    @staticmethod
    def get_closest_divisors(hw: int, aspect_ratio: float) -> tuple[int, int]:
        pairs = tuple(
            (i, hw // i) for i in range(int(math.sqrt(hw)), 1, -1) if hw % i == 0
        )
        pair = min(
            ((i, hw // i) for i in range(2, hw + 1) if hw % i == 0),
            key=lambda x: abs(x[1] / x[0] - aspect_ratio),
        )
        pairs.append(pair)
        return min(pairs, key=lambda x: max(x) / min(x))

    @classmethod
    def calc_optimal_hw(cls, hw: int, aspect_ratio: float) -> tuple[int, int]:
        hcand = round(math.sqrt(hw * aspect_ratio))
        wcand = hw // hcand

        if hcand * wcand == hw:
            return hcand, wcand

        wcand = round(math.sqrt(hw / aspect_ratio))
        hcand = hw // wcand

        if hcand * wcand != hw:
            return cls.get_closest_divisors(hw, aspect_ratio)

        return hcand, wcand

    def random_divisor(
        self,
        value: int,
        min_value: int,
        /,
        max_options: int = 1,
    ) -> int:
        min_value = min(min_value, value)
        # All big divisors of value (inclusive)
        divisors = tuple(i for i in range(min_value, value + 1) if value % i == 0)
        ns = tuple(value // i for i in divisors[:max_options])  # has at least 1 element
        idx = self.rng.randint(0, len(ns) - 1) if len(ns) - 1 > 0 else 0
        return ns[idx]

    def check_timestep(self, extra_options):
        current_timestep = self.model.model.model_sampling.timestep(
            extra_options["sigmas"][0],
        ).item()
        return current_timestep <= self.start_step and current_timestep >= self.end_step

    def attn1_in(self, q, k, v, extra_options):
        if not self.check_timestep(extra_options):
            self.temp = None
            return q, k, v

        model_chans = q.shape[-2]
        orig_shape = extra_options["original_shape"]

        apply_to = tuple(
            (orig_shape[-2] / (2**i)) * (orig_shape[-1] / (2**i))
            for i in range(self.max_depth + 1)
        )
        if model_chans not in apply_to:
            return q, k, v

        aspect_ratio = orig_shape[-1] / orig_shape[-2]

        hw = q.size(1)
        h, w = (
            round(math.sqrt(hw * aspect_ratio)),
            round(math.sqrt(hw / aspect_ratio)),
        )

        factor = (2 ** apply_to.index(model_chans)) if self.scale_depth else 1

        nh = self.random_divisor(h, self.latent_tile_size * factor, self.swap_size)
        nw = self.random_divisor(w, self.latent_tile_size * factor, self.swap_size)

        if nh * nw <= 1:
            return q, k, v

        q = rearrange(
            q,
            "b (nh h nw w) c -> (b nh nw) (h w) c",
            h=h // nh,
            w=w // nw,
            nh=nh,
            nw=nw,
        )
        self.temp = (nh, nw, h, w)
        return q, k, v

    def attn1_out(self, out, _extra_options):
        if self.temp is None:
            return out
        nh, nw, h, w = self.temp
        self.temp = None
        out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
        return rearrange(
            out,
            "b nh nw (h w) c -> b (nh h nw w) c",
            h=h // nh,
            w=w // nw,
        )


class HyperTileBleh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "tile_size": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "swap_size": ("INT", {"default": 2, "min": 1, "max": 128}),
                "max_depth": ("INT", {"default": 0, "min": 0, "max": 10}),
                "scale_depth": ("BOOLEAN", {"default": False}),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 1000,
                        "step": 0.1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    def patch(
        self,
        model,
        seed,
        tile_size,
        swap_size,
        max_depth,
        scale_depth,
        start_step,
        end_step,
    ):
        return (
            HyperTile(
                model.clone(),
                seed,
                tile_size,
                swap_size,
                max_depth,
                scale_depth,
                start_step,
                end_step,
            ).patch(),
        )
