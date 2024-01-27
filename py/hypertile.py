# The chain of yoinks grows ever longer.
# Originally taken from: https://github.com/tfernd/HyperTile/
# Modified version of ComfyUI main code
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_hypertile.py

import math
import random

import torch

from einops import rearrange
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, th, apply_control

def get_closest_divisors(hw: int, aspect_ratio: float) -> tuple[int, int]:
    pairs = [(i, hw // i) for i in range(int(math.sqrt(hw)), 1, -1) if hw % i == 0]
    pair = min(((i, hw // i) for i in range(2, hw + 1) if hw % i == 0),
               key=lambda x: abs(x[1] / x[0] - aspect_ratio))
    pairs.append(pair)
    res = min(pairs, key=lambda x: max(x) / min(x))
    return res


def calc_optimal_hw(hw: int, aspect_ratio: float) -> tuple[int, int]:
    hcand = round(math.sqrt(hw * aspect_ratio))
    wcand = hw // hcand

    if hcand * wcand != hw:
        wcand = round(math.sqrt(hw / aspect_ratio))
        hcand = hw // wcand

        if hcand * wcand != hw:
            return get_closest_divisors(hw, aspect_ratio)

    return hcand, wcand

def random_divisor(value: int, min_value: int, /, max_options: int = 1, rand_obj=random.Random()) -> int:
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    if len(ns) - 1 > 0:
        idx = rand_obj.randint(0, len(ns) - 1)
    else:
        idx = 0

    return ns[idx]

class HyperTileBleh:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
          "model": ("MODEL",),
          "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
          "tile_size": ("INT", {"default": 256, "min": 1, "max": 2048}),
          "swap_size": ("INT", {"default": 2, "min": 1, "max": 128}),
          "max_depth": ("INT", {"default": 0, "min": 0, "max": 10}),
          "scale_depth": ("BOOLEAN", {"default": False}),
          "start_step": ("INT", { "default": 0, "min": 0, "max": 1000, "step": 1, "display": "number" }),
          "end_step": ("INT", { "default": 1000, "min": 0, "max": 1000, "step": 0.1, }),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "bleh/model_patches"

    def patch(self, model, seed, tile_size, swap_size, max_depth, scale_depth, start_step, end_step):
        latent_tile_size = max(32, tile_size) // 8
        temp = None

        rand_obj = random.Random()
        rand_obj.seed(seed)

        def hypertile_in(q, k, v, extra_options):
            nonlocal temp

            current_timestep = model.model.model_sampling.timestep(extra_options['sigmas'][0]).item()
            if current_timestep > start_step or current_timestep < end_step:
              temp = None
              return q, k, v

            model_chans = q.shape[-2]
            orig_shape = extra_options['original_shape']

            apply_to = []
            for i in range(max_depth + 1):
                apply_to.append((orig_shape[-2] / (2 ** i)) * (orig_shape[-1] / (2 ** i)))
            if model_chans not in apply_to:
              return q, k, v

            aspect_ratio = orig_shape[-1] / orig_shape[-2]

            hw = q.size(1)
            h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))

            factor = (2 ** apply_to.index(model_chans)) if scale_depth else 1
            nh = random_divisor(h, latent_tile_size * factor, swap_size, rand_obj)
            nw = random_divisor(w, latent_tile_size * factor, swap_size, rand_obj)

            if nh * nw > 1:
                q = rearrange(q, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                temp = (nh, nw, h, w)
            return q, k, v

        def hypertile_out(out, extra_options):
            nonlocal temp
            if temp is not None:
                nh, nw, h, w = temp
                temp = None
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
            return out

        m = model.clone()
        m.set_model_attn1_patch(hypertile_in)
        m.set_model_attn1_output_patch(hypertile_out)
        return (m, )
