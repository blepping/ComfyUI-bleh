# Adapted from the ComfyUI built-in node

import bisect
import importlib as il

import comfy
import numpy as np
import torch

dare = il.import_module("custom_nodes.ComfyUI-DareMerge.ddare.merge")

UPSCALE_METHODS = (
    "bicubic",
    "nearest-exact",
    "bilinear",
    "area",
    "bislerp",
    "bislerp_alt",
    "colorize",
    "hslerp",
    "bibislerp",
)

FILTER_SIZES = (
    np.array([1.0]),
    np.array([1.0, 1.0]),
    np.array([1.0, 2.0, 1.0]),
    np.array([1.0, 3.0, 3.0, 1.0]),
    np.array([1.0, 4.0, 6.0, 4.0, 1.0]),
    np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]),
    np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]),
)


def make_filter(channels, dtype, size=3):
    a = FILTER_SIZES[size - 1]
    filt = torch.tensor(a[:, None] * a[None, :], dtype=dtype)
    filt = filt / torch.sum(filt)
    return filt[None, None, :, :].repeat((channels, 1, 1, 1))


def scale_samples(
    samples,
    width,
    height,
    mode="bicubic",
    mode_h=None,
    antialias_size=0,
):
    if mode_h is None:
        mode_h = mode
    if mode in ("bicubic", "nearest-exact", "bilinear", "area"):
        return torch.nn.functional.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            antialias=antialias_size > 0,
        )

    result = biderp(samples, width, height, mode, mode_h)
    if antialias_size > 0:
        return result
    channels = result.shape[1]
    filt = make_filter(channels, result.dtype, antialias_size).to(result.device)
    return torch.nn.functional.conv2d(result, filt, groups=channels, padding="same")


# Modified from ComfyUI
def biderp(samples, width, height, mode="bislerp", mode_h=None):
    def slerp_orig(b1, b2, r):
        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
            1,
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
            1,
        ) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    # Modified from DARE merge
    def hslerp(a, b, t):
        interpolation_tensor = torch.zeros(
            1,
            a.shape[-1],
            device=a.device,
            dtype=a.dtype,
        )

        interpolation_tensor[0, 0] = 1.0

        result = (1 - t) * a + t * b
        norm = (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
        norm[t < 0.5, None] *= -1
        result += norm
        return result

    if mode_h is None:
        mode_h = mode

    modes = {
        "bislerp_alt": dare.merge_tensors_slerp,
        "colorize": dare.merge_tensors_colorize,
        "hslerp": hslerp,
        "bislerp": slerp_orig,
        "bibislerp": dare.merge_tensors_bislerp,
    }
    derp_w, derp_h = modes.get(mode, slerp_orig), modes.get(mode_h, slerp_orig)

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1),
        )
        coords_1 = torch.nn.functional.interpolate(
            coords_1,
            size=(1, length_new),
            mode="bilinear",
        )
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape(
                (1, 1, 1, -1),
            )
            + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2,
            size=(1, length_new),
            mode="bilinear",
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = derp_w(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = derp_h(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


class DeepShrinkBleh:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

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
                "downscale_w": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 32.0, "step": 0.1},
                ),
                "downscale_h": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 32.0, "step": 0.1},
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
                "downscale_method_w": (UPSCALE_METHODS,),
                "downscale_method_h": (UPSCALE_METHODS,),
                "upscale_method_w": (UPSCALE_METHODS,),
                "upscale_method_h": (UPSCALE_METHODS,),
                "antialias_downscale_size": ("INT", {"default": 0, "max": 7}),
                "antialias_upscale_size": ("INT", {"default": 0, "max": 7}),
            },
        }

    def patch(
        self,
        model,
        commasep_block_numbers,
        downscale_w,
        downscale_h,
        start_percent,
        start_fadeout_percent,
        end_percent,
        downscale_after_skip,
        downscale_method_w,
        downscale_method_h,
        upscale_method_w,
        upscale_method_h,
        antialias_downscale_size,
        antialias_upscale_size,
    ):
        # if downscale_factor == 1 or start_percent == 1 or end_percent == 0:
        #     return (model.clone(),)
        block_numbers = tuple(
            int(x) for x in commasep_block_numbers.split(",") if x.strip()
        )
        # if downscale_factor < 1 and start_fadeout_percent != 1:
        #     raise ValueError(
        #         "BlehDeepShrink: Setting scale factor below 1 cannot be combined with fadeout",
        #     )
        downscale_w, downscale_h = 1.0 / downscale_w, 1.0 / downscale_h
        if not (block_numbers and all(val > 0 and val <= 32 for val in block_numbers)):
            raise ValueError(
                "BlehDeepShrink: Bad value for block numbers: must be comma-separated list of numbers between 1-32",
            )
        # antialias_downscale = antialias_downscale and downscale_method in (
        #     "bicubic",
        #     "bilinear",
        # )
        # antialias_upscale = antialias_upscale and upscale_method in (
        #     "bicubic",
        #     "bilinear",
        # )
        if start_fadeout_percent < start_percent:
            start_fadeout_percent = start_percent
        elif start_fadeout_percent > end_percent:
            # No fadeout.
            start_fadeout_percent = 1000.0

        sigma_start = model.model.model_sampling.percent_to_sigma(
            max(0.001, start_percent),
        )
        sigma_end = model.model.model_sampling.percent_to_sigma(end_percent)
        sigma_min = model.model.model_sampling.percent_to_sigma(1)
        sigma_max = model.model.model_sampling.percent_to_sigma(0.001)
        sigma_adj = sigma_max - sigma_min
        print("GOT", sigma_min, sigma_max, sigma_adj)

        # Arbitrary number that should have good enough precision
        pct_steps = 400
        pct_incr = 1.0 / pct_steps
        sig2pct = tuple(
            model.model.model_sampling.percent_to_sigma(x / pct_steps)
            for x in range(pct_steps, -1, -1)
        )
        # print(sig2pct)

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
            pct2 = 1.0 - (sigma / sigma_max)
            print(">>>", pct, pct2, "--", sigma, sigma_start, sigma_end, sigma_adj)
            if (
                pct < start_fadeout_percent
                or start_fadeout_percent > end_percent
                or pct > end_percent
            ):
                scaled_scale_w, scaled_scale_h = downscale_w, downscale_h
            else:
                # May or not be accurate but the idea is to scale the downscale factor by the percentage
                # of the start fade to end deep shrink we have currently traversed. It at least sort of works.
                downscale_pct = 1.0 - (
                    (pct - start_fadeout_percent)
                    / (end_percent - start_fadeout_percent)
                )
                scaled_scale_w = 1.0 - ((1.0 - downscale_w) * downscale_pct)
                scaled_scale_h = 1.0 - ((1.0 - downscale_h) * downscale_pct)
            orig_width, orig_height = h.shape[-1], h.shape[-2]
            width, height = (
                round(orig_width * scaled_scale_w),
                round(orig_height * scaled_scale_h),
            )
            # if scaled_scale >= 0.98 or width >= orig_width or height >= orig_height:
            #     return h
            if (0.98 <= scaled_scale_w <= 1.02 and 0.98 <= scaled_scale_h <= 1.02) or (
                width == orig_width and height == orig_height
            ):
                return h
            return scale_samples(
                h,
                width,
                height,
                mode=downscale_method_w,
                mode_h=downscale_method_h,
                antialias_size=antialias_downscale_size,
            )

        def output_block_patch(h, hsp, _transformer_options):
            if h.shape[2] == hsp.shape[2] and h.shape[3] == hsp.shape[3]:
                return h, hsp
            return scale_samples(
                h,
                hsp.shape[-1],
                hsp.shape[-2],
                mode=upscale_method_w,
                mode_h=upscale_method_h,
                antialias_size=antialias_upscale_size,
            ), hsp

        m = model.clone()
        if start_percent >= 1.0:
            return (m,)
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m,)


class BlehLatentUpscaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (UPSCALE_METHODS,),
                "scale_width": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
                "scale_height": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
                "antialias": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, scale_width, scale_height):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_width)
        height = round(samples["samples"].shape[2] * scale_height)
        s["samples"] = scale_samples(
            samples["samples"],
            width,
            height,
            upscale_method,
        )
        return (s,)
