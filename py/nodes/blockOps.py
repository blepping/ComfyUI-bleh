# Adapted from the ComfyUI built-in node

import bisect
import importlib as il
import math
from enum import Enum, auto

import numpy as np
import torch
import yaml
from torch import fft

dare = il.import_module("custom_nodes.ComfyUI-DareMerge.ddare.merge")


def normalize(latent, target_min=None, target_max=None):
    min_val = latent.min()
    max_val = latent.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (latent - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


def hslerp(a, b, t):
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)

    interpolation_tensor = torch.zeros(
        1,
        num_channels,
        1,
        1,
        device=a.device,
        dtype=a.dtype,
    )
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result


blending_modes = {
    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor
    # Interpolates between tensors a and b using normalized linear interpolation.
    "bislerp": lambda a, b, t: normalize((1 - t) * a + t * b),
    # Transfer the color from `b` to `a` by t` factor
    "colorize": lambda a, b, t: a + (b - a) * t,
    # Interpolates between tensors a and b using cosine interpolation.
    "cosinterp": lambda a, b, t: (
        a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))
    )
    / 2,
    # Interpolates between tensors a and b using cubic interpolation.
    "cuberp": lambda a, b, t: a + (b - a) * (3 * t**2 - 2 * t**3),
    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    "hslerp": hslerp,
    # Adds tensor b to tensor a, scaled by t.
    "inject": lambda a, b, t: a + b * t,
    # Interpolates between tensors a and b using linear interpolation.
    "lerp": lambda a, b, t: (1 - t) * a + t * b,
    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    "lineardodge": lambda a, b, t: normalize(a + b * t),
}

mscales = {
    "none": (),
    "bandpass": (
        (5, 0.0),  # Low-pass filter
        (15, 1.0),  # Pass-through filter (allows mid-range frequencies)
        (25, 0.0),  # High-pass filter
    ),
    "lowpass": (
        (
            10,
            1.0,
        ),
    ),  # Allows low-frequency components, suppresses high-frequency components
    "highpass": (
        (
            10,
            0.0,
        ),
    ),  # Suppresses low-frequency components, allows high-frequency components
    "passthrough": ((10, 1.0),),  # Passes all frequencies unchanged, no filtering
    "gaussianblur": (
        (
            10,
            0.5,
        ),  # Blurs the image by allowing a range of frequencies with a Gaussian shape
    ),
    "edge": (
        (
            10,
            2.0,
        )
    ),  # Enhances edges and high-frequency features while suppressing low-frequency details
    "sharpen": (
        (
            10,
            1.5,
        )
    ),  # Increases the sharpness of the image by emphasizing high-frequency components
    "multilowpass": ((5, 1.0), (10, 0.5), (15, 0.2)),  # Multi-scale low-pass filter
    "multihighpass": ((5, 0.0), (10, 0.5), (15, 0.8)),  # Multi-scale high-pass filter
    "multipassthrough": (
        (5, 1.0),
        (10, 1.0),
        (15, 1.0),
    ),  # Pass-through at different scales
    "multigaussianblur": ((5, 0.5), (10, 0.8), (15, 0.2)),  # Multi-scale Gaussian blur
    "multiedge": ((5, 1.2), (10, 1.5), (15, 2.0)),  # Multi-scale edge enhancement
    "multisharpen": ((5, 1.5), (10, 2.0), (15, 2.5)),  # Multi-scale sharpening
}

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
    "cosinterp",
    "cuberp",
    "inject",
    "lerp",
    "lineardodge",
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
        result = torch.nn.functional.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            antialias=antialias_size > 7,
        )
    else:
        result = biderp(samples, width, height, mode, mode_h)
    if antialias_size < 1 or antialias_size > 7:
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
        "colorize": blending_modes["colorize"],
        "hslerp": hslerp,
        "bislerp": slerp_orig,
        "bibislerp": blending_modes["bislerp"],
        "inject": blending_modes["inject"],
        "lerp": blending_modes["lerp"],
        "lineardodge": blending_modes["lineardodge"],
        "cosinterp": blending_modes["cosinterp"],
        "cuberp": blending_modes["cuberp"],
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


def ffilter(x, threshold, scale, scales=None, strength=1.0):
    # FFT
    if isinstance(x, list):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        return x
    x_freq = fft.fftn(x.float(), dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    _batch, _channels, height, width = x_freq.shape
    mask = torch.ones(x_freq.shape, device=x.device)

    crow, ccol = height // 2, width // 2
    mask[
        ...,
        crow - threshold : crow + threshold,
        ccol - threshold : ccol + threshold,
    ] = scale

    if scales:
        for scale_params in scales:
            for scale_threshold, scale_value in scale_params:
                scaled_scale_value = scale_value * strength
                scale_mask = torch.ones(x_freq.shape, device=x.device)
                scale_mask[
                    ...,
                    crow - scale_threshold : crow + scale_threshold,
                    ccol - scale_threshold : ccol + scale_threshold,
                ] = scaled_scale_value
                mask = mask + (scale_mask - mask) * strength

    x_freq *= mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    return fft.ifftn(x_freq, dim=(-2, -1)).real.to(x.dtype)


class OpType(Enum):
    # scale, strength, blend, blend mode, use hidden mean
    SLICE = auto()

    # scale, filter, filter strength, threshold
    FFILTER = auto()

    # type (bicubic, nearest, bilinear, area), scale width, scale height, antialias
    SCALE_TORCH = auto()

    # type (bicubic, nearest, bilinear, area), antialias
    UNSCALE_TORCH = auto()

    # type width (slerp, slerp_alt, hslerp, colorize), type height, scale width, scale height, antialias size
    SCALE = auto()

    # type width (slerp, slerp_alt, hslerp, colorize), type height, antialias size
    UNSCALE = auto()

    # direction (h, v)
    FLIP = auto()

    # count
    ROT90 = auto()

    # true/false - only makes sense with output block
    TARGET_SKIP = auto()

    # factor
    MULTIPLY = auto()


class CondType(Enum):
    TYPE = auto()
    BLOCK = auto()
    STAGE = auto()
    FROM_PERCENT = auto()
    TO_PERCENT = auto()
    PERCENT = auto()
    COUNT = auto()


class BlockType(Enum):
    INPUT = auto()
    INPUT_AFTER_SKIP = auto()
    MIDDLE = auto()
    OUTPUT = auto()


class BlockCond:
    def __init__(self, typ, value):
        self.typ = getattr(CondType, typ.upper().strip())
        self.value = set(value if isinstance(value, (list, tuple)) else (value,))

    def test(self, state, count=0):
        match self.typ:
            case CondType.FROM_PERCENT:
                pct = state[CondType.PERCENT]
                result = all(pct >= v for v in self.value)
            case CondType.TO_PERCENT:
                pct = state[CondType.PERCENT]
                result = all(pct <= v for v in self.value)
            case CondType.COUNT:
                result = all(count < v for v in self.value)
            case _:
                result = state[self.typ] in self.value
        return result

    def __repr__(self):
        return f"<Cond({self.typ}): {self.value}>"


class BlockConds:
    def __init__(self, conds, count=0):
        self.count = count
        if not conds:
            self.conds = ()
            return
        self.conds = tuple(BlockCond(ct, cv) for ct, cv in conds.items())

    def test(self, state):
        count = self.count
        result = all(c.test(state, count) for c in self.conds)
        if result:
            self.count += 1
        return result

    def __repr__(self):
        return f"<Conds[{self.count}]: {self.conds}"


def hidden_mean(h):
    hidden_mean = h.mean(1).unsqueeze(1)
    b = hidden_mean.shape[0]
    hidden_max, _ = torch.max(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    hidden_min, _ = torch.min(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    return (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        hidden_max - hidden_min
    ).unsqueeze(2).unsqueeze(3)


class BlockOp:
    def __init__(self, typ, *args):
        self.typ = getattr(OpType, typ.upper().strip())
        self.args = args

    def eval(self, state):
        t = out = state[state["target"]]
        match self.typ:
            case OpType.SCALE_TORCH | OpType.UNSCALE_TORCH:
                if self.typ == OpType.SCALE_TORCH:
                    mode, scale_w, scale_h, antialias = self.args
                    width, height = (
                        round(t.shape[-1] * scale_w),
                        round(t.shape[-2] * scale_h),
                    )
                else:
                    hsp = state["hsp"]
                    if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                        return
                    mode, antialias = self.args
                    width, height = hsp.shape[-1], hsp.shape[-2]
                out = scale_samples(
                    t,
                    width,
                    height,
                    mode,
                    antialias_size=8 if antialias else 0,
                )
            case OpType.SCALE | OpType.UNSCALE:
                if self.typ == OpType.SCALE:
                    mode_w, mode_h, scale_w, scale_h, antialias_size = self.args
                    width, height = (
                        round(t.shape[-1] * scale_w),
                        round(t.shape[-2] * scale_h),
                    )
                else:
                    hsp = state["hsp"]
                    if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                        return
                    mode_w, mode_h, antialias_size = self.args
                    width, height = hsp.shape[-1], hsp.shape[-2]
                print("SCALE", width, height, mode_w, mode_h, antialias_size)
                out = scale_samples(
                    t,
                    width,
                    height,
                    mode=mode_w,
                    mode_h=mode_h,
                    antialias_size=antialias_size,
                )
            case OpType.FLIP:
                out = torch.flip(
                    t,
                    dims=(2 if self.args == "h" else 3,),
                )
            case OpType.ROT90:
                print("ROTBEFORE", t.shape)
                out = torch.rot90(t, self.args[0], dims=(3, 2))
                print("ROTAFTER", out.shape)
            case OpType.TARGET_SKIP:
                state["target"] = "hsp" if self.args[0] is True else "h"
                return
            case OpType.FFILTER:
                scale, filt, strength, threshold = self.args
                if isinstance(filt, str):
                    filt = mscales[filt]
                out = ffilter(t, threshold, scale, filt, strength)
            case OpType.SLICE:
                scale, strength, blend, mode, use_hm = self.args
                slice_size = round(t.shape[1] * scale)
                print("SLICE", slice_size)
                sliced = t[:, :slice_size]
                if use_hm:
                    result = sliced * ((strength - 1) * hidden_mean(t) + 1)
                else:
                    result = sliced * strength
                if blend != 1:
                    result = blending_modes[mode](sliced, result, blend)
                out[:, :slice_size] = result
            case OpType.MULTIPLY:
                out *= self.args[0]
            case _:
                raise ValueError("Unhandled")
        state[state["target"]] = out

    def __repr__(self):
        return f"<Op({self.typ}): {self.args!r}>"


class BlockRule:
    @classmethod
    def from_dict(cls, val):
        if not isinstance(val, (list, tuple)):
            val = (val,)

        return tuple(
            cls(
                conds=d.get("if", ()),
                ops=d.get("ops", ()),
                matched=d.get("then", ()),
                nomatched=d.get("else", ()),
            )
            for d in val
        )

    def __init__(self, conds=(), ops=(), matched=(), nomatched=()):
        self.conds = BlockConds(conds)
        self.ops = tuple(BlockOp(o[0], *o[1:]) for o in ops)
        self.matched = BlockRule.from_dict(matched)
        self.nomatched = BlockRule.from_dict(nomatched)

    def get_all_types(self):
        result = {c.value for c in self.conds if c.typ == CondType.TYPE}
        for r in self.matched:
            result |= r.get_all_types()
        for r in self.nomatched:
            result |= r.get_all_types()
        return result

    def eval(self, state):
        # print("EVAL", state | {"h": None, "hsp": None})
        if not self.conds.test(state):
            for r in self.nomatched:
                r.eval(state)
            return
        state["target"] = "h"
        for o in self.ops:
            o.eval(state)
        for r in self.matched:
            r.eval(state)

    def __repr__(self):
        return f"<Rule: IF({self.conds}) THEN({self.ops}, {self.matched}) ELSE({self.nomatched})>"


class BlehBlockOps:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rules": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    def patch(
        self,
        model,
        rules,
    ):
        rules = rules.strip()
        print("PARSING", repr(rules))
        if len(rules) == 0:
            return (model.clone(),)
        parsed_rules = yaml.safe_load(rules)
        rules = tuple(BlockRule.from_dict(r)[0] for r in parsed_rules)
        print("RULES:", rules)

        # Arbitrary number that should have good enough precision
        pct_steps = 400
        pct_incr = 1.0 / pct_steps
        sig2pct = tuple(
            model.model.model_sampling.percent_to_sigma(x / pct_steps)
            for x in range(pct_steps, -1, -1)
        )

        def get_pct(topts):
            sigma = topts["sigmas"][0].item()
            # This is obviously terrible but I couldn't find a better way to get the percentage from the current sigma.
            idx = bisect.bisect_right(sig2pct, sigma)
            if idx >= len(sig2pct):
                # Sigma out of range somehow?
                return None
            return pct_incr * (pct_steps - idx)

        stages = (1280, 640, 320)

        def make_state(typ, topts, h, hsp=None):
            pct = get_pct(topts)
            if pct is None:
                return None

            stage = stages.index(h.shape[1]) + 1 if h.shape[1] in stages else -1
            return {
                CondType.TYPE: typ,
                CondType.PERCENT: pct,
                CondType.BLOCK: topts["block"][1],
                CondType.STAGE: stage,
                "h": h,
                "hsp": hsp,
                "target": "h",
            }

        def block_patch(typ, h, topts):
            state = make_state(typ, topts, h)
            if state is None:
                return h
            for rule in rules:
                rule.eval(state)
            return state["h"]

        def output_block_patch(h, hsp, transformer_options):
            state = make_state("output", transformer_options, h, hsp)
            if state is None:
                return h
            for rule in rules:
                rule.eval(state)
            return state["h"], state["hsp"]

        m = model.clone()
        m.set_model_input_block_patch_after_skip(
            lambda *args: block_patch("input_after_skip", *args),
        )
        m.set_model_input_block_patch(lambda *args: block_patch("input", *args))
        m.set_model_patch(
            lambda *args: block_patch("middle", *args),
            "middle_block_patch",
        )
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
