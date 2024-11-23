# Credits:
#   Blending, slice and filtering functions based on https://github.com/WASasquatch/FreeU_Advanced
from __future__ import annotations

import contextlib
import math
import os
from functools import partial

import kornia.filters as kf
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import FloatTensor, LongTensor, fft

OVERRIDE_NO_SCALE = "COMFYUI_BLEH_OVERRIDE_NO_SCALE" in os.environ
USE_ORIG_NORMALIZE = "COMFYUI_BLEH_ORIG_NORMALIZE" in os.environ


def normalize_orig(latent, target_min=None, target_max=None, **_unused_kwargs: dict):
    min_val = latent.min()
    max_val = latent.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (latent - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


def normalize(latent, *, reference_latent=None, dim=(-3, -2, -1)):
    if reference_latent is None:
        return latent
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    target_min, target_max = (
        reference_latent.amin(dim=dim, keepdim=True),
        reference_latent.amax(dim=dim, keepdim=True),
    )

    normalized = (latent - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


if USE_ORIG_NORMALIZE:
    normalize = normalize_orig


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
    interpolation_tensor[0, 0, 0, 0] = 1.0 if t < 0.5 else -1.0

    result = (1 - t) * a + t * b
    result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result


# This may be far from correct.
def hslerp_alt(a, b, t):
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")
    interp = torch.zeros(
        1,
        a.size(1),
        device=a.device,
        dtype=a.dtype,
    )
    interp[0, 0] = 1.0
    result = (1 - t) * a + t * b
    norm = (torch.norm(b - a, dim=1, keepdim=True) / 6) * interp
    norm[t.broadcast_to(norm.shape) < 0.5] *= -1
    return result.add_(norm)


# This should be more correct but the results are worse. :(
def hslerp_alt2(a, b, t, *, sign_order=(1.0, -1.0), sign_threshold=0.5):
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")
    t_expanded = t.broadcast_to(a.shape[-2:])
    while t_expanded.ndim < a.ndim:
        t_expanded = t_expanded.unsqueeze(0)
    return (
        ((1 - t) * a)
        .add_(t * b)
        .add_(
            torch.norm(b - a, dim=1, keepdim=True).div_(6)
            * torch.where(t_expanded.abs() < sign_threshold, *sign_order),
        )
    )


# Copied from ComfyUI
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


# From https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
def altslerp(  # noqa: PLR0914
    v0: FloatTensor,
    v1: FloatTensor,
    t: float | FloatTensor,
    dot_threshold=0.9995,
    dim=-1,
):
    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = torch.linalg.norm(v0, dim=dim)
    v1_norm: FloatTensor = torch.linalg.norm(v1, dim=dim)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(dim)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(dim)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(dim)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > dot_threshold)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = (
        max(0, t.dim() - v0.dim()) if isinstance(t, torch.Tensor) else 0
    )
    t_batch_dims: torch.Size = (
        t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
    )
    out: FloatTensor = torch.zeros_like(v0.expand(*t_batch_dims, *(dim,) * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: FloatTensor = torch.lerp(v0, v1, t)

        out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(dim), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        # Calculate initial angle between v0 and v1
        theta_0: FloatTensor = dot.arccos().unsqueeze(dim)
        sin_theta_0: FloatTensor = theta_0.sin()
        # Angle at timestep t
        theta_t: FloatTensor = theta_0 * t
        sin_theta_t: FloatTensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: FloatTensor = sin_theta_t / sin_theta_0
        slerped: FloatTensor = s0 * v0 + s1 * v1

        out: FloatTensor = slerped.where(can_slerp.unsqueeze(dim), out)

    return out


class BlendMode:
    __slots__ = ("allow_scale", "f", "norm", "norm_dims", "rev")

    class _Empty:
        pass

    def __init__(
        self,
        f,
        norm=None,
        norm_dims=(-3, -2, -1),
        rev=False,
        allow_scale=True,
    ):
        self.f = f
        self.norm = norm
        self.norm_dims = norm_dims
        self.rev = rev
        self.allow_scale = allow_scale

    def edited(
        self,
        *,
        f=_Empty,
        norm=_Empty,
        norm_dims=_Empty,
        rev=_Empty,
        allow_scale=_Empty,
    ):
        empty = self._Empty
        return self.__class__(
            f if f is not empty else self.f,
            norm=norm if norm is not empty else self.norm,
            norm_dims=norm_dims if norm_dims is not empty else self.norm_dims,
            rev=rev if rev is not empty else self.rev,
            allow_scale=allow_scale if allow_scale is not empty else self.allow_scale,
        )

    def __call__(self, a, b, t):
        if self.rev:
            a, b = b, a
        if self.norm is None:
            return self.f(a, b, t)
        ref = (1 - t) * a + t * b
        return self.norm(self.f(a, b, t), reference_latent=ref, dim=self.norm_dims)


BLENDING_MODES = {
    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor
    # Interpolates between tensors a and b using normalized linear interpolation.
    "bislerp": BlendMode(
        lambda a, b, t: ((1 - t) * a).add_(t * b),
        normalize,
    ),
    # "nbislerp": BlendMode(lambda a, b, t: (1 - t) * a + t * b, normalize),
    "slerp": BlendMode(lambda a, b, t: altslerp(a, b, t, dim=-1)),
    # Transfer the color from `b` to `a` by t` factor
    "colorize": BlendMode(lambda a, b, t: (b - a).mul_(t).add_(a)),
    # Interpolates between tensors a and b using cosine interpolation.
    "cosinterp": BlendMode(
        lambda a, b, t: (
            (a + b).sub_((a - b).mul_(torch.cos(t * torch.tensor(math.pi))))
        ).div_(2),
    ),
    # Interpolates between tensors a and b using cubic interpolation.
    "cuberp": BlendMode(lambda a, b, t: (b - a).mul_(3 * t**2 - 2 * t**3).add_(a)),
    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    "hslerp": BlendMode(hslerp),
    "hslerpalt": BlendMode(hslerp_alt2),
    "hslerpalt110x": BlendMode(partial(hslerp_alt2, sign_order=(1.1, -1.1))),
    "hslerpalt125x": BlendMode(partial(hslerp_alt2, sign_order=(1.25, -1.25))),
    "hslerpalt150x": BlendMode(partial(hslerp_alt2, sign_order=(1.5, -1.5))),
    "hslerpalt300x": BlendMode(partial(hslerp_alt2, sign_order=(3.0, -3.0))),
    "hslerpaltflipsign": BlendMode(partial(hslerp_alt2, sign_order=(-1.0, 1.0))),
    "hslerpaltflipsign110x": BlendMode(partial(hslerp_alt2, sign_order=(-1.1, 1.1))),
    "hslerpaltflipsign125x": BlendMode(partial(hslerp_alt2, sign_order=(-1.25, 1.25))),
    "hslerpaltflipsign150x": BlendMode(partial(hslerp_alt2, sign_order=(-1.5, 1.5))),
    "hslerpaltflipsign300x": BlendMode(partial(hslerp_alt2, sign_order=(-3.0, 3.0))),
    # Adds tensor b to tensor a, scaled by t.
    "inject": BlendMode(lambda a, b, t: (b * t).add_(a)),
    "injecthalf": BlendMode(lambda a, b, t: (b * (t * 0.5)).add_(a)),
    "injectquarter": BlendMode(lambda a, b, t: (b * (t * 0.25)).add_(a)),
    # Interpolates between tensors a and b using linear interpolation.
    "lerp": BlendMode(lambda a, b, t: ((1 - t) * a).add_(t * b)),
    "lerp050x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(0.5)),
    "lerp075x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(0.75)),
    "lerp110x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.1)),
    "lerp125x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.25)),
    "lerp150x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.5)),
    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    "lineardodge": BlendMode(lambda a, b, t: (b * t).add_(a)),
    # "nlineardodge": BlendMode(lambda a, b, t: a + b * t, normalize),
    # Simulates a brightening effect by dividing a by (1 - b) with a small epsilon to avoid division by zero.
    "colordodge": BlendMode(lambda a, b, _t: a / (1 - b + 1e-6), allow_scale=False),
    "difference": BlendMode(
        lambda a, b, t: abs(a - b) * t,
        normalize,
        allow_scale=False,
    ),
    "exclusion": BlendMode(
        lambda a, b, t: (a + b - 2 * a * b) * t,
        normalize,
        allow_scale=False,
    ),
    "glow": BlendMode(
        lambda a, b, _t: torch.where(
            a <= 1,
            a**2 / (1 - b + 1e-6),
            b * (a - 1) / (a + 1e-6),
        ),
        allow_scale=False,
    ),
    "hardlight": BlendMode(
        lambda a, b, t: (
            2 * a * b * (a < 0.5).float()
            + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()
        )
        * t,
        allow_scale=False,
    ),
    "linearlight": BlendMode(
        lambda a, b, _t: torch.where(b <= 0.5, a + 2 * b - 1, a + 2 * (b - 0.5)),
    ),
    "multiply": BlendMode(
        lambda a, b, t: (a * t).mul_(b * t),
        normalize,
        allow_scale=False,
    ),
    "overlay": BlendMode(
        lambda a, b, t: (2 * a * b + a**2 - 2 * a * b * a) * t
        if torch.all(b < 0.5)
        else (1 - 2 * (1 - a) * (1 - b)) * t,
        allow_scale=False,
    ),
    # Combines tensors a and b using the Pin Light formula.
    "pinlight": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 0.5,
            torch.min(a, 2 * b),
            torch.max(a, 2 * b - 1),
        ),
    ),
    "reflect": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 1,
            b**2 / (1 - a + 1e-6),
            a * (b - 1) / (b + 1e-6),
        ),
        allow_scale=False,
    ),
    "screen": BlendMode(
        lambda a, b, t: 1 - (1 - a) * (1 - b) * (1 - t),
        allow_scale=False,
    ),
    "subtract": BlendMode(lambda a, b, t: a * t - b * t, allow_scale=False),
    "vividlight": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 0.5,
            a / (1 - 2 * b + 1e-6),
            (a + 2 * b - 1) / (2 * (1 - b) + 1e-6),
        ),
        allow_scale=False,
    ),
}

with contextlib.suppress(ImportError):
    import importlib

    ddare = importlib.import_module("ComfyUI-DareMerge.ddare.merge")
    BLENDING_MODES |= {
        f"DARE{k}": BlendMode(partial(ddare.merge_tensors, k)) for k in ddare.METHODS
    }
    for k_ in ("slice", "cyclic", "gradient", "hslerp"):
        k = f"DARE{k_}"
        derp = BLENDING_MODES[k]
        BLENDING_MODES[f"DARE{k_}max"] = derp.edited(
            f=lambda a, b, t, _f=derp.f: _f(a, b, t.max()),
        )
        BLENDING_MODES[f"DARE{k_}min"] = derp.edited(
            f=lambda a, b, t, _f=derp.f: _f(a, b, t.min()),
        )
        BLENDING_MODES[f"DARE{k_}std"] = derp.edited(
            f=lambda a, b, t, _f=derp.f: _f(a, b, t.std()),
        )
        derp.f = lambda a, b, t, _f=derp.f: _f(a, b, t.mean())


BLENDING_MODES |= {
    f"norm{k}": v.edited(norm=normalize)
    for k, v in BLENDING_MODES.items()
    if k != "hslerp" and v.norm is None
}

BLENDING_MODES |= {f"rev{k}": v.edited(rev=True) for k, v in BLENDING_MODES.items()}

BIDERP_MODES = {
    k: v.edited(norm_dims=0)
    for k, v in BLENDING_MODES.items()
    if (v.allow_scale or OVERRIDE_NO_SCALE) and ("DARE" in k or not k.endswith("slerp"))
}

BIDERP_MODES |= {
    "hslerp": hslerp_alt,
    "bislerp": slerp_orig,
    "altbislerp": altslerp,
    "revaltbislerp": lambda a, b, t: altslerp(b, a, t),
    "bibislerp": BLENDING_MODES["bislerp"].edited(norm_dims=0),
    "revhslerp": lambda a, b, t: hslerp_alt(b, a, t),
    "revbislerp": lambda a, b, t: slerp_orig(b, a, t),
    "revbibislerp": BLENDING_MODES["revbislerp"].edited(norm_dims=0),
}

FILTER_PRESETS = {
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
        ),
    ),  # Enhances edges and high-frequency features while suppressing low-frequency details
    "sharpen": (
        (
            10,
            1.5,
        ),
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


ENHANCE_METHODS = (
    "lowpass",
    "multilowpass",
    "highpass",
    "multihighpass",
    "bandpass",
    "randhilowpass",
    "randmultihilowpass",
    "randhibandpass",
    "randlowbandpass",
    "gaussianblur",
    "multigaussianblur",
    "edge",
    "multiedge",
    "sharpen",
    "multisharpen",
    "korniabilateralblur",
    "korniagaussianblur",
    "korniasharpen",
    "korniaedge",
    "korniarevedge",
    "korniarandblursharp",
    "renoise1",
    "renoise2",
)

UPSCALE_METHODS = (
    "bicubic",
    "nearest-exact",
    "bilinear",
    "area",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "fractional_max_pool2d",
    "lp_pool2d_1",
    "lp_pool2d_2",
    "lp_pool2d_4",
    *BIDERP_MODES.keys(),
    *(
        f"{meth}+{enh}"
        for meth in ("bicubic", "bislerp", "hslerp", "random")
        for enh in ENHANCE_METHODS
    ),
    "random",
    "randomaa",
)


RAND_UPSCALE_METHODS = (
    "bicubic",
    "colorize",
    "bislerp",
    "revcosinterp",
    "bilinear",
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
    filt /= torch.sum(filt)
    return filt[None, None, :, :].repeat((channels, 1, 1, 1))


def antialias_tensor(x, antialias_size):
    channels = x.shape[1]
    filt = make_filter(channels, x.dtype, antialias_size).to(x.device)
    return nnf.conv2d(x, filt, groups=channels, padding="same")


def enhance_tensor(  # noqa: PLR0911
    x,
    name,
    scale=1.0,
    sigma=None,
    *,
    skip_multiplier=1,
    adjust_scale=True,
):
    randitems = None
    orig_scale = scale
    randskip = 0
    match name:
        case "randmultihilowpass":
            scale *= 0.1
            randskip = 4
            randitems = ("multilowpass", "multihighpass")
        case "randhilowpass":
            scale *= 0.1
            randskip = 6
            randitems = ("lowpass", "highpass")
        case "randlowbandpass":
            scale *= 0.25
            randskip = 1
            randitems = ("lowpass", "multilowpass", "bandpass")
        case "randhibandpass":
            scale *= 0.25
            randskip = 1
            randitems = ("highpass", "multihighpass", "bandpass")
        case "bandpass":
            scale *= 0.2
        case "renoise1" | "renoise2":
            if sigma is None:
                return x
            noise_scale = (
                min(sigma / 6.0, 2.0 / max(sigma, 1e-05))
                if name == "renoise1"
                else sigma / 8.0
            )
            if noise_scale < 1e-04:
                return x
            noise = torch.randn_like(x)
            return noise.mul_(noise_scale).add_(x)
    if not adjust_scale:
        scale = orig_scale
    randskip = int(randskip * skip_multiplier)
    if randitems:
        ridx = torch.randint(len(randitems) + randskip, (1,), device="cpu").item()
        if ridx >= len(randitems):
            return x
        return enhance_tensor(x, randitems[ridx], scale=scale)
    fpreset = FILTER_PRESETS.get(name)
    if fpreset is not None:
        if not adjust_scale:
            scale *= 2
        return ffilter(x, 1, 1.0, fpreset, 0.5 * scale)
    match name:
        case "korniabilateralblur":
            return x + (kf.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5)) - x) * (
                scale * 2.0
            )
        case "korniagaussianblur":
            return kf.gaussian_blur2d(x, (3, 3), (1.5, 1.5)) * scale
        case "korniasharpen":
            return x + (kf.unsharp_mask(x, (3, 3), (1.5, 1.5)) - x) * (scale / 2.0)
        case "korniaedge" | "korniarevedge":
            blur = kf.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5)) - x
            sharpened = kf.unsharp_mask(x, (3, 3), (1.5, 1.5)) - x
            if name == "korniarevedge":
                scale *= -1.0
            return x + (sharpened + blur) * (scale / 2.0)
        case "korniarandblursharp":
            return enhance_tensor(
                x,
                "korniagaussianblur"
                if torch.rand(1, device="cpu").item() < 0.5
                else "korniasharpen",
                scale=scale,
            )
        case _:
            raise ValueError("Unknown enhancement")


@torch.no_grad()
def scale_samples(
    samples,
    width,
    height,
    *,
    mode="bicubic",
    mode_h=None,
    antialias_size=0,
    post_effect_strength=1.0,
    sigma=None,
):
    if mode_h is None:
        mode_h = mode
    mode, *enhancement = mode.split("+", 1)
    mode_h = mode_h.split("+", 1)[0]
    modes = (mode, mode_h)
    if "randomaa" in modes:
        raasize, useraa = torch.rand(2, device="cpu").detach()
        antialias_size = (int(raasize * 7) + 1) * int(useraa * 2)
    if "random" in modes or "randomaa" in modes:
        ridxs = torch.randint(
            len(RAND_UPSCALE_METHODS),
            (2,),
            dtype=torch.uint8,
        ).tolist()
        mode, mode_h = (
            m if mode not in {"random", "randomaa"} else RAND_UPSCALE_METHODS[ridx]
            for ridx, m in zip(ridxs, (mode, mode_h))
        )
        mode_h = mode
    if mode in {"bicubic", "nearest-exact", "bilinear", "area"}:
        result = nnf.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            antialias=antialias_size > 7,
        )
    elif mode == "adaptive_avg_pool2d":
        result = nnf.adaptive_avg_pool2d(samples, (height, width))
    elif mode == "adaptive_max_pool2d":
        result = nnf.adaptive_max_pool2d(samples, (height, width))
    elif mode == "fractional_max_pool2d":
        h, w = samples.shape[-2:]
        result = nnf.fractional_max_pool2d(
            samples,
            kernel_size=3,
            output_ratio=(height / h, width / w),
        )
    elif mode.startswith("lp_pool2d_"):
        h, w = samples.shape[-2:]
        result = nnf.lp_pool2d(
            samples,
            float(mode.rsplit("_", 1)[1]),
            kernel_size=(int(h // height), int(w // width)),
            ceil_mode=True,
        )
    else:
        result = biderp(samples, width, height, mode, mode_h)
    if enhancement:
        result = enhance_tensor(
            result,
            enhancement[-1],
            scale=post_effect_strength,
            sigma=sigma,
        )
    if antialias_size < 1 or antialias_size > 7:
        return result
    return antialias_tensor(result, antialias_size)


# Modified from ComfyUI
def biderp(samples, width, height, mode="bislerp", mode_h=None):  # noqa: PLR0914
    if mode_h is None:
        mode_h = mode

    derp_w, derp_h = (
        BIDERP_MODES.get(mode, slerp_orig),
        BIDERP_MODES.get(mode_h, slerp_orig),
    )

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1),
        )
        coords_1 = nnf.interpolate(
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
        coords_2 = nnf.interpolate(
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
        raise TypeError("Expected tensor")
        # return x
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
        for scale_threshold, scale_value in scales:
            scaled_scale_value = scale_value * strength
            scale_mask = torch.ones(x_freq.shape, device=x.device)
            scale_mask[
                ...,
                crow - scale_threshold : crow + scale_threshold,
                ccol - scale_threshold : ccol + scale_threshold,
            ] = scaled_scale_value
            mask += (scale_mask - mask) * strength

    x_freq *= mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    return fft.ifftn(x_freq, dim=(-2, -1)).real.to(x.dtype)
