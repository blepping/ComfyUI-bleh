# Credits:
#   Blending, slice and filtering functions based on https://github.com/WASasquatch/FreeU_Advanced
from __future__ import annotations

import math
import os
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import kornia.filters as kf
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import FloatTensor, LongTensor, fft
from tqdm import tqdm

from . import wavelet_functions as wavef

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

OVERRIDE_NO_SCALE = "COMFYUI_BLEH_OVERRIDE_NO_SCALE" in os.environ
USE_ORIG_NORMALIZE = "COMFYUI_BLEH_ORIG_NORMALIZE" in os.environ


def pass_kwargs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if args:
        if not all(isinstance(a, dict) for a in args):
            raise ValueError("Can only pass a single dict positionally")
        a0 = args[0]
        for a in args[1:]:
            a0.update(a)
        a0.update(kwargs)
        kwargs = a0
    return {k.removesuffix("_"): v for k, v in kwargs.items()}


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


def normalize_to_scale(
    latent: torch.Tensor,
    target_min: float,
    target_max: float,
    *,
    dim=(-3, -2, -1),
    eps: float = 1e-07,
) -> torch.Tensor:
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    normalized = latent - min_val
    normalized /= (max_val - min_val).add_(eps)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


def soft_clamp(
    t: torch.Tensor,
    min_val: torch.Tensor | float = 0.0,
    max_val: torch.Tensor | float = 1.0,
    *,
    # We define stiffness as a multiplier (beta) for the softplus function.
    # Higher stiffness = sharper transition.
    stiffness: float = 1.0,
    safe: bool = True,
) -> torch.Tensor:
    if stiffness < 1e-04:
        return t.clamp(min_val, max_val)
    if not isinstance(min_val, torch.Tensor):
        min_val = t.new_tensor(min_val)
    if not isinstance(max_val, torch.Tensor):
        max_val = t.new_tensor(max_val)

    # Calculate how much we are exceeding the Max
    # softplus(beta * x) / beta
    upper_overshoot = nnf.softplus((t - max_val).mul_(stiffness)).div_(-stiffness)

    # Calculate how much we are falling short of the Min
    lower_undershoot = nnf.softplus((min_val - t).mul_(stiffness)).div_(stiffness)

    # Apply corrections:
    # Original - (Amount over max) + (Amount under min)
    t = upper_overshoot.add_(t).add_(lower_undershoot)
    return t.clamp_(min_val, max_val) if safe else t


def force_gaussian_distribution(
    t: torch.Tensor,
    *,
    start_dim: int = 1,
    end_dim: int = -1,
    # Invert the argsorts, option for crazy people. Not recommended.
    invert1: bool = False,
    invert2: bool = False,
    eps: float = 1e-08,
) -> torch.Tensor:
    if start_dim < 0:
        start_dim = t.ndim + start_dim
    orig_shape = t.shape
    t_flat = t.flatten(start_dim=start_dim, end_dim=end_dim).movedim(start_dim, -1)

    # Get the rank of each element (0 to N-1)
    # Double argsort safely returns the rank of the original elements
    ranks = (
        t_flat.argsort(dim=-1, descending=invert1)
        .argsort(dim=-1, descending=invert2)
        .to(t)
    )

    # Map ranks to a uniform distribution (0.0 to 1.0 exclusive)
    # then to a Gaussian curve.
    factor = max(eps, t_flat.shape[-1] / 2)
    gaussian = ranks.div_(factor).add_(0.5 / factor - 1).erfinv_().mul_(2**0.5)

    return gaussian.movedim(-1, start_dim).reshape(orig_shape)


# Forces source to the distribution of reference.
def match_distribution(
    source: torch.Tensor,
    *,
    reference: torch.Tensor,
    start_dim: int = 1,
    end_dim: int = -1,
    # Invert the sorts, option for crazy people. Not recommended.
    invert1: bool = False,
    invert2: bool = False,
    invert3: bool = False,
) -> torch.Tensor:
    if source is reference:
        return source.clone()
    if start_dim < 0:
        start_dim = source.ndim + start_dim
    orig_shape = source.shape
    s_flat = source.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)
    r_flat = reference.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)

    r_sorted = r_flat.sort(dim=-1, descending=invert1).values
    s_ranks = s_flat.argsort(
        dim=-1,
        descending=invert2,
    ).argsort(dim=-1, descending=invert3)

    # 4. Give the source elements the values from the reference.
    return (
        r_sorted.gather(dim=-1, index=s_ranks)
        .movedim(-1, start_dim)
        .reshape(orig_shape)
    )


# Scales the source tensor to match the median and variance of the reference.
def robust_scale_match(
    source: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
    # Default MAD if the reference is not passed. Targets the Gaussian distribution.
    mad: float = 0.6745,
    start_dim: int = 1,
    end_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    if start_dim < 0:
        start_dim = source.ndim + start_dim
    orig_shape = source.shape
    source = source.flatten(start_dim=start_dim, end_dim=end_dim).movedim(start_dim, -1)
    # Find the median and spread (MAD) of the source
    src_sub_median = source - source.median(dim=-1, keepdim=True).values
    s_mad = (
        src_sub_median.abs()
        .median(
            dim=-1,
            keepdim=True,
        )
        .values.clamp_min_(eps)
    )

    # If no reference, target a Standard Gaussian scale.
    # (A standard Gaussian has a median of 0 and a MAD of ~0.6745)
    if reference is None:
        mad = min(-eps, mad) if mad < 0 else max(eps, mad)
        return (
            src_sub_median.mul_(s_mad.reciprocal_().mul_(mad))
            .movedim(-1, start_dim)
            .reshape(orig_shape)
        )
    reference = reference.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)

    # Find the reference median and spread
    r_median = reference.median(dim=-1, keepdim=True).values
    r_mad = (reference - r_median).abs_().median(dim=-1, keepdim=True).values

    # Stretch the source to match the reference
    return (
        src_sub_median.mul_(r_mad.div_(s_mad))
        .add_(r_median)
        .movedim(-1, start_dim)
        .reshape(orig_shape)
    )


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
    result += (
        torch.linalg.vector_norm(b - a, dim=1, keepdim=True) / 6
    ) * interpolation_tensor

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
    norm = (torch.linalg.vector_norm(b - a, dim=1, keepdim=True) / 6) * interp
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
            torch.linalg.vector_norm(b - a, dim=1, keepdim=True).div_(6)
            * torch.where(t_expanded.abs() < sign_threshold, *sign_order),
        )
    )


# Copied from ComfyUI
def slerp_orig(b1, b2, r):
    c = b1.shape[-1]

    # norms
    b1_norms = torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
    b2_norms = torch.linalg.vector_norm(b2, dim=-1, keepdim=True)

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
def altslerp(
    v0: FloatTensor,
    v1: FloatTensor,
    t: float | FloatTensor,
    *,
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


def stochasistic_blend(
    a,
    b,
    t,
    *,
    cpu=False,
    fuzz=0.1,
    clamp_t: bool | tuple = True,
    blend=torch.lerp,
    **kwargs: Any,
):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor((t,), dtype=a.dtype, device=a.device)
    t_orig = t
    t = t.broadcast_to(a.shape)
    tadj = torch.rand(
        *t.shape,
        dtype=a.dtype,
        layout=a.layout,
        device="cpu" if cpu else a.device,
    )
    if tadj.device != a.device:
        tadj = tadj.to(a.device)
    tadj = tadj.mul_(fuzz * 2).sub_(fuzz)
    tadj += t
    if isinstance(clamp_t, tuple):
        tadj = tadj.clamp_(*clamp_t)
    elif clamp_t:
        tmin, tmax = t_orig.aminmax()
        tadj = tadj.clamp_(min(0, tmin), max(1.0, tmax))
    return blend(a, b, tadj, **pass_kwargs(kwargs))


def gaussian_smoothing(
    t: torch.Tensor,
    kernel_size,
    sigma: float | tuple | list,
) -> torch.Tensor:
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = (kernel_size,)
    if not isinstance(sigma, (list, tuple)):
        sigma = (sigma,)
    ndim = t.ndim
    ts = t.shape
    if ndim == 1:
        gk = kf.kernels.gaussian(
            kernel_size[0],
            torch.tensor(sigma, dtype=t.dtype, device=t.device),
            device=t.device,
            dtype=t.dtype,
        )[None, None, ...]
        return nnf.conv2d(
            t[None, None, None, ...],
            gk,
            padding=(0, gk.numel() // 2),
        ).view(t.numel())
    if ndim == 2:
        t = t[None, None, ...]
    elif ndim == 3:
        t = t[None, ...]
    elif ndim == 5:
        t = t.reshape(ts[0], ts[1] * ts[2], *ts[3:])
    elif ndim != 4:
        raise ValueError("Can't handle tensor shape")
    if len(kernel_size) == 1:
        kernel_size = kernel_size * 2
    if len(sigma) == 1:
        sigma = sigma * 2
    result = kf.gaussian_blur2d(t, kernel_size, sigma)
    if ndim == 5:
        return result.reshape(*ts)
    while result.ndim > ndim and result.shape[0] == 1:
        result = result.squeeze(0)
    return result


class ProbBlend:
    @staticmethod
    def output(a: torch.Tensor, b: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return torch.where(b_t.to(device=a.device, dtype=torch.bool), b, a)

    def __call__(
        self,
        a,
        b,
        t,
        *,
        cpu=False,
        collapse_dims=(),
        **kwargs: Any,
    ):
        t_device = torch.device("cpu") if cpu else a.device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor((t,), dtype=a.dtype, device=t_device)
        elif t.device != t_device:
            t = t.detach().clone().to(t_device)
        tmin, tmax = t.aminmax()
        tmin, tmax = min(tmin, 0.0), max(tmax, 1.0)
        t = t - tmin
        tdiv = tmax - tmin
        if tdiv != 0:
            t /= tdiv
        if collapse_dims:
            dims = a.ndim
            prob_shape = list(a.shape)
            for didx in collapse_dims:
                if didx >= dims:
                    continue
                prob_shape[didx] = 1
        else:
            prob_shape = a.shape
        t = torch.bernoulli(t.clamp_(0, 1).broadcast_to(prob_shape)).to(a)
        return self.output(a, b, t, **pass_kwargs(kwargs))


class ProbBlendSmoothed(ProbBlend):
    @staticmethod
    def output(
        a: torch.Tensor,
        b: torch.Tensor,
        b_t: torch.Tensor,
        *,
        output_blend=torch.lerp,
        kernel_size: int | tuple | list = 3,
        sigma: float | tuple | list = 1.0,
    ) -> torch.Tensor:
        t = b_t.to(device=a.device, dtype=a.dtype)
        t = gaussian_smoothing(t, kernel_size, sigma)
        return output_blend(a, b, t)


prob_blend = ProbBlend()
prob_blend_smoothed = ProbBlendSmoothed()


# Originally referenced from https://github.com/54rt1n/ComfyUI-DareMerge
# Doesn't handle non-scalar t very well.
def gradient_blend_(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    dim=-1,
    scaling_constant=0.9,
    blend_function=torch.lerp,
    **kwargs: Any,
) -> torch.Tensor:
    dim = max(0, min(a.ndim - 1, a.ndim + dim if dim < 0 else dim))
    if not isinstance(t, torch.Tensor):
        t = a.new_full((1,), t)
    if t.ndim > 0 and t.numel() > 1:
        t = t.broadcast_to(a.shape).mean(dim=dim, keepdim=True)
    count = a.shape[dim]
    peak_idx = int(count * (1 - t))
    ratios = a.new_zeros(count)
    torch.arange(peak_idx, out=ratios[:peak_idx]).div_(peak_idx)
    torch.arange(count - peak_idx - 1, -1, -1, out=ratios[peak_idx:]).div_(
        count - peak_idx,
    )
    if scaling_constant != 1:
        ratios *= scaling_constant
    ratios = ratios.view(tuple(1 if i != dim else -1 for i in range(a.ndim)))
    return blend_function(a, b, ratios, **pass_kwargs(kwargs))


def gradient_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    flatten_start_dim=1,
    scaling_constant=0.9,
    blend_function=torch.lerp,
    **kwargs: Any,
) -> torch.Tensor:
    shape = a.shape
    # print("\nBLEND:", t)
    if isinstance(t, torch.Tensor) and t.ndim > 0 and t.numel() > 1:
        t = t.mean()
    if a.ndim > 2:
        a = a.flatten(start_dim=flatten_start_dim)
        b = b.flatten(start_dim=flatten_start_dim)
    count = a.shape[-1]
    peak_idx = int(count * (1 - t))
    ratios = a.new_zeros(count)
    torch.arange(peak_idx, out=ratios[:peak_idx]).div_(peak_idx)
    torch.arange(count - peak_idx - 1, -1, -1, out=ratios[peak_idx:]).div_(
        count - peak_idx,
    )
    if scaling_constant != 1:
        ratios *= scaling_constant
    result = blend_function(a, b, ratios, **pass_kwargs(kwargs))
    if result.shape != shape:
        return result.reshape(*shape).contiguous()
    return result


def slice_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    flatten=True,
    dim=1,
    flip_a=False,
    flip_b=False,
    flip_out=False,
) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        t = t.mean().clamp(0, 1)
    else:
        t = a.new_full((1,), t).clamp(0, 1)
    if t == 0:
        return a
    if t == 1:
        return b
    orig_shape = a.shape
    if a.ndim > 2 and flatten:
        a = a.flatten(start_dim=dim)
        b = b.flatten(start_dim=dim)
    elsb = int(a.shape[dim] * t)
    elsa = a.shape[dim] - elsb
    astart, aend = (None, elsa) if not flip_a else (a.shape[dim] - elsa, None)
    bstart, bend = (None, elsb) if flip_b else (a.shape[dim] - elsb, None)
    aslice = tuple(
        slice(None) if i != dim else slice(astart, aend) for i in range(a.ndim)
    )
    bslice = tuple(
        slice(None) if i != dim else slice(bstart, bend) for i in range(a.ndim)
    )
    achunk, bchunk = a[aslice], b[bslice]
    result = torch.cat((bchunk, achunk) if flip_out else (achunk, bchunk), dim=dim)
    return result.reshape(orig_shape)


def slice_blend_smooth(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    flatten: bool = True,
    dim: int = 1,
    fade_percent_l: float = 0.1,
    fade_percent_r: float = 0.1,
    always_fade: bool = False,
    b_start_percent: float = 1.0,
    b_blend_max: float = 1.0,
    invert: bool = False,  # Doesn't work propertly at the moment.
    blend_function=torch.lerp,
    **kwargs: Any,
) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        t = t.mean().clamp(0, 1)
    else:
        t = a.new_full((1,), t).clamp_(0, 1)
    if invert:
        t = 1 - t
        b, a = a, b
    if t == 0:
        return a
    b_start_percent = max(0.0, min(1.0, b_start_percent))
    fade_percent_l = (
        max(0.0, min(1.0, fade_percent_l))
        if b_start_percent > 0 and not always_fade
        else 0.0
    )
    fade_percent_r = (
        max(0.0, min(1.0, fade_percent_r))
        if b_start_percent < 1 and not always_fade
        else 0.0
    )
    fade_mul = 1.0 / max(1.0, fade_percent_l + fade_percent_r)
    orig_shape = a.shape
    if flatten and dim < a.ndim - 1:
        a = a.flatten(start_dim=dim)
        b = b.flatten(start_dim=dim)
    dim_els = a.shape[dim]
    els_b = int(dim_els * t)
    if invert:
        els_b += int((dim_els - els_b) * (fade_percent_l + fade_percent_r) * fade_mul)

    els_a = dim_els - els_b
    b_start = int(els_a * b_start_percent)
    b_end = b_start + els_b
    elslfade, elsrfade = (
        int(els_b * fade_percent_l * fade_mul),
        int(els_b * fade_percent_r * fade_mul),
    )
    blend_mask = a.new_zeros(dim_els)
    blend_mask[b_start:b_end] = b_blend_max
    if elslfade > 0:
        blend_mask[b_start : b_start + elslfade] = torch.linspace(
            0.0,
            b_blend_max,
            steps=elslfade + 2,
            device=blend_mask.device,
            dtype=blend_mask.dtype,
        )[1:-1]
    if elsrfade > 0:
        rfade_start = b_end - elsrfade
        blend_mask[rfade_start : rfade_start + elsrfade] = torch.linspace(
            b_blend_max,
            0.0,
            steps=elsrfade + 2,
            device=blend_mask.device,
            dtype=blend_mask.dtype,
        )[1:-1]
    blend_mask = blend_mask.view(
        tuple(dim_els if d == dim else 1 for d in range(a.ndim)),
    )
    return blend_function(a, b, blend_mask, **pass_kwargs(kwargs)).reshape(orig_shape)


def lop_lerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.tensor | float,
    *,
    a_ratio=1.0,
    b_ratio=1.0,
):
    if not isinstance(t, torch.Tensor):
        t = a.new_full((1,), t)
    return (a_ratio - t.clamp(max=a_ratio)).mul(a).add_(b * (t * b_ratio))


# Thanks, ChatGPT though you did get the ratio reversed.
def cosine_similarity_blend_chatgpt_orig(
    b: torch.Tensor,
    a: torch.Tensor,
    ratio: float | torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    a_n = a / (a.norm(dim=dim, keepdim=True).clamp_min(eps))
    b_n = b / (b.norm(dim=dim, keepdim=True).clamp_min(eps))

    c = torch.sum(a_n * b_n, dim=dim, keepdim=True)

    s = 2 * ratio - 1
    if not torch.is_tensor(s):
        s = a.new_tensor(s)

    a_ = 1 - c
    alpha = a_ * (a_ - 2 * s**2)
    beta = 2 * a_ * (c + s**2)
    gamma = c**2 - s**2

    disc = beta**2 - 4 * alpha * gamma

    disc = disc.clamp_min(0.0)
    sqrt_disc = torch.sqrt(disc)

    lam1 = (-beta + sqrt_disc) / (2 * alpha).clamp_min(eps)
    lam2 = (-beta - sqrt_disc) / (2 * alpha).clamp_min(eps)

    lam = torch.where((lam1 >= 0) & (lam1 <= 1), lam1, lam2)
    lam = torch.where((lam >= 0) & (lam <= 1), lam, ratio)

    return torch.lerp(b, a, lam.expand_as(a))


def cosine_similarity_blend_chatgpt(  # noqa: PLR0914
    a: torch.Tensor,
    b: torch.Tensor,
    ratio: float,
    *,
    dim: int = -1,
    eps: float = 1e-8,
    small_angle: float = 1e-4,
    opp_eps: float = 1e-6,
) -> torch.Tensor:
    # --- normalize directions ---
    mag_a = a.norm(dim=dim, keepdim=True).clamp_min(eps)
    mag_b = b.norm(dim=dim, keepdim=True).clamp_min(eps)
    a_n = a / mag_a
    b_n = b / mag_b

    # cosine & angle between a and b
    cos_ab = (a_n * b_n).sum(dim=dim, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_ab)

    # map blend ratio -> fraction along the arc
    # we want angle from a -> out = theta * t
    t = ratio if torch.is_tensor(ratio) else a.new_tensor(ratio)

    # handle exact-opposite case: fallback to lerp then renorm
    opp_mask = torch.abs(cos_ab + 1) < opp_eps
    if opp_mask.any():
        # simple normalized lerp + renormalize
        lerp_dir = (1 - t) * a_n + t * b_n
        lerp_dir /= lerp_dir.norm(dim=dim, keepdim=True).clamp_min(eps)
        # magnitude later will apply
        a_n = torch.where(opp_mask, lerp_dir, a_n)
        b_n = torch.where(opp_mask, b_n, b_n)  # no-op but keeps shapes aligned
        theta = torch.where(
            opp_mask,
            torch.acos((a_n * b_n).sum(dim=dim, keepdim=True)),
            theta,
        )

    # for very small angles, do lerp+renormalize
    lerp_mask = theta < small_angle
    if lerp_mask.any():
        lerp_dir = (1 - t) * a_n + t * b_n
        lerp_dir /= lerp_dir.norm(dim=dim, keepdim=True).clamp_min(eps)
        # override only where theta is small
        a_n = torch.where(lerp_mask, lerp_dir, a_n)
        b_n = torch.where(lerp_mask, b_n, b_n)
        cos_ab = (a_n * b_n).sum(dim=dim, keepdim=True).clamp(-1, 1)
        theta = torch.acos(cos_ab)

    # now true SLERP coefficients
    sin_theta = torch.sin(theta).clamp_min(eps)
    coef_a = torch.sin((1 - t) * theta) / sin_theta
    coef_b = torch.sin(t * theta) / sin_theta

    dir_out = coef_a * a_n + coef_b * b_n

    # --- geometric magnitude interpolation ---
    log_a = torch.log(mag_a)
    log_b = torch.log(mag_b)
    log_out = (1 - t) * log_a + t * log_b
    mag_out = torch.exp(log_out)

    return dir_out * mag_out


def cosine_similarity_blend_deepseek(  # noqa: PLR0914
    a: torch.Tensor,
    b: torch.Tensor,
    ratio: float,
    *,
    dim: int = -1,
    eps=1e-08,
    threshold=1e-06,
) -> torch.Tensor:
    if not torch.is_tensor(ratio):
        ratio = a.new_tensor(ratio)

    # Compute magnitudes of a and b along the specified dimension
    mag_a = torch.linalg.vector_norm(a, p=2, dim=dim, keepdim=True).add_(eps)
    mag_b = torch.linalg.vector_norm(b, p=2, dim=dim, keepdim=True).add_(eps)

    # Avoid division by zero during normalization
    a_norm = a / mag_a
    b_norm = b / mag_b

    # Compute cosine similarity (dot product of normalized vectors)
    d = (a_norm * b_norm).sum(dim=dim, keepdim=True).clamp_(-1.0, 1.0)

    # Compute angle between a_norm and b_norm
    theta = torch.acos(d)

    # Calculate desired cosine similarity with b (s_b) from blend ratio
    s_b = (2.0 * ratio - 1.0).clamp_(-1.0, 1.0)

    # Compute angle from result to b based on s_b
    angle_from_b = torch.acos(s_b)

    # Calculate interpolation parameter t_val
    t_val = (1.0 - angle_from_b / theta).clamp_(0.0, 1.0)

    # Precompute sin_theta for slerp
    sin_theta = torch.sin(theta)

    # Linear interpolation fallback for small sin_theta
    linear_part_norm = torch.lerp(a_norm, b_norm, t_val)
    # linear_part_norm = (1.0 - t_val) * a_norm + t_val * b_norm

    # Slerp computation
    sin_t_theta = torch.sin(t_val * theta)
    sin_comp_theta = torch.sin((1.0 - t_val) * theta)
    slerp_denom = sin_theta + eps  # Avoid division by zero
    slerp_part_norm = (sin_comp_theta / slerp_denom) * a_norm + (
        sin_t_theta / slerp_denom
    ) * b_norm

    # Choose slerp unless sin_theta is too small (use linear then)
    v_norm = torch.where(sin_theta < threshold, linear_part_norm, slerp_part_norm)

    # Linearly interpolate magnitude
    mag = torch.lerp(mag_a, mag_b, ratio)

    # Scale normalized vector by interpolated magnitude
    return v_norm * mag


DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND = "chatgpt"
COSINE_SIMILARITY_BLEND_BACKENDS = {
    "altslerp": altslerp,
    "deepseek": cosine_similarity_blend_deepseek,
    "chatgpt": cosine_similarity_blend_chatgpt,
    "chatgpt_orig": cosine_similarity_blend_chatgpt_orig,
}


def cosine_similarity_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *args: Any,
    backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    **kwargs: Any,
) -> torch.Tensor:
    fun = COSINE_SIMILARITY_BLEND_BACKENDS.get(backend)
    if fun is None:
        errstr = f"Bad cosine similarity blend backend {backend}, must be one of {tuple(COSINE_SIMILARITY_BLEND_BACKENDS)}"
        raise ValueError(errstr)
    return fun(a, b, t, *args, **kwargs)


def cosine_similarity_blend_avg(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    dims=(-1, -2),
) -> torch.Tensor:
    blend_fun = partial(cosine_similarity_blend, backend=backend)
    multiplier = 1.0 / len(dims)
    result = None
    for dim in dims:
        curr_result = blend_fun(a, b, t, dim=dim).mul_(multiplier)
        result = curr_result if result is None else result.add_(curr_result)
    return result


def cosine_similarity_blend_flat(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    start_dim=0,
    end_dim=1,
    backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("Tensor shape mismatch, a and b must be the same shape")
    if start_dim < 0:
        start_dim = a.ndim + start_dim
    if end_dim < 0:
        end_dim = a.ndim + end_dim
    if start_dim < 0 or end_dim < 0 or start_dim >= a.ndim or end_dim >= a.ndim:
        raise ValueError("Bad start/end_dim parameters")
    orig_shape = a.shape
    a = a.flatten(start_dim=start_dim, end_dim=end_dim)
    b = b.flatten(start_dim=start_dim, end_dim=end_dim)
    if isinstance(t, torch.Tensor) and t.ndim == len(orig_shape):
        t = t.flatten(start_dim=start_dim, end_dim=end_dim)
    return cosine_similarity_blend(a, b, t, dim=start_dim, backend=backend).reshape(
        orig_shape,
    )


def blend_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    blend_mode_a: str = "lerp",
    blend_mode_b="cosinesimilarity_flat_spatdims",
    blend_blend: float | torch.Tensor = 0.5,
    blend_mode_blend: str = "lerp",
    blend_a_kwargs: dict | None = None,
    blend_b_kwargs: dict | None = None,
    blend_blend_kwargs: dict | None = None,
) -> torch.Tensor:
    fun_a = BLENDING_MODES[blend_mode_a]
    fun_b = BLENDING_MODES[blend_mode_b]
    fun_blend = BLENDING_MODES[blend_mode_blend]
    if not torch.is_tensor(blend_blend):
        blend_blend = a.new_tensor(blend_blend)
    return fun_blend(
        fun_a(a, b, t, **({} if blend_a_kwargs is None else blend_a_kwargs)),
        fun_b(a, b, t, **({} if blend_b_kwargs is None else blend_b_kwargs)),
        blend_blend,
        **({} if blend_blend_kwargs is None else blend_blend_kwargs),
    )


def ortho_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float,
    *,
    blend_mode: str | Callable | None = None,
    proj_scale: float = -1.0,
    ortho_scale: float = 1.0,
    start_dim: int = 1,
    end_dim: int = -1,
    rescale_limit: float = 0.0,
    # a, b, blend or None
    rescale_result_mode: str | None = None,
    # When rescale_target mode is blend, will use blend_mode if None.
    rescale_result_blend_mode: str | Callable | None = None,
    # LERP if None.
    dyn_result_blend_mode: str | Callable | None = None,
    dyn_ortho_mode: bool = False,
    dyn_min_scale: float = 0.0,
    dyn_max_scale: float = 1.0,
    # Can only be used when the flattened tensor has 4 dimensions left.
    smooth_factor_kernel_size: int | tuple[int, ...] = 0,
    ortho_verbose: bool = False,
    eps: float = 1e-06,
) -> torch.Tensor:
    orig_shape = a.shape
    if not isinstance(smooth_factor_kernel_size, tuple):
        smooth_factor_kernel_size = (smooth_factor_kernel_size | 1,)
    else:
        smooth_factor_kernel_size = tuple(sz | 1 for sz in smooth_factor_kernel_size)
    ndim = a.ndim
    if start_dim < 0:
        start_dim = max(0, min(ndim + start_dim, ndim - 1))
    if end_dim < 0:
        end_dim = max(0, min(ndim + end_dim, ndim - 1))
    if start_dim > end_dim:
        start_dim, end_dim = end_dim, start_dim
    if not isinstance(t, torch.Tensor):
        t = a.new_tensor(t)
        sync_t = False
    else:
        t = t.broadcast_to(a.shape)
        sync_t = True
    if sync_t:
        t = t.flatten(start_dim=start_dim, end_dim=end_dim)
    a = a.flatten(start_dim=start_dim, end_dim=end_dim)
    b = b.flatten(start_dim=start_dim, end_dim=end_dim)
    if end_dim != ndim - 1:
        a = a.movedim(start_dim, -1)
        b = b.movedim(start_dim, -1)
        if sync_t:
            t = t.movedim(start_dim, -1)
    if start_dim == 0:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        if sync_t:
            t = t.unsqueeze(0)
    b_normed = b.norm(dim=-1, keepdim=True) if rescale_limit else None
    dot_ba = (b * a).sum(dim=-1, keepdim=True)
    dot_aa = (a**2).sum(dim=-1, keepdim=True)
    proj = (dot_ba / (dot_aa + eps)) * a
    proj *= proj_scale
    b_ortho = proj.add_(b if ortho_scale == 1.0 else b * ortho_scale)
    if b_normed is not None:
        rescale_limit = abs(rescale_limit)
        if rescale_limit == 1:
            rescale_limit += eps
        b_ortho_normed = b_ortho.norm(dim=-1, keepdim=True)
        b_ortho_normed += eps
        b_normed /= b_ortho_normed
        b_normed = b_normed.clamp_(-rescale_limit, rescale_limit)
        b_ortho *= b_normed
    if blend_mode is None:

        def blend_function(a, b, t):
            return (b * t).add_(a)
    else:
        blend_function = (
            BLENDING_MODES[blend_mode] if isinstance(blend_mode, str) else blend_mode
        )
    ortho_result = blend_function(a, b_ortho, t)
    rr_blend_function = None
    if rescale_result_mode == "a":
        rescale_result_target = a
    elif rescale_result_mode == "b":
        rescale_result_target = b
    elif rescale_result_mode == "blend":
        rr_blend_function = (
            blend_function
            if rescale_result_blend_mode is None
            else (
                BLENDING_MODES[rescale_result_blend_mode]
                if isinstance(rescale_result_blend_mode, str)
                else rescale_result_blend_mode
            )
        )
        rescale_result_target = rr_blend_function(a, b, t)
    else:
        rescale_result_target = None
    if rr_blend_function is None:
        rr_blend_function = blend_function
    if rescale_result_target is not None:
        result_norm = ortho_result.norm(dim=-1, keepdim=True).add_(eps)
        target_norm = rescale_result_target.norm(dim=-1, keepdim=True)
        target_norm /= result_norm
        ortho_result *= target_norm
    if b_normed is not None and dyn_ortho_mode:
        vanilla_result = (
            rr_blend_function(a, b, t)
            if rescale_result_mode != "blend"
            else rescale_result_target
        )
        dyn_blend_function = (
            torch.lerp
            if dyn_result_blend_mode is None
            else (
                BLENDING_MODES[dyn_result_blend_mode]
                if isinstance(dyn_result_blend_mode, str)
                else dyn_result_blend_mode
            )
        )
        ortho_factor = (
            (1.0 - ((b_normed - 1.0) / (rescale_limit - 1.0)).clamp_(0.0, 1.0))
            .add_(dyn_min_scale)
            .mul_(dyn_max_scale - dyn_min_scale)
        )
        if smooth_factor_kernel_size not in {0, 1}:
            if ortho_factor.ndim < 3:
                raise ValueError(
                    f"Can't use smooth_factor_kernel_size when ortho_factor has less than 3 dimensions. It has shape: {ortho_factor.shape}",
                )
            pad_sizes = tuple((sz - 1) // 2 for sz in smooth_factor_kernel_size)
            target_dim = -3 if ortho_factor.ndim > 3 else -2
            pool_fun = (
                torch.nn.functional.avg_pool2d
                if target_dim == -3
                else torch.nn.functional.avg_pool1d
            )
            ortho_factor = (
                pool_fun(
                    ortho_factor.movedim(-1, target_dim),
                    kernel_size=smooth_factor_kernel_size,
                    stride=1,
                    padding=pad_sizes,
                )
                .movedim(target_dim, -1)
                .clamp_(dyn_min_scale, dyn_max_scale)
            )
        ortho_result = dyn_blend_function(vanilla_result, ortho_result, ortho_factor)
        if ortho_verbose:
            tqdm.write(
                f"ORTHO BLEND: b_norm min/max={b_normed.aminmax()}, avg: {ortho_factor.mean().item():.5f}, min: {ortho_factor.min().item():.5f}, max: {ortho_factor.max().item():.5f}",
            )
    if end_dim != ndim - 1:
        ortho_result = ortho_result.movedim(-1, start_dim)
    return ortho_result.reshape(orig_shape)


def symmetric_ortho_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    *,
    symmetric_strength: float = 1.0,
    symmetric_deduce_mode: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    blended = ortho_blend(a, b, t, **kwargs)
    if symmetric_strength == 0.0:
        return blended
    b_ortho = blended.sub_(a)
    if symmetric_deduce_mode:
        b_proj = b - b_ortho
        # Projection would theoretically be the same for both, in the simple case at least?
        # Actually, probably not. Oh well, this is here as an option now.
        a_ortho = a - b_proj
    else:
        a_ortho = ortho_blend(b, a, a.new_tensor(1.0), **kwargs) - b
    a_proj = a - a_ortho
    return a_proj.mul_(1.0 - symmetric_strength).add_(a_ortho).add_(b_ortho)


def contrastive_ortho_cfg_base_a(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor = 1.0,
    *,
    a_blend: float = 1.0,
    b_blend: float = 1.0,
    b_blend_sub_a_scale: float = 1.0,
    a_blend_sub_b_scale: float = 1.0,
    a_ortho_scale: torch.Tensor | float = 1.0,
    b_ortho_scale: torch.Tensor | float = 1.0,
    a_blend_kwargs: dict | None = None,
    b_blend_kwargs: dict | None = None,
    # If a is cond and b is cond - uncond (CFG diff), you could
    # set this to -1.0 to get b=uncond
    b_from_a_scale: float = 0.0,
    # One of: add, lerp, add_t, lerp_t
    # When using _t modes, swaps t and b_from_a_scale.
    b_from_a_mode: str = "add",
    output_base_mode: str = "a",
    # a, b, None or a tensor reference
    final_rescale_target: str | torch.Tensor | None = None,
    final_rescale_energy: float = 1.0,
    # Should be left at 0.
    final_rescale_ortho_blend: float = 0.0,
    final_rescale_strength: float = 1.0,
    final_rescale_kwargs: dict | None = None,
    diff_only: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    kwargs = pass_kwargs(kwargs)
    t_is_tensor = isinstance(t, torch.Tensor)
    if not t_is_tensor:
        if t == 0.0:
            return a
    else:
        t = t.broadcast_to(a.shape)

    if b_from_a_scale != 0:
        if b_from_a_mode.endswith("_t"):
            b_from_a_mode = b_from_a_mode[:-2]
            t, b_from_a_scale = b_from_a_scale, t
            t_is_tensor = isinstance(t, torch.Tensor)
            if isinstance(b_from_a_scale, torch.Tensor):
                b_from_a_scale = b_from_a_scale.broadcast_to(a.shape)
        if b_from_a_mode == "add":
            b = (b * b_from_a_scale).add_(a)
        elif b_from_a_mode == "lerp":
            b = a.lerp(b, b_from_a_scale)
        else:
            raise ValueError("Bad b_from_a_mode")

    # Extract features unique to b (b orthogonal to a)
    b_blend_kwargs = kwargs if b_blend_kwargs is None else kwargs | b_blend_kwargs
    b_ortho = ortho_blend(a, b, a.new_tensor(b_blend), **b_blend_kwargs).sub_(
        a if b_blend_sub_a_scale == 1.0 else a * b_blend_sub_a_scale,
    )
    if isinstance(b_ortho_scale, torch.Tensor) or b_ortho_scale != 1.0:
        b_ortho *= (
            b_ortho_scale.broadcast_to(b_ortho)
            if isinstance(b_ortho_scale, torch.Tensor)
            else b_ortho_scale
        )

    # Extract features unique to a (a orthogonal to b)
    a_blend_kwargs = kwargs if a_blend_kwargs is None else kwargs | a_blend_kwargs
    a_ortho = ortho_blend(b, a, a.new_tensor(a_blend), **a_blend_kwargs).sub_(
        b if a_blend_sub_b_scale == 1.0 else b * a_blend_sub_b_scale,
    )
    if isinstance(a_ortho_scale, torch.Tensor) or a_ortho_scale != 1.0:
        a_ortho *= (
            a_ortho_scale.broadcast_to(a_ortho)
            if isinstance(a_ortho_scale, torch.Tensor)
            else a_ortho_scale
        )

    # Create the contrastive guidance vector
    # Push towards the a-unique features, pull away from the b-unique features
    guidance = a_ortho.sub_(b_ortho)

    if t_is_tensor or t != 1.0:
        guidance *= t
    output_base = a if output_base_mode == "a" else b
    if (
        not isinstance(final_rescale_target, torch.Tensor)
        and final_rescale_target not in {"a", "b", "mid"}
    ) or final_rescale_strength == 0.0:
        return guidance if diff_only else guidance.add_(output_base)
    result = guidance.add_(output_base)
    final_rescale_kwargs = (
        kwargs | final_rescale_kwargs
        if final_rescale_kwargs is not None
        else kwargs.copy()
    )
    final_rescale_kwargs["rescale_result_mode"] = "b"
    if "rescale_limit" not in final_rescale_kwargs:
        final_rescale_kwargs["rescale_limit"] = 2.0
    if isinstance(final_rescale_target, torch.Tensor):
        target_b = final_rescale_target.broadcast_to(result.shape)
    elif final_rescale_target == "b":
        target_b = b
    elif final_rescale_target == "mid":
        target_b = a.lerp(b, 0.5)
    else:
        target_b = a
    if final_rescale_energy != 1.0:
        target_b = target_b * final_rescale_energy
    final_result = ortho_blend(
        result,
        target_b,
        result.new_tensor(final_rescale_ortho_blend),
        **final_rescale_kwargs,
    )
    if final_rescale_strength != 1.0:
        final_result = (final_result - result).mul_(final_rescale_strength).add_(result)
    return final_result - output_base if diff_only else final_result


class WaveletBlend:
    wavelet: wavef.Wavelet | None = None
    use_float64: bool = False

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        use_float64: bool = False,
        **kwargs: Any,
    ):
        self.device = device
        self.wavelet_kwargs = kwargs
        self.use_float64 = use_float64

    def get_wavelet(self, *, device: str | torch.device | None = None) -> wavef.Wavelet:
        if self.wavelet is None:
            self.wavelet = wavef.Wavelet(
                device=device if device is not None else self.device,
                **self.wavelet_kwargs,
            ).to(dtype=torch.float64 if self.use_float64 else torch.float32)
            self.device = device
            return self.wavelet
        if device is not None and self.wavelet.device != device:
            self.wavelet = self.wavelet.to(device=device)
            self.device = device
        return self.wavelet

    @staticmethod
    def maybe_offset(
        yl: torch.Tensor,
        yh: Sequence[torch.Tensor],
        offset_yl: float | torch.Tensor | None,
        offset_yh: float | Sequence[float | Sequence[float]] | None,
        *,
        in_place: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if offset_yl in {None, 1.0} and offset_yh in {None, 1.0}:
            return (yl, tuple(yh))
        return wavef.wavelet_scaling(
            yl,
            yh,
            yl_scale=offset_yl if offset_yl is not None else 1.0,
            yh_scales=offset_yh,
            in_place=in_place,
        )

    def wavelet_blend(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: float | torch.Tensor,
        *,
        blend_mode_yl: str | Callable = torch.lerp,
        blend_mode_yh: str | Callable | None = None,
        a_offset_yl: float | torch.Tensor | None = None,
        a_offset_yh: float | Sequence[float | Sequence[float]] | None = None,
        b_offset_yl: float | torch.Tensor | None = None,
        b_offset_yh: float | Sequence[float | Sequence[float]] | None = None,
        out_offset_yl: float | torch.Tensor | None = None,
        out_offset_yh: float | Sequence[float | Sequence[float]] | None = None,
        blend_yl_offset: float = 1.0,
        blend_yh_offset: float | torch.Tensor = 1.0,
        two_step_inverse: bool = False,
        in_place_offset: bool = True,
    ) -> torch.Tensor:
        if isinstance(blend_mode_yl, str):
            blend_mode_yl = BLENDING_MODES[blend_mode_yl]
        if blend_mode_yh is None:
            blend_mode_yh = blend_mode_yl
        elif isinstance(blend_mode_yh, str):
            blend_mode_yh = BLENDING_MODES[blend_mode_yh]
        wavelet = self.get_wavelet(device=a.device)
        dtype = a.dtype
        if a.ndim != b.ndim:
            raise ValueError(
                f"Tensor a ndim ({a.ndim}) must match tensor b ndim ({b.ndim})"
            )
        orig_shape = a.shape
        # FIXME: This reshaping logic is almost certainly not reliable.
        if a.ndim > 4:
            a = a.reshape(a.shape[0], -1, *a.shape[-2:])
        if b.ndim > 4:
            b = a.reshape(b.shape[0], -1, *b.shape[-2:])
        a = a.to(dtype=torch.float64 if self.use_float64 else torch.float32)
        b = b.to(a)
        t = a.new_tensor(t) if not isinstance(t, torch.Tensor) else t.to(a)
        if t.ndim > 4:
            t = a.reshape(t.shape[0], -1, *t.shape[-2:])
        aw_l, aw_h = self.maybe_offset(
            *wavelet.forward(a),
            a_offset_yl,
            a_offset_yh,
            in_place=in_place_offset,
        )
        bw_l, bw_h = self.maybe_offset(
            *wavelet.forward(b),
            b_offset_yl,
            b_offset_yh,
            in_place=in_place_offset,
        )
        blend_yl_offset = t if blend_yl_offset == 1 else t * blend_yl_offset
        blend_yh_offset = t if blend_yh_offset == 1 else t * blend_yh_offset
        outw_l, outw_h = self.maybe_offset(
            *wavef.wavelet_blend(
                (aw_l, aw_h),
                (bw_l, bw_h),
                yl_factor=blend_yl_offset,
                yh_factor=blend_yh_offset,
                blend_function=blend_mode_yl,
                yh_blend_function=blend_mode_yh,
            ),
            offset_yl=out_offset_yl,
            offset_yh=out_offset_yh,
            in_place=in_place_offset,
        )
        result = wavelet.inverse(outw_l, outw_h, two_step_inverse=two_step_inverse)
        result = result[tuple(slice(None, dsize) for dsize in a.shape)]
        return result.to(dtype=dtype).reshape(orig_shape)


WAVELET_BLEND_CACHE: dict[frozenset[tuple[str, Any]], WaveletBlend] = {}


def wavelet_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    blend_mode_yl: str | Callable = torch.lerp,
    blend_mode_yh: str | Callable | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if isinstance(blend_mode_yl, str):
        blend_mode_yl = BLENDING_MODES[blend_mode_yl]
    if blend_mode_yh is None:
        blend_mode_yh = blend_mode_yl
    _ = kwargs.pop("device", None)
    wavelet_kwargs = {
        k: kwargs.pop(k)
        for k in (
            "wave",
            "level",
            "mode",
            "use_1d_dwt",
            "use_dtcwt",
            "biort",
            "qshift",
            "inv_wave",
            "inv_mode",
            "inv_biort",
            "inv_qshift",
            "two_step_inverse",
            "use_float64",
        )
        if k in kwargs
    }
    cache_key = frozenset(
        (
            wavelet_kwargs
            | {"blend_mode_yl": blend_mode_yl, "blend_mode_yh": blend_mode_yh}
        ).items(),
    )
    print(f"\nWAVELET BLEND: cache key: {cache_key}")
    wb = WAVELET_BLEND_CACHE.get(cache_key)
    if wb is None:
        wb = WaveletBlend(device=a.device, **wavelet_kwargs)
        WAVELET_BLEND_CACHE[cache_key] = wb
    return wb.wavelet_blend(
        a,
        b,
        t,
        blend_mode_yl=blend_mode_yl,
        blend_mode_yh=blend_mode_yh,
        **kwargs,
    )


class TieredBlendWrapper:
    def __init__(
        self,
        blend_function: Callable,
        *,
        tiers: int = 16,
        start_dim: int = 1,
        end_dim: int = -1,
        descending: bool = True,
        abs_mode: bool = False,
        # a, b, add, sub, lerp (50% LERP)
        sort_target: str = "a",
        pad_value: float = 0.0,
    ):
        """Wraps any blending function to operate on 'Probability Tiers'.

        :param blend_function: Callable with signature (a, b, blend_ratio)
        :param tiers: Number of fake 'channels' or tiers to divide the data into.
        :param start_dim: The first dimension to flatten.
        :param end_dim: The last dimension to flatten.
        :param descending: Sort highest-to-lowest (True) or lowest-to-highest (False).
        :param pad_value: Value to pad with if the flattened size isn't divisible by tiers.
                          (For logits, -math.inf might be better, but 0.0 is safe for latents).
        """
        self.blend_function = blend_function
        self.tiers = tiers
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.descending = descending
        self.abs_mode = abs_mode
        self.sort_target = sort_target
        self.pad_value = pad_value

    def get_target(self, a_flat: torch.Tensor, b_flat: torch.Tensor) -> torch.Tensor:
        starget = self.sort_target
        if starget == "a":
            return a_flat
        if starget == "b":
            return b_flat
        if starget == "add":
            return a_flat + b_flat
        if starget == "sub":
            return a_flat - b_flat
        if starget == "lerp":
            return a_flat.lerp(b_flat, 0.5)
        raise ValueError("Invalid sort target")

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor | float,
        t: torch.Tensor | float,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.tiers < 1:
            return self.blend_function(a, b, t, **kwargs)

        orig_shape = a.shape

        start_dim, end_dim = (
            d if d >= 0 else a.ndim + d for d in (self.start_dim, self.end_dim)
        )
        if any(d < 0 or d > a.ndim for d in (start_dim, end_dim)):
            raise ValueError("Dimension out of range")

        b = (
            b.broadcast_to(orig_shape)
            if isinstance(b, torch.Tensor)
            else torch.full_like(a, fill_value=b)
        )

        a_flat = a.flatten(start_dim=start_dim, end_dim=end_dim)
        b_flat = b.flatten(start_dim=start_dim, end_dim=end_dim)

        t_is_tensor = isinstance(t, torch.Tensor) and t.numel() > 1
        if t_is_tensor:
            t_flat = t.broadcast_to(orig_shape).flatten(
                start_dim=start_dim,
                end_dim=end_dim,
            )
        else:
            t_flat = t

        a_flat = a_flat.transpose(start_dim, -1)
        b_flat = b_flat.transpose(start_dim, -1)
        if t_is_tensor:
            t_flat = t_flat.transpose(start_dim, -1)

        length = a_flat.shape[-1]

        target = self.get_target(a_flat, b_flat)
        if self.abs_mode:
            target = target.abs()

        target_vals, indices = torch.sort(target, dim=-1, descending=self.descending)
        del target
        if not self.abs_mode and self.sort_target == "a":
            a_vals = target_vals
        else:
            del target_vals
            a_vals = torch.gather(a_flat, dim=-1, index=indices)
        b_vals = torch.gather(b_flat, dim=-1, index=indices)
        if t_is_tensor:
            t_vals = torch.gather(t_flat, dim=-1, index=indices)

        # PAD (If length is not divisible by tiers)
        pad_len = (self.tiers - (length % self.tiers)) % self.tiers
        if pad_len > 0:
            a_vals = nnf.pad(a_vals, (0, pad_len), value=self.pad_value)
            b_vals = nnf.pad(b_vals, (0, pad_len), value=self.pad_value)
            if t_is_tensor:
                t_vals = nnf.pad(t_vals, (0, pad_len), value=self.pad_value)

        # Reshape into tiers (e.g., [..., length] -> [..., tiers, features])
        new_shape = (*a_vals.shape[:-1], self.tiers, -1)
        a_tiered = a_vals.reshape(new_shape)
        b_tiered = b_vals.reshape(new_shape)
        effective_t = t_vals.reshape(new_shape) if t_is_tensor else t

        blended_tiered = self.blend_function(a_tiered, b_tiered, effective_t, **kwargs)

        blended_flat = blended_tiered.reshape(a_vals.shape)

        if pad_len > 0:
            blended_flat = blended_flat[..., :-pad_len]

        # Scatter back to original element positions
        result_flat = torch.empty_like(blended_flat)
        result_flat.scatter_(dim=-1, index=indices, src=blended_flat)

        return result_flat.transpose(start_dim, -1).reshape(orig_shape)


def tiered_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float,
    *,
    tiers_blend_mode: str | Callable = "lerp",
    tiers_blend_kwargs: dict | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if isinstance(tiers_blend_mode, str):
        tiers_blend_mode = BLENDING_MODES[tiers_blend_mode]
    tw_kwargs_map = {
        "tiers": "tiers",
        "tiers_start_dim": "start_dim",
        "tiers_end_dim": "end_dim",
        "tiers_descending": "descending",
        "tiers_abs_mode": "abs_mode",
        "tiers_sort_target": "sort_target",
        "tiers_pad_value": "pad_value",
    }
    tw_kwargs = {tk: kwargs.pop(k) for k, tk in tw_kwargs_map.items() if k in kwargs}
    wrapped_blend_function = TieredBlendWrapper(tiers_blend_mode, **tw_kwargs)
    if tiers_blend_kwargs is not None:
        kwargs = kwargs | tiers_blend_kwargs
    return wrapped_blend_function(a, b, t, **pass_kwargs(kwargs))


# Shortest path circular interpolation (with the default params)
def sp_circular_interpolation(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float,
    *,
    period: float | None = 2.0 * torch.pi,
    period_scale: float = 1.0,
    # Optional: lock the final output inside the bounds
    wrap_result: bool = False,
    # Only needed if wrap_result is True
    lower_bound: float | None = None,
    lower_bound_scale: float = 1.0,
    start_dim: int = 1,
    end_dim: int | None = None,
    elementwise: bool = False,
    minimize_range: bool = False,
    diff_preserve_sign: bool = False,
    result_preserve_sign: bool = False,
    eps: float = 1e-08,
) -> torch.Tensor:
    if period is None:
        if start_dim < 1:
            start_dim = a.ndim + start_dim
        end_dim = (
            a.ndim
            if end_dim is None
            else (a.ndim + end_dim if end_dim < 0 else end_dim)
        ) + 1
        dims = tuple(range(start_dim, end_dim))
        period = (torch.minimum if minimize_range else torch.maximum)(
            a.abs()
            if elementwise
            else a.abs().amax(
                dim=dims,
                keepdim=True,
            ),
            b.abs()
            if elementwise
            else b.abs().amax(
                dim=dims,
                keepdim=True,
            ),
        )
        period = period.mul_(2.0).clamp_min_(eps)
    else:
        period = max(eps, abs(period))
    if period_scale != 1.0:
        period = period * period_scale
    if wrap_result and lower_bound is None:
        lower_bound = period * (-0.5 * lower_bound_scale)
    diff_orig = diff = b - a
    half_period = period * 0.5

    # Wrap the difference to the shortest path around the "circle"
    diff = diff + half_period
    diff %= period
    diff -= half_period
    if diff_preserve_sign:
        diff = diff.copysign_(diff_orig)

    diff *= t.broadcast_to(a.shape) if isinstance(t, torch.Tensor) else t
    result = diff.add_(a)

    if wrap_result:
        result_orig = result
        result = result - lower_bound
        result %= period
        result += lower_bound
        if result_preserve_sign:
            result = result.copysign_(result_orig)
    return result


# Computes the matrix logarithm using complex eigendecomposition.
# Safe for batched Rotation matrices (Vh) and Covariance matrices.
# Matrice must be diagonalizable.
def matrix_log(
    m: torch.Tensor,
    *,
    eps: float = 1e-06,
    ieps: complex = 1e-08j,
    dtype: torch.dtype | None = torch.complex128,
    keep_dtype: bool = True,
) -> torch.Tensor:
    if m.ndim not in {2, 3} or m.shape[-2] != m.shape[-1]:
        raise ValueError("matrix_log only supports diagonalizable square matrices")
    # 1. Add a tiny diagonal epsilon to prevent singular matrix errors / log(0)
    eye = torch.eye(
        m.shape[-1],
        device=m.device,
        dtype=m.dtype if dtype is None else dtype,
    ).mul_(eps)
    m_safe = m.to(dtype=eye.dtype) + eye

    # 2. Eigendecomposition
    # L = Eigenvalues, V = Eigenvectors
    el, ev = torch.linalg.eig(m_safe)

    # 3. Take the natural logarithm of the complex eigenvalues
    # With a tiny complex epsilon to prevent log(0+0j) NaNs
    log_l = el.add_(ieps).log_()

    # 4. Reconstruct the matrix: V @ diag(log_L) @ V^-1
    v_inv = torch.linalg.solve(
        ev,
        torch.eye(ev.shape[-1], device=ev.device, dtype=ev.dtype),
    )
    log_m = (ev * log_l.unsqueeze(-2)) @ v_inv

    # 5. The result should be purely real (the imaginary parts cancel out to ~0)
    return log_m.real if keep_dtype else log_m.real.to(dtype=m.dtype)


# Suitable for blending coordinate spaces like the Vh component of SVD, covariance, etc.
# Something like 0.5 is similar to LERP.
# Values above 1 should be similar to CFG, but for for coordinate spaces.
def geodesic_square_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float,
    *,
    use_pinv: bool = True,
    # Allow non square matrices (but only operate on the square subset)
    lax: bool = True,
) -> torch.Tensor:
    if a.shape != b.shape:
        errstr = f"Input shape mismatch, A {a.shape} != B {b.shape}"
        raise ValueError(errstr)
    if a.ndim == 2:
        x, y = a.shape
    elif a.ndim == 3:
        x, y = a.shape[1:]
    else:
        x = y = None
    if not (x and y) or (not lax and x != y):
        raise ValueError(
            "geodesic_square_matrix only supports square matrices (batch dimension optional)",
        )
    a_orig = a
    if x != y:
        minsz = min(x, y)
        a = a[..., :minsz, :minsz]
        b = b[..., :minsz, :minsz]
    inv_op = torch.linalg.pinv if use_pinv else torch.linalg.inv
    mlg = matrix_log(b @ inv_op(a))
    velocity = torch.linalg.matrix_exp(
        mlg * (t.to(dtype=mlg.dtype) if isinstance(t, torch.Tensor) else t),
    )
    result = (velocity @ a.to(velocity.dtype)).to(dtype=a.dtype)
    if x == y:
        return result
    a_orig = a_orig.clone()
    a_orig[..., :minsz, :minsz] = result
    return a_orig


def fft_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float,
    *,
    fft_dims: Sequence[int] = (-2, -1),
    avg_t_frequency_dims: bool = True,
    a_phase_offset: float = 0.0,
    a_magnitude_multiplier: float = 1.0,
    a_magnitude_power: float = 0.0,
    b_phase_offset: float = 0.0,
    b_magnitude_multiplier: float = 1.0,
    b_magnitude_power: float = 0.0,
    blended_phase_offset: float = 0.0,
    blended_magnitude_multiplier: float = 1.0,
    blended_magnitude_power: float = 0.0,
    phase_blend_mode: str | Callable = sp_circular_interpolation,
    phase_blend_kwargs: dict | None = None,
    phase_blend_multiplier: float = 1.0,
    phase_blend_offset: float = 0.0,
    magnitude_blend_mode: str | Callable = torch.lerp,
    magnitude_blend_kwargs: dict | None = None,
    magnitude_blend_multiplier: float = 1.0,
    magnitude_blend_offset: float = 0.0,
    magnitude_eps: float = 1e-08,
    **kwargs: Any,
) -> torch.Tensor:
    fft_dims = tuple(fft_dims)
    if a.ndim < 3:
        raise ValueError("fft blend can only handle tensors with 3+ dimensions.")
    kwargs = pass_kwargs(kwargs)
    p_blend = (
        BLENDING_MODES[phase_blend_mode]
        if isinstance(phase_blend_mode, str)
        else phase_blend_mode
    )
    m_blend = (
        BLENDING_MODES[magnitude_blend_mode]
        if isinstance(magnitude_blend_mode, str)
        else magnitude_blend_mode
    )
    a_f = torch.fft.rfftn(a, dim=fft_dims)
    a_phase = torch.atan2(a_f.imag, a_f.real)
    a_mag = a_f.abs()
    if a_phase_offset != 0.0:
        a_phase += a_phase_offset
    if a_magnitude_multiplier != 1.0:
        a_mag *= a_magnitude_multiplier
    if a_magnitude_power != 0.0:
        a_mag = a_mag.add_(magnitude_eps).pow_(a_magnitude_power)
    b_f = torch.fft.rfftn(b, dim=fft_dims)
    b_phase = torch.atan2(b_f.imag, b_f.real)
    b_mag = b_f.abs()
    if b_phase_offset != 0.0:
        b_phase += b_phase_offset
    if b_magnitude_multiplier != 1.0:
        b_mag *= b_magnitude_multiplier
    if b_magnitude_power != 0.0:
        b_mag = b_mag.add_(magnitude_eps).pow_(b_magnitude_power)
    if not isinstance(t, torch.Tensor):
        t = a.new_tensor(t)
    elif t.ndim > 1:
        # FIXME: This probably doesn't work.
        t = t.broadcast_to(a_f.shape)
        if avg_t_frequency_dims:
            t = t.mean(dim=fft_dims, keepdim=True)
    magnitude_blend_kwargs = kwargs | (
        magnitude_blend_kwargs if magnitude_blend_kwargs is not None else {}
    )
    phase_blend_kwargs = kwargs | (
        phase_blend_kwargs if phase_blend_kwargs is not None else {}
    )
    t_mag = t if magnitude_blend_multiplier == 1.0 else t * magnitude_blend_multiplier
    if magnitude_blend_offset != 0.0:
        t_mag = magnitude_blend_offset + t_mag
    t_phase = t if phase_blend_multiplier == 1.0 else t * phase_blend_multiplier
    if phase_blend_offset != 0.0:
        t_phase = phase_blend_offset + t_phase
    blended_mag = m_blend(a_mag, b_mag, t_mag, **magnitude_blend_kwargs).abs()
    blended_phase = p_blend(a_phase, b_phase, t_phase, **phase_blend_kwargs)
    if blended_phase_offset != 0.0:
        blended_phase += blended_phase_offset
    if blended_magnitude_multiplier != 1.0:
        blended_mag *= blended_magnitude_multiplier
    if blended_magnitude_power != 0.0:
        blended_mag = blended_mag.add_(magnitude_eps).pow_(blended_magnitude_power)
    return torch.fft.irfftn(
        torch.polar(blended_mag, blended_phase),
        s=tuple(a.shape[d] for d in fft_dims),
        dim=fft_dims,
    )


class DecompBlend:
    @staticmethod
    def decomp(
        t: torch.Tensor,
        *,
        mode: str = "svd",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tqdm.write(f"DECOMP: shape1={t.shape}")
        if t.ndim > 3:
            raise ValueError(
                "Can only handle batched or unbatched matrices (2 or 3 dimensions)",
            )
        tqdm.write(f"DECOMP: shape={t.shape}")
        if mode == "svd":
            return torch.linalg.svd(t, full_matrices=False)
        if mode == "svd_lowrank":
            q = kwargs.pop("q", None)
            u, s, v = torch.svd_lowrank(
                t,
                q=q if q is not None else t.shape[-1],
                **kwargs,
            )
            return u, s, v.mT
        if mode == "qr":
            u, r_mat = torch.linalg.qr(t)
            s = r_mat.diagonal(dim1=-2, dim2=-1)
            vh = (1.0 / s).masked_fill_(s == 0.0, 1.0 / 1e-08).unsqueeze(-1) * r_mat
            return u, s, vh
        raise ValueError("Bad decomp mode")

    @staticmethod
    def align(
        left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        align_mode: str = "joint",
        invert: bool = False,
        eps: float = 1e-06,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        align_mode = align_mode.strip().lower()
        if align_mode not in {"u", "v", "vh", "joint"}:
            raise ValueError("Align mode must be one of u, v[h], or joint")
        align_mode = align_mode[0]

        ul, sl, vl = left
        ur, sr, vr = right

        def get_sim(t_right: torch.Tensor, t_left: torch.Tensor) -> torch.Tensor:
            norm_right, norm_left = (
                torch.linalg.vector_norm(t, dim=-2, keepdim=True).clamp_min_(eps)
                for t in (t_right, t_left)
            )
            return (t_right / norm_right).mT @ (t_left / norm_left)

        sim_u = get_sim(ur, ul) if align_mode in "uj" else None
        sim_vh = get_sim(vr.mT, vl.mT) if align_mode in "vj" else None
        sim = (
            sim_u * sim_vh
            if align_mode == "j"
            else (sim_u if sim_u is not None else sim_vh)
        )

        match_idx = (sim.abs().argmin if invert else sim.abs().argmax)(dim=-1)
        signs = (
            (sim if align_mode != "j" else sim_u)
            .gather(
                dim=-1,
                index=match_idx.unsqueeze(-1),
            )
            .sign_()
        )
        return (
            ul.gather(
                dim=-1,
                index=match_idx.unsqueeze(-2).expand(*ul.shape[:-1], sr.shape[-1]),
            )
            * signs.mT,
            sl.gather(dim=-1, index=match_idx),
            vl.gather(
                dim=-2,
                index=match_idx.unsqueeze(-1).expand(
                    *vl.shape[:-2],
                    sr.shape[-1],
                    vl.shape[-1],
                ),
            )
            * signs,
        )

    @staticmethod
    def get_size_with_offset(
        *,
        rank: int,
        size: float,
        offset: float = -1,
    ) -> tuple[int, int]:
        size = int(size) if abs(size) >= 1.0 else math.ceil(size * rank)
        if size < 0:
            size = rank + size
        elif size == 0:
            size = rank
        offset = int(offset) if abs(offset) >= 1.0 else math.ceil(offset * rank)
        if offset < 0:
            offset = rank + offset
        size = max(0, min(rank, size))
        offset = min(rank - size, max(0, offset))
        return (size, offset)

    @classmethod
    def rank_slice_blend(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        # Negative values count from the end. Values > -1.0, < 1.0 are interpreted
        # as percentage of ranks. Values outside of that range are truncated and treated
        # as absolute rank indexes. Offsets are specified the same way.
        t: float | torch.Tensor,
        *,
        feature_dim: int = 1,
        # You will always get the exact size you specify and the offset will be adjusted
        # if there isn't enough space.
        rank_offset: float = -1.0,
        decomp_mode: str = "svd",
        align_mode: str = "joint",
        align_invert: bool = False,
        blend_components: str = "usv",
        n_iter: int = 6,
        q: int | None = None,
    ) -> torch.Tensor:
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("Can only only handle 2+ dimensional tensors")
        t = t.mean().detach().cpu().item() if isinstance(t, torch.Tensor) else float(t)

        tqdm.write(f"ORIG SHAPE: {a.shape}")
        a = a.movedim(feature_dim, -1)
        flat_start_dim = 1 if a.ndim > 2 else 0
        adj_shape = a.shape
        a = a.flatten(start_dim=flat_start_dim, end_dim=-2)
        b = b.movedim(feature_dim, -1).flatten(start_dim=flat_start_dim, end_dim=-2)

        dl = cls.decomp(a, mode=decomp_mode, niter=n_iter, q=q)
        dr = cls.decomp(b, mode=decomp_mode, niter=n_iter, q=q)

        size, offset = cls.get_size_with_offset(
            rank=dl[1].shape[-1],
            size=t,
            offset=rank_offset,
        )
        rs = slice(offset, offset + size)

        if (align_mode := align_mode.strip().lower()) in {"joint", "u", "v", "vh"}:
            dl = cls.align(
                dl,
                dr,
                align_mode=align_mode,
                invert=align_invert,
            )
        ((ua, sa, vha), (ub, sb, vhb)) = dl, dr
        blend_components = blend_components.strip().lower()
        if "u" in blend_components:
            ua[..., rs] = ub[..., rs]
        if "s" in blend_components:
            sa[..., rs] = sb[..., rs]
        if "v" in blend_components:
            vha[..., rs, :] = vhb[..., rs, :]
        result = ua @ sa.diag_embed() @ vha
        return result.reshape(adj_shape).movedim(-1, feature_dim).contiguous()

    @staticmethod
    def normalizing_in(
        t: torch.Tensor,
        *,
        centering_strength: float,
        centering_restore_strength: float,
        variance_normalizing: bool,
        aug_scale: float,
        dim: int | Sequence[int],
        orig_features: int,
        in_place: bool = True,
        trim_features: bool = True,
        eps: float = 1e-08,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not in_place:
            t = t.clone()
        if not isinstance(dim, int):
            dim = tuple(dim)
        mean = (
            t.mean(dim=dim, keepdim=True)
            if centering_strength != 0.0 or centering_restore_strength != 0.0
            else None
        )
        if centering_strength != 0 and mean is not None:
            t -= mean * centering_strength if centering_strength != 1 else mean
        if mean is not None and trim_features:
            mean = mean[..., :orig_features]
        if variance_normalizing:
            std = t.std(dim=dim, keepdim=True).clamp_min_(eps)
            t /= std
            if trim_features:
                std = std[..., :orig_features]
        else:
            std = None
        if aug_scale != 0 and t.shape[-1] > orig_features:
            t[..., orig_features:] *= aug_scale
        return t, mean, std

    @staticmethod
    def normalizing_out(
        t: torch.Tensor,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        centering_restore_strength: float,
        in_place: bool = True,
    ) -> torch.Tensor:
        if mean is None and std is None:
            return t
        if not in_place:
            t = t.clone()
        t_slices = tuple(slice(None, sz) for sz in t.shape)
        mean, std = (None if temp is None else temp[t_slices] for temp in (mean, std))
        if std is not None:
            t *= std
        if centering_restore_strength != 0 and mean is not None:
            t += (
                mean.mul_(centering_restore_strength)
                if centering_restore_strength != 1
                else mean
            )
        return t

    @classmethod
    def rank_blend(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        t: float | torch.Tensor,
        *,
        # Controls whether changes get applied to a or b.
        base_a: bool = True,
        blend_mode: Callable | str = torch.lerp,
        # If None, uses addition (inject) in diff mode or LERP in blend mode.
        result_blend_mode: Callable | str | None = None,
        # One of diff, slice, blend
        blend_strategy: str = "diff",
        # Negative values count from the end. Values > -1.0, < 1.0 are interpreted
        # as percentage of ranks. Values outside of that range are truncated and treated
        # as absolute rank indexes. Offsets are specified the same way.
        # Since the ranks parameter is a size, 0 means all ranks, -2 to means total_ranks - 2, etc.
        ranks: float = 0.5,
        # You will always get the exact size you specify and the offset will be adjusted
        # if there isn't enough space.
        rank_offset: float = -1.0,
        rank_start_scale: float = 1.0,
        rank_end_scale: float = 1.0,
        rank_ramp_power: float = 0.0,
        use_log_rank_scales: bool = False,
        # rmula, rmulb, amulb, rroll, rmulroll, rmulrollc
        feature_augmentations: Sequence[str] = ("rmula", "rmulb", "amulb"),
        feature_augmentation_scale: float = 0.0,
        feature_dim: int = 1,
        # Flattening occurs after the feature dim is moved to the end.
        flatten_start_dim: int = 1,
        flatten_end_dim: int = -2,
        decomp_mode: str = "svd",
        centering_strength: float = 0.0,
        centering_restore_strength: float = 0.0,
        result_scale: float = 1.0,
        variance_normalizing: bool = False,
        # Alignment only applies to the slice blend strategy.
        align_mode: str = "joint",
        align_invert: bool = False,
        align_base: bool = True,
        decomp_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if blend_strategy not in {"diff", "blend", "slice"}:
            raise ValueError("Bad blend_strategy")
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("Can only only handle 2+ dimensional tensors")
        if isinstance(blend_mode, str):
            blend_mode = BLENDING_MODES[blend_mode]
        if isinstance(result_blend_mode, str):
            result_blend_mode = BLENDING_MODES[result_blend_mode]
        elif result_blend_mode is None and blend_strategy == "blend":
            result_blend_mode = torch.lerp

        kwargs = pass_kwargs(kwargs)
        feature_augmentations = tuple(feature_augmentations)

        use_aug = feature_augmentations and feature_augmentation_scale != 0
        base = a if base_a else b
        blend_result: torch.Tensor = blend_mode(a, b, t, **kwargs)
        if blend_strategy == "diff":
            blend_result -= base
        blend_result = blend_result.movedim(feature_dim, -1)
        adj_shape = blend_result.shape
        orig_features = adj_shape[-1]
        blend_result = blend_result.flatten(
            start_dim=flatten_start_dim,
            end_dim=flatten_end_dim,
        )
        if use_aug:
            aug_list = [blend_result]
            flat_a = a.movedim(feature_dim, -1).flatten(
                start_dim=flatten_start_dim,
                end_dim=flatten_end_dim,
            )
            flat_b = b.movedim(feature_dim, -1).flatten(
                start_dim=flatten_start_dim,
                end_dim=flatten_end_dim,
            )
            aug = None
            for augtype in feature_augmentations:
                if augtype == "rmula":
                    aug = blend_result * flat_a
                elif augtype == "rmulb":
                    aug = blend_result * flat_b
                elif augtype == "amulb":
                    aug = flat_a * flat_b
                elif augtype == "rroll":
                    aug = blend_result.roll(shifts=1, dims=-2)
                elif augtype == "rmulroll":
                    aug = blend_result.roll(shifts=1, dims=-2).mul_(blend_result)
                elif augtype == "rmulrollc":
                    aug = blend_result.roll(shifts=1, dims=-1).mul_(blend_result)
                else:
                    errstr = f"Unknown augmentation type: {augtype}"
                    raise ValueError(errstr)
                aug_list.append(aug)
            if len(aug_list) > 1:
                blend_result = torch.cat(aug_list, dim=-1)
            del aug, aug_list, flat_a, flat_b
        norm_in = partial(
            cls.normalizing_in,
            centering_strength=centering_strength,
            centering_restore_strength=centering_restore_strength,
            variance_normalizing=variance_normalizing,
            aug_scale=feature_augmentation_scale,
            dim=flatten_start_dim,
            orig_features=orig_features,
        )
        if blend_strategy == "slice":
            flat_base = blend_result.clone()
            flat_base[..., :orig_features] = base.movedim(feature_dim, -1).flatten(
                start_dim=flatten_start_dim,
                end_dim=flatten_end_dim,
            )
            flat_base, base_mean, base_std = norm_in(flat_base)
        else:
            flat_base = base_mean = base_std = None
        blend_result, mean, std = norm_in(blend_result)
        dr = cls.decomp(blend_result, mode=decomp_mode, **(decomp_kwargs or {}))
        if blend_strategy == "slice" and flat_base is not None:
            db = cls.decomp(flat_base, mode=decomp_mode, **(decomp_kwargs or {}))
            if (align_mode := align_mode.strip().lower()) in {"joint", "u", "v", "vh"}:
                temp = cls.align(
                    db if align_base else dr,
                    dr if align_base else db,
                    align_mode=align_mode,
                    invert=align_invert,
                )
                db, dr = (temp, dr) if align_base else (db, temp)
                del temp

        u, s, vh = dr
        size, offset = cls.get_size_with_offset(
            rank=s.shape[-1],
            size=ranks,
            offset=rank_offset,
        )
        if rank_start_scale != rank_end_scale:
            if rank_offset < 0:
                rank_start_scale, rank_end_scale = rank_end_scale, rank_start_scale
            rank_scales = torch.linspace(
                rank_start_scale,
                rank_end_scale,
                steps=size,
                dtype=a.dtype,
                device=a.device,
            ).unsqueeze(0)
            if rank_ramp_power != 0:
                rank_scales = torch.where(
                    rank_scales == 0,
                    0,
                    rank_scales.abs().pow_(rank_ramp_power).copysign_(rank_scales),
                )
        else:
            rank_scales = rank_start_scale
        rs = slice(offset, offset + size)
        tqdm.write(
            f"RANK BLEND: slice={rs}, scales={rank_scales}, shape={blend_result.shape}, adj={adj_shape}",
        )
        u, s, vh = u[..., rs], s[..., rs], vh[..., rs, :]
        if not isinstance(rank_scales, float) or rank_scales != 1.0:
            if use_log_rank_scales:
                s = s.abs().log1p_().copysign_(s)
            s *= rank_scales
            if use_log_rank_scales:
                s = s.abs().expm1_().copysign_(s)
        result = (u @ s.diag_embed() @ vh)[..., :orig_features]
        if result_scale != 1.0 and result_blend_mode is None:
            if std is None:
                result *= result_scale
            else:
                std *= result_scale
        result = cls.normalizing_out(
            result,
            mean=mean,
            std=std,
            centering_restore_strength=centering_restore_strength,
        )
        if blend_strategy == "slice" and db is not None:
            bu, bs, bvh = db
            size_before = rs.start
            size_after = bs.shape[-1] - rs.stop
            base_result = (
                (
                    bu[..., :size_before]
                    @ bs[..., :size_before].diag_embed()
                    @ bvh[..., :size_before, :]
                )
                if size_before > 0
                else None
            )
            if size_after > 0:
                temp = (
                    bu[..., rs.stop :]
                    @ bs[..., rs.stop :].diag_embed()
                    @ bvh[..., rs.stop :, :]
                )
                base_result = (
                    base_result.add_(temp) if base_result is not None else temp
                )
            if base_result is not None:
                base_result = cls.normalizing_out(
                    base_result[..., :orig_features],
                    mean=base_mean,
                    std=base_std,
                    centering_restore_strength=centering_restore_strength,
                )
                result += base_result

        result = result.reshape(adj_shape).movedim(-1, feature_dim).contiguous()
        if blend_strategy == "slice":
            return result
        if result_blend_mode is not None:
            return result_blend_mode(base, result, result_scale, **kwargs)
        if blend_strategy == "diff":
            return result.add_(base)
        raise RuntimeError("Unhandled blend_strategy")


def chain_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    *args: Any,
    chain_iterations: int = 1,
    chain_blend_mode: str | Callable = torch.lerp,
    **kwargs: Any,
) -> torch.Tensor:
    if chain_iterations < 1:
        return a.clone()
    fun = (
        BLENDING_MODES[chain_blend_mode]
        if isinstance(chain_blend_mode, str)
        else chain_blend_mode
    )
    kwargs = pass_kwargs(kwargs)
    for _ in range(chain_iterations):
        b = fun(a, b, *args, **kwargs)
    return b


def pct_limit_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    *args: Any,
    base_a: bool = True,
    diff_limit: float = 0.25,
    eps: float = 1e-07,
    blend_mode: str | Callable = torch.lerp,
    dim: int | Sequence[int] | None = None,
    # Only applies in elementwise mode (dim=None)
    prevent_sign_flip: bool = False,
    # Negative values disable soft clamp for that specific constraint.
    # Higher stiffness -> approach the limit more closely before values warp.
    pct_clamp_stiffness: float = 10.0,
    sign_clamp_stiffness: float = 10.0,
    **kwargs: Any,
) -> torch.Tensor:
    if a.ndim < 2:
        raise ValueError("Blend function requires 2+ dimensions")
    if diff_limit < 0:
        raise ValueError("diff_limit must be positive")

    blend_function = (
        BLENDING_MODES[blend_mode] if isinstance(blend_mode, str) else blend_mode
    )

    br = blend_function(a, b, *args, **pass_kwargs(kwargs))
    base = a if base_a else b

    if diff_limit == 0:
        return base.clone()

    if dim is not None:
        diff = br.sub_(base)
        diff_norm = torch.linalg.vector_norm(diff, dim=dim, keepdim=True)
        max_norm = (
            torch.linalg.vector_norm(base, dim=dim, keepdim=True)
            .mul_(diff_limit)
            .clamp_min_(eps)
        )

        if pct_clamp_stiffness >= 0:
            # Soft clamp the magnitude of the difference
            soft_diff_norm = soft_clamp(
                diff_norm,
                min_val=0.0,
                max_val=max_norm,
                stiffness=pct_clamp_stiffness,
            )
            scale = soft_diff_norm.div_(diff_norm.clamp_min_(eps))
        else:
            # Hard clamp the scale
            scale = max_norm.div_(diff_norm.clamp_min_(eps)).clamp_max_(1.0)

        return diff.mul_(scale).add_(base)

    # Elementwise handling.
    if prevent_sign_flip:
        # Create bounds using infinity so we ONLY restrict the zero-crossing
        mask = base >= 0
        sign_lower = torch.where(mask, 0.0, -torch.inf)
        sign_upper = torch.where(mask, torch.inf, 0.0)

        if sign_clamp_stiffness >= 0:
            br = soft_clamp(br, sign_lower, sign_upper, stiffness=sign_clamp_stiffness)
        else:
            br = br.clamp_(min=sign_lower, max=sign_upper)

    max_diff = base.abs().mul_(diff_limit).clamp_min_(eps)
    lower_bound = base - max_diff
    upper_bound = base + max_diff

    if pct_clamp_stiffness < 0:
        return br.clamp_(min=lower_bound, max=upper_bound)
    return soft_clamp(br, lower_bound, upper_bound, stiffness=pct_clamp_stiffness)


def moment_aligned_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    *args: Any,
    blend_mode: str | Callable = torch.lerp,
    base_a: bool = True,
    mean_scale: float = 1.0,
    std_scale: float = 1.0,
    dim: int | Sequence[int] | None = None,
    eps: float = 1e-07,
    **kwargs: Any,
) -> torch.Tensor:
    blend_function = (
        BLENDING_MODES[blend_mode] if isinstance(blend_mode, str) else blend_mode
    )
    kwargs = pass_kwargs(kwargs)

    if mean_scale == 0.0 and std_scale == 0.0:
        return blend_function(a, b, *args, **kwargs)

    if dim is None:
        dim = tuple(range(1, a.ndim))

    target, base = (b, a) if base_a else (a, b)
    mean_target = target.mean(dim=dim, keepdim=True)

    # Centering here is always necessary. We will add the mean back if mean_scale is 0.
    aligned = target - mean_target
    if std_scale != 0.0:
        std_target = target.std(dim=dim, keepdim=True).clamp_min_(eps)
        std_base = base.std(dim=dim, keepdim=True).clamp_min_(eps)
        std_goal = std_base if std_scale == 1 else std_target.lerp(std_base, std_scale)
        aligned *= std_goal.div_(std_target)
        del std_target, std_base, std_goal
    if mean_scale != 0.0:
        mean_base = base.mean(dim=dim, keepdim=True)
        mean_goal = (
            mean_base if mean_scale == 1 else mean_target.lerp_(mean_base, mean_scale)
        )
        del mean_base
    else:
        mean_goal = mean_target
    aligned += mean_goal
    del mean_goal, mean_target, base, target

    return blend_function(
        a if base_a else aligned,
        aligned if base_a else b,
        *args,
        **kwargs,
    )


def distro_aligned_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    *args: Any,
    blend_mode: str | Callable = torch.lerp,
    reference_blend_mode: str | Callable = torch.lerp,
    # When using a reference and unset, will use the original ratio.
    reference_blend: float | torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    # One of da, db, dr(eference), sa, sb, sr, dg(aussian), sg
    # 's' vs 'd' determines whether distro or robust scale matching is used.
    align_a: str | None = None,
    align_b: str | None = None,
    align_result: str | None = None,
    start_dim: int = 1,
    end_dim: int = -1,
    scale_match_mad: float = 0.6745,
    **kwargs: Any,
) -> torch.Tensor:
    if len(args) == 0:
        raise ValueError("Missing ratio positional parameter")
    kwargs = pass_kwargs(kwargs)
    blend_function = (
        BLENDING_MODES[blend_mode] if isinstance(blend_mode, str) else blend_mode
    )
    if align_a is None and align_b is None and align_result is None:
        return blend_function(a, b, *args, **kwargs)
    need_ref = reference is None and any(
        val in {"dr", "sr"} for val in (align_a, align_b, align_result)
    )
    if need_ref:
        ref_blend_function = (
            BLENDING_MODES[reference_blend_mode]
            if isinstance(reference_blend_mode, str)
            else reference_blend_mode
        )
        ref_args = (
            reference_blend if reference_blend is not None else args[0],
            *args[1:],
        )
        reference = ref_blend_function(a, b, *ref_args, **kwargs)

    align_targets = {
        "da": a,
        "db": b,
        "dr": reference,
        "sa": a,
        "sb": b,
        "sr": reference,
    }
    if any(
        val not in {None, "dg", "sg"} and align_targets.get(val) is None
        for val in (align_a, align_b, align_result)
    ):
        raise ValueError("Invalid align target")

    def do_align(t: torch.Tensor, amode: str | None) -> torch.Tensor:
        if amode is None:
            return t
        if amode == "dg":
            return force_gaussian_distribution(t, start_dim=start_dim, end_dim=end_dim)
        if amode == "sg":
            return robust_scale_match(
                t,
                start_dim=start_dim,
                end_dim=end_dim,
                mad=scale_match_mad,
            )
        if amode.startswith("d"):
            return match_distribution(
                t,
                reference=align_targets[amode],
                start_dim=start_dim,
                end_dim=end_dim,
            )
        return robust_scale_match(
            t,
            reference=align_targets[amode],
            start_dim=start_dim,
            end_dim=end_dim,
        )

    a, b = (do_align(item, amode) for item, amode in ((a, align_a), (b, align_b)))

    return do_align(blend_function(a, b, *args, **kwargs), align_result)


# Standard LERP, but the weights are forced to preserve a variance of 1.
def pythagorean_lerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    eps: float = 1e-08,
) -> torch.Tensor:
    w_a = 1.0 - t
    w_b = t

    # Calculate how much the variance would shrink.
    if isinstance(w_a, torch.Tensor):
        variance_shrink = (w_a**2).add_(w_b**2).sqrt_().clamp_min_(eps)
    else:
        variance_shrink = max(eps, (w_a**2 + w_b**2) ** 0.5)

    # Then scale the weights to compensate.
    return a.mul(w_a / variance_shrink).add_(b * (w_b / variance_shrink))


def rms_interpolation(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    # To only use magnitude, enable this and set power to 1.
    abs_inputs: bool = False,
    # One of:
    #   blend, reference (reference must be supplied), a, b, leave,
    sign_mode: str = "blend",
    reference: torch.Tensor | None = None,
    blend_mode: str | Callable = torch.lerp,
    sign_blend_mode: str | Callable | None = None,
    power: float | torch.Tensor = 2.0,
    inv_power: float | torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    kwargs = pass_kwargs(kwargs)
    blend_function = (
        BLENDING_MODES[blend_mode] if isinstance(blend_mode, str) else blend_mode
    )
    a_orig, b_orig = a, b
    if abs_inputs:
        a, b = a.abs(), b.abs()
    if power != 1.0:
        if inv_power is None:
            inv_power = 1 / power
        a, b = a**power, b**power
    blend_result = blend_function(a, b, t, **kwargs)
    if power != 1.0:
        if sign_mode == "leave":
            return blend_result.abs().pow_(inv_power).copysign_(blend_result)
        blend_result = blend_result.abs_().pow_(inv_power)
    elif sign_mode == "leave":
        return blend_result
    if sign_mode == "blend":
        sign_blend_function = (
            (
                BLENDING_MODES[sign_blend_mode]
                if isinstance(sign_blend_mode, str)
                else sign_blend_mode
            )
            if sign_blend_mode is not None
            else blend_function
        )
        reference = sign_blend_function(a_orig, b_orig, t, **kwargs)
        return blend_result.copysign_(reference)
    if sign_mode == "reference":
        if reference is None:
            raise ValueError("sign mode reference requires a reference to be supplied")
        return blend_result.copysign_(reference.to(blend_result))
    if sign_mode == "a":
        return blend_result.copysign_(a_orig)
    if sign_mode == "b":
        return blend_result.copysign_(b_orig)
    errstr = f"Unhandled sign mode: {sign_mode}"
    raise ValueError(errstr)


def orbit_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    # One of leave, clamp, bsuba, rsuba, rsubb
    excess_mode: str = "leave",
    invert_excess: bool = False,
) -> torch.Tensor:
    t_orig = t
    if excess_mode == "clamp":
        t = t.clamp(-1, 1) if isinstance(t, torch.Tensor) else max(-1.0, min(1.0, t))
    angle = t * (math.pi / 2.0)
    if isinstance(angle, torch.Tensor):
        orbit = a.mul(angle.cos()).add_(b.mul(angle.sin()))
    else:
        orbit = (a * math.cos(angle)).add_(b * math.sin(angle))

    if excess_mode not in {"leave", "clamp"}:
        have_excess = (
            torch.any(t_orig.abs() > 1).detach().cpu().item()
            if isinstance(t_orig, torch.Tensor)
            else abs(t_orig) > 1
        )
    else:
        have_excess = False

    if not have_excess:
        return orbit

    if excess_mode == "bsuba":
        tangent_vector = b - a
    elif excess_mode == "rsuba":
        tangent_vector = orbit - a
    elif excess_mode == "rsubb":
        tangent_vector = orbit - b
    else:
        errstr = f"Unknown excess mode: {excess_mode}"
        raise ValueError(errstr)
    if invert_excess:
        tangent_vector = tangent_vector.neg_()

    # 2. How far out of bounds are we? (0 if t <= 1)
    if isinstance(t_orig, torch.Tensor):
        excess = (t_orig.abs() - 1.0).clamp_min_(0.0).copysign_(t_orig)
    else:
        excess = math.copysign(max(0.0, abs(t_orig) - 1.0), t_orig)

    return orbit.add_(tangent_vector.mul_(excess))


class BlendMode:
    __slots__ = (
        "allow_scale",
        "f",
        "f_kwargs",
        "f_raw",
        "force_rescale",
        "fork_rng",
        "invert_scale",
        "norm",
        "norm_dims",
        "rescale_dims",
        "rescale_max",
        "rescale_min",
        "rev",
        "scale_multiplier",
        "visible",
    )

    class _Empty:
        pass

    def __init__(
        self,
        f,
        norm=None,
        norm_dims: tuple = (-3, -2, -1),
        rev: bool = False,
        allow_scale: bool = True,
        rescale_dims: tuple = (-3, -2, -1),
        rescale_min: float = 0.0,
        rescale_max: float = 1.0,
        force_rescale: bool = False,
        fork_rng: bool = False,
        invert_scale: float | None = None,
        scale_multiplier: float = 1.0,
        visible: bool = True,
        **kwargs: dict,
    ):
        self.f_raw = f
        self.f = f if not kwargs else partial(f, **kwargs)
        self.f_kwargs = kwargs
        if norm is True:
            norm = normalize
        elif norm is False:
            norm = None
        self.norm = norm
        self.norm_dims = norm_dims
        self.rev = rev
        self.allow_scale = allow_scale
        self.rescale_dims = rescale_dims
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        self.force_rescale = force_rescale
        self.fork_rng = fork_rng
        self.invert_scale = invert_scale
        self.scale_multiplier = scale_multiplier
        self.visible = visible

    def edited(self, *, f=_Empty, preserve_kwargs=True, **kwargs: dict) -> BlendMode:
        empty = self._Empty
        kwargs = (self.f_kwargs | kwargs) if preserve_kwargs else kwargs
        kwargs |= {
            k: v if (v := kwargs.get(k, empty)) is not empty else getattr(self, k)
            for k in (
                "norm",
                "norm_dims",
                "rev",
                "allow_scale",
                "rescale_dims",
                "rescale_min",
                "rescale_max",
                "force_rescale",
                "fork_rng",
                "invert_scale",
                "scale_multiplier",
                "visible",
            )
        }
        return self.__class__(f if f is not empty else self.f_raw, **kwargs)

    def rescale(self, t, *, rescale_dims=_Empty):
        if t.ndim > 2:
            rescale_dims = (
                self.rescale_dims if rescale_dims is self._Empty else rescale_dims
            )
        else:
            # Meh.
            rescale_dims = -1
        tmin = torch.amin(t, keepdim=True, dim=rescale_dims)
        tmax = torch.amax(t, keepdim=True, dim=rescale_dims)
        return (
            (t - tmin).div_(tmax - tmin).clamp_(self.rescale_min, self.rescale_max),
            tmin,
            tmax,
        )

    def _blend_internal(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
        *,
        norm_dims=_Empty,
        rescale_min_blend: float = 0.5,
        rescale_max_blend: float = 0.5,
        rescale_min_blend_function: Callable = torch.lerp,
        rescale_max_blend_function: Callable = torch.lerp,
        **kwargs: dict,
    ) -> torch.Tensor:
        if self.force_rescale:
            a, amin, amax = self.rescale(a)
            b, bmin, bmax = self.rescale(b)
        with torch.random.fork_rng(devices=(a.device, b.device), enabled=self.fork_rng):
            result = self.__call__internal(a, b, t, norm_dims=norm_dims, **kwargs)
        if not self.force_rescale:
            return result
        rmin = rescale_min_blend_function(amin, bmin, rescale_min_blend)
        rmax = rescale_max_blend_function(amax, bmax, rescale_max_blend)
        return result.mul_(rmax.sub_(rmin)).add_(rmin)

    _AT = TypeVar("_AT", torch.Tensor, float)

    def __call__(
        self,
        a: _AT,
        b: torch.Tensor | float,
        t: torch.Tensor | float,
        *,
        norm_dims=_Empty,
        **kwargs: Any,
    ) -> _AT:
        float_a = not isinstance(a, torch.Tensor)
        if float_a:
            a = torch.tensor(a, dtype=torch.float64, device="cpu")
        if not isinstance(b, torch.Tensor):
            b = a.new_tensor(b)
        if b.ndim > 1:
            b = b.broadcast_to(a.shape)
        if not isinstance(t, torch.Tensor):
            t = a.new_tensor(t)
        if t.ndim > 1:
            t = t.broadcast_to(a.shape)
        if float_a and (b.numel() != 1 or t.numel() != 1):
            raise ValueError(
                "When passing the 'a' parameter as a float, 'b' and 't' must either be float or 1-element tensors.",
            )
        result = self._blend_internal(a, b, t, norm_dims=norm_dims, **kwargs)
        return result.mean().detach().cpu().item() if float_a else result

    def __call__internal(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor | float,
        *,
        norm_dims=_Empty,
        **kwargs: dict,
    ) -> torch.Tensor:
        if self.rev:
            a, b = b, a
        if self.invert_scale is not None:
            t = self.invert_scale - t
        if self.scale_multiplier != 1.0:
            t = t * self.scale_multiplier
        if self.norm is None:
            return self.f(a, b, t, **kwargs)
        return self.norm(
            self.f(a, b, t),
            reference_latent=torch.lerp(a, b, t),
            dim=self.norm_dims if norm_dims is self._Empty else norm_dims,
        )


class BlendingModes:
    BLEH = True

    def __init__(self, builtins=None):
        self.builtins = {} if builtins is None else builtins
        self.cache = {}

    def get_dict_key(self, k: dict):
        ds = frozenset(k.items())
        cached = self.cache.get(ds)
        if cached is not None:
            return cached
        name = k.get("name")
        if name is None:
            raise ValueError(
                "When passing a blend mode key as dict, a string 'name' key must exist."
            )
        name = name.strip()
        base_bm = self.builtins.get(name)
        if base_bm is None:
            errstr = f"Unknown mode {name} for extended blend specification"
            raise ValueError(errstr)
        bm_kwargs = k.copy()
        del bm_kwargs["name"]
        bm = base_bm.edited(**bm_kwargs)
        self.cache[k] = bm
        return bm

    def get(self, k: str | dict, default=None):
        if isinstance(k, dict):
            return self.get_dict_key(k)
        result = self.builtins.get(k)
        if result is not None:
            return result
        result = self.cache.get(k)
        if result is not None:
            return result
        return self.try_extended(k, default=default)

    _simple_value_map: ClassVar = {
        "true": True,
        "false": False,
        "()": (),
        "none": None,
    }

    def parse_value(self, k: str, v: str):
        k = k.strip().lower()
        v = v.strip()
        if not v:
            raise ValueError("Empty value")
        if len(v) > 1 and v[0] == "^":
            literal_mode = True
            v = v[1:]
        else:
            literal_mode = False
        vl = v.lower()
        result = self._simple_value_map.get(vl, vl)
        if result is not vl:
            return result
        v0 = v[0]
        if v0.isdigit() or v0 in "-+":
            if "," in v:
                result = tuple(
                    self.parse_value(k, subv)
                    for subv in (_subv for _subv in v.split(",") if _subv.strip())
                )
                if len(result) > 1 and not all(
                    subv.__class__ is result[0].__class__ for subv in result[1:]
                ):
                    errstr = f"Mismatched items in list for key {k}"
                    raise ValueError(errstr)
                return result
            return float(v) if "." in v else int(v)
        if not literal_mode and k.startswith("blend"):
            # It won't be a numeric value here.
            result = self.builtins.get(v)
            if result is None:
                errstr = f"Unknown blend mode {v}"
                raise ValueError(errstr)
            return result
        return v

    def parse_arg(self, s: str, idx: int) -> tuple:
        kv = s.split("=", 1)
        if len(kv) != 2:
            errstr = f"Failed to parse argument at position {idx}"
            raise ValueError(errstr)
        k, v = kv[0].strip(), kv[1].strip()
        if not k:
            errstr = f"Empty key at argument position {idx}"
            raise ValueError(errstr)
        try:
            v_out = self.parse_value(k, v)
        except ValueError as exc:
            errstr = f"Parse failed at argument position {idx}: {exc}"
            raise ValueError(errstr) from exc
        return (k, v_out)

    def try_extended(self, k: str, default=None) -> object:
        if ":" not in k:
            return default
        name, *arglist = k.strip().split(":")
        name = name.strip()
        base_bm = self.builtins.get(name)
        if base_bm is None:
            errstr = f"Unknown mode {name} for extended blend specification"
            raise ValueError(errstr)
        bm_kwargs = dict(self.parse_arg(arg, idx) for idx, arg in enumerate(arglist))
        bm = base_bm.edited(**bm_kwargs)
        self.cache[k] = bm
        return bm

    def items(self):
        return ((k, v) for k, v in self.builtins.items() if v.visible)

    def values(self):
        return (v for v in self.builtins.values() if v.visible)

    def __contains__(self, k: str) -> bool:
        return self.get(k) is not None

    def __iter__(self):
        return (k for k, _v in self.items())

    keys = __iter__

    def __setitem__(self, k: str, v) -> None:
        self.builtins[k] = v if isinstance(v, BlendMode) else BlendMode(v)

    def __getitem__(self, k: str) -> BlendMode:
        result = self.get(k)
        if result is None:
            raise KeyError(k)
        return result

    def __ior__(self, other: dict | object):
        if isinstance(other, dict):
            self.builtins |= {
                k: v if isinstance(v, BlendMode) else BlendMode(v)
                for k, v in other.items()
            }
            return self
        self.builtins |= other.builtins
        self.cache |= other.cache
        return self

    def __or__(self, other: dict | object) -> object:
        clone = self.__class__()
        clone.builtins = self.builtins.copy()
        clone.cache = self.cache.copy()
        if isinstance(other, dict):
            clone.builtins |= {
                k: v if isinstance(v, BlendMode) else BlendMode(v)
                for k, v in other.items()
            }
            return clone
        clone.builtins |= other.builtins
        clone.cache |= other.cache
        return clone

    def copy(self):
        return self | {}


BLENDING_MODES = {
    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor
    "a_only": BlendMode(lambda a, _b, t: a * t, allow_scale=False),
    "b_only": BlendMode(lambda _a, b, t: b * t, allow_scale=False),
    # Interpolates between tensors a and b using normalized linear interpolation.
    # This definitely isn't biSLERP.
    "bislerp_wrong": BlendMode(torch.lerp, normalize),
    # "^"bislerp": BlendMode(lambda a, b, t: (1 - t) * a + t * b, normalize),
    "slerp": BlendMode(altslerp),
    # Transfer the color from `b` to `a` by t` factor
    "colorize": BlendMode(torch.lerp),
    # Interpolates between tensors a and b using cosine interpolation.
    "cosinterp": BlendMode(
        lambda a, b, t: ((a + b).sub_((a - b).mul_((t * torch.pi).cos()))).div_(2),
    ),
    # Interpolates between tensors a and b using cubic interpolation.
    "cuberp": BlendMode(
        lambda a, b, t: (b - a).mul_((3 * t**2).sub_(2 * t**3)).add_(a),
    ),
    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    "hslerp": BlendMode(hslerp),
    "hslerpalt": BlendMode(hslerp_alt2),
    "hslerpalt110x": BlendMode(hslerp_alt2, sign_order=(1.1, -1.1)),
    "hslerpalt125x": BlendMode(hslerp_alt2, sign_order=(1.25, -1.25)),
    "hslerpalt150x": BlendMode(hslerp_alt2, sign_order=(1.5, -1.5)),
    "hslerpalt300x": BlendMode(hslerp_alt2, sign_order=(3.0, -3.0)),
    "hslerpaltflipsign": BlendMode(hslerp_alt2, sign_order=(-1.0, 1.0)),
    "hslerpaltflipsign110x": BlendMode(hslerp_alt2, sign_order=(-1.1, 1.1)),
    "hslerpaltflipsign125x": BlendMode(hslerp_alt2, sign_order=(-1.25, 1.25)),
    "hslerpaltflipsign150x": BlendMode(hslerp_alt2, sign_order=(-1.5, 1.5)),
    "hslerpaltflipsign300x": BlendMode(hslerp_alt2, sign_order=(-3.0, 3.0)),
    "problerp0.25": BlendMode(stochasistic_blend, fuzz=0.25),
    "problerp0.1": BlendMode(stochasistic_blend, fuzz=0.1),
    "problerp0.025": BlendMode(stochasistic_blend, fuzz=0.025),
    "probselect": BlendMode(prob_blend),
    "probselect_channels": BlendMode(prob_blend, collapse_dims=(1,)),
    "probselectsmoothed": BlendMode(prob_blend_smoothed),
    "probselectsmoothed_channels": BlendMode(
        prob_blend_smoothed,
        collapse_dims=(1,),
    ),
    "probselectsmoothed_ks5": BlendMode(prob_blend_smoothed, kernel_size=5),
    "probselectsmoothed_ks9": BlendMode(prob_blend_smoothed, kernel_size=9),
    "probselectsmoothed_ks9_sigma3": BlendMode(
        prob_blend_smoothed,
        kernel_size=9,
        sigma=3.0,
    ),
    "probinject": BlendMode(
        lambda a, b, t, **kwargs: prob_blend(torch.zeros_like(b), b, t, **kwargs).add_(
            a,
        ),
    ),
    "probsubtract_b": BlendMode(
        lambda a, b, t, **kwargs: a - prob_blend(torch.zeros_like(b), b, t, **kwargs),
    ),
    "gradient": BlendMode(gradient_blend),
    # Adds tensor b to tensor a, scaled by t.
    "inject": BlendMode(lambda a, b, t: (b * t).add_(a)),
    "injecthalf": BlendMode(lambda a, b, t: (b * (t * 0.5)).add_(a)),
    "injectquarter": BlendMode(lambda a, b, t: (b * (t * 0.25)).add_(a)),
    "inject_difference": BlendMode(lambda a, b, t: (a - b).mul_(t).add_(a)),
    "inject_copysign_a": BlendMode(lambda a, b, t: (b * t).add_(a).copysign_(a)),
    "inject_copysign_b": BlendMode(lambda a, b, t: (b * t).add_(a).copysign_(b)),
    "inject_avoidsign_a": BlendMode(lambda a, b, t: (b * t).add_(a).copysign_(a.neg())),
    "inject_avoidsign_b": BlendMode(lambda a, b, t: (b * t).add_(a).copysign_(b.neg())),
    "cfg": BlendMode(torch.lerp),
    "cfg_base_a": BlendMode(lambda a, b, t: (a - b).mul_(t).add_(a)),
    # Interpolates between tensors a and b using linear interpolation.
    "lerp": BlendMode(torch.lerp),
    "lerp050x": BlendMode(lambda a, b, t: a.lerp(b, t).mul_(0.5)),
    "lerp075x": BlendMode(lambda a, b, t: a.lerp(b, t).mul_(0.75)),
    "lerp110x": BlendMode(lambda a, b, t: a.lerp(b, t).mul_(1.1)),
    "lerp125x": BlendMode(lambda a, b, t: a.lerp(b, t).mul_(1.25)),
    "lerp150x": BlendMode(lambda a, b, t: a.lerp(b, t).mul_(1.5)),
    "lerp_copysign_a": BlendMode(
        lambda a, b, t: a.lerp(b, t).copysign_(a),
    ),
    "lerp_copysign_b": BlendMode(
        lambda a, b, t: a.lerp(b, t).copysign_(b),
    ),
    "lerp_avoidsign_a": BlendMode(
        lambda a, b, t: a.lerp(b, t).copysign_(a.neg()),
    ),
    "lerp_avoidsign_b": BlendMode(
        lambda a, b, t: a.lerp(b, t).copysign_(b.neg()),
    ),
    "weighted_average": BlendMode(
        lambda a, b, t: (b * t).add_(a) / (1.0 + abs(t)),
    ),
    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    "lineardodge": BlendMode(lambda a, b, t: (b * t).add_(a)),
    "copysign": BlendMode(lambda a, b, _t: torch.copysign(a, b)),
    "probcopysign": BlendMode(
        lambda a, b, t: prob_blend(a, b, t).copysign_(a),
    ),
    "slice_flat_d1": BlendMode(slice_blend, dim=1, flatten=True),
    "slice_flat_d2": BlendMode(slice_blend, dim=2, flatten=True),
    "slice_d1": BlendMode(slice_blend, dim=1, flatten=False),
    "slice_d2": BlendMode(slice_blend, dim=2, flatten=False),
    "slice_d3": BlendMode(slice_blend, dim=3, flatten=False),
    "slice_d1_flip": BlendMode(
        slice_blend,
        dim=1,
        flatten=False,
        flip_a=True,
        flip_b=True,
        flip_out=True,
    ),
    "slice_d2_flip": BlendMode(
        slice_blend,
        dim=2,
        flatten=False,
        flip_a=True,
        flip_b=True,
        flip_out=True,
    ),
    "slice_d3_flip": BlendMode(
        slice_blend,
        dim=3,
        flatten=False,
        flip_a=True,
        flip_b=True,
        flip_out=True,
    ),
    "slicesmooth_d1": BlendMode(slice_blend_smooth, dim=1, flatten=False),
    "slicesmooth_d2": BlendMode(slice_blend_smooth, dim=2, flatten=False),
    "slicesmooth_d3": BlendMode(slice_blend_smooth, dim=3, flatten=False),
    "loplerp_a098": BlendMode(lop_lerp, a_ratio=0.98),
    "loplerp_a101": BlendMode(lop_lerp, a_ratio=1.01),
    "loplerp_a102": BlendMode(lop_lerp, a_ratio=1.02),
    "loplerp_a105": BlendMode(lop_lerp, a_ratio=1.05),
    "cosinesimilarity": BlendMode(
        cosine_similarity_blend,
        backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    ),
    "cosinesimilarity_flat": BlendMode(
        cosine_similarity_blend_flat,
        start_dim=1,
        end_dim=-1,
        backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    ),
    "cosinesimilarity_flat_spatdims": BlendMode(
        cosine_similarity_blend_flat,
        start_dim=-2,
        end_dim=-1,
        backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    ),
    "cosinesimilarity_avg_spatdims": BlendMode(
        cosine_similarity_blend_avg,
        dims=(-1, -2),
        backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    ),
    "hybrid_lerp_cosinesimilarity": BlendMode(
        blend_blend,
        blend_mode_a="lerp",
        blend_mode_b="cosinesimilarity_flat_spatdims",
        blend_mode_blend="lerp",
        blend_blend=0.5,
    ),
    # Simulates a brightening effect by dividing a by (1 - b) with a small epsilon to avoid division by zero.
    "colordodge": BlendMode(
        lambda a, b, _t: a / (1 - b + 1e-6),
        allow_scale=False,
        force_rescale=True,
    ),
    "difference": BlendMode(
        lambda a, b, t: abs(a - b) * t,
        # normalize,
        allow_scale=False,
        force_rescale=True,
    ),
    "exclusion": BlendMode(
        lambda a, b, t: (a + b - 2 * a * b) * t,
        # normalize,
        allow_scale=False,
        force_rescale=True,
    ),
    "glow": BlendMode(
        lambda a, b, _t: torch.where(
            a <= 1,
            a**2 / (1 - b + 1e-6),
            b * (a - 1) / (a + 1e-6),
        ),
        allow_scale=False,
        force_rescale=True,
    ),
    "hardlight": BlendMode(
        lambda a, b, t: (
            (
                2 * a * b * (a < 0.5).float()
                + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()
            )
            * t
        ),
        allow_scale=False,
        force_rescale=True,
    ),
    "linearlight": BlendMode(
        lambda a, b, _t: torch.where(b <= 0.5, a + 2 * b - 1, a + 2 * (b - 0.5)),
        force_rescale=True,
    ),
    "multiply": BlendMode(
        lambda a, b, t: (a * t).mul_(b * t),
        normalize,
        allow_scale=False,
    ),
    "multiply_by_b": BlendMode(
        lambda a, b, _t: a * b,
        allow_scale=False,
    ),
    "overlay": BlendMode(
        lambda a, b, t: (
            (2 * a * b + a**2 - 2 * a * b * a) * t
            if torch.all(b < 0.5)
            else (1 - 2 * (1 - a) * (1 - b)) * t
        ),
        allow_scale=False,
        force_rescale=True,
    ),
    # Combines tensors a and b using the Pin Light formula.
    "pinlight": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 0.5,
            torch.min(a, 2 * b),
            torch.max(a, 2 * b - 1),
        ),
        force_rescale=True,
    ),
    "reflect": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 1,
            b**2 / (1 - a + 1e-6),
            a * (b - 1) / (b + 1e-6),
        ),
        allow_scale=False,
        force_rescale=True,
    ),
    "screen": BlendMode(
        lambda a, b, t: 1 - (1 - a) * (1 - b) * (1 - t),
        allow_scale=False,
        force_rescale=True,
    ),
    "subtract": BlendMode(lambda a, b, t: a * t - b * t, allow_scale=False),
    "subtract_b": BlendMode(lambda a, b, t: a - b * t, allow_scale=False),
    "subtract_b_scaleup_a": BlendMode(
        lambda a, b, t: a * (1.0 + t) - b * t,
        allow_scale=False,
    ),
    "vividlight": BlendMode(
        lambda a, b, _t: torch.where(
            b <= 0.5,
            a / (1 - 2 * b + 1e-6),
            (a + 2 * b - 1) / (2 * (1 - b) + 1e-6),
        ),
        allow_scale=False,
        force_rescale=True,
    ),
    "wavelet_b_hi_100_lo_0": BlendMode(
        f=wavelet_blend,
        blend_yl_offset=0.0,
        blend_yh_offset=1.0,
        wave="db4",
        level=8,
    ),
    "wavelet_b_hi_0_lo_100": BlendMode(
        f=wavelet_blend,
        blend_yl_offset=1.0,
        blend_yh_offset=0.0,
        wave="db4",
        level=8,
    ),
    "ortho": BlendMode(ortho_blend),
    "ortho_rescaled": BlendMode(ortho_blend, rescale_limit=2.0),
    "ortho_rescaled_lerpish": BlendMode(
        ortho_blend,
        rescale_limit=2.0,
        rescale_result_blend_mode="lerp",
        rescale_result_mode="blend",
    ),
    "ortho_lerp": BlendMode(ortho_blend, blend_mode="lerp"),
    "ortho_dyn_lerp": BlendMode(
        ortho_blend,
        blend_mode="lerp",
        rescale_result_mode="blend",
        rescale_limit=4.0,
        dyn_ortho_mode=True,
    ),
    "ortho_dyn_lerp_inverted": BlendMode(
        ortho_blend,
        blend_mode="lerp",
        rescale_result_mode="blend",
        rescale_limit=2.0,
        dyn_ortho_mode=True,
        rev=True,
        invert_scale=1.0,
    ),
    "ortho_lerp_rescaled": BlendMode(
        ortho_blend,
        blend_mode="lerp",
        rescale_result_mode="blend",
        rescale_limit=2.0,
    ),
    "ortho_cfg": BlendMode(
        lambda a, b, t, **kwargs: ortho_blend(b, a - b, t, **kwargs),
    ),
    "ortho_cfg_base_a": BlendMode(
        lambda a, b, t, **kwargs: ortho_blend(a, a - b, t, **kwargs),
    ),
    "symmetric_ortho": BlendMode(symmetric_ortho_blend),
    "symmetric_ortho_rescaled": BlendMode(symmetric_ortho_blend, rescale_limit=2.0),
    "contrastive_ortho_cfg": BlendMode(
        lambda a, b, t, **kwargs: contrastive_ortho_cfg_base_a(b, a, t, **kwargs),
    ),
    "contrastive_ortho_cfg_base_a": BlendMode(contrastive_ortho_cfg_base_a),
    # These next two probably aren't actually useful.
    "symmetric_ortho_cfg": BlendMode(
        lambda a, b, t, **kwargs: symmetric_ortho_blend(b, a - b, t, **kwargs),
        visible=False,
    ),
    "symmetric_ortho_cfg_base_a": BlendMode(
        lambda a, b, t, **kwargs: symmetric_ortho_blend(a, a - b, t, **kwargs),
        visible=False,
    ),
    "tiered_blend": BlendMode(
        tiered_blend,
        visible=False,
    ),
    "sp_circular_interpolation": BlendMode(sp_circular_interpolation),
    "geodesic_square_matrix": BlendMode(geodesic_square_matrix),
    "fft_blend": BlendMode(fft_blend),
    "fft_phase_blend": BlendMode(partial(fft_blend, magnitude_blend_multiplier=0.0)),
    "fft_magnitude_blend": BlendMode(partial(fft_blend, phase_blend_multiplier=0.0)),
    "decomp_rank_blend": BlendMode(DecompBlend.rank_slice_blend),
    "decomp_diff": BlendMode(DecompBlend.rank_blend),
    "chain": BlendMode(chain_blend, visible=False),
    "pct_limited_025": BlendMode(partial(pct_limit_blend, diff_limit=0.25)),
    "moment_aligned": BlendMode(moment_aligned_blend),
    "distro_aligned": BlendMode(partial(distro_aligned_blend, align_b="da")),
    "distro_aligned_result": BlendMode(
        partial(distro_aligned_blend, align_result="da"),
    ),
    "gaussian_aligned_result": BlendMode(
        partial(distro_aligned_blend, align_result="dg"),
    ),
    "gaussian_aligned": BlendMode(
        partial(
            distro_aligned_blend,
            align_a="dg",
            align_b="dg",
            align_result="dg",
        ),
    ),
    "magnitude_interpolation_lerpsign": BlendMode(
        partial(rms_interpolation, sign_mode="blend", abs_inputs=True, power=1.0),
    ),
    "rms_interpolation_lerpsign": BlendMode(
        partial(rms_interpolation, sign_mode="blend"),
    ),
    "pythagorean_lerp": BlendMode(pythagorean_lerp),
    "orbit": BlendMode(orbit_blend),
}

BLENDING_MODES |= {
    f"norm{k}": v.edited(norm=normalize)
    for k, v in BLENDING_MODES.items()
    if k != "hslerp" and v.norm is None
}

BLENDING_MODES |= {f"rev{k}": v.edited(rev=True) for k, v in BLENDING_MODES.items()}

BLENDING_MODES = BlendingModes(BLENDING_MODES)

BIDERP_MODES = {
    k: v.edited(norm_dims=0)
    for k, v in BLENDING_MODES.items()
    if (v.allow_scale or OVERRIDE_NO_SCALE) and not k.endswith("slerp")
}

BIDERP_MODES |= {
    "hslerp": hslerp_alt,
    "bislerp": slerp_orig,
    "altbislerp": altslerp,
    "revaltbislerp": lambda a, b, t: altslerp(b, a, t),
    "bibislerp": BLENDING_MODES["bislerp_wrong"].edited(norm_dims=0),
    "revhslerp": lambda a, b, t: hslerp_alt(b, a, t),
    "revbislerp": lambda a, b, t: slerp_orig(b, a, t),
    "revbibislerp": BLENDING_MODES["revbislerp_wrong"].edited(norm_dims=0),
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
    if name == "randmultihilowpass":
        scale *= 0.1
        randskip = 4
        randitems = ("multilowpass", "multihighpass")
    elif name == "randhilowpass":
        scale *= 0.1
        randskip = 6
        randitems = ("lowpass", "highpass")
    elif name == "randlowbandpass":
        scale *= 0.25
        randskip = 1
        randitems = ("lowpass", "multilowpass", "bandpass")
    elif name == "randhibandpass":
        scale *= 0.25
        randskip = 1
        randitems = ("highpass", "multihighpass", "bandpass")
    elif name == "bandpass":
        scale *= 0.2
    elif name in {"renoise1", "renoise2"}:
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
    if name == "korniabilateralblur":
        return x + (kf.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5)) - x) * (scale * 2.0)
    if name == "korniagaussianblur":
        return kf.gaussian_blur2d(x, (3, 3), (1.5, 1.5)) * scale
    if name == "korniasharpen":
        return x + (kf.unsharp_mask(x, (3, 3), (1.5, 1.5)) - x) * (scale / 2.0)
    if name in {"korniaedge", "korniarevedge"}:
        blur = kf.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5)) - x
        sharpened = kf.unsharp_mask(x, (3, 3), (1.5, 1.5)) - x
        if name == "korniarevedge":
            scale *= -1.0
        return x + (sharpened + blur) * (scale / 2.0)
    if name == "korniarandblursharp":
        return enhance_tensor(
            x,
            "korniagaussianblur"
            if torch.rand(1, device="cpu").item() < 0.5
            else "korniasharpen",
            scale=scale,
        )
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
            for ridx, m in zip(ridxs, (mode, mode_h), strict=True)
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
def biderp(samples, width, height, mode="bislerp", mode_h=None):
    if mode_h is None:
        mode_h = mode

    derp_w = (BIDERP_MODES if ":" not in mode else BLENDING_MODES).get(mode, slerp_orig)
    derp_h = (BIDERP_MODES if ":" not in mode_h else BLENDING_MODES).get(
        mode_h, slerp_orig
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
