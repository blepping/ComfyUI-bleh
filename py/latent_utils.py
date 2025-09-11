# Credits:
#   Blending, slice and filtering functions based on https://github.com/WASasquatch/FreeU_Advanced
from __future__ import annotations

import math
import os
from functools import partial
from typing import ClassVar

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
    return blend(a, b, tadj)


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
        kernel_size = kernel_size * 2  # noqa: PLR6104
    if len(sigma) == 1:
        sigma = sigma * 2  # noqa: PLR6104
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
        **kwargs: dict,
    ):
        t_device = torch.device("cpu") if cpu else a.device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor((t,), dtype=a.dtype, device=t_device)
        elif t.device != t_device:
            t = t.detach().clone().to(t_device)
        tmin, tmax = t.aminmax()
        tmin, tmax = min(tmin, 0.0), max(tmax, 1.0)
        t = t - tmin  # noqa: PLR6104
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
        return self.output(a, b, t, **kwargs)


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
    return blend_function(a, b, ratios)


def gradient_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float | torch.Tensor,
    *,
    flatten_start_dim=1,
    scaling_constant=0.9,
    blend_function=torch.lerp,
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
    result = blend_function(a, b, ratios)
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


def slice_blend_smooth(  # noqa: PLR0914
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
    return blend_function(a, b, blend_mask).reshape(orig_shape)


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


# # Thanks, ChatGPT though you did get the ratio reversed.
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
    mag_a = torch.norm(a, p=2, dim=dim, keepdim=True).add_(eps)
    mag_b = torch.norm(b, p=2, dim=dim, keepdim=True).add_(eps)

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
    *args: list,
    backend=DEFAULT_COSINE_SIMILARITY_BLEND_BACKEND,
    **kwargs: dict,
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


class BlendMode:
    __slots__ = (
        "allow_scale",
        "f",
        "f_kwargs",
        "f_raw",
        "force_rescale",
        "norm",
        "norm_dims",
        "rescale_dims",
        "rev",
    )

    class _Empty:
        pass

    def __init__(  # noqa: PLR0917
        self,
        f,
        norm=None,
        norm_dims=(-3, -2, -1),
        rev=False,
        allow_scale=True,
        rescale_dims=(-3, -2, -1),
        force_rescale=False,
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
        self.force_rescale = force_rescale

    def edited(
        self,
        *,
        f=_Empty,
        norm=_Empty,
        norm_dims=_Empty,
        rev=_Empty,
        allow_scale=_Empty,
        rescale_dims=_Empty,
        force_rescale=_Empty,
        preserve_kwargs=True,
        **kwargs: dict,
    ) -> object:
        empty = self._Empty
        kwargs = (self.f_kwargs | kwargs) if preserve_kwargs else kwargs
        return self.__class__(
            f if f is not empty else self.f_raw,
            norm=norm if norm is not empty else self.norm,
            norm_dims=norm_dims if norm_dims is not empty else self.norm_dims,
            rev=rev if rev is not empty else self.rev,
            allow_scale=allow_scale if allow_scale is not empty else self.allow_scale,
            rescale_dims=rescale_dims
            if rescale_dims is not empty
            else self.rescale_dims,
            force_rescale=force_rescale
            if force_rescale is not empty
            else self.force_rescale,
            **kwargs,
        )

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
        return (t - tmin).div_(tmax - tmin).clamp_(0, 1), tmin, tmax

    def __call__(self, a, b, t, *, norm_dims=_Empty) -> torch.Tensor:
        if not self.force_rescale:
            return self.__call__internal(a, b, t, norm_dims=norm_dims)
        a, amin, amax = self.rescale(a)
        b, bmin, bmax = self.rescale(b)
        result = self.__call__internal(a, b, t, norm_dims=norm_dims)
        del a, b
        rmin, rmax = torch.lerp(amin, bmin, 0.5), torch.lerp(amax, bmax, 0.5)
        del amin, amax, bmin, bmax
        return result.mul_(rmax.sub_(rmin)).add_(rmin)

    def __call__internal(self, a, b, t, *, norm_dims=_Empty) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) and isinstance(a, torch.Tensor):
            t = a.new_full((1,), t)
        if self.rev:
            a, b = b, a
        if self.norm is None:
            return self.f(a, b, t)
        return self.norm(
            self.f(a, b, t),
            reference_latent=torch.lerp(a, b, t),
            dim=self.norm_dims if norm_dims is self._Empty else norm_dims,
        )


class BlendingModes:
    def __init__(self, builtins=None):
        self.builtins = {} if builtins is None else builtins
        self.cache = {}

    def get(self, k: str, default=None):
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
        return self.builtins.items()

    def values(self):
        return self.builtins.values()

    def __contains__(self, k: str) -> bool:
        return self.get(k) is not None

    def __iter__(self):
        return self.builtins.__iter__()

    keys = __iter__

    def __setitem__(self, k: str, v) -> str:
        self.builtins[k] = v if isinstance(v, BlendMode) else BlendMode(v)

    def __getitem__(self, k: str):
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
    "bislerp": BlendMode(
        lambda a, b, t: ((1 - t) * a).add_(t * b),
        normalize,
    ),
    # "nbislerp": BlendMode(lambda a, b, t: (1 - t) * a + t * b, normalize),
    "slerp": BlendMode(altslerp),
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
    # Interpolates between tensors a and b using linear interpolation.
    # "lerp": BlendMode(lambda a, b, t: ((1.0 - t) * a).add_(t * b)),
    "lerp": BlendMode(torch.lerp),
    "lerp050x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(0.5)),
    "lerp075x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(0.75)),
    "lerp110x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.1)),
    "lerp125x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.25)),
    "lerp150x": BlendMode(lambda a, b, t: (((1 - t) * a).add_(t * b)).mul_(1.5)),
    "lerp_copysign_a": BlendMode(
        lambda a, b, t: ((1.0 - t) * a).add_(t * b).copysign_(a),
    ),
    "lerp_copysign_b": BlendMode(
        lambda a, b, t: ((1.0 - t) * a).add_(t * b).copysign_(b),
    ),
    "lerp_avoidsign_a": BlendMode(
        lambda a, b, t: ((1.0 - t) * a).add_(t * b).copysign_(a.neg()),
    ),
    "lerp_avoidsign_b": BlendMode(
        lambda a, b, t: ((1.0 - t) * a).add_(t * b).copysign_(b.neg()),
    ),
    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    "lineardodge": BlendMode(lambda a, b, t: (b * t).add_(a)),
    "copysign": BlendMode(lambda a, b, _t: torch.copysign(a, b)),
    "probcopysign": BlendMode(lambda a, b, t: torch.copysign(a, prob_blend(a, b, t))),
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
            2 * a * b * (a < 0.5).float()
            + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()
        )
        * t,
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
    "overlay": BlendMode(
        lambda a, b, t: (2 * a * b + a**2 - 2 * a * b * a) * t
        if torch.all(b < 0.5)
        else (1 - 2 * (1 - a) * (1 - b)) * t,
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

    derp_w = (BIDERP_MODES if ":" not in mode else BLENDING_MODES).get(mode, slerp_orig)
    derp_h = (BIDERP_MODES if ":" not in mode_h else BLENDING_MODES).get(mode_h, slerp_orig)

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
