# Credits:
#   Blending, slice and filtering functions based on https://github.com/WASasquatch/FreeU_Advanced
from __future__ import annotations

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


# def prob_blend_(a, b, t, *, cpu=False):
#     if not isinstance(t, torch.Tensor):
#         t = torch.tensor((t,), dtype=a.dtype, device=a.device)
#     tmin, tmax = t.aminmax()
#     tmin, tmax = min(tmin, 0.0), max(tmax, 1.0)
#     t = t - tmin
#     tdiv = tmax - tmin
#     if tdiv != 0:
#         t /= tdiv
#     t = t.broadcast_to(a.shape)
#     probs = torch.rand(
#         *a.shape,
#         dtype=a.dtype,
#         layout=a.layout,
#         device="cpu" if cpu else a.device,
#     )
#     if probs.device != a.device:
#         probs = probs.to(a.device)
#     return torch.where(probs > t, a, b)


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


def prob_blend(a, b, t, *, cpu=False):
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
    t = t.clamp_(0, 1).broadcast_to(a.shape)
    return torch.where(torch.bernoulli(t).to(device=a.device, dtype=torch.bool), b, a)


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


def prob_blend_smoothed(
    a,
    b,
    t,
    *,
    cpu: bool = False,
    blend=torch.lerp,
    kernel_size: int | tuple | list = 3,
    sigma: float | tuple | list = 1.0,
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
    t = t.clamp_(0, 1).broadcast_to(a.shape)
    t = torch.bernoulli(t).to(device=a.device, dtype=a.dtype)
    t = gaussian_smoothing(t, kernel_size, sigma)
    return blend(a, b, t)


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
    if isinstance(t, torch.Tensor) and t.ndim > 0 and t.numel() > 1:
        t = t.mean()
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
    # print(
    #     f"\nBLENDING: astart={astart}, aend={aend}, bstart={bstart}, bend={bend}, shape={orig_shape}, ashape={achunk.shape}, bshape={bchunk.shape}, aslice={aslice}, bslice={bslice}",
    # )
    result = torch.cat((bchunk, achunk) if flip_out else (achunk, bchunk), dim=dim)
    # print(f"OUT SHAPE: {result.shape}")
    return result.reshape(orig_shape)


class BlendMode:
    __slots__ = (
        "allow_scale",
        "f",
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
    ):
        self.f = f
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
    ):
        empty = self._Empty
        return self.__class__(
            f if f is not empty else self.f,
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

    def __call__(self, a, b, t, *, norm_dims=_Empty):
        if not self.force_rescale:
            return self.__call__internal(a, b, t, norm_dims=norm_dims)
        a, amin, amax = self.rescale(a)
        b, bmin, bmax = self.rescale(b)
        result = self.__call__internal(a, b, t, norm_dims=norm_dims)
        del a, b
        rmin, rmax = torch.lerp(amin, bmin, 0.5), torch.lerp(amax, bmax, 0.5)
        del amin, amax, bmin, bmax
        return result.mul_(rmax.sub_(rmin)).add_(rmin)

    def __call__internal(self, a, b, t, *, norm_dims=_Empty):
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
    "problerp0.25": BlendMode(partial(stochasistic_blend, fuzz=0.25)),
    "problerp0.1": BlendMode(partial(stochasistic_blend, fuzz=0.1)),
    "problerp0.025": BlendMode(partial(stochasistic_blend, fuzz=0.025)),
    "probselect": BlendMode(prob_blend),
    "probselectsmoothed": BlendMode(prob_blend_smoothed),
    "probselectsmoothed_ks5": BlendMode(partial(prob_blend_smoothed, kernel_size=5)),
    "probselectsmoothed_ks9": BlendMode(partial(prob_blend_smoothed, kernel_size=9)),
    "probselectsmoothed_ks9_sigma3": BlendMode(
        partial(
            prob_blend_smoothed,
            kernel_size=9,
            sigma=3.0,
        ),
    ),
    "gradient": BlendMode(gradient_blend),
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
    "copysign": BlendMode(lambda a, b, _t: torch.copysign(a, b)),
    "probcopysign": BlendMode(lambda a, b, t: torch.copysign(a, prob_blend(a, b, t))),
    "slice_flat_d1": BlendMode(partial(slice_blend, dim=1, flatten=True)),
    "slice_flat_d2": BlendMode(partial(slice_blend, dim=2, flatten=True)),
    "slice_d1": BlendMode(partial(slice_blend, dim=1, flatten=False)),
    "slice_d2": BlendMode(partial(slice_blend, dim=2, flatten=False)),
    "slice_d3": BlendMode(partial(slice_blend, dim=3, flatten=False)),
    "slice_d1_flip": BlendMode(
        partial(
            slice_blend,
            dim=1,
            flatten=False,
            flip_a=True,
            flip_b=True,
            flip_out=True,
        ),
    ),
    "slice_d2_flip": BlendMode(
        partial(
            slice_blend,
            dim=2,
            flatten=False,
            flip_a=True,
            flip_b=True,
            flip_out=True,
        ),
    ),
    "slice_d3_flip": BlendMode(
        partial(
            slice_blend,
            dim=3,
            flatten=False,
            flip_a=True,
            flip_b=True,
            flip_out=True,
        ),
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
