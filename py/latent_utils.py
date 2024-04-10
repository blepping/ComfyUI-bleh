# Credits:
# Blending, slice and filtering functions based on https://github.com/WASasquatch/FreeU_Advanced

import math

import numpy as np
import torch
from torch import fft


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

    norm = (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    if t < 0.5:
        result += norm
    else:
        result -= norm

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


BLENDING_MODES = {
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


def antialias_tensor(x, antialias_size):
    channels = x.shape[1]
    filt = make_filter(channels, x.dtype, antialias_size).to(x.device)
    return torch.nn.functional.conv2d(x, filt, groups=channels, padding="same")


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
    return antialias_tensor(result, antialias_size)


# Modified from ComfyUI
def biderp(samples, width, height, mode="bislerp", mode_h=None):
    if mode_h is None:
        mode_h = mode

    modes = {
        "colorize": BLENDING_MODES["colorize"],
        "hslerp": hslerp_alt,
        "bislerp": slerp_orig,
        "bibislerp": BLENDING_MODES["bislerp"],
        "inject": BLENDING_MODES["inject"],
        "lerp": BLENDING_MODES["lerp"],
        "lineardodge": BLENDING_MODES["lineardodge"],
        "cosinterp": BLENDING_MODES["cosinterp"],
        "cuberp": BLENDING_MODES["cuberp"],
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
            mask = mask + (scale_mask - mask) * strength

    x_freq *= mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    return fft.ifftn(x_freq, dim=(-2, -1)).real.to(x.dtype)
