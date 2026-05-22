from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import pytorch_wavelets as ptwav
    import pywt

    HAVE_WAVELETS = True
except ImportError:
    ptwav = None
    pywt = None
    HAVE_WAVELETS = False


_V, _D = TypeVar("_V"), TypeVar("_D")


def fallback(val: _V | _D, default: _D = None) -> _V | _D:
    return val if val is not None else default


class Wavelet:
    DEFAULT_MODE = "symmetric"
    DEFAULT_LEVEL = 3
    DEFAULT_WAVE = "db4"
    DEFAULT_USE_1D_DWT = False
    DEFAULT_USE_DTCWT = False
    DEFAULT_QSHIFT = "qshift_a"
    DEFAULT_BIORT = "near_sym_a"

    def __init__(
        self,
        *,
        wave: str = DEFAULT_WAVE,
        level: int = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
        use_1d_dwt: bool = DEFAULT_USE_1D_DWT,
        use_dtcwt: bool = DEFAULT_USE_DTCWT,
        biort: str = DEFAULT_BIORT,
        qshift: str = DEFAULT_QSHIFT,
        inv_wave: str | None = None,
        inv_mode: str | None = None,
        inv_biort: str | None = None,
        inv_qshift=None,
        device: str | torch.device | None = None,
    ):
        if not HAVE_WAVELETS:
            raise RuntimeError(
                "Wavelet use requires the pytorch_wavelets package to be installed in your Python environment",
            )
        inv_wave = fallback(inv_wave, wave)
        inv_mode = fallback(inv_mode, mode)
        inv_biort = fallback(inv_biort, biort)
        inv_qshift = fallback(inv_qshift, qshift)
        if use_dtcwt:
            fwdfun, invfun = ptwav.DTCWTForward, ptwav.DTCWTInverse
        elif use_1d_dwt:
            fwdfun, invfun = ptwav.DWT1DForward, ptwav.DWT1DInverse
        else:
            fwdfun, invfun = ptwav.DWTForward, ptwav.DWTInverse
        if use_dtcwt:
            self._wavelet_forward = fwdfun(
                J=level,
                mode=mode,
                biort=biort,
                qshift=qshift,
            )
            self._wavelet_inverse = invfun(
                mode=inv_mode,
                biort=inv_biort,
                qshift=inv_qshift,
            )
        else:
            self._wavelet_forward = fwdfun(J=level, wave=wave, mode=mode)
            self._wavelet_inverse = invfun(wave=inv_wave, mode=inv_mode)
        self.device = device
        if device is not None:
            self._wavelet_forward = self._wavelet_forward.to(device=device)
            self._wavelet_inverse = self._wavelet_inverse.to(device=device)

    def forward(
        self,
        t: torch.Tensor,
        *,
        forward_function: Callable | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return fallback(forward_function, self._wavelet_forward)(t)

    def inverse(
        self,
        yl: torch.Tensor,
        yh: tuple[torch.Tensor, ...],
        *,
        inverse_function: Callable | None = None,
        two_step_inverse: bool = False,
    ) -> torch.Tensor:
        inverse_function = fallback(inverse_function, self._wavelet_inverse)
        if not two_step_inverse:
            return inverse_function((yl, yh))
        result = inverse_function((torch.zeros_like(yl), yh))
        result += inverse_function(
            (
                yl,
                tuple(torch.zeros_like(yh_band) for yh_band in yh),
            )
        )
        return result

    def to(self, *args: list, copy: bool = False, **kwargs: dict) -> Wavelet:
        o = Wavelet.__new__(Wavelet) if copy else self
        o._wavelet_forward = self._wavelet_forward.to(*args, **kwargs)  # noqa: SLF001
        o._wavelet_inverse = self._wavelet_inverse.to(*args, **kwargs)  # noqa: SLF001
        o.device = kwargs.get("device")
        return o

    @staticmethod
    def wavelist() -> tuple:
        return tuple(pywt.wavelist()) if HAVE_WAVELETS else ()

    @staticmethod
    def biortlist() -> tuple:
        return (
            ("near_sym_a", "near_sym_b", "antonini", "legall") if HAVE_WAVELETS else ()
        )

    @staticmethod
    def qshiftlist() -> tuple:
        return (
            ("qshift_a", "qshift_b", "qshift_c", "qshift_d", "qshift_06")
            if HAVE_WAVELETS
            else ()
        )

    @staticmethod
    def modelist() -> tuple:
        return (
            (
                "symmetric",
                "zero",
                "reflect",
                "replicate",
                "periodization",
                "periodic",
                "constant",
            )
            if HAVE_WAVELETS
            else ()
        )


def expand_yh_scales(
    yh: Sequence,
    *,
    yh_scales: float | Sequence = 1.0,
) -> float | tuple:
    yhlen = len(yh)
    yh_shape = yh[0].shape
    # Doesn't make sense to target orientations for 1D DWD (3D here).
    olen = yh_shape[2] if len(yh_shape) > 3 else 1
    # print(f"\nSIZES: yhlen={yhlen}, olen={olen}, yh_shape={yh[0].shape}")
    if isinstance(yh_scales, (float, int)):
        return ((float(yh_scales),) * olen,) * yhlen
    otemplate = (1.0,) * olen
    yh_scales = tuple(
        (float(band),) * olen
        if isinstance(band, (float, int))
        else (
            (
                *(float(i) for i in band[:olen]),
                *otemplate[: olen - len(band[:olen])],
            )
            if isinstance(band, (tuple, list))
            else band
        )
        for band in yh_scales
    )
    if "fill" in yh_scales:
        fillidx = yh_scales.index("fill")
        if "fill" in yh_scales[fillidx + 1 :]:
            raise ValueError("Only one fill allowed.")
        if fillidx == 0 or len(yh_scales) < 2:
            raise ValueError(
                "Invalid fill value, cannot be in the first position or the only item.",
            )
        yhslen = len(yh_scales)
        if yhslen - 1 < yhlen:
            # Need to pad.
            fill = (yh_scales[fillidx - 1],) * (yhlen - (len(yh_scales) - 1))
            yh_scales = (*yh_scales[:fillidx], *fill, *yh_scales[fillidx + 1 :])
        else:
            # Just remove the "fill".
            yh_scales = (*yh_scales[:fillidx], *yh_scales[fillidx + 1 :])
    return yh_scales[:yhlen]


def wavelet_scaling(
    yl: torch.Tensor,
    yh: Sequence[torch.Tensor],
    yl_scale: float | torch.Tensor,
    yh_scales: float | Sequence[float | Sequence[float]] | None,
    *,
    in_place: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if not in_place:
        yl = yl.clone()
        yh = tuple(yhband.clone() for yhband in yh)
    if yl_scale != 1.0:
        yl *= yl_scale
    yh_scales = expand_yh_scales(
        yh,
        yh_scales=yh_scales if yh_scales is not None else 1.0,
    )
    for hscale, ht in zip(yh_scales, yh):
        if isinstance(hscale, (int, float)):
            ht *= hscale  # noqa: PLW2901
            continue
        for lidx in range(min(ht.shape[2], len(hscale))):
            ht[:, :, lidx] *= hscale[lidx]
    return (yl, yh)


def wavelet_blend(
    a: tuple,
    b: tuple,
    *,
    yl_factor: torch.Tensor | float,
    blend_function: Callable,
    yh_factor: torch.Tensor | float | None = None,
    yh_blend_function: Callable | None = None,
) -> tuple:
    if not isinstance(yl_factor, torch.Tensor):
        yl_factor = a[0].new_full((1,), yl_factor)
    if yh_factor is None:
        yh_factor = yl_factor
    elif not isinstance(yh_factor, torch.Tensor):
        yh_factor = a[0].new_full((1,), yh_factor)
    yh_blend_function = fallback(yh_blend_function, blend_function)
    return (
        blend_function(a[0], b[0], yl_factor),
        tuple(yh_blend_function(ta, tb, yh_factor) for ta, tb in zip(a[1], b[1])),
    )
