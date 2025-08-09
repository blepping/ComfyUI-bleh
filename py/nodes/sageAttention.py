# ruff: noqa: PLR6104
from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING

import comfy.ldm.modules.attention as comfyattn
import torch
import yaml
from comfy.samplers import KSAMPLER

try:
    import sageattention

    sageattn_default_function = sageattention.sageattn
except ImportError:
    sageattention = None
    sageattn_default_head_sizes = None
    sageattn_default_function = None

try:
    import spas_sage_attn
except ImportError:
    spas_sage_attn = None


if TYPE_CHECKING:
    import collections
    from collections.abc import Callable


if sageattention is not None:
    try:
        sageattn_version = importlib.metadata.version("sageattention")
    except Exception:  # noqa: BLE001
        sageattn_version = "unknown"
    if (
        sageattn_version == "unknown"
        or sageattn_version.startswith("1.")
        or sageattn_version == "2.0.0"
    ):
        sageattn_default_head_sizes = {64, 96, 128}
    else:
        # SageAttention 2.0.1 (and later one would assume) supports up to 128.
        sageattn_default_head_sizes = set(range(1, 129))
else:
    sageattn_version = "unknown"


def attention_bleh(  # noqa: PLR0914
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    orig_attention: Callable,
    sageattn_allow_head_sizes: collections.abc.Collection
    | None = sageattn_default_head_sizes,
    sageattn_function: collections.abc.Callable = sageattn_default_function,
    sageattn_version: str = sageattn_version,
    sageattn_verbose: bool = False,
    **kwargs: dict[str],
) -> torch.Tensor:
    old_sageattn = sageattn_version[:2] in {"1.", "un"}
    orig_attn_kwargs = {
        k: kwargs.pop(k)
        for k in ("mask", "skip_reshape", "skip_output_reshape", "attn_precision")
        if k in kwargs.copy()
    }
    bleh_kwargs = {
        k: kwargs.pop(k)
        for k in kwargs.copy()
        if k.startswith("sm_scale_")
        or k in {"q_multiplier", "k_multiplier", "v_multiplier", "output_multiplier"}
    }
    mask = orig_attn_kwargs.get("mask")
    skip_reshape = orig_attn_kwargs.get("skip_reshape", False)
    skip_output_reshape = orig_attn_kwargs.get("skip_output_reshape", False)
    batch = q.shape[0]
    dim_head = q.shape[-1] // (1 if skip_reshape else heads)
    enabled = sageattn_allow_head_sizes is None or dim_head in sageattn_allow_head_sizes
    if enabled and old_sageattn:
        enabled = all(t.shape == q.shape for t in (k, v))
    if sageattn_verbose:
        print(
            f"\n>> SAGE({enabled}): reshape={not skip_reshape}, output_reshape={not skip_output_reshape}, dim_head={q.shape[-1]}, heads={heads}, adj_heads={dim_head}, q={q.shape}, k={k.shape}, v={v.shape}, orig_attn_args={orig_attn_kwargs}, args: {kwargs}\n",
        )

    if not enabled:
        return orig_attention(q, k, v, heads, **orig_attn_kwargs)
    tensor_layout = kwargs.pop("tensor_layout", None)
    if old_sageattn:
        tensor_layout = "HND"
    elif tensor_layout is None:
        tensor_layout = "HND" if skip_reshape else "NHD"
    tensor_layout = tensor_layout.strip().upper()
    if tensor_layout not in {"NHD", "HND"}:
        raise ValueError("Bad tensor_layout, must be one of NHD, HND")
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, ...]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
    if not skip_reshape:
        if tensor_layout == "HND":
            q, k, v = (
                t.view(batch, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)
            )
            do_transpose = True
        else:
            q, k, v = (t.view(batch, -1, heads, dim_head) for t in (q, k, v))
            do_transpose = False
    else:
        do_transpose = not skip_output_reshape
        if not old_sageattn and tensor_layout == "NHD":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            do_transpose = skip_output_reshape
    if not old_sageattn:
        kwargs["tensor_layout"] = tensor_layout
    sm_scale_hd = bleh_kwargs.pop(f"sm_scale_{dim_head}", None)
    if sm_scale_hd is not None:
        kwargs["sm_scale"] = sm_scale_hd
    q_multiplier = bleh_kwargs.get("q_multiplier", 1.0)
    k_multiplier = bleh_kwargs.get("k_multiplier", 1.0)
    v_multiplier = bleh_kwargs.get("v_multiplier", 1.0)
    output_multiplier = bleh_kwargs.get("output_multiplier", 1.0)
    if q_multiplier != 1.0:
        q = q * q_multiplier
    if k_multiplier != 1.0:
        k = k * k_multiplier
    if v_multiplier != 1.0:
        v = v * v_multiplier
    kwargs = {"is_causal": False, "dropout_p": 0.0, "attn_mask": mask} | kwargs
    result = (
        torch.zeros_like(q)
        if output_multiplier == 0
        else sageattn_function(q, k, v, **kwargs)
    )
    if output_multiplier not in {0, 1}:
        result *= output_multiplier
    if do_transpose:
        result = result.transpose(1, 2)
    if not skip_output_reshape:
        result = result.reshape(batch, -1, heads * dim_head)
    return result


def copy_funattrs(fun, dest=None):
    if dest is None:
        dest = fun.__class__(fun.__code__, fun.__globals__)
    for k in (
        "__code__",
        "__defaults__",
        "__kwdefaults__",
        "__module__",
    ):
        setattr(dest, k, getattr(fun, k))
    return dest


def make_attn_wrapper(
    *,
    orig_attn,
    sageattn_function: str = "sageattn",
    **kwargs: dict,
):
    outer_kwargs = kwargs
    if sageattn_function.startswith("sparge") and spas_sage_attn is None:
        raise ValueError(
            "SpargeAttention is not available, make sure you have the spas_sage_attn Python package installed",
        )
    if sageattn_function == "sparge":
        sageattn_function = spas_sage_attn.spas_sage2_attn_meansim_cuda
    elif sageattn_function == "sparge1":
        sageattn_function = spas_sage_attn.spas_sage_attn_meansim_cuda
    else:
        sageattn_function = getattr(sageattention, sageattn_function)

    def attn(
        *args: list,
        _bleh_outer_kwargs=outer_kwargs,
        _bleh_orig_attention=orig_attn,
        _bleh_attn_function=sageattn_function,
        _bleh_attn=attention_bleh,
        **kwargs: dict,
    ) -> torch.Tensor:
        return _bleh_attn(
            *args,
            orig_attention=_bleh_orig_attention,
            sageattn_function=_bleh_attn_function,
            **_bleh_outer_kwargs,
            **kwargs,
        )

    return attn


@contextlib.contextmanager
def sageattn_context(
    enabled: bool,
    **kwargs: dict,
):
    if not enabled:
        yield None
        return
    orig_attn = copy_funattrs(comfyattn.optimized_attention)
    attn = make_attn_wrapper(orig_attn=orig_attn, **kwargs)
    try:
        copy_funattrs(attn, comfyattn.optimized_attention)
        yield None
    finally:
        copy_funattrs(orig_attn, comfyattn.optimized_attention)


def get_yaml_parameters(yaml_parameters: str | None = None) -> dict:
    if not yaml_parameters:
        return {}
    extra_params = yaml.safe_load(yaml_parameters)
    if extra_params is None:
        return {}
    if not isinstance(extra_params, dict):
        raise ValueError(  # noqa: TRY004
            "BlehSageAttention: yaml_parameters must either be null or an object",
        )
    return extra_params


class BlehGlobalSageAttention:
    DESCRIPTION = "Deprecated: Prefer using BlehSageAttentionSampler if possible. This node allows globally replacing ComfyUI's attention with SageAtteniton (performance enhancement). Requires SageAttention to be installed into the ComfyUI Python environment. IMPORTANT: This is not a normal model patch. For settings to apply (including toggling on or off) the node must actually be run. If you toggle it on, run your workflow and then bypass or mute the node this will not actually disable SageAttention."
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "hacks"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. These are mostly passed directly to the SageAttention function with no error checking. Must be empty or a YAML object.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    orig_attn = None

    @classmethod
    def go(
        cls,
        *,
        model: object,
        enabled: bool,
        yaml_parameters: str | None = None,
    ) -> tuple:
        if not enabled:
            if cls.orig_attn is not None:
                copy_funattrs(cls.orig_attn, comfyattn.optimized_attention)
                cls.orig_attn = None
            return (model,)
        if sageattention is None:
            raise RuntimeError(
                "sageattention not installed to Python environment: SageAttention feature unavailable",
            )
        if not cls.orig_attn:
            cls.orig_attn = copy_funattrs(comfyattn.optimized_attention)
        attn = make_attn_wrapper(
            orig_attn=cls.orig_attn,
            **get_yaml_parameters(yaml_parameters),
        )
        copy_funattrs(attn, comfyattn.optimized_attention)
        return (model,)


def sageattn_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    sageattn_sampler_options: tuple,
    **kwargs: dict,
) -> torch.Tensor:
    sampler, start_percent, end_percent, sageattn_kwargs = sageattn_sampler_options
    ms = model.inner_model.inner_model.model_sampling
    start_sigma, end_sigma = (
        round(ms.percent_to_sigma(start_percent), 4),
        round(ms.percent_to_sigma(end_percent), 4),
    )
    del ms

    def model_wrapper(
        x: torch.Tensor,
        sigma: torch.Tensor,
        **extra_args: dict[str],
    ) -> torch.Tensor:
        sigma_float = float(sigma.max().detach().cpu())
        enabled = end_sigma <= sigma_float <= start_sigma
        with sageattn_context(
            enabled=enabled,
            **sageattn_kwargs,
        ):
            return model(x, sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **sampler.extra_options,
    )


class BlehSageAttentionSampler:
    DESCRIPTION = "Sampler wrapper that enables using SageAttention (performance enhancement) while sampling is in progress. Requires SageAttention to be installed into the ComfyUI Python environment."
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
            },
            "optional": {
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Time the effect becomes active as a percentage of sampling, not steps.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Time the effect ends (inclusive) as a percentage of sampling, not steps.",
                    },
                ),
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. These are mostly passed directly to the SageAttention function with no error checking. Must be empty or a YAML object.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        sampler: object,
        *,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        yaml_parameters: str | None = None,
    ) -> tuple:
        if sageattention is None:
            raise RuntimeError(
                "sageattention not installed to Python environment: SageAttention feature unavailable",
            )
        return (
            KSAMPLER(
                sageattn_sampler,
                extra_options={
                    "sageattn_sampler_options": (
                        sampler,
                        start_percent,
                        end_percent,
                        get_yaml_parameters(yaml_parameters),
                    ),
                },
            ),
        )
