from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING

import comfy.ldm.modules.attention as comfyattn
import yaml
from comfy.samplers import KSAMPLER

try:
    import sageattention

    sageattn_default_function = sageattention.sageattn
except ImportError:
    sageattention = None
    sageattn_default_head_sizes = None
    sageattn_default_function = None


if TYPE_CHECKING:
    import collections
    from collections.abc import Callable

    import torch


if sageattention is not None:
    try:
        sageattn_version = importlib.metadata.version("sageattention")
    except Exception:  # noqa: BLE001
        sageattn_version = None
    if (
        sageattn_version is None
        or sageattn_version.startswith("1.")
        or sageattn_version == "2.0.0"
    ):
        sageattn_default_head_sizes = {64, 96, 128}
    else:
        # SageAttention 2.0.1 (and later one would assume) supports up to 128.
        sageattn_default_head_sizes = set(range(1, 129))


def attention_sage(  # noqa: PLR0917
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: torch.dtype | None = None,
    skip_reshape: bool = False,
    *,
    orig_attention: Callable,
    sageattn_allow_head_sizes: collections.abc.Collection
    | None = sageattn_default_head_sizes,
    sageattn_function: collections.abc.Callable = sageattn_default_function,
    sageattn_verbose: bool = False,
    **kwargs: dict[str],
) -> torch.Tensor:
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
    enabled = sageattn_allow_head_sizes is None or dim_head in sageattn_allow_head_sizes
    if sageattn_verbose:
        print(
            f"\n>> SAGE({enabled}): reshape={not skip_reshape}, dim_head={q.shape[-1]}, heads={heads}, adj_heads={dim_head}, q={q.shape}, k={k.shape}, v={v.shape}, args: {kwargs}\n",
        )
    if not enabled:
        return orig_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
        )
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, ...]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
    if not skip_reshape:
        if kwargs.get("tensor_layout") != "NHD":
            q, k, v = (
                t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)
            )
            do_transpose = True
        else:
            q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
            do_transpose = False
    else:
        do_transpose = True
    sm_scale_hd = kwargs.pop(f"sm_scale_{dim_head}", None)
    if sm_scale_hd is not None:
        kwargs["sm_scale"] = sm_scale_hd
    result = sageattn_function(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=mask,
        dropout_p=0.0,
        **kwargs,
    )
    if do_transpose:
        result = result.transpose(1, 2)
    return result.reshape(b, -1, heads * dim_head)


def copy_funattrs(fun, dest=None):
    if dest is None:

        def dest():
            pass

    for k in ("__code__", "__defaults__", "__kwdefaults__"):
        setattr(dest, k, getattr(fun, k))
    return dest


def make_sageattn_wrapper(
    *,
    orig_attn,
    sageattn_function: str = "sageattn",
    **kwargs: dict,
):
    outer_kwargs = kwargs
    sageattn_function = getattr(sageattention, sageattn_function)

    def attn(
        *args: list,
        _sage_outer_kwargs=outer_kwargs,
        _sage_orig_attention=orig_attn,
        _sage_sageattn_function=sageattn_function,
        _sage_attn=attention_sage,
        **kwargs: dict,
    ) -> torch.Tensor:
        return _sage_attn(
            *args,
            orig_attention=_sage_orig_attention,
            sageattn_function=_sage_sageattn_function,
            **_sage_outer_kwargs,
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
    attn = make_sageattn_wrapper(orig_attn=orig_attn, **kwargs)
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
            "DiffuseHighSampler: yaml_parameters must either be null or an object",
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
        attn = make_sageattn_wrapper(
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
