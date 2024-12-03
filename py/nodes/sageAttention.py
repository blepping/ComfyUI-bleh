from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import yaml
from comfy.ldm.modules import attention as comfy_attention
from comfy.samplers import KSAMPLER

try:
    import sageattention
except ImportError:
    sageattention = None

if TYPE_CHECKING:
    import torch

orig_attention = comfy_attention.optimized_attention


def attention_sage(  # noqa: PLR0917
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    sageattn_allow_head_sizes: set | tuple | list | None = None,
    sageattn_function=sageattention.sageattn if sageattention is not None else None,
    sageattn_verbose=False,
    **kwargs: dict[str],
):
    if sageattn_allow_head_sizes is None:
        sageattn_allow_head_sizes = {64, 96, 128}
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
    enabled = dim_head in sageattn_allow_head_sizes
    if sageattn_verbose:
        print(
            f"\n>> SAGE({enabled}): reshape={not skip_reshape}, dim_head={dim_head}, heads={heads}, adj_heads={q.shape[-1] // heads}, args: {kwargs}\n",
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
    if not skip_reshape:
        if kwargs.get("tensor_layout") != "NHD":
            q, k, v = (
                t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)
            )
            do_transpose = True
        else:
            q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
            do_transpose = False
    sm_scale_hd_key = f"sm_scale_{dim_head}"
    sm_scale_hd = kwargs.get(sm_scale_hd_key)
    if sm_scale_hd is not None:
        del kwargs[sm_scale_hd_key]
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


def monkeypatch_attention(enabled: bool, **kwargs: dict[str]):
    sageattn_function = getattr(
        sageattention,
        kwargs.pop("sageattn_function", "sageattn"),
    )
    comfy_attention.optimized_attention = (
        orig_attention
        if not enabled
        else partial(
            attention_sage,
            sageattn_function=sageattn_function,
            **kwargs,
        )
    )


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
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
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

    @classmethod
    def go(cls, *, model: object, enabled: bool, yaml_parameters: str | None = None):
        if sageattention is None:
            raise RuntimeError(
                "sageattention not installed to Python environment: SageAttention feature unavailable",
            )
        monkeypatch_attention(enabled, **get_yaml_parameters(yaml_parameters))
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

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict[str]):
        sigma_float = float(sigma.max().detach().cpu())
        enabled = end_sigma <= sigma_float <= start_sigma
        backup_attn = comfy_attention.optimized_attention
        if enabled:
            monkeypatch_attention(enabled=True, **sageattn_kwargs)
        else:
            comfy_attention.optimized_attention = orig_attention
        try:
            result = model(x, sigma, **extra_args)
        finally:
            comfy_attention.optimized_attention = backup_attn
        return result

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
        start_percent=0.0,
        end_percent=1.0,
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
