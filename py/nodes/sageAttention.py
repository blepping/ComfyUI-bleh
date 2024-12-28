from __future__ import annotations

import contextlib
import importlib
import sys
import weakref
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import comfy
import comfy_extras
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


class ComfyModules(NamedTuple):
    comfy = comfy
    comfy_extras = comfy_extras
    custom_nodes = SimpleNamespace()


_attention_paths = (
    # Base attention
    "comfy.ldm.modules.attention",
    # Comfy modules
    "comfy_extras.nodes_sag",
    "comfy.ldm.audio.dit",
    "comfy.ldm.aura.mmdit",
    "comfy.ldm.cascade.common",
    "comfy.ldm.flux.math",
    "comfy.ldm.genmo.join_model.asymm_models_joint",
    "comfy.ldm.genmo.vae.model",
    "comfy.ldm.hunyuan_video.model",
    "comfy.ldm.hydit.attn_layers",
    "comfy.ldm.hydit.poolers",
    "comfy.ldm.modules.diffusionmodules.mmdit",
    "comfy.ldm.pixart.blocks",
    # Fluxtapoz custom node
    "custom_nodes.fluxtapoz.rave_attention",
    "custom_nodes.fluxtapoz.rave_rope_attention",
    "custom_nodes.fluxtapoz.flux.layers",
    "custom_nodes.fluxtapoz.nodes.apply_pag_node",
    # sd-perturbed-attention custom node
    "custom_nodes.sd_perturbed_attention.pag_nodes",
)


def get_obj_path(obj: Any, path: str) -> Any:  # noqa: ANN401
    try:
        for pitem in path.split("."):
            obj = getattr(obj, pitem, None)
            if obj is None:
                return None
    except ReferenceError:
        return None
    return obj


_attention_modules = {}

orig_attentions = {}

_integrations_map = (
    ("sd_perturbed_attention", "sd-perturbed-attention"),
    ("fluxtapoz", "ComfyUI-Fluxtapoz"),
)


def build_integrations() -> None:
    modules = sys.modules.copy()
    for key, module_name in _integrations_map:
        module = modules.get(module_name)
        if module is not None:
            setattr(ComfyModules.custom_nodes, key, weakref.proxy(module))


def build_attention_modules(force: bool = False) -> None:
    if _attention_modules and not force:
        return
    build_integrations()
    _attention_modules.clear()
    _attention_modules.update({
        path: module
        for path, module in (
            (path, get_obj_path(ComfyModules, path)) for path in _attention_paths
        )
        if module is not None
    })


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


def save_attentions() -> dict:
    build_attention_modules()
    return {
        path: (module, fun)
        for path, module, fun in (
            (path, module, getattr(module, "optimized_attention", None))
            for path, module in _attention_modules.items()
        )
        if fun is not None
    }


def build_attentions(
    orig_attentions: dict,
    *,
    sageattn_function: str = "sageattn",
    **kwargs: dict,
) -> dict:
    build_attention_modules()
    sageattn_function = getattr(sageattention, sageattn_function)
    return {
        path: (
            module,
            partial(
                attention_sage,
                orig_attention=fun,
                sageattn_function=sageattn_function,
                **kwargs,
            ),
        )
        for path, (module, fun) in orig_attentions.items()
    }


def monkeypatch_attention(
    enabled: bool,
    *,
    orig_attentions: dict,
    new_attentions: dict | None = None,
    **kwargs: dict[str],
) -> None:
    if not enabled:
        new_attentions = orig_attentions
    elif new_attentions is None:
        new_attentions = build_attentions(orig_attentions, **kwargs)
    for module, fun in new_attentions.values():
        module.optimized_attention = fun


@contextlib.contextmanager
def sageattn_context(
    enabled: bool,
    *,
    orig_attentions: dict | None = None,
    new_attentions: dict | None = None,
    **kwargs: dict,
):
    if not enabled:
        yield None
        return
    if orig_attentions is None:
        orig_attentions = save_attentions()
    elif len(orig_attentions) == 0:
        orig_attentions.update(save_attentions())
    if new_attentions == {}:
        new_attentions.update(build_attentions(orig_attentions, **kwargs))
    try:
        monkeypatch_attention(
            enabled,
            orig_attentions=orig_attentions,
            new_attentions=new_attentions,
            **kwargs,
        )
        yield None
    finally:
        monkeypatch_attention(enabled=False, orig_attentions=orig_attentions)


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

    @classmethod
    def go(
        cls,
        *,
        model: object,
        enabled: bool,
        yaml_parameters: str | None = None,
    ) -> tuple:
        if sageattention is None:
            raise RuntimeError(
                "sageattention not installed to Python environment: SageAttention feature unavailable",
            )
        if enabled and not orig_attentions:
            orig_attentions.update(save_attentions())
        monkeypatch_attention(
            enabled,
            orig_attentions=orig_attentions,
            **get_yaml_parameters(yaml_parameters),
        )
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

    new_attentions = {}
    backup_attentions = {}

    def model_wrapper(
        x: torch.Tensor,
        sigma: torch.Tensor,
        **extra_args: dict[str],
    ) -> torch.Tensor:
        sigma_float = float(sigma.max().detach().cpu())
        enabled = end_sigma <= sigma_float <= start_sigma
        with sageattn_context(
            enabled=enabled,
            orig_attentions=backup_attentions,
            new_attentions=new_attentions,
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
