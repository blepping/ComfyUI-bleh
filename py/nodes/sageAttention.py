# ruff: noqa: PLR6104
from __future__ import annotations

import contextlib
import importlib
import math
from enum import Enum, auto
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Any, NamedTuple

import comfy.ldm.modules.attention as comfyattn
import torch
import yaml
from comfy.samplers import KSAMPLER
from tqdm import tqdm

from ..latent_utils import BLENDING_MODES

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


HAVE_ATTN_OVERRIDE = hasattr(comfyattn, "register_attention_function")


# class AttnsConfig(NamedTuple):
#     name: str
#     version: str
#     supported_head_sizes: collections.abc.Collection

# class AttentionRule(NamedTuple):


# class AttentionRules(NamedTuple):
#     orig_attn: Callable
#     start_sigma: float = math.inf
#     end_sigma: float = 0.0
#     verbose: bool = False
#     rules: tuple[AttentionRule, ...] = ()


def attention_bleh(  # noqa: PLR0914
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
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
    orig_keys = tuple(kwargs)
    orig_attn_kwargs = {
        k: kwargs.pop(k)
        for k in (
            "skip_reshape",
            "skip_output_reshape",
            "attn_precision",
            "transformer_options",
            "_inside_attn_wrapper",
        )
        if k in orig_keys
    }
    orig_attn_kwargs["mask"] = mask
    orig_keys = tuple(kwargs)
    bleh_kwargs = {
        k: kwargs.pop(k)
        for k in orig_keys
        if k.startswith("sm_scale_")
        or k in {"q_multiplier", "k_multiplier", "v_multiplier", "output_multiplier"}
    }
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
    return result.contiguous()


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
    if sageattn_function in {"sparge", "sparge2"}:
        sageattn_function = spas_sage_attn.spas_sage2_attn_meansim_cuda
    elif sageattn_function == "sparge2_topk":
        sageattn_function = spas_sage_attn.spas_sage2_attn_meansim_topk_cuda
    elif sageattn_function == "sparge1":
        sageattn_function = spas_sage_attn.spas_sage_attn_meansim_cuda
    elif sageattn_function == "sparge1_topk":
        sageattn_function = spas_sage_attn.spas_sage_attn_meansim_topk_cuda
    else:
        sageattn_function = getattr(sageattention, sageattn_function)

    if HAVE_ATTN_OVERRIDE:

        def attn(
            comfy_orig_attn: Callable,
            *args: list,
            _bleh_outer_kwargs=outer_kwargs,
            _bleh_orig_attention=orig_attn,
            _bleh_attn_function=sageattn_function,
            _bleh_attn=attention_bleh,
            **kwargs: dict,
        ) -> torch.Tensor:
            return _bleh_attn(
                *args,
                orig_attention=_bleh_orig_attention or comfy_orig_attn,
                sageattn_function=_bleh_attn_function,
                **_bleh_outer_kwargs,
                **kwargs,
            )
    else:

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


if HAVE_ATTN_OVERRIDE:

    @contextlib.contextmanager
    def sageattn_context(
        enabled: bool,
        **kwargs: dict,
    ):
        yield make_attn_wrapper(orig_attn=None, **kwargs) if enabled else None

else:

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
        if HAVE_ATTN_OVERRIDE:
            raise RuntimeError(
                "BlehGlobalAttention does not currently support the new ComfyUI attention changes."
            )
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


class TimeMode(Enum):
    PERCENT = auto()
    SIGMA = auto()


class SageAttnOptions(NamedTuple):
    sampler: object
    attn_kwargs: dict[str, Any]
    start_time: float = math.inf
    end_time: float = 0.0
    time_mode: TimeMode = TimeMode.PERCENT


class BlehModelWrapper:
    def __init__(self, model: object, model_call: Callable):
        self.__bleh_model = model
        self.__bleh_model_call = model_call

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.__bleh_model_call(self.__bleh_model, *args, **kwargs)

    def __getattr__(self, k: str):
        return getattr(self.__bleh_model, k)


def sageattn_sampler(
    config: SageAttnOptions,
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    # *,
    # sageattn_sampler_options: tuple,
    **kwargs: dict,
) -> torch.Tensor:
    # sampler, start_percent, end_percent, sageattn_kwargs = sageattn_sampler_options
    if config.time_mode == TimeMode.PERCENT:
        ms = model.inner_model.inner_model.model_sampling
        start_sigma, end_sigma = (
            round(ms.percent_to_sigma(config.start_time), 4),
            round(ms.percent_to_sigma(config.end_time), 4),
        )
        del ms
    else:
        start_sigma = config.start_time
        end_sigma = config.end_time

    def model_call(
        model: object,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: dict[str],
    ) -> torch.Tensor:
        sigma_float = float(sigma.max().detach().cpu())
        enabled = end_sigma <= sigma_float <= start_sigma
        with sageattn_context(
            enabled=enabled,
            **config.attn_kwargs,
        ) as attn_override:
            if enabled and HAVE_ATTN_OVERRIDE:
                model_options = kwargs.pop("model_options", {}).copy()
                transformer_options = model_options.pop(
                    "transformer_options", {}
                ).copy()
                transformer_options["optimized_attention_override"] = attn_override
                model_options["transformer_options"] = transformer_options
                kwargs["model_options"] = model_options
            return model(x, sigma, **kwargs)

    return config.sampler.sampler_function(
        BlehModelWrapper(model, model_call),
        x,
        sigmas,
        **kwargs,
        **config.sampler.extra_options,
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
                update_wrapper(
                    partial(
                        sageattn_sampler,
                        SageAttnOptions(
                            start_time=start_percent,
                            end_time=end_percent,
                            time_mode=TimeMode.PERCENT,
                            sampler=sampler,
                            attn_kwargs=get_yaml_parameters(yaml_parameters),
                        ),
                    ),
                    sampler.sampler_function,
                ),
            ),
        )


class AdvancedAttnRule(NamedTuple):
    attn_function: Callable | None
    attn_kwargs: dict[str, Any]
    blend_function: Callable | None = None
    start_sigma: float = math.inf
    end_sigma: float = 0.0
    check_nan: bool = False
    q_multiplier: float = 1.0
    k_multiplier: float = 1.0
    v_multiplier: float = 1.0
    output_multiplier: float = 1.0
    device: torch.device | str | None = None
    dtype: torch.dtype | str | None = None
    blend: float = 1.0
    op_q: str | None = None
    op_k: str | None = None
    op_v: str | None = None
    op_current_result_preblend: str | None = None
    op_result_preblend: str | None = None
    op_result_postblend: str | None = None
    op_result_postblend_diff: str | None = None

    @classmethod
    def build(
        cls,
        *,
        attn_function: str | Callable | None = None,
        blend_mode: str | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> NamedTuple:
        blend_function = BLENDING_MODES[blend_mode] if blend_mode is not None else None
        my_kwargs = {k: kwargs.pop(k) for k in cls._fields if k in kwargs}
        if attn_function == "default":
            attn_function = None
        elif isinstance(attn_function, str):
            kwargs["sageattn_function"] = attn_function
            attn_function = make_attn_wrapper(orig_attn=None, **kwargs)
        if isinstance(dtype, str):
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float64": torch.float64,
            }.get(dtype)
        return cls(
            device=device,
            dtype=dtype,
            blend_function=blend_function,
            attn_function=attn_function,
            attn_kwargs=kwargs,
            **my_kwargs,
        )


class AdvancedAttnConfig(NamedTuple):
    sampler: object
    verbose: bool = False
    rules: tuple[AdvancedAttnRule, ...] = ()
    start_time: float = math.inf
    end_time: float = 0.0
    call_indexes: frozenset[float] = frozenset()
    time_mode: TimeMode = TimeMode.PERCENT
    min_cond_batch: int = 0
    batch_slice: tuple | str | None = None
    max_idx: int = -1
    delegate_override: bool = True
    op_result: str | None = None
    latent_ops: dict[str, Callable] = {}

    @classmethod
    def build(
        cls,
        *,
        rules=(),
        time_mode: str | TimeMode | None = None,
        call_indexes=(),
        **kwargs,
    ) -> NamedTuple:
        fs = frozenset(cls._fields)
        rules = tuple(AdvancedAttnRule.build(**r) for r in rules)
        if isinstance(time_mode, str):
            time_mode = getattr(TimeMode, time_mode.strip().upper())
        call_indexes = frozenset(
            i
            if math.isnan(i) or i == math.inf or not isinstance(i, float)
            else int(i) + 0.5
            for i in call_indexes
        )
        kwargs = {k: v for k, v in kwargs.items() if k in fs}
        return cls(
            rules=rules,
            time_mode=time_mode,
            call_indexes=call_indexes,
            **kwargs,
        )

    def call_op(
        self, op_key: str | None, t: torch.Tensor, *, sigma: float
    ) -> torch.Tensor:
        op = None if op_key is None else self.latent_ops.get(op_key)
        if op is None:
            return t
        return (
            op(t)
            if not hasattr(op, "EXTENDED_LATENT_OPERATION")
            else op(t, sigma=sigma)
        )

    def attn_wrapper(
        self,
        sigma: float,
        currattncall: CurrAttnCall,
        old_override: Callable | None,
        comfy_orig_attn: Callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        fallback_attn = (
            partial(old_override, comfy_orig_attn)
            if old_override is not None and self.delegate_override
            else comfy_orig_attn
        )
        result = None
        rules = self.rules
        call_idx = currattncall.idx
        rev_idx = (
            -abs(currattncall.max_idx - currattncall.idx)
            if currattncall.max_idx >= 0
            else math.nan
        )
        ci = self.call_indexes
        fallthrough_match = math.inf in ci
        exclude = (call_idx + 0.5) in ci or (
            not math.isnan(rev_idx) and (rev_idx + 0.5) in ci
        )
        matched = not exclude and (fallthrough_match or call_idx in ci or rev_idx in ci)
        if self.verbose:
            tqdm.write(
                f"[BLEH] AdvancedAttn wrapper({currattncall.idx:<3}): sigma={sigma:.4f}, max_idx={currattncall.max_idx:<3}, rev_idx={rev_idx:<3}, matched={matched}, exclude={exclude}, q.shape={q.shape}",
            )
        currattncall.idx += 1
        rules = self.rules if matched else ()
        for rule in rules:
            if not rule.end_sigma <= sigma <= rule.start_sigma:
                continue
            currq, currk, currv = q, k, v
            if rule.q_multiplier != 1:
                currq = currq * rule.q_multiplier
            if rule.k_multiplier != 1:
                currk = currk * rule.k_multiplier
            if rule.v_multiplier != 1:
                currv = currv * rule.v_multiplier
            if rule.dtype is not None or rule.device is not None:
                currq = currq.to(device=rule.device, dtype=rule.dtype)
                currk = currk.to(device=rule.device, dtype=rule.dtype)
                currv = currv.to(device=rule.device, dtype=rule.dtype)
            currq = self.call_op(rule.op_q, currq, sigma=sigma)
            currk = self.call_op(rule.op_k, currk, sigma=sigma)
            currv = self.call_op(rule.op_v, currv, sigma=sigma)
            attn_function = (
                partial(rule.attn_function, fallback_attn)
                if rule.attn_function
                else fallback_attn
            )
            curr_result = attn_function(currq, currk, currv, *args, **kwargs)
            del currq, currk, currv
            if rule.check_nan and curr_result.isnan().any():
                del curr_result
                continue
            if rule.output_multiplier != 1:
                curr_result *= rule.output_multiplier
            curr_result = self.call_op(
                rule.op_current_result_preblend,
                curr_result,
                sigma=sigma,
            )
            if curr_result.dtype != q.dtype or curr_result.device != q.device:
                curr_result = curr_result.to(q)
            if result is not None:
                result = self.call_op(rule.op_result_preblend, result, sigma=sigma)
            if result is None or rule.blend_function is None:
                result = curr_result
                del curr_result
                continue
            prev_result = result
            result = rule.blend_function(result, curr_result, rule.blend)
            if rule.op_result_postblend_diff is not None:
                result = prev_result + self.call_op(
                    rule.op_result_postblend_diff,
                    result - prev_result,
                    sigma=sigma,
                )
            del curr_result, prev_result
            result = self.call_op(rule.op_result_postblend, result, sigma=sigma)
        if result is None:
            result = fallback_attn(q, k, v, *args, **kwargs)
        return self.call_op(self.op_result, result, sigma=sigma)


class CurrAttnCall:
    def __init__(self, idx: int = 0, max_idx: int = -1):
        self.idx = idx
        self.max_idx = max_idx


def advancedattn_sampler(
    config: AdvancedAttnConfig,
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    **kwargs: dict,
) -> torch.Tensor:
    if config.time_mode == TimeMode.PERCENT:
        ms = model.inner_model.inner_model.model_sampling
        start_sigma, end_sigma = (
            round(ms.percent_to_sigma(config.start_time), 4),
            round(ms.percent_to_sigma(config.end_time), 4),
        )
        del ms
    else:
        start_sigma = config.start_time
        end_sigma = config.end_time

    max_idx = -1

    def model_call(
        model: object,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        nonlocal max_idx
        sigma_float = float(sigma.max().detach().cpu())
        enabled = end_sigma <= sigma_float <= start_sigma
        if not enabled:
            return model(x, sigma, **kwargs)
        calltracker = CurrAttnCall(idx=0, max_idx=max_idx)
        if config.verbose:
            tqdm.write(f"[BLEH] AdvancedAttn: Config: {config}")
        model_options = kwargs.pop("model_options", {}).copy()
        transformer_options = model_options.pop("transformer_options", {}).copy()
        old_override = transformer_options.pop("optimized_attention_override", None)
        attn_override = partial(
            config.attn_wrapper,
            sigma_float,
            calltracker,
            old_override,
        )
        transformer_options["optimized_attention_override"] = attn_override
        model_options["transformer_options"] = transformer_options
        kwargs["model_options"] = model_options
        result = model(x, sigma, **kwargs)
        max_idx = max(max_idx, calltracker.idx)
        return result

    return config.sampler.sampler_function(
        BlehModelWrapper(model, model_call),
        x,
        sigmas,
        **kwargs,
        **config.sampler.extra_options,
    )


class BlehAdvancedAttentionSampler:
    DESCRIPTION = "TBD"
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    DEFAULT_YAML_PARAMS = """verbose: false
start_time: 0.0
end_time: 1.0
# One of: percent, sigma
time_mode: percent
# .inf means match everything. Whole float values exclude an index. I.E 2.0
# Call index as in the Nth call to attention this model evaluation.
# Negative indexes count from the end but can only match after a pass through the model.
call_indexes: [.inf]
# Can be set to null (everything), cond, uncond or a list.
batch_slice: null
# Requires cond batch information to be passed and at least this many items.
min_cond_batch: 0
rules:
    # Passed as sageattn_function unless set to default.
    # Keys not in this list are passed through like with the SageAttention node:
    #   attn_function, blend_mode, blend
  - attn_function: default
    # You can set whatever other keys you want here.
  - attn_function: sageattn
    # Blends target the last attention result and are ignored
    # if it's missing.
    # The default blend means:
    #   sageattn + (defaultattn - sageattn) * 2
    blend_mode: cfg
    blend: 2.0
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "yaml_parameters": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_YAML_PARAMS,
                        "tooltip": "Allows specifying custom parameters via YAML. These are mostly passed directly to the SageAttention function with no error checking. Must be empty or a YAML object.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
            "optional": {
                "op_0": ("LATENT_OPERATION",),
                "op_1": ("LATENT_OPERATION",),
                "op_2": ("LATENT_OPERATION",),
                "op_3": ("LATENT_OPERATION",),
                "op_4": ("LATENT_OPERATION",),
                "op_5": ("LATENT_OPERATION",),
                "op_6": ("LATENT_OPERATION",),
                "op_7": ("LATENT_OPERATION",),
                "op_8": ("LATENT_OPERATION",),
                "op_9": ("LATENT_OPERATION",),
            },
        }

    @classmethod
    def go(
        cls,
        sampler: object,
        yaml_parameters: str,
        **kwargs: dict,
    ) -> tuple:
        if sageattention is None:
            raise RuntimeError(
                "sageattention not installed to Python environment: SageAttention feature unavailable",
            )
        if not HAVE_ATTN_OVERRIDE:
            raise RuntimeError(
                "This node only supports recent ComfyUI versions that support attention overrides.",
            )
        params = get_yaml_parameters(yaml_parameters)
        params["latent_ops"] = {
            k: v for k, v in kwargs.items() if k.startswith("op_") and v is not None
        }
        return (
            KSAMPLER(
                update_wrapper(
                    partial(
                        advancedattn_sampler,
                        AdvancedAttnConfig.build(sampler=sampler, **params),
                    ),
                    sampler.sampler_function,
                ),
            ),
        )
