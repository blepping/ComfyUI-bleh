# ruff: noqa: TID252
from __future__ import annotations

import contextlib
import copy
import math
import operator
import random
from decimal import Decimal
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple, Sequence

import torch
from comfy import model_management
from comfy.model_management import throw_exception_if_processing_interrupted

from .. import latent_utils as lutils
from ..better_previews.previewer import PREVIEWER_STATE, ensure_previewer

if TYPE_CHECKING:
    from collections.abc import Callable


class DiscardPenultimateSigma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "sigmas": ("SIGMAS", {"forceInput": True}),
            },
        }

    FUNCTION = "go"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    DESCRIPTION = "Discards the next to last sigma in the list."

    @classmethod
    def go(cls, enabled: bool, sigmas: torch.Tensor) -> tuple[torch.Tensor]:
        if not enabled or len(sigmas) < 2:
            return (sigmas,)
        return (torch.cat((sigmas[:-2], sigmas[-1:])),)


class SeededDisableNoise:
    def __init__(self, seed: int, seed_offset: int = 1):
        self.seed = seed
        self.seed_offset = seed_offset

    def generate_noise(self, latent):
        samples = latent["samples"]
        torch.manual_seed(self.seed)
        random.seed(self.seed)  # For good measure.
        shape = samples.shape
        device = model_management.get_torch_device()
        for _ in range(self.seed_offset):
            _ = random.random()  # noqa: S311
            _ = torch.randn(
                *shape,
                dtype=samples.dtype,
                layout=samples.layout,
                device=device,
            )
        return torch.zeros(
            samples.shape,
            dtype=samples.dtype,
            layout=samples.layout,
            device="cpu",
        )


class BlehDisableNoise:
    DESCRIPTION = "Allows setting a seed even when disabling noise. Used for SamplerCustomAdvanced or other nodes that take a NOISE input."
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/noise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "seed_offset": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 200,
                        "tooltip": "Advances the RNG this many times to avoid the mistake of using the same noise for sampling as the initial noise. I recommend leaving this at 1 (or higher) but you can set it to 0. to disable",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        noise_seed: int,
        seed_offset: int | None = 1,
    ) -> tuple[SeededDisableNoise]:
        return (
            SeededDisableNoise(
                noise_seed,
                seed_offset if seed_offset is not None else 0,
            ),
        )


class Wildcard(str):
    __slots__ = ()

    def __ne__(self, _unused):
        return False


class BlehPlug:
    DESCRIPTION = "This node can be used to plug up an input but act like the input was not actually connected. Can be used to prevent something like Use Everywhere nodes from supplying an input without having to set up blacklists or other configuration."
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "hacks"

    WILDCARD = Wildcard("*")
    RETURN_TYPES = (WILDCARD,)

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def go(cls):
        return (None,)


class BlehCast:
    DESCRIPTION = "UNSAFE: This node allows casting its input to any type. NOTE: This does not actually change the data in any way, it just allows you to connect its output to any input. Only use if you know for sure the data is compatible."
    FUNCTION = "go"
    CATEGORY = "hacks"

    WILDCARD = Wildcard("*")
    RETURN_TYPES = (WILDCARD,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (
                    cls.WILDCARD,
                    {
                        "forceInput": True,
                        "description": "You can connect any type of input here, but take to ensure that you connect the output from this node to an input that is compatible.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, any_input):
        return (any_input,)


class BlehSetSigmas:
    DESCRIPTION = "Advanced node that allows manipulating SIGMAS. For example, you can manually enter a list of sigmas, insert some new sigmas into existing SIGMAS, etc."
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/sigmas"
    RETURN_TYPES = ("SIGMAS",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "Start index for modifying sigmas, zero-based. May be set to a negative value to index from the end, i.e. -1 is the last item, -2 is the penultimate item.",
                    },
                ),
                "mode": (
                    ("replace", "insert", "multiply", "add", "subtract", "divide"),
                    {
                        "default": "replace",
                        "tooltip": "",
                    },
                ),
                "order": (
                    ("AB", "BA"),
                    {
                        "default": "AB",
                        "tooltip": "Only applies to add, subtract, multiply and divide operations. Controls the order of operations. For example if order AB then add means A*B, if order BA then add means B*A.",
                    },
                ),
                "commasep_sigmas_b": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Exclusive with sigmas_b. Enter a comma-separated list of sigma values here. For non-insert mode, the input sigmas will be padded with zeros if necessary. Example: start_index=2 (3rd item), mode=replace, input sigmas 4,3,2,1 and you used replace mode with 0.3,0.2,0.1 the output would be 4,3,0.3,0.2,0.1",
                    },
                ),
            },
            "optional": {
                "sigmas_a": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": "Optional input as long as commasep_sigmas is not also empty. If not supplied, an initial sigmas list of the appropriate size will be generated filled with zeros.",
                    },
                ),
                "sigmas_b": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": "Optionally populate this or commasep_sigmas_b but not both.",
                    },
                ),
            },
        }

    OP_MAP = {  # noqa: RUF012
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    @classmethod
    def go(
        cls,
        *,
        start_index: int,
        mode: str,
        order: str,
        commasep_sigmas_b: str,
        sigmas_a: torch.Tensor | None = None,
        sigmas_b: torch.Tensor | None = None,
    ) -> tuple:
        new_sigmas_list = tuple(
            Decimal(val) for val in commasep_sigmas_b.strip().split(",") if val.strip()
        )
        if new_sigmas_list and sigmas_b is not None:
            raise ValueError(
                "Must populate one of sigmas_b or commasep_sigmas_b but not both.",
            )
        if sigmas_b is not None:
            sigmas_b = sigmas_b.to(dtype=torch.float64, device="cpu", copy=True)
        else:
            sigmas_b = torch.tensor(new_sigmas_list, device="cpu", dtype=torch.float64)
        newlen = sigmas_b.numel()
        if sigmas_a is None or sigmas_a.numel() == 0:
            sigmas_a = None
            if not newlen:
                raise ValueError(
                    "sigmas_a, commasep_sigmas_b and sigmas_b can't all be empty.",
                )
            if start_index < 0:
                raise ValueError(
                    "Negative start_index doesn't make sense when input sigmas are empty.",
                )
        if newlen == 0:
            return (sigmas_a.to(dtype=torch.float, copy=True),)
        oldlen = 0 if sigmas_a is None else sigmas_a.numel()
        if start_index < 0:
            start_index = oldlen + start_index
            if start_index < 0:
                raise ValueError(
                    "Negative start index points past the beginning of sigmas_a",
                )
        past_end = 0 if start_index < oldlen else start_index + 1 - oldlen
        if past_end and mode == "insert":
            mode = "replace"
        if past_end:
            outlen = oldlen + newlen + past_end - 1
        elif mode == "insert":
            outlen = oldlen + newlen
        else:
            outlen = oldlen + max(0, newlen - (oldlen - start_index))
        sigmas_out = torch.zeros(outlen, device="cpu", dtype=torch.float64)
        if mode == "insert":
            sigmas_out[:start_index] = sigmas_a[:start_index]
            sigmas_out[start_index : start_index + newlen] = sigmas_b
            sigmas_out[start_index + newlen :] = sigmas_a[start_index:]
        else:
            if oldlen:
                sigmas_out[:oldlen] = sigmas_a
            if mode == "replace":
                sigmas_out[start_index : start_index + newlen] = sigmas_b
            else:
                opfun = cls.OP_MAP.get(mode)
                if opfun is None:
                    raise ValueError("Bad mode")
                arga = sigmas_out[start_index : start_index + newlen]
                if order == "BA":
                    arga, argb = sigmas_b, arga
                else:
                    argb = sigmas_b
                sigmas_out[start_index : start_index + newlen] = opfun(arga, argb)
        return (sigmas_out.to(torch.float),)


class BlehEnsurePreviewer:
    DESCRIPTION = "This node ensures Bleh is used for previews. Can be used if other custom nodes overwrite the Bleh previewer. It will pass through any value unchanged."
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "hacks"

    WILDCARD = Wildcard("*")
    RETURN_TYPES = (WILDCARD,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (
                    cls.WILDCARD,
                    {
                        "forceInput": True,
                        "description": "You can connect any type of input here, but take to ensure that you connect the output from this node to an input that is compatible.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, any_input):
        ensure_previewer()
        return (any_input,)


class BlehImageAsLatent:
    DESCRIPTION = "This node allows you to rearrange an IMAGE to look like a LATENT. Can be useful if you want to apply some latent operations to an IMAGE. Can be reversed with the BlehLatentAsImage node."
    FUNCTION = "go"
    CATEGORY = "latent/advanced"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "rescale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, will rescale the image values which usually are from 0 to 1 to -1 to 1.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, image: torch.Tensor, rescale: bool) -> tuple:
        image = image.to(device="cpu", dtype=torch.float32, copy=True)
        if image.ndim == 3:
            image = image[None]
        elif image.ndim != 4:
            raise ValueError("Unexpected number of dimensions in image")
        image = image.movedim(-1, 1)
        if rescale:
            image = image.sub_(0.5).mul_(2.0)
        return ({"samples": image},)


class BlehLatentAsImage:
    DESCRIPTION = "This node lets you rearrange a LATENT to look like an IMAGE. Note: It does not respect anything like masks or latent selection metadata (from nodes like LatentFromBatch) that might exist."
    FUNCTION = "go"
    CATEGORY = "latent/advanced"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent": ("LATENT",),
                "values_mode": (
                    ("rescale", "rescale_perchannel", "clamp"),
                    {"default": "rescale"},
                ),
                "channels_into_batch": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, will create a greyscale image for each latent channel.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        latent: dict,
        values_mode: str,
        channels_into_batch: bool,
    ) -> tuple:
        samples = latent["samples"]
        if samples.ndim != 4:
            raise ValueError("Expected a 4D latent but didn't get one")
        if channels_into_batch:
            samples = samples.reshape(-1, *samples.shape[2:]).unsqueeze(1)
            samples = samples.expand(samples.shape[0], 3, *samples.shape[2:])
        image = samples.movedim(1, -1).to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )[..., :4]
        if values_mode == "clamp":
            return (image.clamp(0.0, 1.0),)
        image = lutils.normalize_to_scale(
            image,
            0.0,
            1.0,
            dim=(2, 3) if values_mode == "rescale_perchannel" else (1, 2, 3),
        )
        return (image,)


class BlehModelPatchFastTerminate:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "hacks"
    DESCRIPTION = "Patches a model to check if processing is interrupted at the start of every block. Makes interrupting generations more responsive on supported models (mainly useful for video models that might take 40+ second for a step). Should support most existing models."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    @classmethod
    def go(cls, model):
        m = model.clone()

        def wrap_transformer_forward(orig_forward, *args: list, **kwargs: dict):
            throw_exception_if_processing_interrupted()
            return orig_forward(*args, **kwargs)

        found = 0
        for bt in (
            "blocks",
            "single_blocks",
            "double_blocks",
            "transformer_blocks",
            "vace_blocks",
            "double_stream_blocks",
            "single_stream_blocks",
        ):
            bn = 0
            while True:
                k = f"diffusion_model.{bt}.{bn}"
                try:
                    block = model.get_model_object(k)
                except AttributeError:
                    block = None
                if block is None:
                    k = f"diffusion_model.{bt}.block{bn}"
                    with contextlib.suppress(AttributeError):
                        block = model.get_model_object(k)
                bn += 1
                if block is None:
                    break
                orig_forward = getattr(block, "forward", None)
                if orig_forward is None:
                    continue
                m.add_object_patch(
                    f"{k}.forward",
                    partial(wrap_transformer_forward, orig_forward),
                )
                found += 1

        if found > 0:
            # Appears to be a transformer-based model so we're done.
            return (m,)

        # Fallthough to handling normal SD models.
        def input_block_patch(h, _transformer_options):
            throw_exception_if_processing_interrupted()
            return h

        def output_block_patch(h, hsp, _transformer_options):
            throw_exception_if_processing_interrupted()
            return h, hsp

        m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)

        return (m,)


class BlehModelProcessLatentIn:
    DESCRIPTION = "Advanced node that can be used to scale a raw latent for model input. Generally only needed if you're doing something that bypasses the normal latent input mechanisms."
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = "latent/advanced"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
            },
        }

    @classmethod
    def go(cls, *, model, latent: dict) -> tuple[dict]:
        latent_format = model.model.latent_format
        samples = (
            latent["samples"].detach().to(device="cpu", dtype=torch.float32, copy=True)
        )
        return (latent | {"samples": latent_format.process_in(samples)},)


class BlehModelProcessLatentOut:
    DESCRIPTION = "Advanced node that can be used to scale a latent to the correct range for output. Generally only needed if you're doing something that bypasses the normal latent output mechanisms."
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = "latent/advanced"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
            },
        }

    @classmethod
    def go(cls, *, model, latent: dict) -> tuple[dict]:
        latent_format = model.model.latent_format
        samples = (
            latent["samples"].detach().to(device="cpu", dtype=torch.float32, copy=True)
        )
        return (latent | {"samples": latent_format.process_out(samples)},)


class PreviewFixGuider:
    def __init__(self, guider, **kwargs: Any):
        self.__guider = guider
        self.__state_overrides = {k: v for k, v in kwargs.items() if v is not None}

    def __getattr__(self, k):
        return getattr(self.__guider, k)

    def sample(self, noise, latent_image, *args: Any, **kwargs: Any):
        latent_shapes = (
            (tuple(latent_image.shape),)
            if not latent_image.is_nested
            else tuple(tuple(t.shape) for t in latent_image.unbind())
        )
        PREVIEWER_STATE.last_latent_shapes = latent_shapes
        soverrides = self.__state_overrides
        saved_state = {}
        if soverrides:
            saved_state |= {k: getattr(PREVIEWER_STATE, k, None) for k in soverrides}
            for k, v in soverrides.items():
                setattr(PREVIEWER_STATE, k, v)
        try:
            return self.__guider.sample(noise, latent_image, *args, **kwargs)
        finally:
            PREVIEWER_STATE.last_latent_shapes = None
            if saved_state:
                for k, v in saved_state.items():
                    setattr(PREVIEWER_STATE, k, v)


class BlehFixGuiderPreviewing:
    DESCRIPTION = "Wraps a guider to give the Bleh previewing system a hint about the latent shapes. Only necessary for models like LTX-2 which use nested tensors."
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "hacks"

    RETURN_TYPES = ("GUIDER",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "fps_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 9999.0,
                        "tooltip": "Can be used to override the FPS when previewing with video models. Disabled if set to 0.",
                    },
                ),
            },
            "optional": {
                "prefer_previewer": (
                    ("default", "ltxav", "ltxav23", "ltxav23wide"),
                    {
                        "default": "default",
                        "tooltip": "This is mostly only useful for LTX 2.3 since there isn't a way for the internal logic to know what latent format is being used. Set this to ltxav23 for LTX 2.3 (ltxav23wide for the wide previewer model), otherwise leave on the default. Note: This option sets global state.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        guider,
        fps_override: float | None = None,
        prefer_previewer: str | None = None,
    ) -> tuple:
        if prefer_previewer:
            PREVIEWER_STATE.prefer_previewer = prefer_previewer
        return (
            PreviewFixGuider(
                guider,
                fps_override=fps_override if fps_override != 0.0 else None,
            ),
        )


class ConditioningBlender:
    """Base class for complex chronological conditioning blending."""

    def __call__(self, *args: Any, **kwargs: Any):
        return self.blend(*args, **kwargs)

    @staticmethod
    def get_bounds(meta_dict: dict) -> tuple[float, float]:
        start = meta_dict.get("start_percent", 0.0)
        end = meta_dict.get("end_percent", 1.0)
        if start >= end:
            end = math.nextafter(start, 2.0)
        return start, end

    def get_boundaries(self, cond1: list, cond2: list) -> list[float]:
        boundaries = {0.0, 1.0}
        for item in cond1 + cond2:
            start, end = self.get_bounds(item[1])
            boundaries.add(start)
            boundaries.add(end)
        return sorted(boundaries)

    def get_active_items(self, cond_list: list, t_start: float, t_end: float) -> list:
        active = []
        for item in cond_list:
            c_start, c_end = self.get_bounds(item[1])
            if c_start <= t_start and c_end >= t_end:
                active.append(item)
        return active

    def simple_blend_wrapper(
        self,
        blend_function: Callable,
        cond1: list,
        cond2: list,
        strength: float,
        *,
        new_start_percent: float,
        new_end_percent: float,
        **kwargs: Any,
    ) -> list:
        blended_tensor = blend_function(cond1[0], cond2[0], strength, **kwargs)
        new_meta = copy.deepcopy(cond1[1])
        new_meta |= {
            "start_percent": new_start_percent,
            "end_percent": new_end_percent,
        }
        return [blended_tensor, new_meta]

    def blend(
        self,
        cond1: list,
        cond2: list,
        blend_func: Callable,
        *,
        strength: float = 0.5,
        blend_full_items: bool = False,
        **kwargs: Any,
    ) -> list:
        boundaries = self.get_boundaries(cond1, cond2)
        if not blend_full_items:
            blend_func = partial(self.simple_blend_wrapper, blend_func)

        result_conditioning = []

        for t_start, t_end in pairwise(boundaries):
            if t_start >= t_end:
                continue

            active1 = self.get_active_items(cond1, t_start, t_end)
            active2 = self.get_active_items(cond2, t_start, t_end)
            first_active_cond = active1 or active2

            if not first_active_cond:
                continue

            adjusted_end = math.nextafter(t_end, -1.0) if t_end < 1.0 else t_end

            if not (active1 and active2):
                for c in first_active_cond:
                    new_meta = copy.deepcopy(c[1])
                    new_meta |= {"start_percent": t_start, "end_percent": adjusted_end}
                    result_conditioning.append([c[0].clone(), new_meta])
                continue

            for c1 in active1:
                for c2 in active2:
                    blend_result = blend_func(
                        c1,
                        c2,
                        strength,
                        new_start_percent=t_start,
                        new_end_percent=adjusted_end,
                        **kwargs,
                    )
                    result_conditioning.append(blend_result)

        return result_conditioning


class BlehConditioningBlender(ConditioningBlender):
    """Extended Blender capable of N-dimensional dynamic slicing, padding, and alignment."""

    def __init__(
        self,
        *,
        size_mismatch_strategy: str = "zero",
        size_mismatch_alignment_mode: str = "left",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.strategy = size_mismatch_strategy
        self.alignment = size_mismatch_alignment_mode

    def get_slices(
        self,
        shape: torch.Size | tuple[int, ...],
        target_shape: torch.Size | tuple[int, ...],
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """Generates N-dimensional slice objects for both source and target."""
        src_slices, tgt_slices = [], []
        for sz, tgt_sz in zip(shape, target_shape, strict=True):
            # Slice start, slice end, target start, target end
            ss = se = ts = te = None
            if sz < tgt_sz:  # Padding source into target
                if self.alignment == "left":
                    ts = 0
                elif self.alignment == "right":
                    ts = tgt_sz - sz
                else:  # center
                    ts = (tgt_sz - sz) // 2
                te = ts + sz
            elif sz > tgt_sz:  # Cropping source down to target
                if self.alignment == "left":
                    ss = 0
                elif self.alignment == "right":
                    ss = sz - tgt_sz
                else:  # center
                    ss = (sz - tgt_sz) // 2
                se = ss + tgt_sz
            src_slices.append(slice(ss, se))
            tgt_slices.append(slice(ts, te))
        return tuple(src_slices), tuple(tgt_slices)

    def process_tensor(
        self,
        t: torch.Tensor,
        other_t: torch.Tensor,
        *,
        orig_t1: torch.Tensor,
        orig_t2: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if t.shape == target_shape:
            return t

        if self.strategy == "replicate":
            repeats = []
            for sz, tgt_sz in zip(t.shape, target_shape, strict=True):
                if tgt_sz % sz != 0:
                    errstr = f"Cannot replicate conditioning tensor: target size {tgt_sz} is not evenly divisible by source size {sz}."
                    raise ValueError(errstr)
                repeats.append(tgt_sz // sz)
            return t.repeat(*repeats)

        src_slices, tgt_slices = self.get_slices(t.shape, target_shape)

        if self.strategy == "smaller":
            return t[src_slices]

        # --- Padding Strategies ---
        if self.strategy == "mean_cond_1":
            # Always uses orig_t1, regardless of which tensor is currently 't'
            dims = list(range(1, orig_t1.ndim)) if orig_t1.ndim > 1 else [0]
            mean_val = orig_t1.mean(dim=dims, keepdim=True)
            out = mean_val.expand(target_shape).clone()
        elif self.strategy == "mean_cond_2":
            # Always uses orig_t2, regardless of which tensor is currently 't'
            dims = list(range(1, orig_t2.ndim)) if orig_t2.ndim > 1 else [0]
            mean_val = orig_t2.mean(dim=dims, keepdim=True)
            out = mean_val.expand(target_shape).clone()
        elif self.strategy in {"larger", "match_cond_1", "match_cond_2"}:
            out = other_t.expand(target_shape).clone()
        else:  # "zero"
            out = t.new_zeros(target_shape)

        out[tgt_slices] = t[src_slices]
        return out

    def align_tensors(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aligns two arbitrary N-dimensional tensors based on the instance's strategy and alignment modes."""
        if t1.shape == t2.shape:
            return t1, t2

        if t1.ndim != t2.ndim:
            errstr = f"Can't align tensors with different numbers of dimensions. t1 shape: {t1.shape} ({t1.ndim}), t2 shape: {t2.shape} ({t2.ndim})"
            raise ValueError(errstr)

        if self.strategy == "error":
            errstr = (
                f"Size mismatch: {t1.shape} vs {t2.shape} (Strategy set to 'error')"
            )
            raise ValueError(errstr)

        if self.strategy == "match_cond_1":
            target_shape = t1.shape
        elif self.strategy == "match_cond_2":
            target_shape = t2.shape
        else:
            size_op: Callable = min if self.strategy == "smaller" else max
            target_shape = tuple(
                size_op(s1, s2) for s1, s2 in zip(t1.shape, t2.shape, strict=True)
            )

        # Pass t1 and t2 as the strict originals to preserve the mean math!
        out1 = self.process_tensor(
            t=t1,
            other_t=t2,
            orig_t1=t1,
            orig_t2=t2,
            target_shape=target_shape,
        )
        out2 = self.process_tensor(
            t=t2,
            other_t=t1,
            orig_t1=t1,
            orig_t2=t2,
            target_shape=target_shape,
        )
        return out1, out2


class BlehBlendConditioning:
    DESCRIPTION = "Blends conditioning_1 with conditioning_2. Unlike the ComfyUI builtin node, this can handle multiple conditioning items on both side and respects time ranges but not other potential ranges (I.E. regional conditioning). Note however that conditionings are generally encoded as sequences of tokens, so something like 'a cute dog' blended with 'sketch of a cat' is likely to be doing something like blending 'a' with 'sketch', 'cute' with 'of', etc."
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "blend_mode": (
                    tuple(lutils.BLENDING_MODES.keys()),
                    {"default": "lerp"},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "min": -99999.0,
                        "max": 999999.0,
                        "default": 0.5,
                    },
                ),
                "blend_cond_tensor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether the base conditioning tensor is blended. When disabled, this will use the conditioning tensor from conditioning_1.",
                    },
                ),
                "blend_tensors": (
                    "STRING",
                    {
                        "default": "pooled_output",
                        "tooltip": "Comma-separated list of other tensors to blend if they exist (for example pooled_output)",
                    },
                ),
                "metadata_base_mode": (
                    ("cond_1", "cond_2", "prefer_cond_1", "prefer_cond_2", "empty"),
                    {
                        "default": "prefer_cond_1",
                        "tooltip": "Controls what metadata ends up in the blended conditioning item. Items in the blend_tensors list will always be included. Possible values:\ncond_1 - Uses conditioning_1\ncond_2 - See above\nprefer_cond_1 - Uses the key from conditioning_1 if it exists, otherwise conditioning_2\nprefer_cond_2 - See above.\nempty - Starts with empty metadata.",
                    },
                ),
                "size_mismatch_strategy": (
                    (
                        "zero",
                        "mean_cond_1",
                        "mean_cond_2",
                        "match_cond_1",
                        "match_cond_2",
                        "replicate",
                        "larger",
                        "smaller",
                        "error",
                    ),
                    {
                        "default": "zero",
                        "tooltip": "Handles the case of size mismatches between items to be blended.\nzero - Uses the larger size, fills extra elements with zero (how ComfyUI usually handles it)\nmean_cond_1 - Same as above, uses the mean from conditioning_1.\nmean_cond_2 - See above.\nmatch_cond_1 - Matches the size to conditioning_1. Similar to choosing larger/smaller mode per-dimension (so alignment may apply).\nmatch_cond_2 - Same as above aside from using conditioning_2's sizes.\nreplicate - Replicates the smaller size to match the larger. It is an error if the sizes aren't evenly divisible. May be use for conditioning types like CLIP where mismatches will be increments of the CLIP max tokens size (77).\nlarger - Uses the values from the item with the larger size.\nsmaller - Prunes the tensor to the smaller size.\nerror - Size mismatches are an error.",
                    },
                ),
                "size_mismatch_alignment_mode": (
                    ("left", "right", "center"),
                    {
                        "default": "left",
                        "tooltip": "Only applies to size mismatch strategies that use the larger size and pad.\nleft - Aligns existing values to lower indexes, padding will apply after them. This is ComfyUI's normal behavior, the other options are probably quite weird.\nright - Padding will apply to lower indexes.\ncenter - Tries to center (left-biased when the size isn't divisible by 2) populated values in the space.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        conditioning_1: list,
        conditioning_2: list,
        blend_mode: str,
        strength: float,
        blend_cond_tensor: bool,
        blend_tensors: str,
        metadata_base_mode: str,
        size_mismatch_strategy: str,
        size_mismatch_alignment_mode: str,
    ) -> tuple:
        blend_tensor_list = (s.strip() for s in blend_tensors.split(","))
        blend_tensor_set = {s for s in blend_tensor_list if s}
        blend_function = lutils.BLENDING_MODES[blend_mode]
        mdbm = metadata_base_mode

        # Initialize our stateful blender
        blender = BlehConditioningBlender(
            size_mismatch_strategy=size_mismatch_strategy,
            size_mismatch_alignment_mode=size_mismatch_alignment_mode,
        )

        def blend_wrapper(
            cond1: list,
            cond2: list,
            strength: float,
            *,
            new_start_percent: float,
            new_end_percent: float,
            **kwargs: Any,
        ) -> list:
            c1t, c1d = cond1[:2]
            c2t, c2d = cond2[:2]
            cond1_extra = cond1[2:]

            if blend_cond_tensor:
                c2t = c2t.to(c1t)
                c1t, c2t = blender.align_tensors(c1t, c2t)

            def is_blendable(t: torch.Tensor):
                return isinstance(t, torch.Tensor) and t.dtype.is_floating_point

            c1_tensor_keys = {
                k
                for k, v in c1d.items()
                if k in blend_tensor_set and isinstance(v, torch.Tensor)
            }
            c2_tensor_keys = {
                k
                for k, v in c2d.items()
                if k in blend_tensor_set and isinstance(v, torch.Tensor)
            }
            blendable_tensor_keys = c1_tensor_keys.intersection(c2_tensor_keys)

            if mdbm == "empty":
                md = {}
            elif mdbm == "cond_1":
                md = c1d
            elif mdbm == "cond_2":
                md = c2d
            elif mdbm == "prefer_cond_1":
                md = c2d | c1d
            elif mdbm == "prefer_cond_2":
                md = c1d | c2d
            else:
                raise ValueError("Bad metadata_base_mode")

            if not blend_cond_tensor:
                ct = c1t.clone()
            else:
                ct = blend_function(c1t, c2t, strength, **kwargs)

            if ct is c1t or ct is c2t:
                ct = ct.clone()

            bmd = {}
            for bk in blendable_tensor_keys:
                mt1, mt2 = c1d[bk], c2d[bk]
                if mt1.ndim != mt2.ndim:
                    continue
                mt2 = mt2.to(mt1)

                # Apply the exact same logic to pooled_output / metadata tensors
                mt1, mt2 = blender.align_tensors(mt1, mt2)

                bmresult = blend_function(mt1, mt2, strength, **kwargs)
                if bmresult is mt1 or bmresult is mt2:
                    bmresult = bmresult.clone()
                bmd[bk] = bmresult

            if bmd:
                md = {k: v for k, v in md.items() if k not in bmd}

            md = copy.deepcopy(md)
            md |= {
                "start_percent": new_start_percent,
                "end_percent": new_end_percent,
                **bmd,
            }
            if cond1_extra:
                cond1_extra = copy.deepcopy(cond1_extra)
            return [ct, md, *cond1_extra]

        result = blender.blend(
            conditioning_1,
            conditioning_2,
            blend_wrapper,
            strength=strength,
            blend_full_items=True,
        )
        return (result,)


class ContrastiveOrthoCFG(NamedTuple):
    start_sigma: float = 99999.0
    end_sigma: float = 0.0
    positive_scale: float = 0.2
    negative_scale: float = 1.0
    start_dim: int = 1
    end_dim: int = 1
    use_noise: bool = False
    base_denoised: bool = False

    def extract_common(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return lutils.contrastive_ortho_cfg_base_a(
            cond,
            uncond,
            1.0,
            a_ortho_scale=-1.0,
            b_ortho_scale=1.0,
            start_dim=self.start_dim,
            end_dim=self.end_dim,
        )

    def extract_parts(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common = self.extract_common(cond, uncond)
        unique_cond = cond - common
        unique_uncond = uncond - common
        if self.negative_scale != 1:
            unique_uncond *= self.negative_scale
        if self.positive_scale != 1:
            unique_cond *= self.positive_scale
        return common, unique_cond, unique_uncond

    def check_sigma(self, sigma: torch.Tensor) -> bool:
        sigma_f = sigma.mean().detach().cpu().item()
        return self.end_sigma <= sigma_f <= self.start_sigma

    def pad_sigma(self, sigma: torch.Tensor, ndim: int) -> torch.Tensor:
        return (
            sigma
            if sigma.ndim == 0
            else sigma.reshape(sigma.shape[0], *((1,) * (ndim - 1)))
        )

    def precfg_patch(self, args: dict[str, Any]) -> Sequence[torch.Tensor]:
        x = args["input"]
        sigma = args["sigma"]
        conds_out = args["conds_out"]
        if len(conds_out) < 2 or conds_out[1] is None or not self.check_sigma(sigma):
            return conds_out
        sigma = self.pad_sigma(sigma, x.ndim)
        cond, uncond = conds_out[:2]
        if self.use_noise:
            cond = (x - cond).div_(sigma)
            uncond = (x - uncond).div_(sigma)
        common, cond_new, uncond_new = self.extract_parts(cond, uncond)
        cond_new += common
        uncond_new += common
        if self.use_noise:
            cond_new = cond_new.mul_(-sigma).add_(x)
            uncond_new = uncond_new.mul_(-sigma).add_(x)
        return conds_out.__class__((cond_new, uncond_new, *conds_out[2:]))

    def postcfg_patch(self, args: dict[str, Any]) -> torch.Tensor:
        x = args["input"]
        cond = args["denoised"] if self.base_denoised else args["cond_denoised"]
        uncond = args.get("uncond_denoised")
        sigma = args["sigma"]
        denoised = args["denoised"]
        if not self.check_sigma(sigma) or uncond is cond or uncond is None:
            return denoised
        sigma = self.pad_sigma(sigma, x.ndim)
        if self.use_noise:
            cond, uncond, denoised = (
                (x - t).div_(sigma) for t in (cond, uncond, denoised)
            )
        common, cond_unique, uncond_unique = self.extract_parts(cond, uncond)
        denoised = cond_unique.sub_(uncond_unique).add_(
            common if self.base_denoised else denoised,
        )
        return denoised.mul_(-sigma).add_(x) if self.use_noise else denoised


class BlehContrastiveOrthoCFG:
    DESCRIPTION = "Contrastive orthogonal CFG. CFG variant that allows you to control the scale of negative and positive features individually. For the most predictable effects, use positive scales and pre_cfg mode with use_noise disabled. This is just a slower version of CFG if you use the default parameters and positive/negative scales of 1.0."
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "advanced/guidance"

    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        dc = ContrastiveOrthoCFG()
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "start_sigma": (
                    "FLOAT",
                    {
                        "default": dc.start_sigma,
                        "max": 99999.0,
                        "min": 0.0,
                    },
                ),
                "end_sigma": (
                    "FLOAT",
                    {
                        "default": dc.end_sigma,
                        "max": 99999.0,
                        "min": 0.0,
                    },
                ),
                "positive_scale": (
                    "FLOAT",
                    {
                        "default": dc.positive_scale,
                        "min": -9999.0,
                        "max": 9999.0,
                        "tooltip": "Scale for features unique to the positive prompt (cond). In pre-CFG mode, this gets multiplied by the CFG scale. For example, at CFG 5, the default of 0.2 would result in roughly the same strength as CFG 1.",
                    },
                ),
                "negative_scale": (
                    "FLOAT",
                    {
                        "default": dc.negative_scale,
                        "min": -9999.0,
                        "max": 9999.0,
                        "tooltip": "Scale for features unique to the negative prompt (uncond). This gets subtracted. In pre-CFG mode the scale is effectively multiplied by CFG.",
                    },
                ),
                "patch_mode": (
                    ("pre_cfg", "post_cfg", "post_cfg_base_denoised"),
                    {
                        "default": "pre_cfg",
                        "tooltip": "pre_cfg: This mode applies the positive change to cond and the negative change to uncond and lets the CFG function take care of subtracting the negative part and adding the positive part. Since CFG 1 is normally just cond, at CFG one you will get the positive change applied as expected but the negative side will have no effect. Or in other words, result is basically common + unique_positive * CFG - unique_negative * (CFG - 1).\n\npost_cfg: The unique negative features are subtracted at exactly the scale you specify and the unique positive features are added in the same way. However, these are relative to the original cond/uncond generations but are applied to the result of CFG.\n\npost_cfg_base_denoised: This is like post_cfg mode except the positive side is what's unique to denoised (the result of CFG) and both parts are added to a common base. The results can be weird if you have other CFG type effects (I.E. CFG++) running afterward because what's unique to denoised will also include the negative side of uncond. Experimental mode, generally not recommended.",
                    },
                ),
                "start_dim": (
                    "INT",
                    {
                        "default": dc.start_dim,
                        "min": -999,
                        "max": 999,
                        "tooltip": "Start dimension (zero-based) for determining orthogonal features. Image models typically use dimensions BATCH, CHANNELS, HEIGHT, WIDTH. Video models insert a FRAMES dimension after CHANNELS. The default is to normalize over channels.",
                    },
                ),
                "end_dim": (
                    "INT",
                    {
                        "default": dc.end_dim,
                        "min": -999,
                        "max": 999,
                        "tooltip": "End dimension (zero-based) for determining orthogonal features. Image models typically use dimensions BATCH, CHANNELS, HEIGHT, WIDTH. Video models insert a FRAMES dimension after CHANNELS. The default is to normalize over channels.",
                    },
                ),
                "use_noise": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Apply CFG to the noise prediction instead of the clean image. CFG normally uses the clean image. Experimental option and generally it's harder to separate out what's orthogonal from noise compared to a clean latent.",
                    },
                ),
                "force_uncond_generation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Disables the normal optimization that skip generating uncond (negative prompt) when CFG is 1.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        model,
        patch_mode: str = "pre_cfg",
        force_uncond_generation: bool = False,
        **kwargs: Any,
    ) -> tuple:
        patch_object = ContrastiveOrthoCFG(
            base_denoised=patch_mode.endswith("_base_denoised"),
            **kwargs,
        )
        model = model.clone()
        if patch_mode == "pre_cfg":
            model.set_model_sampler_pre_cfg_function(
                patch_object.precfg_patch,
                disable_cfg1_optimization=force_uncond_generation,
            )
        else:
            model.set_model_sampler_post_cfg_function(
                patch_object.postcfg_patch,
                disable_cfg1_optimization=force_uncond_generation,
            )
        return (model,)
