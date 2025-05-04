from __future__ import annotations

import operator
import random
from decimal import Decimal

import torch
from comfy import model_management

from ..better_previews.previewer import ensure_previewer  # noqa: TID252


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


class Wildcard(str):  # noqa: FURB189
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
