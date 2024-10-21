from __future__ import annotations

import random

import torch
from comfy import model_management


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
        seed_offset: None | int = 1,
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
