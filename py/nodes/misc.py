import random

import torch


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

    def go(self, enabled, sigmas):
        if not enabled or len(sigmas) < 2:
            return (sigmas,)
        return (torch.cat((sigmas[:-2], sigmas[-1:])),)


class SeededDisableNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, latent):
        samples = latent["samples"]
        torch.manual_seed(self.seed)
        random.seed(self.seed)  # For good measure.
        return torch.zeros(
            samples.shape,
            dtype=samples.dtype,
            layout=samples.layout,
            device="cpu",
        )


class BlehDisableNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
        }

    def go(self, noise_seed):
        return (SeededDisableNoise(noise_seed),)

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/noise"


class Wildcard(str):
    __slots__ = ()

    def __ne__(self, _unused):
        return False


class BlehPlug:
    WILDCARD = Wildcard("*")

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    def go(self):
        return (None,)

    RETURN_TYPES = (WILDCARD,)
    FUNCTION = "go"
    OUTPUT_NODE = False
    CATEGORY = "hacks"
