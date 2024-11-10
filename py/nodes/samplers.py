from __future__ import annotations

import contextlib
import importlib
import random
from copy import deepcopy
from functools import partial
from os import environ
from typing import Any, Callable, NamedTuple

import torch
from comfy.samplers import KSAMPLER, KSampler, k_diffusion_sampling
from tqdm import tqdm

from .misc import Wildcard

BLEH_PRESET_LIMIT = 16
BLEH_PRESET_COUNT = 1
with contextlib.suppress(Exception):
    BLEH_PRESET_COUNT = min(
        BLEH_PRESET_LIMIT,
        max(
            0,
            int(environ.get("COMFYUI_BLEH_SAMPLER_PRESET_COUNT", 1)),
        ),
    )


class SamplerChain(NamedTuple):
    prev: SamplerChain | None = None
    steps: int = 0
    sampler: object | None = None
    chain_sampler: Callable | None = None


class BlehInsaneChainSampler:
    RETURN_TYPES = ("SAMPLER", "BLEH_SAMPLER_CHAIN")
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "steps": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "sampler_chain_opt": ("BLEH_SAMPLER_CHAIN",),
            },
        }

    def build(
        self,
        sampler: object | None = None,
        steps: int = 0,
        sampler_chain_opt: SamplerChain | None = None,
    ) -> tuple[KSAMPLER, SamplerChain]:
        if sampler is None:
            raise ValueError("BlehInsaneChainSampler: sampler missing")
        if steps > 0:
            chain = SamplerChain(steps=steps, sampler=sampler, prev=sampler_chain_opt)
        else:
            chain = sampler_chain_opt or SamplerChain()
        return (KSAMPLER(self.sampler, {"sampler_chain": chain}), chain)

    @classmethod
    @torch.no_grad()
    def sampler(
        cls,
        model,
        x,
        sigmas,
        *args: list[Any],
        disable=None,
        sampler_chain=None,
        **kwargs: dict[str, Any],
    ):
        if not sampler_chain:
            return x
        chain = sampler_chain
        remaining_steps = len(sigmas) - 1
        i = 0
        progress = tqdm(total=remaining_steps, disable=disable)
        while remaining_steps > 0 and chain:
            while chain and (chain.steps == 0 or chain.sampler is None):
                chain = chain.prev
            if chain is None or chain.sampler is None:
                raise ValueError("Sampler chain didn't provide a sampler for sampling!")
            steps = min(chain.steps, remaining_steps)
            real_next = chain.prev
            while real_next and (real_next.steps == 0 or real_next.sampler is None):
                real_next = real_next.prev
            if real_next and (real_next.steps == 0 or real_next.sampler is None):
                real_next = None
            if real_next is None:
                steps = remaining_steps + 1
            start_idx = max(i - 1, i)
            end_idx = start_idx + steps
            curr_sigmas = sigmas[start_idx : end_idx + 1]
            x = chain.sampler.sampler_function(
                model,
                x,
                curr_sigmas,
                *args,
                disable=disable,
                **chain.sampler.extra_options,
                **kwargs,
            )
            i += steps
            progress.update(n=min(steps, remaining_steps))
            remaining_steps -= steps
            chain = real_next
        progress.close()
        return x


class BlehForceSeedSampler:
    DESCRIPTION = "ComfyUI has a bug where it will not set any seed if you have add_noise disabled in the sampler. This node is a workaround for that which ensures a seed alway gets set."
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
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

    FUNCTION = "go"

    def go(
        self,
        sampler: object,
        seed_offset: None | int = 1,
    ) -> tuple[KSAMPLER, SamplerChain]:
        return (
            KSAMPLER(
                self.sampler_function,
                extra_options=sampler.extra_options
                | {
                    "bleh_wrapped_sampler": sampler,
                    "bleh_seed_offset": seed_offset,
                },
                inpaint_options=sampler.inpaint_options | {},
            ),
        )

    @classmethod
    @torch.no_grad()
    def sampler_function(
        cls,
        model: object,
        x: torch.Tensor,
        *args: list[Any],
        extra_args: None | dict[str, Any] = None,
        bleh_wrapped_sampler: object | None = None,
        bleh_seed_offset: None | int = 1,
        **kwargs: dict[str, Any],
    ):
        if not bleh_wrapped_sampler:
            raise ValueError("wrapped sampler missing!")
        seed = (extra_args or {}).get("seed")
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            for _ in range(bleh_seed_offset if bleh_seed_offset is not None else 0):
                _ = random.random()  # noqa: S311
                _ = torch.randn_like(x)
        return bleh_wrapped_sampler.sampler_function(
            model,
            x,
            *args,
            extra_args=extra_args,
            **kwargs,
        )


BLEH_PRESET = [None] * BLEH_PRESET_COUNT


def bleh_sampler_preset_wrapper(
    preset_idx,
    model,
    x,
    sigmas,
    *args: list,
    **kwargs: dict,
) -> torch.Tensor:
    if not (0 <= preset_idx < BLEH_PRESET_COUNT):
        raise ValueError("Bleh sampler preset out of range")
    preset = BLEH_PRESET[preset_idx]
    if preset is None:
        errstr = f"Cannot use bleh_preset_{preset_idx} - present not defined. Ensure BlehSetSamplerPreset runs before sampling."
        raise RuntimeError(errstr)
    sampler, override_sigmas = preset
    if override_sigmas is not None:
        sigmas = override_sigmas.detach().clone().to(sigmas)
    return sampler.sampler_function(
        model,
        x,
        sigmas,
        *args,
        **sampler.extra_options,
        **kwargs,
    )


def add_sampler_presets():
    if BLEH_PRESET_COUNT < 1:
        return
    for idx in range(BLEH_PRESET_COUNT):
        key = f"bleh_preset_{idx}"
        KSampler.SAMPLERS.append(key)
        setattr(
            k_diffusion_sampling,
            f"sample_{key}",
            partial(bleh_sampler_preset_wrapper, idx),
        )
    importlib.reload(k_diffusion_sampling)


class BlehSetSamplerPreset:
    WILDCARD = Wildcard("*")
    DESCRIPTION = "This node allows setting a custom sampler as a preset that can be selected in nodes that don't support custom sampling (FaceDetailer for example). This node needs to run at least once with any preset changes before actual sampling begins. The any_input input acts as a passthrough so you can do something like pass your model or latent through before you start sampling to ensure the node runs. You can also connect something like an integer or string to the dummy_opt input and change it to force the node to run again. The number of presets can be adjusted (and the whole feature disabled if desired) by setting the environment variable COMFYUI_BLEH_SAMPLER_PRESET_COUNT. WARNING: Since the input and output are wildcards, this bypasses ComfyUI's normal type checking. Make sure you connect the output to something that actually accepts the input type."
    RETURN_TYPES = (WILDCARD,)
    OUTPUT_TOOLTIPS = (
        "This just returns the value of any_input unchanged. WARNING: ComfyUI's normal typechecking is disabled here, make sure you connect this output to something that allows the input type.",
    )
    CATEGORY = "hacks"
    NOT_IDEMPOTENT = True
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER", {"tooltip": "Sampler to use for this preset."}),
                "any_input": (
                    cls.WILDCARD,
                    {
                        "tooltip": "This input is simply returned as the output. Note: Make sure you connect this node's output to something that supports input connected here.",
                    },
                ),
                "preset": (
                    "INT",
                    {
                        "min": -1,
                        "max": BLEH_PRESET_COUNT - 1,
                        "default": 0 if BLEH_PRESET_COUNT > 0 else -1,
                        "tooltip": "Preset index to set. If set to -1, no preset assignment will be done. The number of presets can be adjusted, see the README.",
                    },
                ),
                "discard_penultimate_sigma": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Advanced option to allow discarding the penultimate sigma. May be needed for some samplers like dpmpp_3m_sde - if it seems like the generation has a bunch of noise added at the very last step then you can try enabling this. Note: Cannot be used when override sigmas are attached.",
                    },
                ),
            },
            "optional": {
                "override_sigmas_opt": (
                    "SIGMAS",
                    {
                        "tooltip": "Advanced option that allows overriding the sigmas used for sampling. Note: Cannot be used with discard_penultimate_sigma. Also this cannot control the noise added by the sampler, so if the schedule used by the sampler starts on a different sigma than the override you will likely run into issues.",
                    },
                ),
                "dummy_opt": (
                    cls.WILDCARD,
                    {
                        "tooltip": "This input can optionally be connected to any value as a way to force the node to run again on demand. See the README.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        sampler,
        any_input,
        preset,
        discard_penultimate_sigma,
        override_sigmas_opt: None | torch.Tensor = None,
        dummy_opt=None,  # noqa: ARG003
    ):
        if not (0 <= preset < BLEH_PRESET_COUNT):
            return (any_input,)
        if discard_penultimate_sigma and override_sigmas_opt is not None:
            raise ValueError(
                "BlehSetSamplerPreset: Cannot override sigmas and also enable discard penultimate sigma",
            )
        dps_samplers = getattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS", None)
        if dps_samplers is not None:
            key = f"bleh_preset_{preset}"
            if discard_penultimate_sigma:
                dps_samplers.update(key)
            else:
                dps_samplers -= {key}
        sigmas = (
            None
            if override_sigmas_opt is None
            else override_sigmas_opt.detach().clone().cpu()
        )
        BLEH_PRESET[preset] = (deepcopy(sampler), sigmas)
        return (any_input,)
