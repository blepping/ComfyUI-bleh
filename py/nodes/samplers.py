from __future__ import annotations

import contextlib
import importlib
import math
import random
from copy import deepcopy
from functools import partial, update_wrapper
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
            int(environ.get("COMFYUI_BLEH_SAMPLER_PRESET_COUNT", "1")),
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

    def go(self, sampler: object, seed_offset: int | None = 1) -> tuple[KSAMPLER]:
        return (
            KSAMPLER(
                update_wrapper(
                    partial(
                        self.sampler_function,
                        bleh_wrapped_sampler=sampler,
                        bleh_seed_offset=seed_offset,
                    ),
                    sampler.sampler_function,
                ),
                extra_options=sampler.extra_options.copy(),
                inpaint_options=sampler.inpaint_options.copy(),
            ),
        )

    @classmethod
    @torch.no_grad()
    def sampler_function(
        cls,
        model: object,
        x: torch.Tensor,
        *args: list[Any],
        extra_args: dict[str, Any] | None = None,
        bleh_wrapped_sampler: object | None = None,
        bleh_seed_offset: int | None = 1,
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
        if key in KSampler.SAMPLERS:
            print(
                f"\n** ComfyUI-bleh: Warning: {key} already exists in sampler list, skipping adding preset samplers.",
            )
            if idx == 0:
                return
            break
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
        override_sigmas_opt: torch.Tensor | None = None,
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


class BlehCFGInitSampler:
    DESCRIPTION = "Sampler wrapper that allows skipping some number of initial steps, similar to CFGZeroStar zero-init."
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": "Connect the sampler you want to wrap here. It will be called to sample as normal after the configured number of skipped steps.",
                    },
                ),
                "steps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 9999.0,
                        "tooltip": "Number of steps to skip before sampling. You can use a fractional value here, but it may not work well. Whole skipped steps do not require a model call.",
                    },
                ),
                "mode": (
                    ("zero", "afs", "afs_flow_hack", "scale_down"),
                    {
                        "default": "zero",
                        "tooltip": "zero: Works like CFGZeroStar zero init (just skips steps).\nafs: Analytical first step mode. A method of scaling down the initial noise to match skipped steps.\nafs_hack: A version of AFS mode that may work better for flow models.\nscale_down: The simplest approach to scaling down the latent to match the skipped steps.",
                    },
                ),
            },
        }

    FUNCTION = "go"

    def go(
        self,
        *,
        sampler: object,
        steps: float,
        mode: str,
    ) -> tuple[KSAMPLER]:
        if steps == 0:
            return (sampler,)
        sampler_function = update_wrapper(
            partial(
                self.sampler_function,
                bleh_ci_wrapped_sampler=sampler,
                bleh_ci_mode=mode,
                bleh_ci_steps=steps,
            ),
            sampler.sampler_function,
        )
        return (
            KSAMPLER(
                sampler_function,
                extra_options=sampler.extra_options.copy(),
                inpaint_options=sampler.inpaint_options.copy(),
            ),
        )

    @staticmethod
    def sampler_function(
        model: object,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *args: list[Any],
        extra_args: dict[str, Any] | None = None,
        bleh_ci_wrapped_sampler: object | None = None,
        bleh_ci_mode: str = "zero",
        bleh_ci_steps: float = 0,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if not bleh_ci_wrapped_sampler:
            raise ValueError("Wrapped sampler missing!")
        sigmas = sigmas.clone()
        whole_steps = int(bleh_ci_steps)
        steps = math.ceil(bleh_ci_steps)
        step_fraction = bleh_ci_steps - whole_steps
        for idx in range(steps if bleh_ci_mode != "zero" else 0):
            sigma, sigma_next = sigmas[idx : idx + 2]
            dt = sigma_next - sigma
            if idx == whole_steps:
                dt *= step_fraction
            # From https://arxiv.org/abs/2210.05475
            if bleh_ci_mode == "afs":
                d = x / (1.0 + sigma**2) ** 0.5
            elif bleh_ci_mode == "afs_flow_hack":
                d = x / ((1.0 + (sigma * 10.0) ** 2) ** 0.5) / 10.0
            elif bleh_ci_mode == "scale_down":
                d = x / sigma
            else:
                raise ValueError("Bad CFG init mode")
            x = x + d * dt  # noqa: PLR6104

        if steps > 0 and step_fraction != 0:
            sigmas[whole_steps] += (sigmas[steps] - sigmas[whole_steps]) * step_fraction

        return bleh_ci_wrapped_sampler.sampler_function(
            model,
            x,
            sigmas[whole_steps:],
            *args,
            extra_args=extra_args,
            **kwargs,
        )
