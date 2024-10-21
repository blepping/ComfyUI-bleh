from __future__ import annotations

import random
from typing import Any, Callable, NamedTuple

import torch
from comfy.samplers import KSAMPLER
from tqdm import tqdm


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
