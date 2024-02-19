from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import torch
from comfy.samplers import KSAMPLER
from tqdm import tqdm


class SamplerChain(NamedTuple):
    prev: SamplerChain | None = None
    steps: int = 0
    sampler: object | None = None
    chain_sampler: Callable | None = None


class BlehInsaneChainSampler:
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

    RETURN_TYPES = ("SAMPLER", "BLEH_SAMPLER_CHAIN")
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "build"

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


class WrappedKSAMPLER:
    def __init__(self, wrapped_sampler):
        for k in ("sampler_function", "extra_options", "inpaint_options"):
            setattr(self, k, getattr(wrapped_sampler, k))
        self.wrapped_sampler = wrapped_sampler

    def sample(self, *args: Sequence[Any], **kwargs: dict[str, Any]):
        seed = kwargs.get("extra_args", {}).get("seed")
        if seed is not None:
            torch.manual_seed(seed)
        return self.wrapped_sampler.sample(*args, **kwargs)


class BlehForceSeedSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "go"

    def go(self, sampler: object) -> tuple[WrappedKSAMPLER]:
        return (WrappedKSAMPLER(sampler),)
