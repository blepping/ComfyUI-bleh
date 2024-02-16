from __future__ import annotations

from typing import Any

import torch
from comfy.ldm.modules.attention import optimized_attention

# ** transformer_options
# input_block_patch*           : p(h, transformer_options) -> h
# input_block_patch_after_skip*: p(h, transformer_options) -> h
# output_block_patch*          : p(h, hsp, transformer_options) -> h, hsp
# attn1_patch*                 : p(n, context_attn1, value_attn1, extra_options) -> n, context_attn1, value_attn1
# attn1_output_patch*          : p(n, extra_options) -> n
# attn2_patch*                 : p(n, context_attn2, value_attn2, extra_options) -> n, context_attn2, value_attn2
# attn2_output_patch*          : p(n, extra_options) -> n
# middle_patch*                : p(x, extra_options) -> x
# attn1_replace*               : p(n, context_attn1, value_attn1, extra_options) -> n
# attn2_replace*               : p(n, context_attn2, value_attn2, extra_options) -> n

# ** model_options
# model_function_wrapper       : p(model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}) -> output
# sampler_cfg_function         : p({"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}) -> output
# sampler_post_cfg_function*   : p({"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": timestep, "model_options": model_options, "input": x}) -> output


class PatchTypeTransformer:
    def __init__(
        self,
        name,
        nresult=1,
    ):
        self.name = name
        self.nresult = nresult

    def get_patches(self, model_options):
        return (
            model_options.get("transformer_options", {})
            .get("patches", {})
            .get(self.name, [])
        )

    def set_patches(self, model_options, val):
        to = model_options.get("transformer_options", {})
        model_options["transformer_options"] = to
        patches = to.get("patches", {})
        to["patches"] = patches
        patches[self.name] = val

    def exists(self, model_options):
        return len(self.get_patches(model_options)) > 0

    def _call(self, patches, *args: list[Any]):
        result_part, arg_part = args[: self.nresult], args[self.nresult :]
        for p in patches:
            result_part = p(*result_part, *arg_part)
        return result_part

    @torch.no_grad()
    def __call__(self, model_options, *args: list[Any]):
        result = self._call(self.get_patches(model_options), *args)
        return result[0] if isinstance(result, (tuple, list)) and len(result) == 1 else result


class PatchTypeTransformerReplace(PatchTypeTransformer):
    def get_patches(self, model_options):
        return (
            model_options.get("transformer_options", {})
            .get("patches_replace", {})
            .get(self.name, {})
        )

    def set_patches(self, model_options, val):
        to = model_options.get("transformer_options", {})
        model_options["transformer_options"] = to
        patches = to.get("patches_replace", {})
        to["patches_replace"] = patches
        patches[self.name] = val

    @torch.no_grad()
    def __call__(self, key, model_options, *args: list[Any]):
        return self._call(key, self.get_patches(model_options), *args)

    @torch.no_grad()
    def _call(self, key, patches, *args: list[Any]):
        p = patches.get(key)
        if p:
            return p(*args)
        return optimized_attention(*args[:-1], heads=args[-1]["n_heads"])


class PatchTypeModel(PatchTypeTransformer):
    def set_patches(self, model_options, val):
        model_options[self.name] = val[0]

    def get_patches(self, model_options):
        return () if self.name not in model_options else (model_options[self.name],)


class PatchTypeModelWrapper(PatchTypeModel):
    @torch.no_grad()
    def _call(self, patches, apply_model, opts):
        if not patches:
            return apply_model(opts["input"], opts["timestep"], **opts["c"])
        return patches[0](apply_model, opts)


class PatchTypeSamplerPostCfgFunction(PatchTypeModel):
    def get_patches(self, model_options):
        return model_options.get(self.name, ())

    def set_patches(self, model_options, val):
        model_options[self.name] = val

    @torch.no_grad()
    def _call(self, patches, opts):
        result = opts["denoised"]
        for p in patches:
            result = p(opts)
            opts["denoised"] = result
        return result


class PatchTypeSamplerCfgFunction(PatchTypeModel):
    @torch.no_grad()
    def _call(self, patches, opts):
        if not patches:
            cond_pred, uncond_pred = opts["cond_denoised"], opts["uncond_denoised"]
            return uncond_pred + (cond_pred - uncond_pred) * opts["cond_scale"]
        return patches[0](opts)


PATCH_TYPES = {
    "input_block_patch": PatchTypeTransformer("input_block_patch"),
    "input_block_patch_after_skip": PatchTypeTransformer(
        "input_block_patch_after_skip",
    ),
    "output_block_patch": PatchTypeTransformer("output_block_patch", nresult=2),
    "attn1_patch": PatchTypeTransformer("attn1_patch", nresult=3),
    "attn1_output_patch": PatchTypeTransformer("attn1_output_patch"),
    "attn2_patch": PatchTypeTransformer("attn2_patch", nresult=3),
    "attn2_output_patch": PatchTypeTransformer("attn2_output_patch"),
    "middle_patch": PatchTypeTransformer("middle_patch"),
    "attn1": PatchTypeTransformerReplace("attn1"),
    "attn2": PatchTypeTransformerReplace("attn2"),
    "model_function_wrapper": PatchTypeModelWrapper("model_function_wrapper"),
    "sampler_cfg_function": PatchTypeSamplerCfgFunction("sampler_cfg_function"),
    "sampler_post_cfg_function": PatchTypeSamplerPostCfgFunction(
        "sampler_post_cfg_function",
    ),
}


class ModelConditionalState:
    def __init__(self):
        self.last_sigma = None
        self.step = None

    def update(self, sigma):
        if self.last_sigma is None or sigma > self.last_sigma:
            self.step = 0
        elif sigma != self.last_sigma:
            self.step += 1
        self.last_sigma = sigma
        return self.step


class ModelPatchConditional:
    def __init__(
        self,
        model_default,
        model_matched,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        interval: int = 1,
        base_on_default: bool = True,
    ):
        self.options_default = dict(**model_default.model_options)
        self.options_matched = dict(**model_matched.model_options)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.interval = interval
        self.patches_orig = {}
        self.base = model_default if base_on_default else model_matched
        self.base_on_default = base_on_default
        self.sigma_end = self.sigma_start = None

    def lazy_calc_steps(self):
        if self.sigma_start is not None:
            return
        model = self.base
        count = 0
        while hasattr(model, "model"):
            count += 1
            if count > 128:
                raise ValueError("I can't handle these insane levels of modelception!")
            model = model.model
        self.sigma_start = model.model_sampling.percent_to_sigma(
            self.start_percent,
        )
        self.sigma_end = model.model_sampling.percent_to_sigma(
            self.end_percent,
        )

    def should_use_patched(self, state, opts):
        self.lazy_calc_steps()
        if "sigmas" in opts:
            sigmas = opts["sigmas"]
        elif "transformer_options" in opts:
            sigmas = opts["transformer_options"]["sigmas"]
        elif "c" in opts:
            sigmas = opts["c"]["transformer_options"]["sigmas"]
        elif "sigma" in opts:
            sigmas = opts["sigma"]
        else:
            raise ValueError("Cannot determine sigma")
        iv = self.interval
        sigma = sigmas[0].item()
        step = state.update(sigma)
        matched = sigma <= self.sigma_start and sigma >= self.sigma_end
        matched &= (step % iv) == 0 if iv > 0 else ((step + 1) % abs(iv)) > 0
        return matched

    def mk_patch_handler(self, pt, state, key=None):
        def handler(*args: list[Any]):
            matched = self.should_use_patched(state, args[-1])
            # print(f">> {pt.name}, active: {matched}, step: {state.step}")
            opts = self.options_matched if matched else self.options_default
            return pt(key, opts, *args) if key else pt(opts, *args)
        return handler

    def patch(self):
        state = ModelConditionalState()
        base_patched = self.base.clone()
        for pt in PATCH_TYPES.values():
            if not (pt.exists(self.options_default) or pt.exists(self.options_matched)):
                continue
            # print(f"set patch {pt.name}")
            if not isinstance(pt, PatchTypeTransformerReplace):
                pt.set_patches(
                    base_patched.model_options,
                    [self.mk_patch_handler(pt, state)],
                )
                continue
            pt.set_patches(
                base_patched.model_options,
                {
                    k: self.mk_patch_handler(pt, state, key=k)
                    for k in (
                        pt.get_patches(self.options_default).keys()
                        | pt.get_patches(self.options_matched).keys()
                    )
                },
            )
        base_patched.model_options["disable_cfg1_optimization"] = (
            self.options_default.get("disable_cfg1_optimization", False)
            or self.options_matched.get("disable_cfg1_optimization", False)
        )
        return base_patched


class ModelPatchConditionalNode:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_default": ("MODEL",),
                "model_matched": ("MODEL",),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "interval": ("INT", {"default": 1, "min": -999, "max": 999}),
                "base_on_default": ("BOOLEAN", {"default": True}),
            },
        }

    def patch(
        self,
        model_default,
        model_matched=None,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        interval: int = 1,
        base_on_default: bool = True,
    ):
        if not model_matched or start_percent >= 1.0 or interval == 0:
            return (model_default.clone(),)
        mopts = getattr(model_default, "model_options", None)
        if mopts is None or not isinstance(mopts, dict):
            # Not an instance of ModelPatcher, apparently so we can't do anything here.
            return (model_default.clone(),)
        return (
            ModelPatchConditional(
                model_default,
                model_matched,
                start_percent,
                end_percent,
                interval,
                base_on_default=base_on_default,
            ).patch(),
        )
