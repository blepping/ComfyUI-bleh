from __future__ import annotations

import comfy


class BlehRefinerAfter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_timestep": ("INT", {"default": 199, "min": 0, "max": 1000}),
                "model": ("MODEL",),
                "refiner_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "bleh/model_patches"

    FUNCTION = "patch"

    @staticmethod
    def get_real_model(model: object) -> object:
        while hasattr(model, "model"):
            model = model.model
        return model

    def patch(
        self,
        start_timestep: int,
        model: object,
        refiner_model: object,
    ) -> tuple[object]:
        model = model.clone()
        if start_timestep < 1:
            return (model,)
        refiner_model = refiner_model.clone()
        ms = self.get_real_model(model).model_sampling
        real_refiner_model = None

        def unet_wrapper(apply_model, args):
            nonlocal real_refiner_model

            inp, timestep, c = args["input"], args["timestep"], args["c"]
            curr_timestep = ms.timestep(timestep).float().max().item()
            if curr_timestep > start_timestep:
                real_refiner_model = None
                return apply_model(inp, timestep, **c)
            if (
                real_refiner_model is None
                or refiner_model.current_device != refiner_model.load_device
            ):
                comfy.model_management.load_models_gpu([refiner_model])
                real_refiner_model = self.get_real_model(refiner_model)
            return real_refiner_model.apply_model(inp, timestep, **c)

        model.set_model_unet_function_wrapper(unet_wrapper)
        return (model,)
