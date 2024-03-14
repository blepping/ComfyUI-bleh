from __future__ import annotations


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
        import comfy

        model = model.clone()
        comfy.model_management.load_models_gpu([refiner_model])
        refiner_model = self.get_real_model(refiner_model)
        ms = self.get_real_model(model).model_sampling

        def unet_wrapper(apply_model, args):
            inp, timestep, c = args["input"], args["timestep"], args["c"]
            curr_timestep = ms.timestep(timestep).float().max().item()
            return (
                refiner_model.apply_model
                if curr_timestep <= start_timestep
                else apply_model
            )(inp, timestep, **c)

        model.set_model_unet_function_wrapper(unet_wrapper)
        return (model,)
