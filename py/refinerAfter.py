from __future__ import annotations

import comfy


class BlehRefinerAfter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "time_mode": (
                    (
                        "timestep",
                        "percent",
                        "sigma",
                    ),
                ),
                "start_time": ("FLOAT", {"default": 199.0, "min": 0.0, "max": 999.0}),
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

    @staticmethod
    def load_if_needed(model: object) -> bool:
        if model.current_device != model.load_device:
            comfy.model_management.load_models_gpu([model])
            return True
        return False

    def patch(  # noqa: PLR0911
        self,
        start_time: float,
        model: object,
        refiner_model: object,
        time_mode: str = "timestep",
    ) -> tuple[object]:
        model = model.clone()
        refiner_model = refiner_model.clone()
        ms = self.get_real_model(model).model_sampling
        real_refiner_model = None

        match time_mode:
            case "sigma":
                if start_time <= ms.sigma_min():
                    return (model,)
                if start_time >= ms.sigma_max():
                    return (refiner_model,)

                def check_time(sigma):
                    return sigma.item() <= start_time
            case "percent":
                if start_time > 1.0 or start_time < 0.0:
                    raise ValueError(
                        "BlehRefinerAfter: invalid value for percent start time",
                    )
                if start_time >= 1.0:
                    return (model,)
                if start_time <= 0.0:
                    return (refiner_model,)

                def check_time(sigma):
                    return sigma.item() <= ms.percent_to_sigma(start_time)
            case "timestep":
                if start_time <= 0.0:
                    return (model,)
                if start_time >= 999.0:
                    return (refiner_model,)

                def check_time(sigma):
                    return ms.timestep(sigma) <= start_time
            case _:
                raise ValueError("BlehRefinerAfter: invalid time mode")

        def unet_wrapper(apply_model, args):
            nonlocal real_refiner_model

            inp, timestep, c = args["input"], args["timestep"], args["c"]
            if not check_time(timestep.max()):
                real_refiner_model = None
                self.load_if_needed(model)
                return apply_model(inp, timestep, **c)
            if self.load_if_needed(refiner_model) or not real_refiner_model:
                real_refiner_model = self.get_real_model(refiner_model)
            return real_refiner_model.apply_model(inp, timestep, **c)

        model.set_model_unet_function_wrapper(unet_wrapper)
        return (model,)
