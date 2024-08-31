from __future__ import annotations

import comfy.model_management as mm


class BlehRefinerAfter:
    DESCRIPTION = "Allows switching to another model at a certain point in sampling. Only works with models that are closely related as the sampling type and conditioning must match. Can be used to switch to a refiner model near the end of sampling."
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "bleh/model_patches"

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
                    {
                        "tooltip": "Controls how start_time is interpreted. Timestep will be 999 at the start of sampling and 0 at the end - it is basically the inverse of the sampling percentage with a multiplier. Percent is the percent of sampling (not steps) and will be 0.0 at the start of sampling and 1.0 at the end. Sigma is an advanced option - if you don't know what it is, you don't need to use it.",
                    },
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 199.0,
                        "min": 0.0,
                        "max": 999.0,
                        "tooltip": "Time the refiner_model will become active. The type of value you enter here will depend on what time_mode is set to.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch. This will also be the active model until the start_time condition is met.",
                    },
                ),
                "refiner_model": (
                    "MODEL",
                    {
                        "tooltip": "Model to switch to after the start_time condition is met.",
                    },
                ),
            },
        }

    FUNCTION = "patch"

    @staticmethod
    def get_real_model(model: object) -> object:
        while hasattr(model, "model"):
            model = model.model
        return model

    @staticmethod
    def load_if_needed(model: object) -> bool:
        if mm.LoadedModel(model) in mm.current_loaded_models:
            return False
        mm.load_models_gpu([model])
        return True

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
                if start_time <= ms.sigma_min:
                    return (model,)
                if start_time >= ms.sigma_max:
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
