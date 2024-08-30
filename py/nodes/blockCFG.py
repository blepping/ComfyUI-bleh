from functools import partial


class BlockCFGBleh:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"
    DESCRIPTION = (
        "Applies a CFG type effect to the model blocks themselves during evaluation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch",
                    },
                ),
                "commasep_block_numbers": (
                    "STRING",
                    {
                        "default": "i4,m0,o4",
                        "tooltip": "Comma separated list of block numbers, each should start with one of i(input), m(iddle), o(utput)",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Effect strength",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Start time as sampling percentage (not percentage of steps). Percentages are inclusive.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "End time as sampling percentage (not percentage of steps). Percentages are inclusive.",
                    },
                ),
                "skip_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "For output blocks, this causes the effect to apply to the skip connection. For input blocks it patches after the skip connection. No effect for middle blocks.",
                    },
                ),
                "apply_to": (
                    ("cond", "uncond"),
                    {
                        "default": "uncond",
                        "tooltip": "Guides the specified target away from its opposite. cond=positive prompt, uncond=negative prompt.",
                    },
                ),
            },
        }

    @classmethod
    def patch(
        cls,
        *,
        model,
        commasep_block_numbers,
        scale,
        start_percent,
        end_percent,
        skip_mode,
        apply_to,
    ):
        input_blocks = {}
        middle_blocks = {}
        output_blocks = {}
        for item_ in commasep_block_numbers.split(","):
            item = item_.strip().lower()
            if not item:
                continue
            block_type = item[0]
            if block_type not in "imo":
                raise ValueError
            block, *tidx = item[1:].split(".", 1)
            block = int(block)
            tidx = int(tidx) if tidx else -1
            if block_type == "i":
                bd = input_blocks
            else:
                bd = output_blocks if block_type == "o" else middle_blocks
            bd[block] = tidx

        if (
            scale == 0
            or end_percent <= 0
            or start_percent >= 1
            or not (input_blocks or middle_blocks or output_blocks)
        ):
            return (model,)

        ms = model.get_model_object("model_sampling")
        sigma_start = ms.percent_to_sigma(start_percent)
        sigma_end = ms.percent_to_sigma(end_percent)
        reverse = apply_to != "cond"

        def check_applies(block_list, transformer_options):
            block_num = transformer_options["block"][1]
            sigma_tensor = transformer_options["sigmas"].max()
            sigma = sigma_tensor.detach().cpu().item()
            block_def = block_list.get(block_num)
            return (
                sigma_end <= sigma <= sigma_start
                and block_def is not None
                and (
                    block_def == -1
                    or block_def == transformer_options.get("transformer_index")
                )
            )

        def apply_cfg_fun(tensor, primary_offset):
            secondary_offset = 0 if primary_offset == 1 else 1
            if reverse:
                primary_offset, secondary_offset = secondary_offset, primary_offset
            result = tensor.clone()
            batch = tensor.shape[0] // 2
            primary_idxs, secondary_idxs = (
                tuple(range(batch * offs, batch + batch * offs))
                for offs in (primary_offset, secondary_offset)
            )
            # print(f"\nIDXS: cond={primary_idxs}, uncond={secondary_idxs}")
            result[primary_idxs, ...] -= (
                tensor[primary_idxs, ...] - tensor[secondary_idxs, ...]
            ).mul_(scale)
            # result[primary_idxs, ...] = (
            #     tensor[primary_idxs, ...]
            #     + (tensor[primary_idxs, ...] - tensor[secondary_idxs, ...]) * scale
            # )
            return result

        def non_output_block_patch(h, transformer_options, *, block_list):
            cond_or_uncond = transformer_options["cond_or_uncond"]
            if len(cond_or_uncond) != 2 or not check_applies(
                block_list,
                transformer_options,
            ):
                return h
            return apply_cfg_fun(h, cond_or_uncond[0])

        def output_block_patch(h, hsp, transformer_options, *, block_list):
            cond_or_uncond = transformer_options["cond_or_uncond"]
            if len(cond_or_uncond) != 2 or not check_applies(
                block_list,
                transformer_options,
            ):
                return h, hsp
            return (
                (apply_cfg_fun(h, cond_or_uncond[0]), hsp)
                if not skip_mode
                else (h, apply_cfg_fun(hsp, cond_or_uncond[0]))
            )

        m = model.clone()
        if input_blocks:
            (
                m.set_model_input_block_patch
                if skip_mode
                else m.set_model_input_block_patch_after_skip
            )(
                partial(non_output_block_patch, block_list=input_blocks),
            )
        if middle_blocks:
            m.set_model_patch(
                partial(non_output_block_patch, block_list=middle_blocks),
                "middle_block_patch",
            )
        if output_blocks:
            m.set_model_output_block_patch(
                partial(output_block_patch, block_list=output_blocks),
            )
        return (m,)
