import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial, reduce

import torch
from tqdm import tqdm


class BlockType(Enum):
    INPUT = auto()
    OUTPUT = auto()
    MIDDLE = auto()
    ATTN_Q = auto()
    ATTN_K = auto()
    ATTN_V = auto()
    ATTN = auto()


class BlendType(Enum):
    DIFF = auto()
    RESULT = auto()


class CondType(Enum):
    COND = auto()
    UNCOND = auto()
    BOTH = auto()


@dataclass
class BlockCFGItem:
    start_sigma: float
    end_sigma: float
    block_type: BlockType
    target_type: BlendType
    cond_type: CondType
    block_num: int
    scale: float
    skip_mode: bool


class BlockCFG:
    start_sigma: float
    end_sigma: float
    block_types: frozenset[BlockType]
    verbose: bool = True

    def __init__(self, items: tuple[BlockCFGItem, ...]):
        self.start_sigma, self.end_sigma = reduce(
            lambda old, new: (min(old[0], new[0]), max(old[1], new[1])),
            ((i.start_sigma, i.end_sigma) for i in items),
            (math.inf, math.inf * -1),
        )
        self.block_types = frozenset(i.block_type for i in items)

    # def check_applies(self,


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
                        "tooltip": "Comma separated list of block numbers, each should start with one of i(input), m(iddle), o(utput). You may also use * instead of a block number to select all blocks in the category.",
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
        for idx, item_ in enumerate(commasep_block_numbers.split(",")):
            item = item_.strip().lower()
            if not item:
                continue
            block_type = item[0]
            if block_type not in "imo" or len(item) < 2:
                errstr = f"Bad block definition at item {idx}"
                raise ValueError(errstr)
            if item[1] == "*":
                block = tidx = -1
            else:
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
            tqdm.write(f"* BLOCKCFG: tf={transformer_options}")
            cond_or_uncond = transformer_options["cond_or_uncond"]
            if (
                not (0 in cond_or_uncond and 1 in cond_or_uncond)
                or len(cond_or_uncond) != 2
            ):
                return False
            block_num = transformer_options["block"][1]
            sigma_tensor = transformer_options["sigmas"].max()
            sigma = sigma_tensor.detach().cpu().item()
            block_def = block_list.get(block_num)
            ok_time = sigma_end <= sigma <= sigma_start
            if not ok_time:
                return False
            if block_def is None:
                return -1 in block_list
            return block_def in {-1, transformer_options.get("transformer_index")}

        def apply_cfg_fun_(tensor: torch.Tensor, primary_offset: int) -> torch.Tensor:
            full_batch = tensor.shape[0]
            if full_batch % 2:
                raise RuntimeError("Batch size must be multiple of 2")
            batch = full_batch // 2
            diff = tensor[:batch, ...] - tensor[batch:, ...]

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
            return result

        mid_patch = None

        def non_output_block_patch(h, transformer_options, *, block_list):
            nonlocal mid_patch
            # print("\nSET????", mid_patch)
            if mid_patch is not None:
                mid_patch._bleh_set_topts(transformer_options)
            cond_or_uncond = transformer_options["cond_or_uncond"]
            if not block_list or not check_applies(block_list, transformer_options):
                return h
            return apply_cfg_fun(h, cond_or_uncond.index(0))

        def output_block_patch(h, hsp, transformer_options, *, block_list):
            cond_or_uncond = transformer_options["cond_or_uncond"]
            if not check_applies(
                block_list,
                transformer_options,
            ):
                return h, hsp
            cond_idx = cond_or_uncond.index(0)
            return (
                (apply_cfg_fun(h, cond_idx), hsp)
                if not skip_mode
                else (h, apply_cfg_fun(hsp, cond_idx))
            )

        m = model.clone()

        if middle_blocks:
            # print("******** MIDDLE")
            try:
                mb = model.get_model_object("diffusion_model.middle_block.0")
            except AttributeError:
                mb = None
            orig_forward = getattr(mb, "forward", None)
            if mb is None or orig_forward is None:
                raise ValueError("Could not get middle block or forward")

            class MBForward:
                def __init__(self, orig_forward):
                    real_orig_forward = orig_forward
                    while temp := getattr(
                        real_orig_forward, "_bleh_orig_forward", None
                    ):
                        real_orig_forward = temp
                    orig_forward = real_orig_forward
                    self._bleh_orig_forward = orig_forward
                    self._bleh_topts = None

                def _bleh_set_topts(self, transformer_options: dict) -> None:
                    # if self._bleh_topts:
                    #     return
                    cond_or_uncond = transformer_options["cond_or_uncond"]
                    self._bleh_topts = {
                        "cond_or_uncond": cond_or_uncond.clone()
                        if isinstance(cond_or_uncond, torch.Tensor)
                        else cond_or_uncond,
                        "sigmas": transformer_options["sigmas"].clone(),
                        "block": ("middle", 0),
                    }

                def __call__(self, *args: list, **kwargs: dict) -> torch.Tensor:
                    result = self._bleh_orig_forward(*args, **kwargs)
                    try:
                        return non_output_block_patch(
                            result,
                            self._bleh_topts,
                            block_list=middle_blocks,
                        )
                    finally:
                        self._bleh_topts = None

            mid_patch = MBForward(orig_forward)
            m.add_object_patch("diffusion_model.middle_block.0.forward", mid_patch)

        if input_blocks or middle_blocks:
            (
                m.set_model_input_block_patch_after_skip
                if skip_mode
                else m.set_model_input_block_patch
            )(
                partial(non_output_block_patch, block_list=input_blocks),
            )

        if output_blocks:
            m.set_model_output_block_patch(
                partial(output_block_patch, block_list=output_blocks),
            )
        return (m,)
