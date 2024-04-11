# Adapted from the ComfyUI built-in node

import bisect
from enum import Enum, auto

import torch
import yaml

from ..latent_utils import *  # noqa: TID252


class OpType(Enum):
    # scale, strength, blend, blend mode, use hidden mean
    SLICE = auto()

    # scale, filter, filter strength, threshold
    FFILTER = auto()

    # type (bicubic, nearest, bilinear, area), scale width, scale height, antialias
    SCALE_TORCH = auto()

    # type (bicubic, nearest, bilinear, area), antialias
    UNSCALE_TORCH = auto()

    # type width (slerp, slerp_alt, hslerp, colorize), type height, scale width, scale height, antialias size
    SCALE = auto()

    # type width (slerp, slerp_alt, hslerp, colorize), type height, antialias size
    UNSCALE = auto()

    # direction (h, v)
    FLIP = auto()

    # count
    ROT90 = auto()

    # count
    ROLL_CHANNELS = auto()

    # direction (horizontal, vertical, channels) or list of dims, amount (integer or percentage >-1.0 < 1.0)
    ROLL = auto()

    # true/false - only makes sense with output block
    TARGET_SKIP = auto()

    # factor
    MULTIPLY = auto()

    # blend strength, blend_mode, [op]
    BLEND_OP = auto()

    # size
    ANTIALIAS = auto()

    # strength
    NOISE = auto()

    # none
    DEBUG = auto()


class CondType(Enum):
    TYPE = auto()
    BLOCK = auto()
    STAGE = auto()
    FROM_PERCENT = auto()
    TO_PERCENT = auto()
    PERCENT = auto()
    STEP = auto()
    FROM_STEP = auto()
    TO_STEP = auto()
    STEP_INTERVAL = auto()


class BlockType(Enum):
    INPUT = auto()
    INPUT_AFTER_SKIP = auto()
    MIDDLE = auto()
    OUTPUT = auto()


class BlockCond:
    def __init__(self, typ, value):
        self.typ = getattr(CondType, typ.upper().strip())
        self.value = set(value if isinstance(value, (list, tuple)) else (value,))

    def test(self, state):
        match self.typ:
            case CondType.FROM_PERCENT:
                pct = state[CondType.PERCENT]
                result = all(pct >= v for v in self.value)
            case CondType.TO_PERCENT:
                pct = state[CondType.PERCENT]
                result = all(pct <= v for v in self.value)
            case CondType.FROM_STEP:
                step = state[CondType.STEP]
                result = step > 0 and all(step >= v for v in self.value)
            case CondType.TO_STEP:
                step = state[CondType.STEP]
                result = step > 0 and all(step <= v for v in self.value)
            case CondType.STEP_INTERVAL:
                step = state[CondType.STEP]
                result = step > 0 and all(step % v == 0 for v in self.value)
            case _:
                result = state[self.typ] in self.value
        return result

    def __repr__(self):
        return f"<Cond({self.typ}): {self.value}>"


class BlockConds:
    def __init__(self, conds):
        if not conds:
            self.conds = ()
            return
        if isinstance(conds, dict):
            conds = tuple(conds.items())
        self.conds = tuple(BlockCond(ct, cv) for ct, cv in conds)

    def test(self, state):
        return all(c.test(state) for c in self.conds)

    def __repr__(self):
        return f"<Conds[{self.count}]: {self.conds}"


# Copied from https://github.com/WASasquatch/FreeU_Advanced
def hidden_mean(h):
    hidden_mean = h.mean(1).unsqueeze(1)
    b = hidden_mean.shape[0]
    hidden_max, _ = torch.max(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    hidden_min, _ = torch.min(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    return (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        hidden_max - hidden_min
    ).unsqueeze(2).unsqueeze(3)


class BlockOp:
    IDX = 0

    def __init__(self, typ, *args):
        self.typ = getattr(OpType, typ.upper().strip())
        self.args = args

    def eval(self, state):
        t = out = state[state["target"]]
        match self.typ:
            case OpType.SCALE_TORCH | OpType.UNSCALE_TORCH:
                if self.typ == OpType.SCALE_TORCH:
                    mode, scale_w, scale_h, antialias = self.args
                    width, height = (
                        round(t.shape[-1] * scale_w),
                        round(t.shape[-2] * scale_h),
                    )
                else:
                    hsp = state["hsp"]
                    if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                        return
                    mode, antialias = self.args
                    width, height = hsp.shape[-1], hsp.shape[-2]
                out = scale_samples(
                    t,
                    width,
                    height,
                    mode,
                    antialias_size=8 if antialias else 0,
                )
            case OpType.SCALE | OpType.UNSCALE:
                if self.typ == OpType.SCALE:
                    mode_w, mode_h, scale_w, scale_h, antialias_size = self.args
                    width, height = (
                        round(t.shape[-1] * scale_w),
                        round(t.shape[-2] * scale_h),
                    )
                else:
                    hsp = state["hsp"]
                    if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                        return
                    mode_w, mode_h, antialias_size = self.args
                    width, height = hsp.shape[-1], hsp.shape[-2]
                out = scale_samples(
                    t,
                    width,
                    height,
                    mode=mode_w,
                    mode_h=mode_h,
                    antialias_size=antialias_size,
                )
            case OpType.FLIP:
                out = torch.flip(
                    t,
                    dims=(2 if self.args[0] == "v" else 3,),
                )
            case OpType.ROT90:
                out = torch.rot90(t, self.args[0], dims=(3, 2))
            case OpType.ROLL_CHANNELS:
                out = torch.roll(t, self.args[0], dims=(1,))
            case OpType.ROLL:
                dims, amount = self.args
                if isinstance(dims, str):
                    match dims:
                        case "h" | "horizontal":
                            dims = (3,)
                        case "v" | "vertical":
                            dims = (2,)
                        case "c" | "channels":
                            dims = (1,)
                        case _:
                            raise ValueError("Bad roll direction")
                elif isinstance(dims, int):
                    dims = (dims,)
                if isinstance(amount, float) and amount < 1.0 and amount > -1.0:
                    if len(dims) > 1:
                        raise ValueError(
                            "Cannot use percentage based amount with multiple roll dimensions",
                        )
                    amount = int(t.shape[dims[0]] * amount)
                out = torch.roll(t, amount, dims=dims)
            case OpType.TARGET_SKIP:
                state["target"] = "hsp" if self.args[0] is True else "h"
                return
            case OpType.FFILTER:
                scale, filt, strength, threshold = self.args
                if isinstance(filt, str):
                    filt = FILTER_PRESETS[filt]
                out = ffilter(t, threshold, scale, filt, strength)
            case OpType.SLICE:
                scale, strength, blend, mode, use_hm = self.args
                slice_size = round(t.shape[1] * scale)
                sliced = t[:, :slice_size]
                if use_hm:
                    result = sliced * ((strength - 1) * hidden_mean(t) + 1)
                else:
                    result = sliced * strength
                if blend != 1:
                    result = BLENDING_MODES[mode](sliced, result, blend)
                out[:, :slice_size] = result
            case OpType.MULTIPLY:
                out *= self.args[0]
            case OpType.BLEND_OP:
                blend, mode, subops = self.args
                tempname = f"temp{BlockOp.IDX}"
                BlockOp.IDX += 1
                old_target = state["target"]
                state[tempname] = t.clone()
                state["target"] = tempname
                BlockOp(subops[0], *subops[1:]).eval(state)
                state["target"] = old_target
                out = BLENDING_MODES[mode](t, state[tempname], blend)
                del state[tempname]
            case OpType.ANTIALIAS:
                out = antialias_tensor(t, self.args[0])
            case OpType.NOISE:
                # mask = torch.ones(t.shape[2:], device=t.device, dtype=t.dtype)
                # ms = 32
                # mask[ms:-ms, :] = 0
                # mask[:, ms:-ms] = 0
                noise = torch.randn_like(t)  # * mask
                step_scale = state["sigma"] - state["sigma_next"]
                t += noise * step_scale * self.args[0]
            case OpType.DEBUG:
                stcopy = {
                    k: v for k, v in state.items() if not isinstance(v, torch.Tensor)
                }
                stcopy["target_shape"] = t.shape
                print(f">> BlehOps debug: {stcopy!r}")

            case _:
                raise ValueError("Unhandled")
        state[state["target"]] = out

    def __repr__(self):
        return f"<Op({self.typ}): {self.args!r}>"


class BlockRule:
    @classmethod
    def from_dict(cls, val):
        if not isinstance(val, (list, tuple)):
            val = (val,)

        return tuple(
            cls(
                conds=d.get("if", ()),
                ops=d.get("ops", ()),
                matched=d.get("then", ()),
                nomatched=d.get("else", ()),
            )
            for d in val
        )

    def __init__(self, conds=(), ops=(), matched=(), nomatched=()):
        self.conds = BlockConds(conds)
        self.ops = tuple(BlockOp(o[0], *o[1:]) for o in ops)
        self.matched = BlockRule.from_dict(matched)
        self.nomatched = BlockRule.from_dict(nomatched)

    def get_all_types(self):
        result = {c.value for c in self.conds if c.typ == CondType.TYPE}
        for r in self.matched:
            result |= r.get_all_types()
        for r in self.nomatched:
            result |= r.get_all_types()
        return result

    def eval(self, state):
        # print("EVAL", state | {"h": None, "hsp": None})
        if not self.conds.test(state):
            for r in self.nomatched:
                r.eval(state)
            return
        state["target"] = "h"
        for o in self.ops:
            o.eval(state)
        for r in self.matched:
            r.eval(state)

    def __repr__(self):
        return f"<Rule: IF({self.conds}) THEN({self.ops}, {self.matched}) ELSE({self.nomatched})>"


class BlockRules:
    @classmethod
    def from_yaml(cls, s):
        parsed_rules = yaml.safe_load(s)
        return cls(tuple(BlockRule.from_dict(r)[0] for r in parsed_rules))

    def __init__(self, rules):
        self.rules = rules

    def eval(self, state):
        for rule in self.rules:
            rule.eval(state)
        return state


class BlehBlockOps:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rules": ("STRING", {"multiline": True}),
            },
            "optional": {
                "sigmas_opt": ("SIGMAS",),
            },
        }

    def patch(
        self,
        model,
        rules,
        sigmas_opt=None,
    ):
        rules = rules.strip()
        if len(rules) == 0:
            return (model.clone(),)
        rules = BlockRules.from_yaml(rules)

        # Arbitrary number that should have good enough precision
        pct_steps = 400
        pct_incr = 1.0 / pct_steps
        sig2pct = tuple(
            model.model.model_sampling.percent_to_sigma(x / pct_steps)
            for x in range(pct_steps, -1, -1)
        )

        def get_pct(topts):
            sigma = topts["sigmas"][0].item()
            # This is obviously terrible but I couldn't find a better way to get the percentage from the current sigma.
            idx = bisect.bisect_right(sig2pct, sigma)
            if idx >= len(sig2pct):
                # Sigma out of range somehow?
                return None
            return pct_incr * (pct_steps - idx)

        def set_state_step(state, sigma):
            if sigmas_opt is None:
                state[CondType.STEP] = -1
                return st
            step_idx = torch.argmin(torch.abs(sigmas_opt - sigma)).item()
            state |= {
                CondType.STEP: step_idx + 1,
                "sigma": sigmas_opt[step_idx].item(),
                "sigma_next": sigmas_opt[step_idx + 1].item(),
            }
            return state

        stages = (1280, 640, 320)

        def make_state(typ, topts, h, hsp=None):
            pct = get_pct(topts)
            if pct is None:
                return None

            stage = stages.index(h.shape[1]) + 1 if h.shape[1] in stages else -1
            # print(">>", typ, topts["original_shape"], h.shape, stage, topts["block"])
            result = {
                CondType.TYPE: typ,
                CondType.PERCENT: pct,
                CondType.BLOCK: topts["block"][1],
                CondType.STAGE: stage,
                "h": h,
                "hsp": hsp,
                "target": "h",
            }
            set_state_step(result, topts["sigmas"].max().item())
            return result

        def block_patch(typ, h, topts):
            state = make_state(typ, topts, h)
            if state is None:
                return h
            return rules.eval(state)["h"]

        def output_block_patch(h, hsp, transformer_options):
            state = make_state("output", transformer_options, h, hsp)
            if state is None:
                return h
            rules.eval(state)
            return state["h"], state["hsp"]

        def post_cfg_patch(args):
            pct = get_pct({"sigmas": args["sigma"]})
            if pct is None:
                return None
            state = {
                CondType.TYPE: "post_cfg",
                CondType.PERCENT: pct,
                CondType.BLOCK: -1,
                CondType.STAGE: -1,
                "h": args["denoised"],
                "hsp": None,
                "target": "h",
            }
            set_state_step(state, args["sigma"].max().item())
            return rules.eval(state)["h"]

        m = model.clone()
        m.set_model_input_block_patch_after_skip(
            lambda *args: block_patch("input_after_skip", *args),
        )
        m.set_model_input_block_patch(lambda *args: block_patch("input", *args))
        m.set_model_patch(
            lambda *args: block_patch("middle", *args),
            "middle_block_patch",
        )
        m.set_model_output_block_patch(output_block_patch)
        m.set_model_sampler_post_cfg_function(
            post_cfg_patch,
            disable_cfg1_optimization=True,
        )
        return (m,)


class BlehLatentScaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "method_horizontal": (UPSCALE_METHODS,),
                "method_vertical": (("same", *UPSCALE_METHODS),),
                "scale_width": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
                "scale_height": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
                "antialias_size": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(
        self,
        samples,
        method_horizontal,
        method_vertical,
        scale_width,
        scale_height,
        antialias_size,
    ):
        if method_vertical == "same":
            method_vertical = method_horizontal
        samples = samples.copy()
        stensor = samples["samples"]
        width = round(stensor.shape[3] * scale_width)
        height = round(stensor.shape[2] * scale_height)
        samples["samples"] = scale_samples(
            stensor,
            width,
            height,
            method_horizontal,
            method_vertical,
            antialias_size=antialias_size,
        )
        return (samples,)


class BlehLatentOps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "rules": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(
        self,
        samples,
        rules,
    ):
        samples = samples.copy()
        rules = rules.strip()
        if len(rules) == 0:
            return (samples,)
        rules = BlockRules.from_yaml(rules)
        stensor = samples["samples"]
        state = {
            CondType.TYPE: "latent",
            CondType.PERCENT: 0.0,
            CondType.BLOCK: -1,
            CondType.STAGE: -1,
            "h": stensor,
            "hsp": None,
            "target": "h",
        }
        rules.eval(state)
        samples["samples"] = state["h"]
        return (samples,)
