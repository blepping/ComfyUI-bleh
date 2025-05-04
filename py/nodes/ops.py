# Adapted from the ComfyUI built-in node
from __future__ import annotations

import bisect
import importlib
import operator as pyop
from collections import OrderedDict
from enum import Enum, auto
from itertools import starmap

import torch
import yaml

from ..latent_utils import *  # noqa: TID252

try:
    sonar_noise = importlib.import_module("custom_nodes.ComfyUI-sonar.py.noise")
    get_noise_sampler = sonar_noise.get_noise_sampler
except ImportError:

    def get_noise_sampler(noise_type, x, *_args: list, **_kwargs: dict):
        if noise_type != "gaussian":
            raise ValueError("Only gaussian noise supported")
        return lambda _s, _sn: torch.randn_like(x)


class CondType(Enum):
    TYPE = auto()
    BLOCK = auto()
    STAGE = auto()
    FROM_PERCENT = auto()
    TO_PERCENT = auto()
    PERCENT = auto()
    STEP = auto()  # Calculated from closest sigma.
    STEP_EXACT = auto()  # Only exact matching sigma or -1.
    FROM_STEP = auto()
    TO_STEP = auto()
    STEP_INTERVAL = auto()
    COND = auto()


class PatchType(Enum):
    LATENT = auto()
    INPUT = auto()
    INPUT_AFTER_SKIP = auto()
    MIDDLE = auto()
    OUTPUT = auto()
    POST_CFG = auto()
    PRE_APPLY_MODEL = auto()
    POST_APPLY_MODEL = auto()


class CompareType(Enum):
    EQ = auto()
    NE = auto()
    GT = auto()
    LT = auto()
    GE = auto()
    LE = auto()
    NOT = auto()
    OR = auto()
    AND = auto()


class OpType(Enum):
    # scale, strength, blend, blend mode, use hidden mean, dim, scale offset
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

    # scale mode, antialias size, mask example, [op], blend_mode
    MASK_EXAMPLE_OP = auto()

    # size
    ANTIALIAS = auto()

    # scale, type, scale_mode (none, sigma, sigdiff)
    NOISE = auto()

    # none
    DEBUG = auto()

    # mode (constant, reflect, replicate, circular), top, bottom, left, right, constant
    PAD = auto()

    # top, bottom, left, right
    CROP = auto()

    # count, [ops]
    REPEAT = auto()

    # scale, type
    APPLY_ENHANCEMENT = auto()


OP_DEFAULTS = {
    OpType.SLICE: OrderedDict(
        scale=1.0,
        strength=1.0,
        blend=1.0,
        blend_mode="bislerp",
        use_hidden_mean=True,
        dim=1,
        scale_offset=0,
    ),
    OpType.FFILTER: OrderedDict(
        scale=1.0,
        filter="none",
        filter_strength=0.5,
        threshold=1,
    ),
    OpType.SCALE_TORCH: OrderedDict(
        type="bicubic",
        scale_width=1.0,
        scale_height=None,
        antialias=False,
    ),
    OpType.SCALE: OrderedDict(
        type_width="bicubic",
        type_height="bicubic",
        scale_width=1.0,
        scale_height=None,
        antialias_size=0,
    ),
    OpType.UNSCALE_TORCH: OrderedDict(
        type="bicubic",
        antialias=False,
    ),
    OpType.UNSCALE: OrderedDict(
        type_width="bicubic",
        type_height="bicubic",
        antialias_size=0,
    ),
    OpType.FLIP: OrderedDict(direction="h"),
    OpType.ROT90: OrderedDict(count=1),
    OpType.ROLL_CHANNELS: OrderedDict(count=1),
    OpType.ROLL: OrderedDict(direction="c", amount=1),
    OpType.TARGET_SKIP: OrderedDict(active=True),
    OpType.MULTIPLY: OrderedDict(factor=1.0),
    OpType.BLEND_OP: OrderedDict(blend=1.0, blend_mode="bislerp", ops=()),
    OpType.MASK_EXAMPLE_OP: OrderedDict(
        scale_mode="bicubic",
        antialias_size=7,
        mask=(
            (0.5, 0.25, (16, 0.0), 0.25, 0.5),
            ("rep", 18, (20, 0.0)),
            (0.5, 0.25, (16, 0.0), 0.25, 0.5),
        ),
        ops=(),
        blend_mode="lerp",
    ),
    OpType.ANTIALIAS: OrderedDict(size=7),
    OpType.NOISE: OrderedDict(scale=0.5, type="gaussian", scale_mode="sigdiff"),
    OpType.DEBUG: OrderedDict(),
    OpType.PAD: OrderedDict(
        mode="reflect",
        top=0,
        bottom=0,
        left=0,
        right=0,
        constant=None,
    ),
    OpType.CROP: OrderedDict(top=0, bottom=0, left=0, right=0),
    OpType.REPEAT: OrderedDict(count=2, ops=()),
    OpType.APPLY_ENHANCEMENT: OrderedDict(scale=1.0, type="korniabilateralblur"),
}


class Compare:
    VALID_TYPES = {  # noqa: RUF012
        CondType.BLOCK,
        CondType.STAGE,
        CondType.PERCENT,
        CondType.STEP,
        CondType.STEP_EXACT,
    }

    def __init__(self, typ: str, value):
        self.typ = getattr(CompareType, typ.upper().strip())
        if self.typ in {CompareType.OR, CompareType.AND, CompareType.NOT}:
            self.value = tuple(ConditionGroup(v) for v in value)
            self.field = None
            return
        self.field = getattr(CondType, value[0].upper().strip())
        if self.field not in self.VALID_TYPES:
            raise TypeError("Invalid type compare operation")
        self.opfn = getattr(pyop, self.typ.name.lower())
        self.value = value[1:]
        if not isinstance(self.value, (list, tuple)):
            self.value = (self.value,)

    def test(self, state: dict) -> bool:
        if self.typ == CompareType.NOT:
            return all(not v.test(state) for v in self.value)
        if self.typ == CompareType.AND:
            return all(v.test(state) for v in self.value)
        if self.typ == CompareType.OR:
            return any(v.test(state) for v in self.value)
        opfn, fieldval = self.opfn, state[self.field]
        return all(opfn(fieldval, val) for val in self.value)

    def __repr__(self) -> str:
        return f"<Compare({self.typ}): {self.field}, {self.value}>"


class Condition:
    def __init__(self, typ: str, value):
        self.typ = getattr(CondType, typ.upper().strip())
        if self.typ == CondType.TYPE:
            if not isinstance(value, (list, tuple)):
                value = (value,)
            self.value = {getattr(PatchType, pt.strip().upper()) for pt in value}
        elif self.typ is not CondType.COND:
            self.value = set(value if isinstance(value, (list, tuple)) else (value,))
        else:
            self.value = Compare(value[0], value[1:])

    def test(self, state: dict) -> bool:
        if self.typ == CondType.FROM_PERCENT:
            pct = state[CondType.PERCENT]
            result = all(pct >= v for v in self.value)
        elif self.typ == CondType.TO_PERCENT:
            pct = state[CondType.PERCENT]
            result = all(pct <= v for v in self.value)
        elif self.typ == CondType.FROM_STEP:
            step = state[CondType.STEP]
            result = step > 0 and all(step >= v for v in self.value)
        elif self.typ == CondType.TO_STEP:
            step = state[CondType.STEP]
            result = step > 0 and all(step <= v for v in self.value)
        elif self.typ == CondType.STEP_INTERVAL:
            step = state[CondType.STEP]
            result = step > 0 and all(step % v == 0 for v in self.value)
        elif self.typ == CondType.COND:
            result = self.value.test(state)
        else:
            result = state[self.typ] in self.value
        return result

    def __repr__(self) -> str:
        return f"<Cond({self.typ}): {self.value}>"


class ConditionGroup:
    def __init__(self, conds):
        if not conds:
            self.conds = ()
            return
        if isinstance(conds, dict):
            conds = tuple(conds.items())
        if isinstance(conds[0], str):
            conds = (conds,)
        self.conds = tuple(starmap(Condition, conds))

    def test(self, state: dict) -> bool:
        return all(c.test(state) for c in self.conds)

    def get_all_types(self) -> set[str]:
        pass

    def __repr__(self) -> str:
        return f"<ConditionGroup: {self.conds}>"


# Copied from https://github.com/WASasquatch/FreeU_Advanced
def hidden_mean(h):
    hidden_mean = h.mean(1).unsqueeze(1)
    b = hidden_mean.shape[0]
    hidden_max, _ = torch.max(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    hidden_min, _ = torch.min(hidden_mean.view(b, -1), dim=-1, keepdim=True)
    return (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        hidden_max - hidden_min
    ).unsqueeze(2).unsqueeze(3)


class Operation:
    IDX = 0

    def __init__(self, typ: str | OpType, *args: list):
        if isinstance(typ, str):
            typ = getattr(OpType, typ.upper().strip())
        self.typ = typ
        defaults = OP_DEFAULTS[self.typ]
        if len(args) == 1 and isinstance(args[0], dict):
            args = args[0]
            extra = set(args.keys()) - set(defaults.keys())
            if extra:
                errstr = f"Unexpected argument keys for operation {typ}: {extra}"
                raise ValueError(errstr)
            self.args = tuple(starmap(args.get, defaults.items()))
        else:
            if len(args) > len(defaults):
                raise ValueError("Too many arguments for operation")
            self.args = (*args, *tuple(defaults.values())[len(args) :])

    @staticmethod
    def build(typ: str | OpType, *args: list) -> object:
        if isinstance(typ, str):
            typ = getattr(OpType, typ.upper().strip())
        return OP_TO_OPCLASS[typ](typ, *args)

    def eval(self, state: dict):
        out = self.op(state[state["target"]], state)
        state[state["target"]] = out

    def __repr__(self) -> str:
        return f"<Operation({self.typ}): {self.args!r}>"


class SubOpsOperation(Operation):
    SUBOPS_IDXS = ()

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        for argidx in self.SUBOPS_IDXS:
            subops = self.args[argidx]
            if subops and isinstance(subops[0], str):
                # Simple single subop.
                subops = (subops,)
            compiled_subops = []
            for idx in range(len(subops)):
                subop = subops[idx]
                if isinstance(subop, dict):
                    # Compile to rule.
                    subop = subops[idx] = Rule.from_dict(subops[idx])
                elif isinstance(subop, (list, tuple)):
                    # Compile to op.
                    subop = Operation.build(subop[0], *subop[1:])
                compiled_subops.append(subop)
            temp = list(self.args)
            temp[argidx] = compiled_subops
            self.args = tuple(temp)


class OpSlice(Operation):
    def op(self, t, _state):
        out = t
        scale, strength, blend, mode, use_hm, dim, scale_offset = self.args
        if dim < 0:
            dim = t.ndim + dim
        dim_size = t.shape[dim]
        slice_size = max(1, round(dim_size * scale))
        slice_offset = int(dim_size * scale_offset)
        slice_def = tuple(
            slice(None, None)
            if idx != dim
            else slice(slice_offset, slice_offset + slice_size)
            for idx in range(dim + 1)
        )
        sliced = t[slice_def]
        if use_hm:
            result = sliced * ((strength - 1) * hidden_mean(t)[slice_def] + 1)
        else:
            result = sliced * strength
        if blend != 1:
            result = BLENDING_MODES[mode](sliced, result, blend)
        out[slice_def] = result
        return out


class OpFFilter(Operation):
    def op(self, t, _state):
        scale, filt, strength, threshold = self.args
        if isinstance(filt, str):
            filt = FILTER_PRESETS[filt]
        elif filt is None:
            filt = ()
        return ffilter(t, threshold, scale, filt, strength)


class OpScaleTorch(Operation):
    def op(self, t, state):
        if self.typ == OpType.SCALE_TORCH:
            mode, scale_w, scale_h, antialias = self.args
            width, height = (
                round(t.shape[-1] * scale_w),
                round(t.shape[-2] * scale_h),
            )
        else:
            hsp = state["hsp"]
            if hsp is None:
                raise ValueError(
                    "Can only use unscale_torch when HSP is set (output)",
                )
            if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                return t
            mode, antialias = self.args
            width, height = hsp.shape[-1], hsp.shape[-2]
        return scale_samples(
            t,
            width,
            height,
            mode,
            antialias_size=8 if antialias else 0,
            sigma=state.get("sigma"),
        )


class OpUnscaleTorch(OpScaleTorch):
    pass


class OpScale(Operation):
    def op(self, t, state):
        if self.typ == OpType.SCALE:
            mode_w, mode_h, scale_w, scale_h, antialias_size = self.args
            width, height = (
                round(t.shape[-1] * scale_w),
                round(t.shape[-2] * scale_h),
            )
        else:
            hsp = state["hsp"]
            if hsp is None:
                raise ValueError(
                    "Can only use unscale when HSP is set (output)",
                )
            if t.shape[-1] == hsp.shape[-1] and t.shape[-2] == hsp.shape[-2]:
                return t
            mode_w, mode_h, antialias_size = self.args
            width, height = hsp.shape[-1], hsp.shape[-2]
        return scale_samples(
            t,
            width,
            height,
            mode=mode_w,
            mode_h=mode_h,
            antialias_size=antialias_size,
        )


class OpUnscale(OpScale):
    pass


class OpFlip(Operation):
    def op(self, t, _state):
        dimarg = self.args[0]
        if isinstance(dimarg, str):
            dim = dimarg[:1] == "v"
        elif isinstance(dimarg, int):
            dim = (dimarg,)
        else:
            dim = dimarg
        return torch.flip(t, dims=dim)


class OpRot90(Operation):
    def op(self, t, _state):
        return torch.rot90(t, self.args[0], dims=(3, 2))


class OpRollChannels(Operation):
    def op(self, t, _state):
        return torch.roll(t, self.args[0], dims=(1,))


class OpRoll(Operation):
    def op(self, t, _state):
        dims, amount = self.args
        if isinstance(dims, str):
            if dims in {"h", "horizontal"}:
                dims = (3,)
            elif dims in {"v", "vertical"}:
                dims = (2,)
            elif dims in {"c", "channels"}:
                dims = (1,)
            else:
                raise ValueError("Bad roll direction")
        elif isinstance(dims, int):
            dims = (dims,)
        if isinstance(amount, float) and amount < 1.0 and amount > -1.0:
            if len(dims) > 1:
                raise ValueError(
                    "Cannot use percentage based amount with multiple roll dimensions",
                )
            amount = int(t.shape[dims[0]] * amount)
        return torch.roll(t, amount, dims=dims)


class OpTargetSkip(Operation):
    def op(self, t, state):
        if state.get("hsp") is None:
            if state["target"] == "hsp":
                state["target"] = "h"
            return t
        state["target"] = "hsp" if self.args[0] is True else "h"
        return state[state["target"]]


class OpMultiply(Operation):
    def op(self, t, _state):
        return t.mul_(self.args[0])


class OpBlendOp(SubOpsOperation):
    SUBOPS_IDXS = (2,)

    def op(self, t, state):
        blend, mode, subops = self.args
        tempname = f"temp{Operation.IDX}"
        Operation.IDX += 1
        old_target = state["target"]
        state[tempname] = t.clone()
        for subop in subops:
            state["target"] = tempname
            subop.eval(state)
        state["target"] = old_target
        out = BLENDING_MODES[mode](t, state[tempname], blend)
        del state[tempname]
        return out


class OpMaskExampleOp(SubOpsOperation):
    SUBOPS_IDXS = (3,)

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        scale_mode, antialias_size, maskdef, subops, blend_mode = self.args
        blend_function = BLENDING_MODES.get(blend_mode)
        if blend_function is None:
            raise ValueError("Bad blend mode")
        mask = []
        for rowidx in range(len(maskdef)):
            repeats = 1
            rowdef = maskdef[rowidx]
            if rowdef and rowdef[0] == "rep":
                repeats = int(rowdef[1])
                rowdef = rowdef[2:]
            row = []
            for col in rowdef:
                if isinstance(col, (list, tuple)):
                    row += col[1:] * col[0]
                else:
                    row.append(col)
            mask += (row,) * repeats
        mask = torch.tensor(mask, dtype=torch.float32, device="cpu")
        self.args = (
            scale_mode,
            antialias_size,
            mask,
            subops,
            blend_function,
        )

    def op(self, t, state):
        scale_mode, antialias_size, mask, subops, blend_function = self.args
        mask = scale_samples(
            mask.view(1, 1, *mask.shape).to(t.device, dtype=t.dtype),
            t.shape[-1],
            t.shape[-2],
            mode=scale_mode,
            antialias_size=antialias_size,
        ).broadcast_to(t.shape)
        tempname = f"temp{Operation.IDX}"
        Operation.IDX += 1
        old_target = state["target"]
        state[tempname] = t.clone()
        for subop in subops:
            state["target"] = tempname
            subop.eval(state)
        state["target"] = old_target
        out = blend_function(t, state[tempname], mask)
        del state[tempname]
        return out


class OpAntialias(Operation):
    def op(self, t, _state):
        return antialias_tensor(t, self.args[0])


class OpNoise(Operation):
    def op(self, t, state):
        scale, noise_type, scale_mode = self.args
        if scale_mode == "sigma":
            step_scale = state.get("sigma", 1.0)
        elif scale_mode == "sigdiff":
            if "sigma" in state and "sigma_next" in state:
                step_scale = state["sigma"] - state["sigma_next"]
            else:
                step_scale = state.get("sigma", 1.0)
        else:
            step_scale = 1.0
        noise_sampler = get_noise_sampler(
            noise_type,
            t,
            state["sigma_min"],
            state["sigma_max"],
        )
        noise = noise_sampler(state.get("sigma"), state.get("sigma_next"))
        t += noise * step_scale * scale
        return t


class OpDebug(Operation):
    @classmethod
    def op(cls, t, state):
        stcopy = {
            k: v
            if not isinstance(v, torch.Tensor)
            else f"<Tensor: shape={v.shape}, dtype={v.dtype}>"
            for k, v in state.items()
        }
        stcopy["target_shape"] = t.shape
        print(f">> BlehOps debug: {stcopy!r}")
        return t


class OpPad(Operation):
    def op(self, t, _state):
        mode, top, bottom, left, right, constant_value = self.args
        if mode != "constant":
            constant_value = None
        shp = t.shape
        top, bottom = tuple(
            val if isinstance(val, int) else int(shp[-2] * val) for val in (top, bottom)
        )
        left, right = tuple(
            val if isinstance(val, int) else int(shp[-1] * val) for val in (left, right)
        )
        return torch.nn.functional.pad(
            t,
            (left, right, top, bottom),
            mode=mode,
            value=constant_value,
        )


class OpCrop(Operation):
    def op(self, t, _state):
        top, bottom, left, right = self.args
        shp = t.shape
        top, bottom = tuple(
            val if isinstance(val, int) else int(shp[-2] * val) for val in (top, bottom)
        )
        left, right = tuple(
            val if isinstance(val, int) else int(shp[-1] * val) for val in (left, right)
        )
        bottom, right = shp[-2] - bottom, shp[-1] - right
        return t[:, :, top:bottom, left:right]


class OpRepeat(SubOpsOperation):
    SUBOPS_IDXS = (1,)

    def op(self, _t, state):
        count, subops = self.args
        for _ in range(count):
            for subop in subops:
                subop.eval(state)
        return state[state["target"]]


class OpApplyEnhancement(Operation):
    def op(self, t, state):
        scale, typ = self.args
        return enhance_tensor(t, typ, scale=scale, sigma=state.get("sigma"))


OP_TO_OPCLASS = {
    OpType.SLICE: OpSlice,
    OpType.FFILTER: OpFFilter,
    OpType.SCALE_TORCH: OpScaleTorch,
    OpType.UNSCALE_TORCH: OpUnscaleTorch,
    OpType.SCALE: OpScale,
    OpType.UNSCALE: OpUnscale,
    OpType.FLIP: OpFlip,
    OpType.ROT90: OpRot90,
    OpType.ROLL_CHANNELS: OpRollChannels,
    OpType.ROLL: OpRoll,
    OpType.TARGET_SKIP: OpTargetSkip,
    OpType.MULTIPLY: OpMultiply,
    OpType.BLEND_OP: OpBlendOp,
    OpType.MASK_EXAMPLE_OP: OpMaskExampleOp,
    OpType.ANTIALIAS: OpAntialias,
    OpType.NOISE: OpNoise,
    OpType.DEBUG: OpDebug,
    OpType.PAD: OpPad,
    OpType.CROP: OpCrop,
    OpType.REPEAT: OpRepeat,
    OpType.APPLY_ENHANCEMENT: OpApplyEnhancement,
}


class Rule:
    @classmethod
    def from_dict(cls, val) -> object:
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
            if not d.get("disable")
        )

    def __init__(self, conds=(), ops=(), matched=(), nomatched=()):
        self.conds = ConditionGroup(conds)
        if ops and isinstance(ops[0], str):
            ops = (ops,)
        self.ops = tuple(Operation.build(o[0], *o[1:]) for o in ops)
        self.matched = Rule.from_dict(matched)
        self.nomatched = Rule.from_dict(nomatched)

    def get_all_types(self) -> set:
        result = {c.value for c in self.conds if c.typ == CondType.TYPE}
        for r in self.matched:
            result |= r.get_all_types()
        for r in self.nomatched:
            result |= r.get_all_types()
        return result

    def eval(self, state: dict) -> None:
        # print("EVAL", state | {"h": None, "hsp": None})

        if not self.conds.test(state):
            for r in self.nomatched:
                r.eval(state)
            return
        for o in self.ops:
            o.eval(state)
        for r in self.matched:
            r.eval(state)

    def __repr__(self):
        return f"<Rule: IF({self.conds}) THEN({self.ops}, {self.matched}) ELSE({self.nomatched})>"


class RuleGroup:
    @classmethod
    def from_yaml(cls, s: str) -> object:
        parsed_rules = yaml.safe_load(s)
        if parsed_rules is None:
            return cls(())
        return cls(tuple(r for rs in parsed_rules for r in Rule.from_dict(rs)))

    def __init__(self, rules):
        self.rules = rules

    def eval(self, state, toplevel=False):
        for rule in self.rules:
            if toplevel:
                state["target"] = "h"
            rule.eval(state)
        return state

    def __repr__(self) -> str:
        return f"<RuleGroup: {self.rules}>"


class BlehBlockOps:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "bleh/model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rules": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "sigmas_opt": ("SIGMAS",),
            },
        }

    @classmethod
    def patch(
        cls,
        model,
        rules: str,
        sigmas_opt: torch.Tensor | None = None,
    ):
        rules = rules.strip()
        if len(rules) == 0:
            return (model.clone(),)
        rules = RuleGroup.from_yaml(rules)
        # print("RULES", rules)

        # Arbitrary number that should have good enough precision
        pct_steps = 400
        pct_incr = 1.0 / pct_steps
        model_sampling = model.get_model_object("model_sampling")
        sig2pct = tuple(
            model_sampling.percent_to_sigma(x / pct_steps)
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
            sdict = {
                CondType.STEP: -1,
                CondType.STEP_EXACT: -1,
                "sigma": sigma,
                "sigma_min": model_sampling.sigma_min,
                "sigma_max": model_sampling.sigma_max,
            }
            if sigmas_opt is None:
                state |= sdict
                return state
            sigmadiff, idx = torch.min(torch.abs(sigmas_opt[:-1] - sigma), 0)
            idx = idx.item()
            state |= sdict | {
                CondType.STEP: idx + 1,
                CondType.STEP_EXACT: -1 if sigmadiff.item() > 1.5e-06 else idx + 1,
                "sigma_next": sigmas_opt[idx + 1].item(),
            }
            return state

        stages = (1280, 640, 320)

        def make_state(typ: PatchType, topts: dict, h, hsp=None):
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

        def block_patch(typ, h, topts: dict):
            state = make_state(typ, topts, h)
            if state is None:
                return h
            return rules.eval(state, toplevel=True)["h"]

        def output_block_patch(h, hsp, transformer_options: dict):
            state = make_state(PatchType.OUTPUT, transformer_options, h, hsp)
            if state is None:
                return h
            rules.eval(state, toplevel=True)
            return state["h"], state["hsp"]

        def post_cfg_patch(args: dict):
            pct = get_pct({"sigmas": args["sigma"]})
            if pct is None:
                return None
            state = {
                CondType.TYPE: PatchType.POST_CFG,
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
            lambda *args: block_patch(PatchType.INPUT_AFTER_SKIP, *args),
        )
        m.set_model_input_block_patch(lambda *args: block_patch(PatchType.INPUT, *args))
        m.set_model_patch(
            lambda *args: block_patch(PatchType.MIDDLE, *args),
            "middle_block_patch",
        )
        m.set_model_output_block_patch(output_block_patch)
        m.set_model_sampler_post_cfg_function(
            post_cfg_patch,
            disable_cfg1_optimization=True,
        )
        orig_model_function_wrapper = model.model_options.get("model_function_wrapper")

        def pre_model(state):
            state[CondType.TYPE] = PatchType.PRE_APPLY_MODEL
            return rules.eval(state, toplevel=True)["h"]

        def post_model(state, result):
            state[CondType.TYPE] = PatchType.POST_APPLY_MODEL
            state["target"] = "h"
            state["h"] = result
            return rules.eval(state, toplevel=True)["h"]

        def model_unet_function_wrapper(apply_model, args):
            pct = get_pct({"sigmas": args["timestep"]})
            if pct is None:
                return None
            state = {
                CondType.PERCENT: pct,
                CondType.BLOCK: -1,
                CondType.STAGE: -1,
                "h": args["input"],
                "hsp": None,
                "target": "h",
            }
            set_state_step(state, args["timestep"].max().item())
            x = pre_model(state)
            args = args | {"input": x}  # noqa: PLR6104
            if orig_model_function_wrapper is not None:
                result = orig_model_function_wrapper(apply_model, args)
            else:
                result = apply_model(args["input"], args["timestep"], **args["c"])
            return post_model(state, result)

        m.set_model_unet_function_wrapper(model_unet_function_wrapper)

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

    @classmethod
    def upscale(
        cls,
        *,
        samples: dict,
        method_horizontal: str,
        method_vertical: str,
        scale_width: float,
        scale_height: float,
        antialias_size: int,
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
            mode=method_horizontal,
            mode_h=method_vertical,
            antialias_size=antialias_size,
        )
        return (samples,)


class BlehLatentOps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "rules": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "samples_hsp": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent"

    @classmethod
    def go(
        cls,
        *,
        samples: dict,
        rules: str,
        samples_hsp: dict | None = None,
    ):
        samples = samples.copy()
        rules = rules.strip()
        if len(rules) == 0:
            return (samples,)
        rules = RuleGroup.from_yaml(rules)
        stensor = samples["samples"]
        state = {
            CondType.TYPE: PatchType.LATENT,
            CondType.PERCENT: 0.0,
            CondType.BLOCK: -1,
            CondType.STAGE: -1,
            "h": stensor,
            "hsp": None if samples_hsp is None else samples_hsp["samples"],
            "target": "h",
        }
        rules.eval(state, toplevel=True)
        return ({"samples": state["h"]},)


class BlehLatentBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples1": ("LATENT",),
                "samples2": ("LATENT",),
                "samples2_percent": ("FLOAT", {"default": 0.5}),
                "blend_mode": (tuple(BLENDING_MODES.keys()),),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent"

    @classmethod
    def go(
        cls,
        *,
        samples1: dict,
        samples2: dict,
        samples2_percent=0.5,
        blend_mode="lerp",
    ):
        a, b = samples1["samples"], samples2["samples"]
        blend_function = BLENDING_MODES[blend_mode]
        return ({"samples": blend_function(a, b, samples2_percent)},)
