# Modified from https://github.com/madebyollin/taehv/blob/main/taehv.py

# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import nn
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .base import VideoModelInfo

F = torch.nn.functional


class TWorkItem(NamedTuple):
    input_tensor: torch.Tensor
    block_index: int


def conv(
    n_in: int,
    n_out: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    **kwargs: Any,
) -> nn.Conv2d:
    return nn.Conv2d(
        n_in,
        n_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        **kwargs,
    )


class Clamp(nn.Module):
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return (x / 3.0).tanh_().mul_(3.0)


class MemBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        wide: bool = False,
    ):
        super().__init__()
        groups = max(1, n_out // 64) if wide else 1
        if wide:
            if n_out % groups != 0:
                errstr = f"Bad n_out {n_out} parameter for wide MemBlock, must be divisible by 64"
                raise ValueError(errstr)
            self.conv = nn.Sequential(
                conv(n_in * 2, n_out, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                conv(n_out, n_out, groups=groups),
                nn.ReLU(inplace=True),
                conv(n_in, n_out, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                conv(n_out, n_out, groups=groups),
            )
        else:
            self.conv = nn.Sequential(
                conv(n_in * 2, n_out),
                nn.ReLU(inplace=True),
                conv(n_out, n_out),
                nn.ReLU(inplace=True),
                conv(n_out, n_out),
            )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        result = self.conv(torch.cat((x, past), 1))
        result += self.skip(x)
        return self.act(result)


def make_memblocks(n: int, *, count: int = 3, **kwargs: Any) -> tuple[MemBlock, ...]:
    return tuple(MemBlock(n, n, **kwargs) for _ in range(count))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[-3:]
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        return self.conv(x).reshape(-1, *orig_shape[-3:])


class TAEVidContext:
    def __init__(self, model):
        self.model = model
        self.HANDLERS = {
            MemBlock: self.handle_memblock,
            TPool: self.handle_tpool,
            TGrow: self.handle_tgrow,
        }

    def reset(self, x: torch.Tensor) -> None:
        N, T, C, H, W = x.shape
        self.N, self.T = N, T
        self.work_queue = [
            TWorkItem(xt, 0)
            for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))
        ]
        self.mem = [None] * len(self.model)

    def handle_memblock(
        self,
        i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        mem = self.mem
        # mem blocks are simple since we're visiting the graph in causal order
        if mem[i] is None:
            xt_new = b(xt, torch.zeros_like(xt))
            mem[i] = xt
        else:
            xt_new = b(xt, mem[i])
            # inplace might reduce mysterious pytorch memory allocations? doesn't help though
            mem[i].copy_(xt)
        return (xt_new,)

    def handle_tpool(
        self,
        i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        mem = self.mem
        # pool blocks are miserable
        if mem[i] is None:
            mem[i] = []  # pool memory is itself a queue of inputs to pool
        mem[i].append(xt)
        if len(mem[i]) > b.stride:
            # pool mem is in invalid state, we should have pooled before this
            raise RuntimeError("Internal error: Invalid mem state")
        if len(mem[i]) < b.stride:
            # pool mem is not yet full, go back to processing the work queue
            return ()
        # pool mem is ready, run the pool block
        N, C, H, W = xt.shape
        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
        # reset the pool mem
        mem[i] = []
        return (xt,)

    def handle_tgrow(
        self,
        _i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        xt = b(xt)
        C, H, W = xt.shape[1:]
        return reversed(
            xt.view(self.N, b.stride * C, H, W).chunk(b.stride, 1),
        )

    @classmethod
    def handle_default(
        cls,
        _i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        return (b(xt),)

    def handle_block(self, i: int, xt: torch.Tensor, b: nn.Module) -> None:
        handler = self.HANDLERS.get(b.__class__, self.handle_default)
        for xt_new in handler(i, xt, b):
            self.work_queue.insert(0, TWorkItem(xt_new, i + 1))

    def apply(self, x: torch.Tensor, *, show_progress=False) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected 5 dimensional tensor")
        self.reset(x)
        out = []
        work_queue = self.work_queue
        model = self.model
        model_len = len(model)

        with tqdm(range(self.T), disable=not show_progress) as pbar:
            while work_queue:
                xt, i = work_queue.pop(0)
                if i == model_len:
                    # reached end of the graph, append result to output list
                    out.append(xt)
                    continue
                if i == 0:
                    # new source node consumed
                    pbar.update(1)
                self.handle_block(i, xt, model[i])
        return torch.stack(out, 1)


class TAEVidBase(nn.Module):
    temporal_upscale_blocks = 3
    spatial_upscale_blocks = 3
    _nf = (256, 128, 64, 64)

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        vmi: VideoModelInfo,
        image_channels: int = 3,
        device="cpu",
        encoder_time_downscale_level: int = 3,
        decoder_time_upscale_level: int = 3,
        decoder_space_upscale_level: int = 3,
    ):
        super().__init__()
        self.vmi = vmi
        self.image_channels = image_channels
        self.latent_channels = vmi.latent_format.latent_channels
        self.patch_size = vmi.patch_size
        encoder_time_downscale = self._get_encoder_flags(
            time_level=encoder_time_downscale_level,
        )
        decoder_time_upscale, decoder_space_upscale = self._get_decoder_flags(
            time_level=decoder_time_upscale_level,
            space_level=decoder_space_upscale_level,
        )
        encoder_strides = tuple(1 + int(flag) for flag in encoder_time_downscale)
        decoder_strides = tuple(1 + int(flag) for flag in decoder_time_upscale)
        decoder_scale_factors = tuple(1 + int(flag) for flag in decoder_space_upscale)
        self.encoder = self._build_encoder(strides=encoder_strides)
        self.decoder = self._build_decoder(
            strides=decoder_strides,
            scale_factors=decoder_scale_factors,
        )
        self.t_upscale = 2 ** sum(decoder_time_upscale)
        self.t_downscale = 2 ** sum(encoder_time_downscale)
        self.frames_to_trim = self.t_upscale - 1
        if checkpoint_path is None:
            return
        sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(self.patch_tgrow_layers(sd))

    def _get_decoder_flags(
        self,
        *,
        time_level: int = 3,
        space_level: int = 3,
    ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        decoder_time_upscale = tuple(i < time_level for i in range(3))
        decoder_space_upscale = tuple(i < space_level for i in range(3))
        return decoder_time_upscale, decoder_space_upscale

    def _get_encoder_flags(
        self,
        *,
        time_level: int = 3,
    ) -> tuple[bool, ...]:
        return tuple(i < time_level for i in range(3))

    def _build_decoder(
        self,
        *,
        strides: tuple[int, ...],
        scale_factors: tuple[int, ...],
        memblock_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        if memblock_kwargs is None:
            memblock_kwargs = {}
        n_f = self._nf
        return nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            *make_memblocks(n_f[0], **memblock_kwargs),
            nn.Upsample(scale_factor=scale_factors[0]),
            TGrow(n_f[0], strides[0]),
            conv(n_f[0], n_f[1], bias=False),
            *make_memblocks(n_f[1], **memblock_kwargs),
            nn.Upsample(scale_factor=scale_factors[1]),
            TGrow(n_f[1], strides[1]),
            conv(n_f[1], n_f[2], bias=False),
            *make_memblocks(n_f[2], **memblock_kwargs),
            nn.Upsample(scale_factor=scale_factors[2]),
            TGrow(n_f[2], strides[2]),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], self.image_channels * self.patch_size**2),
        )

    def _build_encoder(
        self,
        *,
        strides: tuple[int, ...],
        memblock_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        if memblock_kwargs is None:
            memblock_kwargs = {}
        return nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            nn.ReLU(inplace=True),
            TPool(64, strides[0]),
            conv(64, 64, stride=2, bias=False),
            *make_memblocks(64, **memblock_kwargs),
            TPool(64, strides[1]),
            conv(64, 64, stride=2, bias=False),
            *make_memblocks(64, **memblock_kwargs),
            TPool(64, strides[2]),
            conv(64, 64, stride=2, bias=False),
            *make_memblocks(64, **memblock_kwargs),
            conv(64, self.latent_channels),
        )

    def patch_tgrow_layers(self, sd: dict) -> dict:
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    @classmethod
    def apply_parallel(
        cls,
        x: torch.Tensor,
        model: nn.Module,
        *,
        show_progress=False,
    ) -> torch.Tensor:
        padding = (0, 0, 0, 0, 0, 0, 1, 0)
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress):
            if not isinstance(b, MemBlock):
                x = b(x)
                continue
            nt, c, h, w = x.shape
            t = nt // n
            mem = F.pad(x.reshape(n, t, c, h, w), padding, value=0)[:, :t].reshape(
                x.shape,
            )
            x = b(x, mem)
            del mem
        nt, c, h, w = x.shape
        t = nt // n
        return x.view(n, t, c, h, w)

    def apply(
        self,
        x: torch.Tensor,
        *,
        decode=True,
        parallel=True,
        show_progress=False,
    ) -> torch.Tensor:
        model = self.decoder if decode else self.encoder
        if not decode:
            if self.vmi.patch_size > 1:
                x = F.pixel_unshuffle(x, self.patch_size)
            # Pad handling copied from https://github.com/madebyollin
            if x.shape[1] % self.t_downscale != 0:
                # pad at end to multiple of self.t_downscale
                n_pad = self.t_downscale - x.shape[1] % self.t_downscale
                padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
                x = torch.cat((x, padding), 1)
        if parallel:
            result = self.apply_parallel(x, model, show_progress=show_progress)
        else:
            result = TAEVidContext(model).apply(x, show_progress=show_progress)
        return (
            result
            if not decode or self.vmi.patch_size < 2
            else F.pixel_shuffle(result, self.patch_size)
        )

    def decode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=True, **kwargs)[:, self.frames_to_trim :]

    def encode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class TAEVid(TAEVidBase):
    def _get_decoder_flags(
        self,
        *,
        time_level: int = 3,
        space_level: int = 3,
    ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        tu, su = super()._get_decoder_flags(
            time_level=time_level,
            space_level=space_level,
        )
        return (False, *tu[:2]), su

    def _get_encoder_flags(
        self,
        *,
        time_level: int = 3,
    ) -> tuple[bool, ...]:
        return (*super()._get_encoder_flags(time_level=time_level)[:2], False)


class TAEVidLTX2(TAEVidBase):
    def _get_decoder_flags(
        self,
        *,
        time_level: int = 3,
        space_level: int = 3,
    ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        _tu, su = super()._get_decoder_flags(
            time_level=time_level,
            space_level=space_level,
        )
        return (True, True, True), su

    def _get_encoder_flags(
        self,
        *,
        time_level: int = 3,  # noqa: ARG002
    ) -> tuple[bool, ...]:
        return (True, True, True)


class TAEVidLTX23Wide(TAEVidLTX2):
    def __init__(self, *args: Any, **kwargs: Any):
        self._nf = (1024, 512, 256, 64)
        super().__init__(*args, **kwargs)

    def _build_decoder(
        self,
        *args: Any,
        memblock_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        memblock_kwargs = {} if memblock_kwargs is None else memblock_kwargs.copy()
        memblock_kwargs["wide"] = True
        return super()._build_decoder(*args, memblock_kwargs=memblock_kwargs, **kwargs)
