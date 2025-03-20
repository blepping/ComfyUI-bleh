# Modified from https://github.com/madebyollin/taehv/blob/main/taehv.py

# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

F = torch.nn.functional


class TWorkItem(NamedTuple):
    input_tensor: torch.Tensor
    block_index: int


def conv(n_in: int, n_out: int, **kwargs: dict) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
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
        return self.act(self.conv(torch.cat((x, past), 1)) + self.skip(x))


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

    def apply(self, x: torch.Tensor, *, show_progress_bar=False) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected 5 dimensional tensor")
        self.reset(x)
        out = []
        work_queue = self.work_queue
        model = self.model
        model_len = len(model)

        with tqdm(range(self.T), disable=not show_progress_bar) as pbar:
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


class TAEVid(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        latent_channels: int,
        image_channels: int = 3,
        device="cpu",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.encoder = nn.Sequential(
            conv(image_channels, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            conv(64, latent_channels),
        )
        n_f = (256, 128, 64, 64)
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], image_channels),
        )
        if checkpoint_path is None:
            return
        self.load_state_dict(
            self.patch_tgrow_layers(
                torch.load(checkpoint_path, map_location=device, weights_only=True),
            ),
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
        show_progress_bar=False,
    ) -> torch.Tensor:
        padding = (0, 0, 0, 0, 0, 0, 1, 0)
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
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

    def decode(self, x: torch.Tensor, *, parallel=True) -> torch.Tensor:
        if parallel:
            result = self.apply_parallel(x, self.decoder)
        else:
            result = TAEVidContext(self.decoder).apply(x)
        return result[:, self.frames_to_trim :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)
