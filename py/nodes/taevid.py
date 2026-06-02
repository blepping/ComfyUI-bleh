# ruff: noqa: TID252

import math

import folder_paths
import torch
from comfy import model_management

from ..better_previews.previewer import VIDEO_FORMATS, VideoModelInfo
from ..better_previews.tae_vid import TAEVid


class TAEVideoNodeBase:
    FUNCTION = "go"
    CATEGORY = "latent"

    _download_map = {  # noqa: RUF012
        "hunyuanvideo": ("taehv.pth", "taehv"),
        "ltxv": ("taeltx_2.pth", "taehv"),
        "ltxv23": ("taeltx2_3.pth", "taehv"),
        "ltxv23wide": ("taeltx2_3_wide.pth", "taehv"),
        "mochi": ("taem1.pth", "taem1"),
        "wan21": ("taew2_1.pth", "taehv"),
        "wan22": ("taew2_2.pth", "taehv"),
    }

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent_type": (
                    tuple(cls._download_map),
                    {
                        "tooltip": "Use ltxv for LTX-2 AV.",
                    },
                ),
                "parallel_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Parallel mode may be faster but requires more memory.",
                    },
                ),
            },
        }

    @classmethod
    def get_taevid_model(
        cls,
        latent_type: str,
    ) -> tuple[TAEVid, torch.device, torch.dtype, VideoModelInfo]:
        vmi = VIDEO_FORMATS.get(latent_type)
        if vmi is None or vmi.tae_model is None:
            raise ValueError("Bad latent type")
        tae_model_path = folder_paths.get_full_path("vae_approx", vmi.tae_model)
        if tae_model_path is None:
            dl_info = cls._download_map.get(latent_type)
            if dl_info is None:
                err_string = (
                    f"Unexpected latent type {latent_type}, no information available"
                )
            else:
                filename, reponame = dl_info
                model_src = f"{filename} from https://github.com/madebyollin/{reponame}"
                err_string = f"Missing TAE video model. Download {model_src} and place it in the models/vae_approx directory"
            raise RuntimeError(err_string)
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device=device)
        return (
            vmi.tae_class(checkpoint_path=tae_model_path, vmi=vmi, device=device).to(device),
            device,
            dtype,
            vmi,
        )

    @classmethod
    def go(cls, *, latent, latent_type: str, parallel_mode: bool) -> tuple:
        raise NotImplementedError


class TAEVideoDecode(TAEVideoNodeBase):
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "latent"
    DESCRIPTION = "Fast decoding of Wan, Hunyuan, Mochi and LTX video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "latent": ("LATENT",),
        }
        return result

    @classmethod
    def go(cls, *, latent: dict, latent_type: str, parallel_mode: bool) -> tuple:
        model, device, dtype, vmi = cls.get_taevid_model(latent_type)
        samples = latent["samples"].detach().to(device=device, dtype=dtype, copy=True)
        samples = vmi.latent_format().process_in(samples)
        img = (
            model.decode(
                samples.transpose(1, 2),
                parallel=parallel_mode,
                show_progress=True,
            )
            .movedim(2, -1)
            .to(
                dtype=torch.float,
                device="cpu",
            )
        )
        img = img.reshape(-1, *img.shape[-3:])
        return (img,)


class TAEVideoEncode(TAEVideoNodeBase):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent"
    DESCRIPTION = "Fast encoding of Wan, Hunyuan, Mochi and LTX video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "image": ("IMAGE",),
        }
        return result

    @classmethod
    def go(cls, *, image: torch.Tensor, latent_type: str, parallel_mode: bool) -> tuple:
        model, device, dtype, vmi = cls.get_taevid_model(latent_type)
        image = image.detach().to(device=device, dtype=dtype, copy=True)
        if image.ndim < 5:
            image = image.unsqueeze(0)
        if image.ndim < 5:
            image = image.unsqueeze(0)
        if image.ndim != 5:
            raise ValueError("Unexpected input image dimensions")
        frames = image.shape[1]
        add_frames = (
            math.ceil(frames / vmi.temporal_compression) * vmi.temporal_compression
            - frames
        )
        if add_frames > 0:
            image = torch.cat(
                (
                    image,
                    image[:, frames - 1 :, ...].expand(
                        image.shape[0],
                        add_frames,
                        *image.shape[2:],
                    ),
                ),
                dim=1,
            )
        latent = model.encode(
            image[..., :3].movedim(-1, 2),
            parallel=parallel_mode,
            show_progress=True,
        ).transpose(1, 2)
        latent = (
            vmi.latent_format()
            .process_out(latent)
            .to(
                dtype=torch.float,
                device="cpu",
            )
        )
        return ({"samples": latent},)
