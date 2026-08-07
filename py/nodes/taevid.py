# ruff: noqa: TID252

import math

import folder_paths
import torch
from comfy import model_management

try:
    from comfy import nested_tensor
except (ImportError, ModuleNotFoundError):
    nested_tensor = None

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
        "minimaxh3video": ("taeh3.pth", "taehv"),
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
            vmi.tae_class(
                checkpoint_path=tae_model_path,
                vmi=vmi,
                device=device,
            ).to(device),
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
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            samples = samples.tensors[0]
        samples = samples.detach().to(device=device, dtype=dtype, copy=True)
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
        result["optional"] = {
            "audio_latent_opt": (
                "LATENT",
                {
                    "tooltip": "When connected, will associate an audio latent with the encoded video latent output. This is only useful for audio/video models like LTX and Minimax H3. You may attach a bundled A/V latent here, in which case only the audio part will be attached to the output of this node and the encoded video latent must match the shape of the video side. If that's confusing, another way to put it is if you attach an A/V latent then we know the shape a video latent has to have for that audio. If we encode video and it's a different shape, then we know the audio length can't be valid.\nIMPORTANT: There is no error checking, it will just package whatever you attach here into a nested tensor. If the latent is the wrong format, the length match doesn't exactly match your video length, etc you are going to have a bad time.",
                },
            ),
        }
        return result

    @classmethod
    def go(
        cls,
        *,
        image: torch.Tensor,
        latent_type: str,
        parallel_mode: bool,
        audio_latent_opt: dict | None = None,
    ) -> tuple:
        if audio_latent_opt is not None:
            if nested_tensor is None:
                raise NotImplementedError(
                    "Attaching an audio latent only works with recent ComfyUI versions that have the nested_tensor module. Your ComfyUI version is likely too old. You can still encode video.",
                )
            opt_samples = audio_latent_opt["samples"]
            if getattr(opt_samples, "is_nested"):
                if not hasattr(opt_samples, "tensors"):
                    raise ValueError(
                        "audio_latent_opt seems to be a nested tensor but is missing a tensors property. I can't deal with this!",
                    )
                if len(opt_samples.tensors) != 2:
                    raise ValueError(
                        "audio_latent_opt is a nested tensor but contains an unexpected number of items. Expected exactly two items (video, audio).",
                    )
                video_samples_opt, audio_samples_opt = opt_samples.tensors
            else:
                video_samples_opt = None
                audio_samples_opt = opt_samples
        else:
            audio_samples_opt = video_samples_opt = None

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
        if audio_samples_opt is None:
            return ({"samples": latent},)
        if video_samples_opt is not None and video_samples_opt.shape != latent.shape:
            errstr = f"audio_latent_opt was a bundled A/V latent. Our encoded video latent has shape {latent.shape} which doesn't match the shape of the reference ({video_samples_opt.shape}). The audio latent is the wrong length (shape {audio_samples_opt.shape}) and cannot be attached."
            raise ValueError(errstr)
        latent = nested_tensor.NestedTensor(
            (
                latent,
                audio_samples_opt.to(latent, copy=True),
            ),
        )
        return ({"samples": latent},)
