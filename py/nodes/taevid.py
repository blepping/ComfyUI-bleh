import torch  # noqa: I001

import folder_paths
from comfy import model_management

from ..better_previews.previewer import VIDEO_FORMATS  # noqa: TID252
from ..better_previews.tae_vid import TAEVid  # noqa: TID252


class TAEVideoNodeBase:
    FUNCTION = "go"
    CATEGORY = "latent"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent_type": (("wan21", "hunyuanvideo", "mochi"),),
                "parallel_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Parallel mode is faster but requires more memory.",
                    },
                ),
            },
        }

    @classmethod
    def get_taevid_model(
        cls,
        latent_type: str,
    ) -> tuple[TAEVid, torch.device, torch.dtype]:
        vmi = VIDEO_FORMATS.get(latent_type)
        if vmi is None or vmi.tae_model is None:
            raise ValueError("Bad latent type")
        tae_model_path = folder_paths.get_full_path("vae_approx", vmi.tae_model)
        if tae_model_path is None:
            if latent_type == "wan21":
                model_src = "taew2_1.pth from https://github.com/madebyollin/taehv"
            elif latent_type == "hunyuanvideo":
                model_src = "taehv.pth from https://github.com/madebyollin/taehv"
            else:
                model_src = "taem1.pth from https://github.com/madebyollin/taem1"
            err_string = f"Missing TAE video model. Download {model_src} and place it in the models/vae_approx directory"
            raise RuntimeError(err_string)
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device=device)
        return (
            TAEVid(
                checkpoint_path=tae_model_path,
                latent_channels=vmi.latent_format.latent_channels,
                device=device,
            ).to(device),
            device,
            dtype,
        )

    @classmethod
    def go(cls, *, latent, latent_type: str, parallel_mode: bool) -> tuple:
        pass


class TAEVideoDecode(TAEVideoNodeBase):
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "latent"
    DESCRIPTION = "Fast decoding of Wan, Hunyuan and Mochi video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "latent": ("LATENT",),
        }
        return result

    @classmethod
    def go(cls, *, latent: dict, latent_type: str, parallel_mode: bool) -> tuple:
        model, device, dtype = cls.get_taevid_model(latent_type)
        samples = latent["samples"].detach().to(device=device, dtype=dtype, copy=True)
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
    DESCRIPTION = "Fast encoding of Wan, Hunyuan and Mochi video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "image": ("IMAGE",),
        }
        return result

    @classmethod
    def go(cls, *, image: torch.Tensor, latent_type: str, parallel_mode: bool) -> tuple:
        model, device, dtype = cls.get_taevid_model(latent_type)
        image = image.detach().to(device=device, dtype=dtype, copy=True)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        latent = (
            model.encode(
                image.movedim(-1, 2),
                parallel=parallel_mode,
                show_progress=True,
            )
            .transpose(1, 2)
            .to(
                dtype=torch.float,
                device="cpu",
            )
        )
        return ({"samples": latent},)
