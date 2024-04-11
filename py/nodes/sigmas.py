import torch


class DiscardPenultimateSigma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "sigmas": ("SIGMAS", {"forceInput": True}),
            },
        }

    FUNCTION = "go"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def go(self, enabled, sigmas):
        if not enabled or len(sigmas) < 2:
            return (sigmas,)
        return (torch.cat((sigmas[:-2], sigmas[-1:])),)
