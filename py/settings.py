from pathlib import Path


class Settings:
    def __init__(self):
        self.btp_enabled = False

    def update(self, obj):
        btp = obj.get("betterTaesdPreviews", None)
        self.btp_enabled = btp is not None and btp.get("enabled", True) is True
        if not self.btp_enabled:
            return
        max_size = max(8, btp.get("max_size", 768))
        self.btp_max_width = max(8, btp.get("max_width", max_size))
        self.btp_max_height = max(8, btp.get("max_height", max_size))
        self.btp_max_batch = max(1, btp.get("max_batch", 4))
        self.btp_max_batch_cols = max(1, btp.get("max_batch_cols", 2))
        self.btp_throttle_secs = btp.get("throttle_secs", 1)
        self.btp_throttle_secs_fallback = btp.get("throttle_secs_fallback")
        if self.btp_throttle_secs_fallback is None:
            self.btp_throttle_secs_fallback = self.btp_throttle_secs
        self.btp_skip_upscale_layers = btp.get("skip_upscale_layers", 0)
        self.btp_preview_device = btp.get("preview_device")
        # default, keep, float32, float16, bfloat16
        self.btp_preview_dtype = btp.get("preview_dtype")
        self.btp_maxed_batch_step_mode = btp.get("maxed_batch_step_mode", False)
        self.btp_compile_previewer = btp.get("compile_previewer", False)
        self.btp_oom_fallback = btp.get("oom_fallback", "latent2rgb")
        self.btp_oom_retry = btp.get("oom_retry", True)
        self.btp_whitelist = frozenset(btp.get("whitelist_formats", frozenset()))
        self.btp_blacklist = frozenset(btp.get("blacklist_formats", frozenset()))
        self.btp_video_parallel = btp.get("video_parallel", False)
        self.btp_video_max_frames = btp.get("video_max_frames", -1)
        self.btp_video_temporal_upscale_level = btp.get(
            "video_temporal_upscale_level",
            0,
        )
        self.btp_animate_preview = btp.get("animate_preview", "none")
        self.btp_verbose = btp.get("verbose", False)

    @staticmethod
    def get_cfg_path(filename) -> Path:
        my_path = Path.resolve(Path(__file__).parent)
        return my_path.parent / filename

    def try_update_from_json(self, filename):
        import json  # noqa: PLC0415

        try:
            with Path.open(self.get_cfg_path(filename)) as fp:
                self.update(json.load(fp))
                return True
        except OSError:
            return False

    def try_update_from_yaml(self, filename):
        try:
            import yaml  # noqa: PLC0415

            with Path.open(self.get_cfg_path(filename)) as fp:
                self.update(yaml.safe_load(fp))
                return True
        except (OSError, ImportError):
            return False


SETTINGS = Settings()


def load_settings():
    if not SETTINGS.try_update_from_yaml("blehconfig.yaml"):
        SETTINGS.try_update_from_json("blehconfig.json")
    return SETTINGS
