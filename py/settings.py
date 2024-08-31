from pathlib import Path


class Settings:
    def __init__(self):
        self.btp_enabled = False

    def update(self, obj):
        btp = obj.get("betterTaesdPreviews", None)
        if btp is None:
            self.btp_enabled = False
        else:
            self.btp_enabled = True
            max_size = max(8, btp.get("max_size", 768))
            self.btp_max_width = max(8, btp.get("max_width", max_size))
            self.btp_max_height = max(8, btp.get("max_height", max_size))
            self.btp_max_batch = max(1, btp.get("max_batch", 4))
            self.btp_max_batch_cols = max(1, btp.get("max_batch_cols", 2))
            self.btp_throttle_secs = btp.get("throttle_secs", 1)
            self.btp_skip_upscale_layers = btp.get("skip_upscale_layers", 0)
            self.btp_preview_device = btp.get("preview_device")
            self.btp_maxed_batch_step_mode = btp.get("maxed_batch_step_mode", False)

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
