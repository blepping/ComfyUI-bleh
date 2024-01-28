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
            self.btp_max_size = btp.get("max_size", 768)
            self.btp_max_batch = btp.get("max_batch", 4)
            self.btp_max_batch_cols = btp.get("max_batch_cols", 2)
            self.btp_throttle_secs = btp.get("throttle_secs", 1)
            self.btp_use_cuda = btp.get("use_cuda", True)

    def get_cfg_path(self, filename):
        my_path = Path.resolve(Path(__file__).parent)
        return my_path.parent / filename

    def try_update_from_json(self, filename):
        import json

        try:
            with Path.open(self.get_cfg_path(filename)) as fp:
                self.update(json.load(fp))
        except OSError:
            return False

    def try_update_from_yaml(self, filename):
        try:
            import yaml

            with Path.open(self.get_cfg_path(filename)) as fp:
                self.update(yaml.safe_load(fp))
        except (OSError, ImportError):
            return False


SETTINGS = Settings()


def load_settings():
    if not SETTINGS.try_update_from_yaml("blehconfig.yaml"):
        SETTINGS.try_update_from_json("blehconfig.json")
    return SETTINGS
