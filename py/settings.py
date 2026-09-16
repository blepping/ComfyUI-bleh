from __future__ import annotations

import contextlib
import json
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import yaml

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from typing import Self


class _Empty:
    pass


class Blenum(Enum):
    @classmethod
    def build(cls, val: str | "Self") -> "Self":
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            errstr = f"Bad input type for enum value builder, expected a string or instance of {cls.__name__} but got {val}"
            raise TypeError(errstr)
        result = getattr(cls, val.strip().upper(), None)
        if result is not None:
            return result
        pretty_vals = ", ".join(eitem.name.lower() for eitem in cls)
        errstr = f"Value {val!r} is invalid for type {cls.__name__}, valid values (case-insensitive): {pretty_vals}"
        raise ValueError(errstr)


class OomFallback(Blenum):
    NONE = auto()
    LATENT2RGB = auto()


class PreviewDtype(Blenum):
    KEEP = auto()
    VAE = auto()
    FLOAT32 = auto()
    BFLOAT16 = auto()
    FLOAT16 = auto()
    FLOAT64 = auto()

    @property
    def as_dtype(self) -> torch.dtype | None:
        pdt = PreviewDtype
        if self is pdt.VAE:
            raise ValueError("Cannot convert VAE enum to dtype")
        return {
            pdt.FLOAT64: torch.float64,
            pdt.FLOAT32: torch.float32,
            pdt.FLOAT16: torch.float16,
            pdt.BFLOAT16: torch.bfloat16,
        }.get(self)


class AnimatePreview(Blenum):
    NONE = auto()
    VIDEO = auto()
    BATCH = auto()
    BOTH = auto()


class PreviewSettings(NamedTuple):
    enabled: bool = True
    verbose: bool = False
    max_width: int = 768
    max_height: int = 768
    max_batch: int = 4
    max_batch_cols: int = 2
    throttle_secs: float = 1
    throttle_secs_fallback: float | None = None
    throttle_secs_video: float | None = 10
    maxed_batch_step_mode: bool = False
    preview_device: str | None = None
    preview_dtype: PreviewDtype = PreviewDtype.BFLOAT16
    preview_non_blocking: bool = False
    skip_upscale_layers: int = 0
    compile_previewer: bool | dict = False
    oom_fallback: OomFallback = OomFallback.LATENT2RGB
    oom_retry: bool = True
    whitelist_formats: frozenset[str] = frozenset()
    blacklist_formats: frozenset[str] = frozenset()
    video_parallel: bool = False
    video_max_frames: int = -1
    video_temporal_upscale_level: int = 0
    animate_preview: AnimatePreview = AnimatePreview.VIDEO
    publish_last_preview: bool = False
    publish_last_preview_min_refresh: float = 5
    only_animate_last_preview: bool = True
    preview_interval: int = 1
    preview_offset: int = 0

    def get_throttle(self, *, video: bool = False, fallback: bool = False) -> float:
        if fallback and self.throttle_secs_fallback is not None:
            return self.throttle_secs_fallback
        if video and self.throttle_secs_video is not None:
            return self.throttle_secs_video
        return self.throttle_secs

    @classmethod
    def handle_complex_field(
        cls,
        *,
        key: str,
        field_type: type,
        args: dict,
    ) -> dict:
        val = args.get(key, _Empty)
        if val is not _Empty and not isinstance(val, field_type):
            args[key] = getattr(field_type, "build", field_type)(val)
        return args

    @classmethod
    def build(cls, **kwargs: Any) -> "Self":
        if kwargs.get("preview_dtype", _Empty) is None:
            del kwargs["preview_dtype"]
        for k, dv in cls._field_defaults.items():
            if isinstance(dv, (Blenum, frozenset, tuple)):
                kwargs = cls.handle_complex_field(
                    key=k,
                    field_type=dv.__class__,
                    args=kwargs,
                )
        if (max_size := kwargs.pop("max_size", None)) is not None:
            for k in ("max_width", "max_height"):
                if k not in kwargs:
                    kwargs[k] = max_size
        fs = frozenset(cls._fields)
        kwargs = {k: v for k, v in kwargs.items() if k in fs}
        min_vals = {
            "max_width": 8,
            "max_height": 8,
            "max_batch": 1,
            "max_batch_cols": 1,
        }
        for k, mv in min_vals.items():
            v = kwargs.get(k)
            if isinstance(v, (int, float)):
                kwargs[k] = mv.__class__(max(mv, v))
        return cls(**kwargs)


class Settings(NamedTuple):
    previews: PreviewSettings = PreviewSettings()

    @staticmethod
    def get_config_path(filename: str | Path) -> Path:
        my_path = Path.resolve(Path(__file__).parent)
        return my_path.parent / filename

    @staticmethod
    def load_config_object(base_name: str) -> dict | None:
        base_path = Path.resolve(Path(__file__).parent.parent)
        for ext in ("yaml", "json"):
            filename = base_path / f"{base_name}.{ext}"
            loader = yaml.safe_load if ext.startswith("y") else json.load
            try:
                with Path.open(filename) as fp:
                    loaded = loader(fp)
            except OSError:
                continue
            if loaded is None or isinstance(loaded, dict):
                return loaded
            errstr = f"YAML or JSON config file must be an object if present, got type {type(loaded)}"
            raise TypeError(errstr)
        return None

    @classmethod
    def load(cls, base_name: str = "blehconfig") -> "Self" | None:
        loaded = cls.load_config_object(base_name)
        return cls.build(**loaded) if loaded is not None else None

    @classmethod
    def build(cls, **kwargs: Any) -> "Self":
        btp = kwargs.get("previews") or kwargs.get("betterTaesdPreviews")
        if not btp:
            return cls()
        if not isinstance(btp, dict):
            errstr = f"Configuration previews or betterTaesdPreviews (deprecated) key must be an object or unset, {type(btp)} is invalid."
            raise TypeError(errstr)
        return cls(previews=PreviewSettings.build(**btp))


SETTINGS = Settings()


def load_settings() -> Settings | None:
    global SETTINGS  # noqa: PLW0603

    new_settings = Settings.load()
    if new_settings is not None:
        SETTINGS = new_settings
    return new_settings
