import sys

import nodes

from . import py
from .py import settings
from .py.nodes import samplers

BLEH_VERSION = 2


settings.load_settings()

from .py.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def blep_init():
    bi = sys.modules.get("_blepping_integrations", {})
    if "bleh" in bi:
        return
    bi["bleh"] = sys.modules[__name__]
    sys.modules["_blepping_integrations"] = bi
    nodes._blepping_integrations = bi  # noqa: SLF001
    samplers.add_sampler_presets()


blep_init()

__all__ = ("BLEH_VERSION", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "py")
