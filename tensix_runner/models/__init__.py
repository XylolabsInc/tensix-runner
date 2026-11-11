# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The model families, by the checkpoint's model_type."""

FAMILIES = {
    "qwen3": "qwen3",
    "qwen3_5_text": "qwen35",
    "qwen3_5": "qwen35",
    "gemma4_text": "gemma4",
    "gemma4": "gemma4",
}


def load(export, device, **kw):
    """The model for an export's model_type."""
    mt = export.model_type
    if mt not in FAMILIES:
        raise NotImplementedError(f"model_type {mt!r}: not supported yet ({sorted(FAMILIES)})")
    if FAMILIES[mt] == "qwen3":
        from .qwen3 import Qwen3

        return Qwen3(export, device, **kw)
    if FAMILIES[mt] == "qwen35":
        from .qwen35 import Qwen35

        return Qwen35(export, device, **kw)
    if FAMILIES[mt] == "gemma4":
        from .gemma4 import Gemma4

        return Gemma4(export, device, **kw)
    raise AssertionError(mt)
