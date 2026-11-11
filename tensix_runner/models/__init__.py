# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The model families, by the checkpoint's model_type."""

FAMILIES = {"qwen3": "qwen3"}


def load(export, device, **kw):
    """The model for an export's model_type."""
    mt = export.model_type
    if mt not in FAMILIES:
        raise NotImplementedError(f"model_type {mt!r}: not supported yet ({sorted(FAMILIES)})")
    if FAMILIES[mt] == "qwen3":
        from .qwen3 import Qwen3

        return Qwen3(export, device, **kw)
    raise AssertionError(mt)
