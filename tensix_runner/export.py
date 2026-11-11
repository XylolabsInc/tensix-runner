# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""A Xylolabs Tensix export on disk: a Hugging Face checkpoint whose linear layers already sit on
TT-NN's BFP4 or BFP8 grid, with `quant4.json` naming each projection's format.

The checkpoint's files are bf16 safetensors (sharded, with `model.safetensors.index.json`, or one
`model.safetensors`). Tensors are read one at a time and never held whole: `Export.get(name)`
opens each shard once and reads the tensor it is asked for."""

import hashlib
import json
from pathlib import Path

FORMATS = ("bf16", "bfp8", "bfp4")


class ExportError(ValueError):
    """The directory is not an export this runner can load."""


class Export:
    def __init__(self, root):
        self.root = Path(root).expanduser()
        cfg, man = self.root / "config.json", self.root / "quant4.json"
        for f in (cfg, man):
            if not f.exists():
                raise ExportError(f"{self.root}: no {f.name} (not a Xylolabs Tensix export)")
        self.config = json.loads(cfg.read_text())
        raw = man.read_bytes()
        self.manifest = json.loads(raw)
        self.sha1 = hashlib.sha1(raw).hexdigest()  # names the weight cache
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        elif (self.root / "model.safetensors").exists():
            from safetensors import safe_open

            with safe_open(self.root / "model.safetensors", "pt") as f:
                self.weight_map = dict.fromkeys(f.keys(), "model.safetensors")
        else:
            raise ExportError(f"{self.root}: no model.safetensors.index.json or model.safetensors")
        self._handles = {}
        self.formats = self._formats()
        self.prefix = self._prefix()

    @property
    def text_config(self):
        """The language model's config (multimodal checkpoints nest it as text_config)."""
        return self.config.get("text_config", self.config)

    @property
    def model_type(self):
        return self.text_config.get("model_type") or self.config.get("model_type")

    def _formats(self):
        """{(layer, projection): format} from quant4.json, e.g. (0, "self_attn.q_proj"): "bfp4"."""
        out = {}
        for layer, projs in self.manifest.get("layers", {}).items():
            for proj, info in projs.items():
                info = info if isinstance(info, dict) else {"format": info}
                fmt = info.get("format")
                if fmt not in FORMATS:
                    raise ExportError(
                        f"layer {layer} {proj}: format {fmt!r} is not one of {FORMATS}"
                    )
                if info.get("online"):
                    raise ExportError(
                        f"layer {layer} {proj}: an online rotation, which this runner does not apply"
                    )
                out[(int(layer), proj)] = fmt
        if not out:
            raise ExportError(f"{self.root}: quant4.json lists no layers")
        return out

    @property
    def head_format(self):
        """The LM head's format ("bf16" unless quant4.json says otherwise)."""
        fmt = self.manifest.get("lm_head", "bf16")
        if fmt not in FORMATS:
            raise ExportError(f"lm_head: format {fmt!r} is not one of {FORMATS}")
        return fmt

    def _prefix(self):
        """The text model's tensor prefix: "model." (Qwen3) or "model.language_model." (the
        multimodal checkpoints of Qwen3.5 and Gemma 4)."""
        layer, proj = min(self.formats)
        tail = f"layers.{layer}.{proj}.weight"
        hits = [k[: -len(tail)] for k in self.weight_map if k.endswith(tail) and "mtp." not in k]
        if len(hits) != 1:
            raise ExportError(f"{self.root}: {len(hits)} tensors end in {tail}")
        return hits[0]

    def name(self, layer, proj):
        """The tensor name of layer's projection weight."""
        return f"{self.prefix}layers.{layer}.{proj}.weight"

    @property
    def embed_name(self):
        return f"{self.prefix}embed_tokens.weight"

    @property
    def head_name(self):
        """lm_head.weight, or the embedding where the checkpoint ties its head to it."""
        return "lm_head.weight" if "lm_head.weight" in self.weight_map else self.embed_name

    def matrices(self, layers=None):
        """(layer, projection, format, tensor name) of every projection quant4.json lists, in order."""
        for (layer, proj), fmt in sorted(self.formats.items()):
            if layers is None or layer in layers:
                yield layer, proj, fmt, self.name(layer, proj)

    def _handle(self, name):
        from safetensors import safe_open

        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{self.root}: no tensor {name}")
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.root / shard, "pt")
        return self._handles[shard]

    def get(self, name):
        """One tensor, as stored."""
        return self._handle(name).get_tensor(name)

    def part(self, name):
        """A tensor to read in parts: indexing it (part(name)[:, a:b]) reads only those values."""
        return self._handle(name).get_slice(name)

    def close(self):
        self._handles.clear()
