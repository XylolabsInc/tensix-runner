# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Qwen3 on the device against transformers (TENSIX_RUNNER_EXPORT a Qwen3 export; needs ttnn with a
card or ttsim, and the ref extra): layer 0's prefill of 32 positions and one decode step on its
KV rows, each at PCC 0.99 or better. tensix-runner pcc runs the longer checks."""

import pytest

from tensix_runner.export import Export
from tensix_runner.models import load

pytestmark = [pytest.mark.device, pytest.mark.export]


def test_layer0(export_dir, device):
    pytest.importorskip("transformers")
    from tensix_runner import pcc

    ex = Export(export_dir)
    if ex.model_type != "qwen3":
        pytest.skip(f"{ex.model_type} is not a Qwen3 export")
    model = load(ex, device, max_seq=256, layers=[0], head=False)
    res = pcc.check_layers(model, pcc.prompt_tokens(ex, 33), 32, [0], log=lambda s: None)
    assert set(res) == {(0, "prefill"), (0, 32)}
    assert all(r >= 0.99 for r, _ in res.values()), res
