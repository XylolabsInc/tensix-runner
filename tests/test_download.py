# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Download layout and command line checks with Hugging Face calls stubbed."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest

from tensix_runner import cli


@pytest.mark.parametrize(
    ("files", "variant", "patterns", "folder"),
    [
        (["README.md", "quant4.json"], None, None, None),
        (["quant4.json", "bfp4/quant4.json"], None, None, None),
        (["bfp4/quant4.json", "bfp8/quant4.json"], "bfp4", ["bfp4/*"], "bfp4"),
        (["bfp4/quant4.json", "bfp8/quant4.json"], "bfp4/", ["bfp4/*"], "bfp4"),
        (["bfp8/quant4.json", "archive/old/quant4.json"], None, ["bfp8/*"], "bfp8"),
    ],
)
def test_download_layout(files, variant, patterns, folder):
    assert cli._download_layout(files, variant) == (patterns, folder)


@pytest.mark.parametrize(
    ("files", "variant", "message"),
    [
        (
            ["bfp8/quant4.json", "bfp4/quant4.json"],
            None,
            "Choose an export folder with --variant (bfp4, bfp8).",
        ),
        (
            ["bfp8/quant4.json", "bfp4/quant4.json"],
            "bf16",
            "Unknown variant 'bf16'. Available export folders are bfp4, bfp8.",
        ),
        (["quant4.json"], "bfp4", "Unknown variant 'bfp4'. This repo has one export at its root."),
        (
            ["README.md", "bfp4/config.json"],
            None,
            "No export found. Expected quant4.json at the root or in a top-level folder.",
        ),
        (
            ["archive/bfp4/quant4.json"],
            None,
            "No export found. Expected quant4.json at the root or in a top-level folder.",
        ),
    ],
)
def test_download_layout_errors(files, variant, message):
    with pytest.raises(SystemExit) as error:
        cli._download_layout(files, variant)
    assert str(error.value) == message


@pytest.fixture
def hub(monkeypatch, tmp_path):
    api = Mock()
    snapshot_download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return api.list_repo_files, snapshot_download


def test_download_root_snapshot(hub, tmp_path, capsys):
    list_repo_files, snapshot_download = hub
    list_repo_files.return_value = ["quant4.json"]

    assert cli.cmd_download(Namespace(name="model", repo=None, dir=None, variant=None)) == 0

    list_repo_files.assert_called_once_with("Xylolabs/model")
    snapshot_download.assert_called_once_with("Xylolabs/model", local_dir=Path("exports/model"))
    assert capsys.readouterr().out == f"Xylolabs/model -> {tmp_path}\n"


def test_download_folder_snapshot(hub, tmp_path, capsys):
    list_repo_files, snapshot_download = hub
    list_repo_files.return_value = ["bfp4/quant4.json", "bfp8/quant4.json"]
    args = Namespace(name="model", repo="org/other", dir="custom", variant="bfp4/")

    assert cli.cmd_download(args) == 0

    list_repo_files.assert_called_once_with("org/other")
    snapshot_download.assert_called_once_with(
        "org/other", local_dir=Path("custom"), allow_patterns=["bfp4/*"]
    )
    assert capsys.readouterr().out == f"org/other -> {tmp_path / 'bfp4'}\n"


def test_download_layout_error_skips_snapshot(hub):
    list_repo_files, snapshot_download = hub
    list_repo_files.return_value = ["bfp4/quant4.json", "bfp8/quant4.json"]

    with pytest.raises(SystemExit):
        cli.cmd_download(Namespace(name="model", repo=None, dir=None, variant=None))

    list_repo_files.assert_called_once_with("Xylolabs/model")
    snapshot_download.assert_not_called()


def test_download_cli_accepts_variant(monkeypatch):
    cmd_download = Mock(return_value=0)
    monkeypatch.setattr(cli, "cmd_download", cmd_download)

    assert cli.main(["download", "model", "--variant", "bfp4"]) == 0
    cmd_download.assert_called_once()
    assert cmd_download.call_args.args[0].variant == "bfp4"
