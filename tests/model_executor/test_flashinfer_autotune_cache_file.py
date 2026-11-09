# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FLASHINFER_AUTOTUNE_CACHE_FILE names the FlashInfer autotune cache
file itself, so engines whose caller keys it by their settings share it: it
is used as given, whatever the config's hash, and it is no compile factor."""

from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
)


def test_the_named_file_is_used_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "shared" / "b200-settings.json"
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_FILE", str(target))
    # The config's hash is never asked for: no runner needed.
    assert resolve_flashinfer_autotune_file(None) == target  # type: ignore[arg-type]
    assert target.parent.is_dir()


def test_the_named_file_is_no_compile_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_FILE", "/data/a.json")
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_FILE" not in envs.compile_factors()
