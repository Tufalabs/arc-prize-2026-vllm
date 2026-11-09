# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm.utils.memory_log's sampler at start-up's end: it stops by default,
and with VLLM_MEMORY_LOG_SERVING it keeps sampling into serving."""

import threading

import pytest

from vllm.utils import memory_log


@pytest.fixture
def started(monkeypatch):
    """The module as a worker's sampler leaves it once started (no thread)."""
    monkeypatch.setenv(memory_log.ENV_INTERVAL, "15")
    monkeypatch.setattr(memory_log, "_started", True)
    monkeypatch.setattr(memory_log, "_stopped", threading.Event())
    monkeypatch.setattr(memory_log, "_device", None)


@pytest.mark.usefixtures("started")
def test_the_sampler_stops_at_startups_end_by_default(monkeypatch):
    monkeypatch.delenv(memory_log.ENV_SERVING, raising=False)
    memory_log.stop("compile_or_warm_up_model exit")
    assert memory_log._stopped.is_set()


@pytest.mark.usefixtures("started")
@pytest.mark.parametrize("value", ["1", "yes"])
def test_the_sampler_keeps_going_into_serving_when_asked(monkeypatch, value):
    monkeypatch.setenv(memory_log.ENV_SERVING, value)
    resets = []
    monkeypatch.setattr(memory_log, "_reset_device_peaks", lambda: resets.append(1))
    memory_log.stop("compile_or_warm_up_model exit")
    assert not memory_log._stopped.is_set()
    assert resets == [1]


@pytest.mark.usefixtures("started")
def test_zero_means_startup_only(monkeypatch):
    monkeypatch.setenv(memory_log.ENV_SERVING, "0")
    memory_log.stop("compile_or_warm_up_model exit")
    assert memory_log._stopped.is_set()
