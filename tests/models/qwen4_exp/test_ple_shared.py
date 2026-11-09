# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host's shared Qwen4Exp PLE table (``ple_shared``): built from a
checkpoint's row shards in the engine's layout, waited for and mapped by each
engine, and checked against the engine's own checkpoint as it loads."""

import json
import mmap
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.common import ple_shared

_TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_DIM = 5


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """A safetensors file of uint8 tensors: header length, JSON header, data."""
    header, data, offset = {}, b"", 0
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes()
        header[name] = {
            "dtype": "U8",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def _checkpoint(tmp_path: Path, rows: list[int]) -> tuple[Path, torch.Tensor]:
    """A checkpoint whose PLE shards have ``rows`` rows each, split over two
    files beside an unrelated tensor, and the table they make."""
    generator = torch.Generator().manual_seed(len(rows))
    shards = [
        torch.randint(0, 256, (r, _DIM), dtype=torch.uint8, generator=generator)
        for r in rows
    ]
    names = [f"{_TABLE}.shard_{i}.weight" for i in range(len(shards))]
    half = len(shards) // 2
    files = {
        "model-1.safetensors": {
            **dict(zip(names[:half], shards[:half])),
            "lm_head.weight": torch.zeros(3, 3, dtype=torch.uint8),
        },
        "model-2.safetensors": dict(zip(names[half:], shards[half:])),
    }
    weight_map = {}
    for file_name, tensors in files.items():
        _write_safetensors(tmp_path / file_name, tensors)
        weight_map.update(dict.fromkeys(tensors, file_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tmp_path, torch.cat(shards)


def test_the_table_is_the_shards_in_the_engines_order(tmp_path: Path) -> None:
    checkpoint, expected = _checkpoint(tmp_path, [4, 4, 4, 3])
    table = tmp_path / "table"
    assert ple_shared.build_table(checkpoint, table) == expected.numel()
    built = torch.frombuffer(bytearray(table.read_bytes()), dtype=torch.uint8)
    assert torch.equal(built.view(-1, _DIM), expected)
    assert sorted(p.name for p in tmp_path.iterdir() if "partial" in p.name) == []


def test_shards_the_engine_would_place_elsewhere_are_refused(tmp_path: Path) -> None:
    # Shard i goes to row i * ceil(11 / 3) = 4i: a short first shard leaves a gap.
    checkpoint, _ = _checkpoint(tmp_path, [3, 4, 4])
    with pytest.raises(ValueError, match="even split"):
        ple_shared.build_table(checkpoint, tmp_path / "table")
    assert not (tmp_path / "table").exists()


def test_an_engine_waits_for_the_table_and_reads_it_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, expected = _checkpoint(tmp_path, [4, 4, 2])
    table = tmp_path / "table"
    monkeypatch.setattr(ple_shared, "_register_pinned", lambda weight, nbytes: None)
    builder = threading.Timer(0.5, ple_shared.build_table, (checkpoint, table))
    builder.start()
    started = time.monotonic()
    weight = ple_shared.map_table(table, expected.shape[0], _DIM, torch.uint8)
    builder.join()
    assert time.monotonic() - started >= 0.5
    assert torch.equal(weight, expected)
    assert ple_shared.mapped_by_any(table)
    with pytest.raises(ValueError, match="holds"):
        ple_shared.map_table(table, expected.shape[0] + 1, _DIM, torch.uint8)


def test_an_engine_gives_up_on_a_table_nobody_builds(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="holder"):
        ple_shared.map_table(tmp_path / "table", 4, _DIM, torch.uint8, timeout_s=0.0)


def test_a_table_from_another_checkpoint_fails_the_load(tmp_path: Path) -> None:
    checkpoint, expected = _checkpoint(tmp_path, [4, 4, 4])
    ple_shared.check_shard(expected, expected[4:8], 4, "table")
    other = expected[4:8].clone()
    other[-1, 0] ^= 1
    with pytest.raises(ValueError, match="row 7 differs"):
        ple_shared.check_shard(expected, other, 4, "table")


def test_a_process_mapping_the_table_is_seen(tmp_path: Path) -> None:
    table = tmp_path / "table"
    table.write_bytes(bytes(mmap.PAGESIZE))
    assert not ple_shared.mapped_by_any(table)
    with table.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ):
        assert ple_shared.mapped_by_any(table)


#: A holder that takes the table, leaves one in place and is killed outright.
_KILLED_HOLDER = """
import importlib.util, os, pathlib, signal, sys
spec = importlib.util.spec_from_file_location("ple_shared", sys.argv[1])
ple_shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ple_shared)
table = pathlib.Path(sys.argv[2])
assert ple_shared.take_hold(table) is not None
table.write_bytes(b"stale")
os.kill(os.getpid(), signal.SIGKILL)
"""


def test_one_holder_a_table_and_a_dead_ones_lock_is_taken_over(tmp_path: Path) -> None:
    table = tmp_path / "table"
    held = ple_shared.take_hold(table)
    assert held is not None and ple_shared.take_hold(table) is None
    os.close(held)
    # Killed outright, a holder leaves its lock file naming it, and its table;
    # its lock goes with it.
    killed = subprocess.run(
        [sys.executable, "-c", _KILLED_HOLDER, ple_shared.__file__, str(table)]
    )
    assert killed.returncode == -signal.SIGKILL and table.exists()
    taken = ple_shared.take_hold(table)
    assert taken is not None and not table.exists()
    os.close(taken)


def test_a_holder_holds_the_table_before_its_lock_file_names_it(
    tmp_path: Path,
) -> None:
    """Taking the lock and naming the holder in its file are two steps: a
    contender between them finds the table held, not its lock stale."""
    table = tmp_path / "table"
    held = ple_shared.take_hold(table)
    assert held is not None
    os.ftruncate(held, 0)  # the file as it is before the pid is written
    assert ple_shared.take_hold(table) is None
    os.close(held)


def test_an_engine_started_holder_serves_the_table_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, expected = _checkpoint(tmp_path, [4, 4, 4])
    table = tmp_path / "table"
    monkeypatch.setattr(ple_shared, "_register_pinned", lambda weight, nbytes: None)
    holder = ple_shared.start_holder(checkpoint, table)
    try:
        weight = ple_shared.map_table(
            table, expected.shape[0], _DIM, torch.uint8, timeout_s=60.0
        )
        assert torch.equal(weight, expected)
        assert ple_shared.take_hold(table) is None  # the started holder has it
    finally:
        holder.send_signal(signal.SIGTERM)
        holder.wait(timeout=30)
    assert not table.exists() and not table.with_name("table.holder").exists()


def test_the_table_s_variables_are_declared_and_no_compile_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM warns on a VLLM_ variable it does not declare, and refuses one when
    asked to fail on environment validation; it hashes every one it declares
    into the compile cache key unless told not to. Both of ple_shared's names
    are declared, and where a table lives keys no compiled graph."""
    for name in (ple_shared.SHARED_TABLE_ENV, ple_shared.HOLD_FROM_ENV):
        monkeypatch.setenv(name, "/dev/shm/table")
        assert name in envs.environment_variables
        assert name not in envs.compile_factors()
