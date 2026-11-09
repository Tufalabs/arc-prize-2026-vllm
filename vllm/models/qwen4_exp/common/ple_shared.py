# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Qwen4Exp n-gram PLE table for every engine on a host.

An engine that keeps its PLE table in pinned host memory holds a copy of its
own -- tens of GiB -- and engines on one host serving checkpoints with the
same table hold the same bytes each. Named by ``SHARED_TABLE_ENV``, the table
is instead one file in host shared memory (``/dev/shm``): a holder process
builds it once from a checkpoint and keeps it (``hold``), and each engine maps
it and registers the mapping with CUDA, so its GPU reads the shared pages in
place. The host's memory accounting counts the pages once, against
the holder that wrote them.

The file is the table as an engine lays it out: the checkpoint's row shards in
shard order, shard ``i`` from row ``i * ceil(rows / shards)``. An engine
checks each shard's first and last rows against it while loading, so a table
built from another checkpoint fails the load rather than serve wrong rows.

Importable without torch, so a holder runs anywhere the standard library
does: ``python ple_shared.py hold <checkpoint dir> <table path>``. The holder
is a job of its own, or a process an engine starts in its own job when
``HOLD_FROM_ENV`` names the checkpoint: one engine a host, whose job's memory
then carries the table.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

#: Names the shared table's file; unset, each engine allocates a table of its own.
SHARED_TABLE_ENV = "VLLM_QWEN4EXP_PLE_SHARED_TABLE"
#: On the one engine of a host that holds the table, the checkpoint it builds
#: it from: the engine starts the holder in its own job (``start_holder``).
HOLD_FROM_ENV = "VLLM_QWEN4EXP_PLE_SHARED_TABLE_HOLD_FROM"
#: How long an engine waits for the holder to finish the table.
READY_TIMEOUT_S = 3600.0
#: ``cudaHostRegister`` flags: portable (every context) and mapped (a UVA
#: device pointer). Not read-only, which the B200 hosts' driver refuses
#: (``cudaErrorNotSupported``, 801), so the mapping is writable too: nothing
#: writes it.
_REGISTER_FLAGS = 0x1 | 0x2
#: Bytes a build copies at a time, per shard.
_COPY_CHUNK = 64 << 20
#: Shards a build copies at once.
_COPY_THREADS = 8

_SHARD_NAME = re.compile(
    r"(?P<table>.*\.ple_embedding\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$"
)


def _shards(checkpoint: Path) -> list[tuple[Path, int, int, list[int], str]]:
    """The checkpoint's PLE shards in shard order, each as its safetensors
    file, the byte offset and length of its data, its shape and its dtype."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    tables: set[str] = set()
    found: dict[int, tuple[Path, int, int, list[int], str]] = {}
    for name, file_name in index["weight_map"].items():
        match = _SHARD_NAME.match(name)
        if match is None:
            continue
        tables.add(match["table"])
        if file_name not in headers:
            # A safetensors file: an 8-byte header length, the JSON header,
            # then the data its offsets count from.
            with (checkpoint / file_name).open("rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                headers[file_name] = (8 + length, json.loads(f.read(length)))
        base, header = headers[file_name]
        start, end = header[name]["data_offsets"]
        found[int(match["index"])] = (
            checkpoint / file_name,
            base + start,
            end - start,
            list(header[name]["shape"]),
            header[name]["dtype"],
        )
    if len(tables) != 1:
        raise ValueError(f"{checkpoint} has {len(tables)} PLE tables, not one")
    if sorted(found) != list(range(len(found))):
        raise ValueError(f"{checkpoint}'s PLE shards are not numbered 0 to n-1")
    return [found[i] for i in range(len(found))]


def build_table(checkpoint: Path, table: Path) -> int:
    """Write ``checkpoint``'s PLE table to ``table`` and return its size in
    bytes. Built beside it and renamed in, so a reader finds the whole table
    or none."""
    shards = _shards(checkpoint)
    rows = [shape[0] for _, _, _, shape, _ in shards]
    total = sum(rows)
    shard_rows = -(-total // len(shards))
    # The engine places shard i at row i * ceil(rows / shards): every shard
    # but the last holds exactly that many rows.
    if any(r != shard_rows for r in rows[:-1]) or not 0 < rows[-1] <= shard_rows:
        raise ValueError(f"{checkpoint}'s PLE shards are not an even split: {rows}")
    if len({(tuple(shape[1:]), dtype) for *_, shape, dtype in shards}) != 1:
        raise ValueError(f"{checkpoint}'s PLE shards differ in row shape or dtype")
    row_bytes = shards[0][2] // rows[0]
    size = total * row_bytes
    partial = table.with_name(f"{table.name}.partial-{os.getpid()}")
    with partial.open("wb") as out:
        out.truncate(size)

    def copy(i: int) -> None:
        source, offset, length, _, _ = shards[i]
        with source.open("rb") as src, partial.open("r+b") as dst:
            src.seek(offset)
            dst.seek(i * shard_rows * row_bytes)
            left = length
            while left:
                chunk = src.read(min(_COPY_CHUNK, left))
                if not chunk:
                    raise OSError(f"{source} ends inside PLE shard {i}")
                dst.write(chunk)
                left -= len(chunk)

    try:
        with ThreadPoolExecutor(_COPY_THREADS) as pool:
            list(pool.map(copy, range(len(shards))))
        os.replace(partial, table)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return size


def mapped_by_any(table: Path) -> bool:
    """Whether a process on this host whose mappings this one may read maps
    ``table``."""
    target = str(table)
    for maps in Path("/proc").glob("[0-9]*/maps"):
        try:
            if target in maps.read_text():
                return True
        except OSError:
            continue  # gone, or another user's
    return False


def _lock_path(table: Path) -> Path:
    """The holder's lock file, beside ``table``."""
    return table.with_name(f"{table.name}.holder")


def take_hold(table: Path) -> int | None:
    """Claim ``table`` for this process: an exclusive ``flock`` on the lock
    file beside it, held while the returned descriptor stays open and let go
    by the kernel however the holder ends, so taking it over is as atomic as
    taking it. None when a live holder has it. One holder a table: a second
    would build its own pages beside the first's. A lock file still naming a
    holder, which only one that died without its stop signal leaves, means
    the table it left is removed for this holder to build afresh (an engine
    still mapping it keeps its pages)."""
    lock = _lock_path(table)
    while True:
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        # An ending holder unlinks the file it locked before it lets go: a lock
        # on that file guards nothing, so take the one now at the path.
        try:
            locked = os.fstat(fd)
            current = os.stat(lock)
            same = (locked.st_dev, locked.st_ino) == (current.st_dev, current.st_ino)
        except FileNotFoundError:
            same = False
        if not same:
            os.close(fd)
            continue
        if os.read(fd, 32):
            table.unlink(missing_ok=True)
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode(), 0)
        return fd


def hold(
    checkpoint: Path,
    table: Path,
    *,
    idle_s: float = 600.0,
    poll_s: float = 60.0,
) -> None:
    """Build ``table`` unless it exists, then keep it until the engines that
    mapped it have all let go for ``idle_s``, and remove it. A stop signal
    removes it at once: an engine still mapping it keeps its pages."""

    held = take_hold(table)
    if held is None:
        print(f"ple_shared: {table} already has a holder", flush=True)
        return
    lock = _lock_path(table)

    def remove(signum: int, frame: Any) -> None:
        table.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, remove)
    signal.signal(signal.SIGINT, remove)
    if table.exists():
        print(f"ple_shared: holding the existing {table}", flush=True)
    else:
        started = time.monotonic()
        size = build_table(checkpoint, table)
        print(
            f"ple_shared: built {table} ({size / 2**30:.1f} GiB) from {checkpoint} "
            f"in {time.monotonic() - started:.0f}s",
            flush=True,
        )
    seen = False
    idle_since: float | None = None
    while True:
        time.sleep(poll_s)
        if mapped_by_any(table):
            seen, idle_since = True, None
            continue
        if seen:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= idle_s:
                break
    table.unlink(missing_ok=True)
    lock.unlink(missing_ok=True)
    os.close(held)
    print(f"ple_shared: removed {table}, unmapped for {idle_s:.0f}s", flush=True)


def start_holder(checkpoint: Path, table: Path) -> subprocess.Popen[bytes]:
    """Start a holder for ``table`` in this process's job, in a session of its
    own so that it outlives the engine that starts it, until the job ends and
    its stop signal removes the table. The holder's output is this process's."""
    return subprocess.Popen(
        [sys.executable, "-u", __file__, "hold", str(checkpoint), str(table)],
        start_new_session=True,
    )


def map_table(
    table: Path,
    rows: int,
    dim: int,
    dtype: torch.dtype,
    *,
    timeout_s: float = READY_TIMEOUT_S,
) -> torch.Tensor:
    """The shared table as a ``rows x dim`` host tensor its GPU reads through
    UVA: waits for the holder to finish it, maps it and registers the mapping
    with CUDA as pinned memory."""
    import mmap

    import torch

    deadline = time.monotonic() + timeout_s
    while not table.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no shared PLE table at {table} after {timeout_s:.0f}s: "
                "is its holder running on this host?"
            )
        time.sleep(5.0)
    nbytes = rows * dim * torch.empty((), dtype=dtype).element_size()
    size = table.stat().st_size
    if size != nbytes:
        raise ValueError(
            f"the shared PLE table {table} holds {size} bytes; this engine's "
            f"table is {rows} x {dim} {dtype}, {nbytes} bytes"
        )
    with table.open("r+b") as f:
        mapped = mmap.mmap(f.fileno(), nbytes, access=mmap.ACCESS_WRITE)
    weight = torch.frombuffer(mapped, dtype=torch.uint8)
    weight = weight.view(dtype).view(rows, dim)
    _register_pinned(weight, nbytes)
    return weight


def _register_pinned(weight: torch.Tensor, nbytes: int) -> None:
    """Register ``weight``'s pages with CUDA, so the UVA view reads them in
    place: a table that does not count as pinned would be copied into a
    pinned buffer of the engine's own, the copy sharing exists to avoid."""
    import torch

    error = torch.cuda.cudart().cudaHostRegister(
        weight.data_ptr(), nbytes, _REGISTER_FLAGS
    )
    if int(error) != 0:
        raise RuntimeError(
            f"cudaHostRegister of the shared PLE table failed: CUDA error {int(error)}"
        )
    if not weight.is_pinned():
        raise RuntimeError("the registered shared PLE table does not count as pinned")


def check_shard(
    table: torch.Tensor,
    shard: torch.Tensor,
    checkpoint_start: int,
    path: str,
) -> None:
    """Refuse a shared table whose rows are not this checkpoint shard's,
    compared at the shard's first and last rows."""
    import torch

    for row in {0, shard.shape[0] - 1} - {-1}:
        mine = shard[row].contiguous().view(torch.uint8)
        theirs = table[checkpoint_start + row].contiguous().view(torch.uint8)
        if not torch.equal(mine, theirs):
            raise ValueError(
                f"the shared PLE table {path} is not this checkpoint's: "
                f"row {checkpoint_start + row} differs"
            )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("build", "hold"):
        sys.exit(f"usage: {sys.argv[0]} build|hold <checkpoint dir> <table path>")
    command, checkpoint_dir, table_path = sys.argv[1:]
    if command == "build":
        build_table(Path(checkpoint_dir), Path(table_path))
    else:
        hold(Path(checkpoint_dir), Path(table_path))
