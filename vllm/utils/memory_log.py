# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the host memory went, sampled from inside the engine.

A cgroup OOM kill leaves no traceback: the process is gone and the log stops
mid-line, so the only account of what filled the cgroup is one printed
before it did. Enabled by ``VLLM_MEMORY_LOG_S``, a sampling interval in
seconds; unset, nothing here runs.

It reports the distinction the kernel acts on rather than a total. ``anon``
and ``unevictable`` (pinned pages) cannot be reclaimed at all; ``shmem`` is
charged as page cache but is unreclaimable without swap; ``file`` is
ordinary page cache, freed on demand, except the ``file_mapped`` part, which
needs unmapping first. A cgroup at its limit is healthy or doomed depending
entirely on which of those it holds, and "available memory" distinguishes
none of them.

Stdlib plus torch only, and no vllm imports beyond the logger: this is
called from engine start-up paths, before much of vllm is configured.
"""

import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Sampling interval in seconds. Unset, empty or non-positive disables it.
ENV_INTERVAL = "VLLM_MEMORY_LOG_S"
#: Keep a worker's sampler running after start-up, through serving, with the
#: device's peaks reset where start-up ends: where an engine at the edge of
#: its card's memory fails is in serving, not start-up. Unset or 0: start-up
#: only.
ENV_SERVING = "VLLM_MEMORY_LOG_SERVING"

_CGROUP = "/sys/fs/cgroup"
_STAT_FIELDS = (
    "anon",
    "file",
    "file_mapped",
    "shmem",
    "unevictable",
    "slab",
    "sock",
    "kernel_stack",
)
_GIB = 1024**3
_MIB = 1024**2
_started = False
_stopped = threading.Event()
# The device the worker selected, registered by it: the sampler thread
# cannot inherit it, since the current CUDA device is thread-local.
_device: int | None = None


def bind_device(index: int) -> None:
    """Register the worker's device, from which the GPU figures are read."""
    global _device
    _device = index


def _read_int(path: str) -> int | None:
    """One integer from a cgroup file; None for "max" and for absence."""
    try:
        with open(path) as fh:
            text = fh.read().strip()
    except OSError:
        return None
    return None if text == "max" else int(text)


def _cgroup_stat() -> dict[str, int]:
    """The memory.stat fields reported here, in bytes."""
    out: dict[str, int] = {}
    try:
        with open(f"{_CGROUP}/memory.stat") as fh:
            for line in fh:
                key, _, value = line.partition(" ")
                if key in _STAT_FIELDS:
                    out[key] = int(value)
    except OSError:
        pass
    return out


def _proc_status() -> dict[str, int]:
    """This process's resident, peak-resident, pinned and locked sizes."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM", "VmPin", "VmLck"):
                    out[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return out


def _top_processes(count: int = 4) -> str:
    """The biggest resident processes in this container, largest first.

    The cgroup fields are container-wide while ``VmRSS`` is this process
    only, so a cgroup that grows while ``VmRSS`` does not means somebody
    else -- and without this there is no way to say who from in here.
    """
    rows: list[tuple[int, int, str]] = []
    try:
        pids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return ""
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as fh:
                name = ""
                rss = 0
                for line in fh:
                    if line.startswith("Name:"):
                        name = line.split(maxsplit=1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            continue
        if rss:
            rows.append((rss, int(pid), name))
    if not rows:
        return ""
    rows.sort(reverse=True)
    total = sum(rss for rss, _, _ in rows)
    top = " ".join(f"{pid}:{name}={rss / _GIB:.1f}" for rss, pid, name in rows[:count])
    # The total matters as much as the largest: twenty processes of three
    # GiB fill a cgroup exactly as well as one of sixty, and a list of the
    # biggest few would show neither.
    return f" | procs={len(rows)} rss_sum={total / _GIB:.1f} | top {top}"


def _gpu() -> str:
    """Free/total and what torch holds on the worker's device, once it has one.

    Only after the worker has registered its device and CUDA is initialized
    here: ``mem_get_info`` initializes it otherwise, and a snapshot taken
    before the worker selects its device would put a context on the default
    GPU (and, in the front end, keep one resident beside the worker's) merely
    to observe it. The device is passed explicitly because the sampler runs
    in its own thread, where the current device would be the default one.
    """
    try:
        import torch

        device = _device
        if device is None or not torch.cuda.is_initialized():
            return ""
        free, total = torch.cuda.mem_get_info(device)
        # MiB: an engine near its card's edge differs by tens of them. The
        # peaks are torch's since the last reset (start-up's end, when the
        # sampler keeps going into serving), so a peak between two samples
        # still shows.
        return (
            f" | gpu MiB free={free / _MIB:.0f} total={total / _MIB:.0f} "
            f"alloc={torch.cuda.memory_allocated(device) / _MIB:.0f} "
            f"reserved={torch.cuda.memory_reserved(device) / _MIB:.0f} "
            f"peak_alloc={torch.cuda.max_memory_allocated(device) / _MIB:.0f} "
            f"peak_reserved={torch.cuda.max_memory_reserved(device) / _MIB:.0f}"
        )
    except Exception:  # noqa: BLE001 - instrumentation never breaks a boot
        return ""


def snapshot(label: str) -> str:
    """One line of everything worth knowing, formatted for a log."""
    current = _read_int(f"{_CGROUP}/memory.current")
    limit = _read_int(f"{_CGROUP}/memory.max")
    stat = _cgroup_stat()
    status = _proc_status()

    # The pid, because more than one process samples: the engine core and
    # whatever started it share a cgroup, so their lines carry the same
    # cgroup figures and differ only in VmRSS -- indistinguishable without
    # this, and their elapsed clocks interleave.
    parts = [f"[{os.getpid()}] {label}"]
    if current is not None:
        ceiling = f"/{limit / _GIB:.1f}" if limit else ""
        parts.append(f"cgroup={current / _GIB:.1f}{ceiling} GiB")
    if stat:
        unreclaimable = (
            stat.get("anon", 0) + stat.get("unevictable", 0) + stat.get("shmem", 0)
        )
        parts.append(f"unreclaimable={unreclaimable / _GIB:.1f}")
        parts.append(
            " ".join(f"{k}={stat[k] / _GIB:.1f}" for k in _STAT_FIELDS if k in stat)
        )
    if status:
        parts.append(" ".join(f"{k}={v / _GIB:.1f}" for k, v in status.items()))
    return " | ".join(parts) + _gpu() + _top_processes()


def report(label: str) -> None:
    """Log a snapshot if enabled, never raising: a failure here would hide
    the failure it exists to explain."""
    if not os.environ.get(ENV_INTERVAL, "").strip():
        return
    try:
        logger.info("memory: %s", snapshot(label))
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory: snapshot failed: %s", exc)


#: Seconds to pause at each step marker, so the sampler is guaranteed a
#: reading between two steps rather than straddling them.
ENV_PAUSE = "VLLM_MEMORY_LOG_PAUSE_S"


def mark(label: str) -> None:
    """Report at a step boundary, and hold there if asked to.

    Bracketing a suspect step is only conclusive if a sample lands on each
    side of it; a step shorter than the sampling interval otherwise passes
    between two readings and shows up as a jump with nothing to attribute
    it to.
    """
    report(label)
    raw = os.environ.get(ENV_PAUSE, "").strip()
    if not raw:
        return
    try:
        pause = float(raw)
    except ValueError:
        return
    if pause > 0:
        time.sleep(pause)


def _sampler(interval: float) -> None:
    """Report until asked to stop: the last line before a kill is the
    evidence, and start-up is where the kills have been."""
    elapsed = 0.0
    while not _stopped.wait(interval):
        elapsed += interval
        report(f"t+{elapsed:.0f}s")


def stop(label: str) -> None:
    """End the periodic sampling, after one last reading.

    Start-up is what this was built to watch, and a run serving for nine
    hours after it does not want a line every five seconds. The markers
    still report if anything calls them; only the clock stops.
    """
    if not _started or _stopped.is_set():
        return
    if os.environ.get(ENV_SERVING, "").strip() not in ("", "0"):
        report(f"{label} (sampler continues into serving; device peaks reset)")
        _reset_device_peaks()
        return
    report(f"{label} (sampler stopping)")
    _stopped.set()


def _reset_device_peaks() -> None:
    """Restart torch's peak counters on the worker's device, so the peaks
    the samples carry from here on are serving's, not start-up's."""
    try:
        import torch

        if _device is not None and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(_device)
    except Exception as exc:  # noqa: BLE001 - instrumentation never breaks serving
        logger.warning("memory: resetting the device peaks failed: %s", exc)


def maybe_start() -> None:
    """Start the sampler once per process, if the interval asks for it."""
    global _started
    if _started:
        return
    raw = os.environ.get(ENV_INTERVAL, "").strip()
    if not raw:
        return
    try:
        interval = float(raw)
    except ValueError:
        logger.warning("memory: %s=%r is not a number of seconds", ENV_INTERVAL, raw)
        return
    if interval <= 0:
        return
    _started = True
    report("sampler started")
    threading.Thread(
        target=_sampler, args=(interval,), name="vllm-memory-log", daemon=True
    ).start()
