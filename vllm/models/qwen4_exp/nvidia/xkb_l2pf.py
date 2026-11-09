# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of MoE expert weights on a side stream, switched on by XKB_L2PF=1; it
acts only on MoE layers that run xmoe (XKB_XMOE=1).

After a decoder layer's attention output (out_proj done), `xkb_l2pf_issue` forks a
side stream and launches a small kernel (a few one-warp CTAs) that issues
`cp.async.bulk.prefetch.L2.global` over the first XKB_L2PF_MB megabytes of the
layer's w13 expert codes. The kernel ends as soon as the prefetches are issued; the
bytes arrive in L2 while the main stream runs the HC mix, router and shared expert,
which leave DRAM partly idle, and xmoe's expert kernel then reads those bytes from
L2. `xkb_l2pf_join` after the MoE block makes the main stream wait for the side
stream, so CUDA graph capture sees the fork joined in the same segment. A prefetch
writes nothing, so outputs do not change. layer bench
(run/k16/l2pf/l2pf_moe.py, xmoe, 224-260 rows): 48 MB from 16 CTAs saves 20-22 us an
MoE layer with the router's routing, 8-18 us with skewed routing. engine test
(run/k16/l2pf/fork/l2pf_engine.py, 16 layers, 256 verify rows): all 48 MB at once
from 16 CTAs gives -0.23% wall, because the burst slows the router GEMM and HC mix
in the window; one warp issuing a 64 KB chunk every 100 ns gives -0.90% with 32 MB
and -0.80% with 48 MB, xmoe_mlp 501 -> 485 us. One paced warp issues about one chunk
every 240 ns whatever XKB_L2PF_NS is at 60-150 (about 270 GB/s), so 32 MB takes
about 115 us. With xrouter, nocopy and the shared expert on the aux stream the
window before the expert kernel is shorter; there, 32 MB with mirrored repeats on
one node (job 3625632): 1 warp -0.34 to -0.45% (other nodes), 2 warps at 50 ns
-0.66%, 4 warps at 25 ns -0.75% (57 us, xmoe_mlp 503 -> 486 us), 8 warps at 12 ns
-0.55%; the defaults are 4 warps at 25 ns. The kernel image is the cubin built with
nvcc 13.0 for sm_120a from pf.cu, embedded below and loaded through the CUDA driver
API. Settings (read once at import): XKB_L2PF=1 turns it on. XKB_L2PF_MB (default
32) megabytes of w13 codes from the start of the tensor. XKB_L2PF_CTAS (default 16)
CTAs of 128 threads, each thread one 64 KB prefetch at a time. XKB_L2PF_NS (default
25): when above 0, XKB_L2PF_WARPS CTAs of one warp issue the 64 KB prefetches, one
every XKB_L2PF_NS ns over the grid (kernel l2pf_paced), instead of all at once.
XKB_L2PF_WARPS (default 4): CTAs of the paced kernel, one warp each.
XKB_L2PF_MINROWS, XKB_L2PF_MAXROWS (defaults 64 and 512): token counts outside this
range skip the prefetch.
"""

import base64
import ctypes
import os
import zlib

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("XKB_L2PF", "0") == "1"
NBYTES = int(float(os.environ.get("XKB_L2PF_MB", "32")) * 2**20) // 16 * 16
CTAS = int(os.environ.get("XKB_L2PF_CTAS", "16"))
WARPS = int(os.environ.get("XKB_L2PF_WARPS", "4"))
NS = int(os.environ.get("XKB_L2PF_NS", "25"))
MINROWS = int(os.environ.get("XKB_L2PF_MINROWS", "64"))
MAXROWS = int(os.environ.get("XKB_L2PF_MAXROWS", "512"))
THREADS, CHUNK = 128, 65536

_CUBIN_SHA256 = "fb754e3c7a04b5742088c95a3a7c19d76f55cfca127eaa812ac2d16f488510a4"
_CUBIN_ZB64 = (
    "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"
    "/hcOsyaA"
)

# issued and joined prefetches, and calls that skipped (outside the row range); graph
# replays do not reach Python
counts = {"issued": 0, "joined": 0, "skipped": 0}


class _State:
    """The cubin's kernel in the device's primary context, the side stream and the fork
    and join events.
    """

    def __init__(self, device: torch.device):
        d = ctypes.CDLL("libcuda.so.1")
        d.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        d.cuLaunchKernel.restype = ctypes.c_int
        d.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        d.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        d.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.drv = d
        with torch.cuda.device(device):
            torch.empty(
                1, device=device
            )  # makes torch's primary context current on this thread
            ctx = ctypes.c_void_p()
            self._ck(d.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            if not ctx.value:
                raise RuntimeError("l2pf: no current CUDA context")
            self.image = zlib.decompress(base64.b64decode(_CUBIN_ZB64))
            self.mod = ctypes.c_void_p()
            self._ck(
                d.cuModuleLoadData(ctypes.byref(self.mod), self.image),
                "cuModuleLoadData",
            )
            self.fn = ctypes.c_void_p()
            name = b"l2pf_paced" if NS > 0 else b"l2pf"
            self._ck(
                d.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, name),
                name.decode(),
            )
            self.stream = torch.cuda.Stream(device)
            self.fork = torch.cuda.Event()
            self.join = torch.cuda.Event()
        self.pending = False

    def _ck(self, r: int, what: str) -> None:
        if r != 0:
            s = ctypes.c_char_p()
            self.drv.cuGetErrorString(r, ctypes.byref(s))
            raise RuntimeError(f"l2pf {what}: CUDA driver error {r} {s.value!r}")

    def launch(self, ptr: int, nbytes: int) -> None:
        last = (
            ctypes.c_uint(NS) if NS > 0 else ctypes.c_int(0)
        )  # l2pf_paced: ns a chunk; l2pf: no cache hint
        vals = [
            ctypes.c_void_p(ptr),
            ctypes.c_ulonglong(nbytes),
            ctypes.c_uint(CHUNK),
            last,
        ]
        args = (ctypes.c_void_p * 4)(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals]
        )
        grid, block = (WARPS, 32) if NS > 0 else (CTAS, THREADS)
        self._ck(
            self.drv.cuLaunchKernel(
                self.fn, grid, 1, 1, block, 1, 1, 0, self.stream.cuda_stream, args, None
            ),
            "launch",
        )


_states: dict[int, _State] = {}


def _state(device: torch.device) -> _State:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _states.get(idx)
    if s is None:
        s = _states[idx] = _State(torch.device("cuda", idx))
        logger.info_once(
            "XKB_L2PF: %d MB of w13 prefetched into L2 per xmoe "
            "MoE layer, %d-%d tokens, %s (cubin sha256 %s)",
            NBYTES >> 20,
            MINROWS,
            MAXROWS,
            f"one 64 KB chunk every {NS} ns from {WARPS} warps"
            if NS > 0
            else f"{CTAS} CTAs at once",
            _CUBIN_SHA256[:16],
        )
    return s


def l2pf_target(mlp: torch.nn.Module) -> torch.Tensor | None:
    """The w13 codes of a decoder layer's MoE block when it runs xmoe and the switch is
    on, else None (the layer keeps the stock path).
    """
    if not ENABLED or NBYTES <= 0:
        return None
    experts = getattr(getattr(mlp, "experts", None), "routed_experts", None)
    if experts is None or not getattr(experts, "_xkb_xmoe", False):
        return None
    w = getattr(experts, "w13_weight", None)
    if not isinstance(w, torch.Tensor) or not w.is_cuda:
        return None
    return w


def _issue(x: torch.Tensor, w13: torch.Tensor) -> None:
    rows = x.size(0) if x.dim() > 1 else 1
    s = _state(x.device)
    cur = torch.cuda.current_stream(x.device)
    # a forward that stopped between issue and join (an exception) left its fork
    # open
    if s.pending:
        cur.wait_event(s.join)
        s.pending = False
    if not MINROWS <= rows <= MAXROWS:
        counts["skipped"] += 1
        return
    s.fork.record(cur)
    s.stream.wait_event(s.fork)
    s.launch(w13.data_ptr(), min(NBYTES, w13.numel() * w13.element_size() // 16 * 16))
    s.join.record(s.stream)
    s.pending = True
    counts["issued"] += 1


def _issue_fake(x: torch.Tensor, w13: torch.Tensor) -> None:
    return None


def _join(y: torch.Tensor) -> None:
    s = _state(y.device)
    if s.pending:
        torch.cuda.current_stream(y.device).wait_event(s.join)
        s.pending = False
        counts["joined"] += 1


def _join_fake(y: torch.Tensor) -> None:
    return None


# Both ops declare their tensor input as mutated (they write nothing) so torch.compile
# keeps them and their order.
# With the switch off they are not registered, and l2pf_target returns None, so
# model.py calls neither.
if ENABLED:
    direct_register_custom_op(
        op_name="xkb_l2pf_issue",
        op_func=_issue,
        mutates_args=["x"],
        fake_impl=_issue_fake,
    )
    direct_register_custom_op(
        op_name="xkb_l2pf_join", op_func=_join, mutates_args=["y"], fake_impl=_join_fake
    )
