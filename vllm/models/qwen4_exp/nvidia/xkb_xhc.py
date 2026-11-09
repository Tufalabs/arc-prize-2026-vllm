# SPDX-License-Identifier: Apache-2.0
"""xhc: the HyperConnection mix after the norm (down GEMM, split-K reduce with the HC
silu, up GEMM with the gate mix) in three CUDA kernels on SM 12.0 (RTX PRO 6000
Blackwell), switched on by XKB_XHC=1.

Stock path a call (hyperconnection.py with XKB_HC_FUSED=1): cuBLAS down GEMM [M,
10240] x [336, 10240]^T plus its splitK reduce, _hc_silu_kernel, _hc_up_mix_kernel;
about 31 us at 260 rows in the profiled RTX steps, 97 calls a decode step. hc_down:
warp-specialized split-K GEMM (4 loader warps fill the shared-memory stages, 12 mma
warps), fp32 partials [S, M, 336]. hc_red: sums the partials in split order, rounds
to bf16 as the bf16 GEMM output, then lora = bf16(silu(D / 4)) as _hc_silu_kernel
computes it, and the injection logits = D. hc_up: up GEMM [M, 320] x [10240, 320]^T
for 32 output columns of all 4 streams a block, then bf16(mean over the streams of
sigmoid(gate) * xn) as _hc_up_mix_kernel computes it. hc_up's 144-row tiles fill one
wave only at 145-288 tokens, so at other token counts (XKB_XHC_UP) the stack's up
path runs on hc_red's lora instead: _hc_up_mix_kernel with XKB_HC_FUSED=1, else the
up GEMM and _hc_gate_mix_kernel. The kernels launch with programmatic dependent
launch: each loads its first weight stages before waiting for the previous kernel.
Differences to the stock path: hc_down sums K in another order than cuBLAS, so a
lora or injection value can land one bf16 step away; given equal inputs the silu and
hc_up are bit-equal to the Triton kernels. The kernel image is the cubin built with
nvcc 13.0 for sm_120a from xhc.cu (WS 1, DPW 4, UPW 4, PDL 1), embedded below and
loaded through the CUDA driver API. Settings (read once at import): XKB_XHC=1 serves
GatedResidual layers with 4 streams of 2560, low rank 320, bf16 unquantized
projections and an input of [tokens, 10240] (unit column stride); others keep the
stock path. XKB_XHC_S=M1:S1,M2:S2,...: K splits of hc_down by token count (default:
one wave of the SMs, at most 24 splits). XKB_XHC_UP=A-B,C-D,...: token counts at
which hc_up runs (default 1-32,193-288, where it beat _hc_up_mix_kernel on an SM 12.0 card);
"none" for none.
"""

import base64
import ctypes
import os
import zlib
from typing import cast

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

HC, HD, RK = 4, 2560, 320  # the build's constants (xhc.cu)
_CFG_KEYS = (
    "dmt",
    "dnt",
    "dwm",
    "dwn",
    "dkch",
    "dnst",
    "umt",
    "uns",
    "uwm",
    "uwn",
    "ukch",
    "unst",
    "pdl",
    "mode",
    "dpf",
    "upf",
    "dupf",
    "ws",
    "dpw",
    "upw",
)
_CUBIN_SHA256 = "e5e709272dc17983ef635dbb0fff106f141afcfb860e1ddc8fa57494e78bd27d"
_CUBIN_ZB64 = (
    "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"
)

_ENABLED = os.environ.get("XKB_XHC", "0") == "1"
_S_ENV = os.environ.get("XKB_XHC_S", "")
_S_TABLE = (
    tuple(tuple(int(v) for v in item.split(":")) for item in _S_ENV.split(","))
    if _S_ENV
    else ()
)
_UP_ENV = os.environ.get("XKB_XHC_UP", "1-32,193-288")
_UP_RANGES = (
    ()
    if _UP_ENV == "none"
    else tuple(tuple(int(v) for v in item.split("-")) for item in _UP_ENV.split(","))
)
# XKB_XHC_ROWS (xhc_rows.patch): row counts that run xhc, e.g. "1-32,257-288"; other
# row counts take the stock HC path. Unset: every row count runs xhc, as before.
# The RTX kernel check (xhc_rtx, 10-30) has xhc ahead of the stock chain at 1-16 and
# 260-264 rows and behind at 64-192 and 384-2048.
_ROWS_ENV = os.environ.get("XKB_XHC_ROWS", "")
_ROWS = (
    tuple(tuple(int(v) for v in item.split("-")) for item in _ROWS_ENV.split(","))
    if _ROWS_ENV
    else ()
)
_HC_FUSED = os.environ.get("XKB_HC_FUSED", "0") == "1"  # as hyperconnection.py reads it


class _LaunchAttr(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint),
        ("pad", ctypes.c_char * 4),
        ("val", ctypes.c_ubyte * 64),
    ]


class _LaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gx", ctypes.c_uint),
        ("gy", ctypes.c_uint),
        ("gz", ctypes.c_uint),
        ("bx", ctypes.c_uint),
        ("by", ctypes.c_uint),
        ("bz", ctypes.c_uint),
        ("smem", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_LaunchAttr)),
        ("nattrs", ctypes.c_uint),
    ]


_drv = None


def _driver():
    global _drv
    if _drv is None:
        d = ctypes.CDLL("libcuda.so.1")
        d.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        d.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        d.cuModuleGetGlobal_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        d.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        d.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        d.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        d.cuLaunchKernelEx.argtypes = [
            ctypes.POINTER(_LaunchConfig),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        d.cuLaunchKernelEx.restype = ctypes.c_int
        _drv = d
    return _drv


def _ck(r: int, what: str) -> None:
    if r != 0:
        s = ctypes.c_char_p()
        _driver().cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"xhc {what}: CUDA driver error {r} {s.value!r}")


class _Module:
    """The cubin's three kernels in the device's primary context; launches go to
    torch's current stream with programmatic stream serialization.
    """

    def __init__(self, device: torch.device):
        with torch.cuda.device(device):
            torch.empty(
                1, device=device
            )  # makes torch's primary context current on this thread
            ctx = ctypes.c_void_p()
            _ck(_driver().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            if not ctx.value:
                raise RuntimeError("xhc: no current CUDA context")
            self.image = zlib.decompress(base64.b64decode(_CUBIN_ZB64))
            self.mod = ctypes.c_void_p()
            _ck(
                _driver().cuModuleLoadData(ctypes.byref(self.mod), self.image),
                "cuModuleLoadData",
            )
            self.fn = {}
            for name in ("hc_down", "hc_red", "hc_up"):
                f = ctypes.c_void_p()
                _ck(
                    _driver().cuModuleGetFunction(
                        ctypes.byref(f), self.mod, name.encode()
                    ),
                    name,
                )
                self.fn[name] = f
            ptr, size = ctypes.c_uint64(), ctypes.c_size_t()
            _ck(
                _driver().cuModuleGetGlobal_v2(
                    ctypes.byref(ptr), ctypes.byref(size), self.mod, b"xhc_cfg"
                ),
                "xhc_cfg",
            )
            buf = (ctypes.c_int * 32)()
            _ck(_driver().cuMemcpyDtoH_v2(buf, ptr.value, 128), "read xhc_cfg")
            c = self.cfg = dict(zip(_CFG_KEYS, list(buf)))
            if c["mode"] != 0 or c["ws"] != 1 or c["pdl"] != 1 or c["dupf"] != 0:
                raise RuntimeError(f"xhc: unexpected cubin build {c}")
            self.d_threads = 32 * (c["dwm"] * c["dwn"] + c["dpw"])
            self.u_threads = 32 * (c["uwm"] * c["uwn"] + c["upw"])
            self.d_smem = (
                c["dnst"] * (c["dmt"] + c["dnt"]) * c["dkch"] * 2 + 16 * c["dnst"]
            )  # stages + full/empty mbarriers
            self.u_smem = (
                c["unst"] * (c["umt"] + HC * c["uns"]) * c["ukch"] * 2
                + c["umt"] * (HC * c["uns"] * 2 + 16)
                + 16 * c["unst"]
            )  # stages + xn tile + mbarriers
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _ck(
                _driver().cuFuncSetAttribute(self.fn["hc_down"], 8, self.d_smem),
                "hc_down max dynamic shared memory",
            )
            _ck(
                _driver().cuFuncSetAttribute(self.fn["hc_up"], 8, self.u_smem),
                "hc_up max dynamic shared memory",
            )
            self.sms = torch.cuda.get_device_properties(device).multi_processor_count
            self.attr = _LaunchAttr()
            self.attr.id = 6  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            self.attr.val[0] = 1
            self.device = device
            # One call now, outside any graph capture, so lazy module
            # loading never happens inside a capture.
            xn = torch.zeros(1, HC * HD, dtype=torch.bfloat16, device=device)
            self.mix(
                xn,
                torch.zeros(RK + HC, HC * HD, dtype=torch.bfloat16, device=device),
                torch.zeros(HC * HD, RK, dtype=torch.bfloat16, device=device),
                HC,
                up=True,
            )
            torch.cuda.synchronize(device)

    def _launch(
        self, name: str, grid: tuple[int, int, int], block: int, smem: int, vals: list
    ) -> None:
        ptrs = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals]
        )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        cfg = _LaunchConfig(
            grid[0],
            grid[1],
            grid[2],
            block,
            1,
            1,
            smem,
            ctypes.c_void_p(stream),
            ctypes.pointer(self.attr),
            1,
        )
        _ck(
            _driver().cuLaunchKernelEx(ctypes.byref(cfg), self.fn[name], ptrs, None),
            name,
        )

    def splits(self, M: int) -> int:
        nch = HC * HD // self.cfg["dkch"]
        for m, s in _S_TABLE:
            if m >= M:
                return max(1, min(nch, s))
        if _S_TABLE:
            return max(1, min(nch, _S_TABLE[-1][1]))
        tiles = -(-(RK + HC) // self.cfg["dnt"]) * -(-M // self.cfg["dmt"])
        return max(1, min(nch, 24, self.sms // tiles))

    def mix(
        self,
        xn: torch.Tensor,
        w_down: torch.Tensor,
        w_up: torch.Tensor,
        ninj: int,
        up: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """block_input [M, 2560] and injection logits [M, ninj] (ninj 0 or 4) for xn
        [M, 10240], w_down [N, 10240] (N >= 320 + ninj), w_up [10240, 320].

        up: run hc_up (True) or the stack's up path (False); None picks by
        XKB_XHC_UP.
        """
        M, K = xn.shape
        N = w_down.shape[0]
        inj = torch.empty(M, ninj, dtype=torch.bfloat16, device=xn.device)
        if M == 0:
            return torch.empty(M, HD, dtype=torch.bfloat16, device=xn.device), inj
        if up is None:
            up = any(lo <= M <= hi for lo, hi in _UP_RANGES)
        if xn.data_ptr() % 16 or xn.stride(0) % 8:
            xn = xn.contiguous() if xn.stride(0) % 8 else xn.clone()
        c, v = self.cfg, ctypes.c_void_p
        dnt = c["dnt"]
        NP = -(-N // dnt) * dnt
        S = self.splits(M)
        P = torch.empty(S * M * NP, dtype=torch.float32, device=xn.device)
        lora = torch.empty(M, RK, dtype=torch.bfloat16, device=xn.device)
        i = ctypes.c_int
        self._launch(
            "hc_down",
            (NP // dnt, -(-M // c["dmt"]), S),
            self.d_threads,
            self.d_smem,
            [
                v(xn.data_ptr()),
                i(xn.stride(0)),
                v(w_down.data_ptr()),
                i(N),
                i(M),
                i(K),
                i(S),
                v(P.data_ptr()),
                i(NP),
                v(0),
                i(0),
            ],
        )
        nq = (RK + ninj) // 4
        self._launch(
            "hc_red",
            (-(-M * nq // 256), 1, 1),
            256,
            0,
            [
                v(P.data_ptr()),
                i(S),
                i(M),
                i(NP),
                i(RK),
                i(ninj),
                v(lora.data_ptr()),
                v(inj.data_ptr() if ninj else 0),
                i(ninj),
            ],
        )
        if not up:
            from .ops.hc import hc_gate_mix, hc_up_mix, hc_up_mix_ok

            if _HC_FUSED and hc_up_mix_ok(xn, lora, w_up, HC):
                return hc_up_mix(xn, lora, w_up, HC), inj
            return hc_gate_mix(xn, torch.nn.functional.linear(lora, w_up), HC), inj
        out = torch.empty(M, HD, dtype=torch.bfloat16, device=xn.device)
        self._launch(
            "hc_up",
            (HD // c["uns"], -(-M // c["umt"]), 1),
            self.u_threads,
            self.u_smem,
            [
                v(lora.data_ptr()),
                v(w_up.data_ptr()),
                v(xn.data_ptr()),
                i(xn.stride(0)),
                v(out.data_ptr()),
                i(HD),
                i(M),
            ],
        )
        return out, inj


_modules: dict[int, _Module] = {}


def _get_module(device: torch.device) -> _Module:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    m = _modules.get(idx)
    if m is None:
        m = _modules[idx] = _Module(torch.device("cuda", idx))
    return m


def _xhc_mix(
    xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, ninj: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_module(xn.device).mix(xn, w_down, w_up, ninj)


def _xhc_mix_fake(
    xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, ninj: int
) -> tuple[torch.Tensor, torch.Tensor]:
    del w_down, w_up
    return xn.new_empty((xn.shape[0], HD)), xn.new_empty((xn.shape[0], ninj))


direct_register_custom_op(
    op_name="qwen4_exp_xhc_mix", op_func=_xhc_mix, fake_impl=_xhc_mix_fake
)

_served: dict[str, int] = {}
_kept: dict[str, int] = {}


def _log_counts() -> None:
    logger.info(
        "xhc (cubin %s): layers served %s; stock path kept for %s; rows %s",
        _CUBIN_SHA256[:16],
        _served,
        _kept or "none",
        _ROWS_ENV or "all",
    )


def xhc_enable(layer: torch.nn.Module) -> bool:
    """Whether the GatedResidual layer runs xhc_mix (decided once, at construction)."""
    if not _ENABLED or not current_platform.is_device_capability((12, 0)):
        return False
    down = cast(
        ReplicatedLinear,
        (
            layer.input_mix_weight_down_block_inject
            if layer.use_combine
            else layer.input_mix_weight_down
        ),
    )
    up = cast(ReplicatedLinear, layer.input_mix_weight_up)
    down_weight, up_weight = down.weight, up.weight
    if not isinstance(down_weight, torch.Tensor) or not isinstance(
        up_weight, torch.Tensor
    ):
        return False
    reason = None
    if (layer.hc_count, layer.hidden_size, layer.lora_rank) != (HC, HD, RK):
        reason = f"hc {layer.hc_count}x{layer.hidden_size} rank {layer.lora_rank}"
    elif envs.VLLM_BATCH_INVARIANT:
        reason = "VLLM_BATCH_INVARIANT"
    elif any(
        type(getattr(m, "quant_method", None)) is not UnquantizedLinearMethod
        or getattr(m, "bias", None) is not None
        for m in (down, up)
    ):
        reason = "quantized or biased projection"
    elif (
        (down_weight.dtype, up_weight.dtype) != (torch.bfloat16, torch.bfloat16)
        or tuple(up_weight.shape) != (HC * HD, RK)
        or down_weight.dim() != 2
        or down_weight.shape[1] != HC * HD
        or not RK + (HC if layer.use_combine else 0) <= down_weight.shape[0] <= 336
    ):
        reason = (
            f"weights {tuple(down_weight.shape)} "
            f"{tuple(up_weight.shape)} {down_weight.dtype}"
        )
    if reason is not None:
        _kept[reason] = _kept.get(reason, 0) + 1
        if _kept[reason] == 1:
            _log_counts()
        return False
    if down_weight.is_cuda:
        _get_module(down_weight.device)
    key = "with injection" if layer.use_combine else "final"
    _served[key] = _served.get(key, 0) + 1
    if _served[key] == 1:
        _log_counts()
    return True


def xhc_serves(xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor) -> bool:
    """Static checks on the operands (shapes other than tokens, dtypes, strides), so
    the decision holds under torch.compile.

    With XKB_XHC_ROWS set, also the token count: CUDA graphs fix it per captured
    size, and the pod runs without torch.compile.
    """
    return (
        xn.dim() == 2
        and xn.shape[1] == HC * HD
        and xn.dtype == torch.bfloat16
        and xn.stride(1) == 1
        and xn.is_cuda
        and w_down.is_contiguous()
        and w_up.is_contiguous()
        and w_down.device == xn.device
        and w_up.device == xn.device
        and (not _ROWS or any(lo <= xn.shape[0] <= hi for lo, hi in _ROWS))
    )


def xhc_mix(
    xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, ninj: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_xhc_mix(xn, w_down, w_up, ninj)
