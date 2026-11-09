# SPDX-License-Identifier: Apache-2.0
"""xrouter: the MoE router (gate GEMM, softmax, top-k, renormalize) in two kernels on
SM 12.0 (RTX PRO 6000 Blackwell), switched on by XKB_XROUTER=1.

Stock path a layer: cuBLAS router GEMM [tokens, 2560] x [307, 2560]^T (an unaligned
kernel plus a split-K reduce), then vLLM's moeSoftmax and moeTopK; about 27 us at
260 tokens in the profiled RTX steps. xr_gemm writes fp32 split-K partials of the
logits; xr_topk (one warp a token) sums them, rounds each logit to bf16 as the bf16
gate output is, and runs vLLM's softmax, top-k and renormalization (ties to the
lower expert, padding rows get expert -1). Both kernels launch with programmatic
dependent launch: xr_gemm prefetches router weights before waiting for the previous
kernel, xr_topk starts while xr_gemm drains. The gate returns the routing (weights,
then int32 ids, in one fp32 buffer of [tokens, 20]) in place of logits, and the
layer's router returns it to select_experts unchanged. Differences to the stock
path: the GEMM sums in another order than cuBLAS, so a logit can land one bf16 step
away (99.7-99.9% of logits bit-equal on random inputs) and a near-tie can pick
another expert; the softmax denominator sums in another order (last-bit change of
the weights). The kernel image is the cubin built with nvcc 13.0 for sm_120a from
xrouter.cu (MT 32, NST 3, PDL 1), embedded below and loaded through the CUDA driver
API. Settings (read once at import): XKB_XROUTER=1 serves MoE runners with a bf16
gate of at most 320 experts, softmax top-10 routing (FusedTopKRouter) and a modular
experts kernel; others keep the stock path. XKB_XROUTER_M=LO-HI: token counts served
(default 1-1024); other counts keep the stock path. XKB_XROUTER_S=M1:S1,M2:S2,...: K
splits of the GEMM by token count (default 128:8,384:5,1024:4; S in 1, 2, 4, 5, 8,
10 and hidden size divisible by 64 S).
"""

import base64
import ctypes
import os
import zlib

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
    _get_padding_mask,
)
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.platforms import current_platform

logger = init_logger(__name__)

NP, NT, KCH, TOPK, MT, NST = (
    320,
    64,
    64,
    10,
    32,
    3,
)  # the build's constants (xrouter.cu)
_SPLITS = (1, 2, 4, 5, 8, 10)  # xr_topk's instantiated split counts
_CUBIN_SHA256 = "5673a1693c182cca5cedbb7f1d87a0e1fbb0b8acc114d32fc25a30f1abd2c638"
_CUBIN_ZB64 = (
    "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"
    "8qfZH7T405PX//x3XptMfuuZ/zR56bXTk9OnJ89PniSl/zY9DfMSbdxrE8bM/Poz9O/XJq/Q0Ok7"
    "Lv5TEvcfJpPf/Jnf+r9ef/d/3mXF/4mWevr1CSn09df+4OfJT488+S8mz79CRH2dVEC6YfIFyvNN"
    "/nv6mVcmjFL7tSfp36+QDqKhX/jkn/7vs8t/6fee/+evn37+j37hyfSrk12156+eoSI+OXntC5Nn"
    "Tp9+hAg4ef4viYAvkdqef+YPnn/tpdd4uyfkh1deeeYZFvjtZ0RLfnFVlfX8//nSn/zR682H/uj0"
    "v355d13xPO+K1/bTFVoDVMl/8n/84OTfEtX+VPnV3TXkz58XnfBbpOyfP73XTrjJCyIvT6mc5d1k"
    "H1U+wr6tmD3Xftn4KB0pqaOGCcYPOwxxr0HSJ/9SucB7mb0EfCn1AZL/o6vvss2ep8r4CB1yHhyH"
    "SZ3/0UiZtM7AvsnSf2SqTh+k/4hlqfQ/Tuu0cXr2OsFkoPHpqfIqhi7jF00cH8H8U/F0v/8j0ee9"
    "j8QBf/yftBbVN5V+zmPeHtwWGUcZnGM75dH/e0qdlqNtmhHv6jJPxx/NaNvTs+Ll3Ytbs+OfUncz"
    "mrPzy7s/tubEp2X+9OzyneTtbGZ+JZ81O15yNdIrR1KRLewy5Ro3jZta2EBhHF9B4WhJ/s6S9MvK"
    "P7Kk/Moe69tYkt/dVXm3RST8k2zM8l933b6NJP8nWP4torip8vYbP7O9PD1t4qLy6EBYFP9Udkm8"
    "taS9S/LTb6YL25da7HdLyifdy04/4Hi6J0X9EPNFZJzcROPERH5CxRs8PrXQN89vF/TNKeSbp33h"
    "2+ab3ySfPMs3m8i3Yd9sIt9tzI6f65vNJb7ZXOKbzSW+2cS+NT27fHeObzaR78W+WeY/8M0HvvnA"
    "N79VvvmLaN38Hw988wzfOsc3J+vSZb77rfHNTx345gPffOCb/3b55mX7HqXF/ttY7L+/mHrz/Pd8"
    "/7zMvyu/vBsfrvvpBf78jfjf7JK1tbfE/1uL/XeyNjeX+H9zif9PL1mbpxfPD3P8t5ofkP9e5n/3"
    "6m+X+W93SfnL/He0z/o29ujPO3v019Gu6t+Df6Zk/7Pzz/F/5v7861PO/vzrXuOT+WB38d/f+x7f"
    "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"
    "haObF3TPZz4pG1bH6cdbFBKPNu9vokJWaw=="
)

_ENABLED = os.environ.get("XKB_XROUTER", "0") == "1"
_M_LO, _M_HI = (int(v) for v in os.environ.get("XKB_XROUTER_M", "1-1024").split("-"))
_S_TABLE = tuple(
    tuple(int(v) for v in item.split(":"))
    for item in os.environ.get("XKB_XROUTER_S", "128:8,384:5,1024:4").split(",")
)
assert all(s in _SPLITS for _, s in _S_TABLE), _S_TABLE


def _pick_s(M: int) -> int:
    for m, s in _S_TABLE:
        if m >= M:
            return s
    return _S_TABLE[-1][1]


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
        raise RuntimeError(f"xrouter {what}: CUDA driver error {r} {s.value!r}")


class _Module:
    """The cubin's two kernels in the device's primary context; launches go to torch's
    current stream with programmatic stream serialization.
    """

    def __init__(self, device: torch.device):
        with torch.cuda.device(device):
            torch.empty(
                1, device=device
            )  # makes torch's primary context current on this thread
            ctx = ctypes.c_void_p()
            _ck(_driver().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            if not ctx.value:
                raise RuntimeError("xrouter: no current CUDA context")
            self.image = zlib.decompress(base64.b64decode(_CUBIN_ZB64))
            self.mod = ctypes.c_void_p()
            _ck(
                _driver().cuModuleLoadData(ctypes.byref(self.mod), self.image),
                "cuModuleLoadData",
            )
            self.fn = {}
            for name in ("xr_gemm", "xr_topk"):
                f = ctypes.c_void_p()
                _ck(
                    _driver().cuModuleGetFunction(
                        ctypes.byref(f), self.mod, name.encode()
                    ),
                    name,
                )
                self.fn[name] = f
            self.smem = NST * (MT + NT) * KCH * 2
            _ck(
                _driver().cuFuncSetAttribute(self.fn["xr_gemm"], 8, self.smem),
                "max dynamic shared memory",
            )  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            self.attr = _LaunchAttr()
            self.attr.id = 6  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            self.attr.val[0] = 1
            self.device = device
            # One launch of each kernel now, outside any graph capture, so
            # lazy module loading never happens inside a capture.
            x = torch.zeros(1, 64 * 8, dtype=torch.bfloat16, device=device)
            w = torch.zeros(TOPK, 64 * 8, dtype=torch.bfloat16, device=device)
            P = torch.empty(8 * NP, dtype=torch.float32, device=device)
            out = torch.empty(2 * TOPK, dtype=torch.float32, device=device)
            self.route(x, w, 8, P, out[:TOPK], out[TOPK:].view(torch.int32), None, True)
            torch.cuda.synchronize(device)

    def _launch(
        self, name: str, grid: tuple[int, int, int], smem: int, vals: list
    ) -> None:
        ptrs = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals]
        )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        cfg = _LaunchConfig(
            grid[0],
            grid[1],
            grid[2],
            128,
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

    def route(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        S: int,
        P: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        pad: torch.Tensor | None,
        renorm: bool,
    ) -> None:
        """topk_w [M, 10] fp32 and topk_ids [M, 10] int32 for x [M, K] (bf16, unit
        column stride) and gate weight w [E, K] (bf16, contiguous), E <= 320.
        """
        M, K = x.shape
        E = w.shape[0]
        c = ctypes.c_void_p
        self._launch(
            "xr_gemm",
            (NP // NT, -(-M // MT), S),
            self.smem,
            [
                c(x.data_ptr()),
                ctypes.c_int(x.stride(0)),
                c(w.data_ptr()),
                ctypes.c_int(E),
                ctypes.c_int(M),
                ctypes.c_int(K),
                ctypes.c_int(K // S),
                c(P.data_ptr()),
            ],
        )
        self._launch(
            "xr_topk",
            (-(-M // 4), 1, 1),
            0,
            [
                c(P.data_ptr()),
                ctypes.c_int(S),
                ctypes.c_int(M),
                ctypes.c_int(E),
                ctypes.c_int(int(renorm)),
                c(pad.data_ptr() if pad is not None else 0),
                c(topk_w.data_ptr()),
                c(topk_ids.data_ptr()),
                c(0),
            ],
        )


_modules: dict[int, _Module] = {}


def _get_module(device: torch.device) -> _Module:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    m = _modules.get(idx)
    if m is None:
        m = _modules[idx] = _Module(torch.device("cuda", idx))
    return m


class _LayerRouter:
    """Replaces one runner's gate forward (routing instead of logits) and its router's
    _compute_routing (returns that routing).
    """

    def __init__(
        self, runner: nn.Module, gate: ReplicatedLinear, router: FusedTopKRouter
    ):
        self.runner, self.gate, self.router = runner, gate, router
        weight = gate.weight
        assert isinstance(weight, torch.Tensor)
        self.weight = weight
        self.gate_forward_stock = gate.forward
        self.compute_routing_stock = router._compute_routing
        self.pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.modular: bool | None = (
            None  # decided at the first call, after the experts kernel is set up
        )
        self.stock_calls = 0
        self.served_calls = 0
        gate.forward = self.gate_forward  # type: ignore[method-assign]
        router._compute_routing = self.compute_routing  # type: ignore[method-assign]

    def _serves(self, x: torch.Tensor) -> bool:
        if self.modular is None:
            qm = getattr(
                getattr(self.runner, "routed_experts", None), "quant_method", None
            )
            self.modular = (
                qm is not None and getattr(qm, "is_monolithic", True) is False
            )
            if not self.modular:
                logger.warning(
                    "xrouter: %s has a monolithic experts kernel "
                    "(router logits needed); it keeps the stock "
                    "router",
                    type(qm).__name__,
                )
        w = self.weight
        return (
            self.modular
            and x.dim() == 2
            and _M_LO <= x.shape[0] <= _M_HI
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
            and x.device == w.device
            and x.shape[1] == w.shape[1]
            and x.shape[1] % (KCH * _pick_s(x.shape[0])) == 0
            and w.is_contiguous()
            and w.data_ptr() % 16 == 0
        )

    def gate_forward(self, x: torch.Tensor):
        self.pending = None
        if not self._serves(x):
            self.stock_calls += 1
            return self.gate_forward_stock(x)
        self.served_calls += 1
        M = x.shape[0]
        S = _pick_s(M)
        P = torch.empty(S * M * NP, dtype=torch.float32, device=x.device)
        buf = torch.empty(2 * M * TOPK, dtype=torch.float32, device=x.device)
        topk_w = buf[: M * TOPK].view(M, TOPK)
        topk_ids = buf[M * TOPK :].view(torch.int32).view(M, TOPK)
        pad = _get_padding_mask(M)
        if pad is not None and (
            pad.dtype != torch.bool or not pad.is_contiguous() or pad.numel() < M
        ):
            pad = pad[:M].to(torch.bool).contiguous()
        _get_module(x.device).route(
            x, self.weight, S, P, topk_w, topk_ids, pad, self.router.renormalize
        )
        packed = buf.view(M, 2 * TOPK)
        self.pending = (packed, topk_w, topk_ids)
        return (packed, None) if self.gate.return_bias else packed

    def compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, self.pending = self.pending, None
        if p is not None:
            if router_logits is not p[0]:
                raise RuntimeError(
                    "xrouter: the routing the gate computed did not reach "
                    "select_experts unchanged"
                )
            return p[1], p[2]
        return self.compute_routing_stock(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )


_layers: list[_LayerRouter] = []


def enable_xrouter(model: nn.Module) -> None:
    if not _ENABLED or not current_platform.is_device_capability((12, 0)):
        return
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    kept: dict[str, int] = {}
    served: dict[str, int] = {}
    for child in model.modules():
        if not isinstance(child, MoERunner):
            continue
        gate, router = getattr(child, "gate", None), getattr(child, "router", None)
        w = getattr(gate, "weight", None)
        reason = None
        if not isinstance(gate, ReplicatedLinear) or not isinstance(w, torch.Tensor):
            reason = "no gate in the runner"
        elif getattr(child, "_fse_fuse_gate", False):
            reason = "gate fused with the shared-expert gate"
        elif type(router) is not FusedTopKRouter:
            reason = f"router {type(router).__name__}"
        elif router.scoring_func != "softmax" or router.top_k != TOPK:
            reason = f"{router.scoring_func} top-{router.top_k}"
        elif (
            type(getattr(gate, "quant_method", None)) is not UnquantizedLinearMethod
            or getattr(gate, "bias", None) is not None
        ):
            reason = "quantized or biased gate"
        elif (
            w.dtype != torch.bfloat16
            or w.dim() != 2
            or w.shape[0] > NP
            or any(w.shape[1] % (KCH * s) for _, s in _S_TABLE)
        ):
            reason = f"gate {tuple(w.shape)} {w.dtype}"
        if reason is not None:
            kept[reason] = kept.get(reason, 0) + 1
            continue
        assert isinstance(w, torch.Tensor)
        assert isinstance(gate, ReplicatedLinear)
        assert isinstance(router, FusedTopKRouter)
        if w.is_cuda:
            _get_module(w.device)
        _layers.append(_LayerRouter(child, gate, router))
        key = f"{w.shape[0]}x{w.shape[1]}"
        served[key] = served.get(key, 0) + 1
    logger.info(
        "xrouter (cubin %s) for %d-%d tokens, splits %s: %s; stock router kept for %s",
        _CUBIN_SHA256[:16],
        _M_LO,
        _M_HI,
        ",".join(f"{m}:{s}" for m, s in _S_TABLE),
        served,
        kept or "none",
    )
