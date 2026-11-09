# SPDX-License-Identifier: Apache-2.0
"""CUDA port of the GDN RecoverSSM fused verify (GDN_FUSED_COMMIT) for sm_120, switched
on by XKB_XGDN=1.

`gdn_recoverssm_fused_verify` builds its Triton launch as before and passes it to
`launcher`; with the switch on, the returned function launches `xgdn_verify`
(run/k16/xgdn/xgdn.cu) with the same arguments when this build serves the operands
(K = V = 128, bf16 q/k/v/out/state, float32 records, at most 4 tokens a request,
heads contiguous), else the Triton kernel. The copy of the staged key and gate
records (`_copy_staged_records_kernel`) runs after it as before. Same reads and
writes as the Triton kernel; sums run in a different order, so results can differ in
the last bits ("last-bit rounding"). graph bench (run/k16/xgdn/xgdn_bench.py,
36 layers): verify + copy 5.42 vs 5.80 ms a step at 66 requests and 4.84 vs 5.17 ms
at 59 requests, at about 1.5 TB/s of state read and written. The kernel image is the
cubin built with nvcc 13.0 for sm_120a from xgdn.cu (NT 128, RPT 8, MINB 2),
embedded below and loaded through the CUDA driver API; launches go to torch's
current stream, so CUDA graphs capture them (the argument struct is copied at
launch).
"""

import base64
import ctypes
import os
import zlib

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("XKB_XGDN", "0") == "1"

_CUBIN_SHA256 = "ef9154f0e306e5d8655be39c686acfc0e97b7a1833f7e280f3a0caa77400f414"
_CUBIN_ZB64 = (
    "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"
)

# verify launches by path, counted in Python (graph replays do not reach Python)
counts = {"xgdn": 0, "triton": 0}

_PTRS = [
    "q",
    "k",
    "v",
    "a",
    "b",
    "A_log",
    "dt_bias",
    "state",
    "corr",
    "key",
    "gate",
    "key_stage",
    "gate_stage",
    "row_pending",
    "out",
    "qsl",
    "sidx",
    "cnt",
]
_INTS = [
    "s_q",
    "s_k",
    "s_v",
    "s_a",
    "s_b",
    "s_sb",
    "s_sh",
    "s_sv",
    "s_cb",
    "s_ch",
    "s_cp",
    "s_kb",
    "s_kh",
    "s_kp",
    "s_gb",
    "s_gh",
    "s_gp",
    "s_ksr",
    "s_ksh",
    "s_ksp",
    "s_gsr",
    "s_gsh",
    "s_out",
    "s_qsl",
    "s_sidx",
    "null_block",
    "group",
    "bt",
    "flags",
]


class _Args(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_uint64) for n in _PTRS]
        + [(n, ctypes.c_int64) for n in _INTS]
        + [("scale", ctypes.c_double)]
    )


class _Module:
    """The cubin's kernel in torch's primary context of one device, and its build
    constants.
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
        d.cuModuleGetGlobal_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        d.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.drv = d
        with torch.cuda.device(device):
            torch.empty(
                1, device=device
            )  # makes torch's primary context current on this thread
            self.image = zlib.decompress(base64.b64decode(_CUBIN_ZB64))
            self.mod = ctypes.c_void_p()
            self._ck(
                d.cuModuleLoadData(ctypes.byref(self.mod), self.image),
                "cuModuleLoadData",
            )
            self.fn = ctypes.c_void_p()
            self._ck(
                d.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, b"xgdn_verify"),
                "xgdn_verify",
            )
            ptr, size = ctypes.c_uint64(), ctypes.c_size_t()
            self._ck(
                d.cuModuleGetGlobal_v2(
                    ctypes.byref(ptr), ctypes.byref(size), self.mod, b"xg_cfg"
                ),
                "xg_cfg",
            )
            cfg = (ctypes.c_int * 10)()
            self._ck(
                d.cuMemcpyDtoH_v2(cfg, ptr.value, min(40, size.value)), "xg_cfg copy"
            )
        (
            self.threads,
            self.rpt,
            self.rows,
            self.vsplit,
            self.tmax,
            self.minb,
            self.expmode,
            self.qreg,
            self.fusecopy,
        ) = list(cfg)[:9]
        if self.fusecopy:
            raise RuntimeError(
                "xgdn: a FUSECOPY build needs a workspace this module does not pass"
            )

    def _ck(self, r: int, what: str) -> None:
        if r != 0:
            s = ctypes.c_char_p()
            self.drv.cuGetErrorString(r, ctypes.byref(s))
            raise RuntimeError(f"xgdn {what}: CUDA driver error {r} {s.value!r}")

    def ok(
        self,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        corr,
        key,
        gate,
        key_stage,
        gate_stage,
        out,
        spec_query_len,
    ) -> bool:
        """Whether this build serves these operands (else the caller's Triton kernel
        runs).
        """
        bf, f32 = torch.bfloat16, torch.float32
        if spec_query_len < 1 or (1 << (spec_query_len - 1).bit_length()) > self.tmax:
            return False
        if (
            q.shape[-1] != 128
            or k.shape[-1] != 128
            or v.shape[-1] != 128
            or state.shape[-2:] != (128, 128)
        ):
            return False
        if (
            q.dtype != bf
            or k.dtype != bf
            or v.dtype != bf
            or out.dtype != bf
            or state.dtype != bf
        ):
            return False
        if (
            a.dtype not in (bf, f32)
            or b.dtype != a.dtype
            or dt_bias.dtype not in (bf, f32)
            or A_log.dtype != f32
        ):
            return False
        if any(t.dtype != f32 for t in (corr, key, gate, key_stage, gate_stage)):
            return False
        # the kernel indexes heads as h * 128 and a, b heads as h
        if (
            any(t.stride(-1) != 1 or t.stride(-2) != 128 for t in (q, k, v, out))
            or a.stride(-1) != 1
            or b.stride(-1) != 1
        ):
            return False
        if (
            state.stride(3) != 1
            or state.stride(2) % 8
            or state.stride(1) % 8
            or state.stride(0) % 8
            or state.data_ptr() % 16
        ):
            return False
        if corr.stride(3) != 1 or key.stride(3) != 1 or key_stage.stride(3) != 1:
            return False
        return not (
            q.stride(1) % 4 or k.stride(1) % 4 or q.data_ptr() % 8 or k.data_ptr() % 8
        )

    def verify(
        self,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        corr,
        key,
        gate,
        key_stage,
        gate_stage,
        row_pending,
        out,
        query_start_loc,
        state_indices,
        spec_query_len,
        scale,
        null_block,
    ) -> None:
        batch = state_indices.shape[0]
        num_v_heads = v.shape[2]
        x = _Args()
        for n, t in zip(
            _PTRS,
            (
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                corr,
                key,
                gate,
                key_stage,
                gate_stage,
                row_pending,
                out,
                query_start_loc,
                state_indices,
                None,
            ),
        ):
            setattr(x, n, 0 if t is None else t.data_ptr())
        vals = (
            q.stride(1),
            k.stride(1),
            v.stride(1),
            a.stride(0),
            b.stride(0),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            corr.stride(0),
            corr.stride(1),
            corr.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            key_stage.stride(0),
            key_stage.stride(1),
            key_stage.stride(2),
            gate_stage.stride(0),
            gate_stage.stride(1),
            out.stride(1),
            query_start_loc.stride(0),
            state_indices.stride(0),
            null_block,
            num_v_heads // q.shape[2],
            1 << (spec_query_len - 1).bit_length(),
            (1 if a.dtype == torch.float32 else 0)
            | (2 if dt_bias.dtype == torch.float32 else 0),
        )
        for n, val in zip(_INTS, vals):
            setattr(x, n, int(val))
        x.scale = float(scale)
        args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(x), ctypes.c_void_p))
        stream = torch.cuda.current_stream(q.device).cuda_stream
        self._ck(
            self.drv.cuLaunchKernel(
                self.fn,
                self.vsplit,
                batch,
                num_v_heads,
                self.threads,
                1,
                1,
                0,
                stream,
                args,
                None,
            ),
            "launch",
        )


_modules: dict[int, _Module] = {}


def _module(device: torch.device) -> _Module:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    m = _modules.get(idx)
    if m is None:
        m = _modules[idx] = _Module(torch.device("cuda", idx))
        logger.info_once(
            "XKB_XGDN: GDN fused verify in CUDA (xgdn_verify, %d "
            "threads, %d rows a thread), cubin sha256 %s",
            m.threads,
            m.rpt,
            _CUBIN_SHA256[:16],
        )
    return m


def launcher(triton_launch, spec_query_len: int):
    """`triton_launch` (the Triton kernel bound to its grid) when the switch is off,
    else a function taking its arguments that runs xgdn_verify when it serves them.
    """
    if not ENABLED:
        return triton_launch

    def run(*args, **kwargs):
        (
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state,
            corr,
            key,
            gate,
            key_stage,
            gate_stage,
            row_pending,
            out,
            qsl,
            sidx,
            scale,
            null_block,
        ) = args[:19]
        if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (
            12,
            0,
        ):
            counts["triton"] += 1
            return triton_launch(*args, **kwargs)
        m = _module(q.device)
        if m.ok(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state,
            corr,
            key,
            gate,
            key_stage,
            gate_stage,
            out,
            spec_query_len,
        ):
            m.verify(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                corr,
                key,
                gate,
                key_stage,
                gate_stage,
                row_pending,
                out,
                qsl,
                sidx,
                spec_query_len,
                scale,
                null_block,
            )
            counts["xgdn"] += 1
            return None
        if counts["triton"] == 0:
            logger.warning(
                "XKB_XGDN: operands outside the CUDA build "
                "(spec_query_len %d, q %s %s, state %s %s), the "
                "Triton verify runs",
                spec_query_len,
                tuple(q.shape),
                q.stride(),
                state.dtype,
                tuple(state.shape),
            )
        counts["triton"] += 1
        return triton_launch(*args, **kwargs)

    return run
