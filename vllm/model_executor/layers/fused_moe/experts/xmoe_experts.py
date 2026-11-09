# SPDX-License-Identifier: Apache-2.0
"""NVFP4 weight-only MoE for the Qwen3.8 Flash-Next shapes on SM 12.0 (RTX
PRO 6000), switched on by XKB_XMOE=1.

One persistent kernel (xmoe_mlp) streams each touched expert's weights once with TMA
bulk copies and runs gate/up, SiLU, and down in the same launch. A route kernel
(xmoe_route) groups the (token, slot) pairs by expert, largest experts first, at
most 32 pairs a group. The weights are repacked at load time, in place, into
k-chunks of 16 KB of fp4 codes in MMA fragment order (in the codes tensor) and 2 KB
of block-scale bytes (in the scales tensor). The repack allocates no new weight
memory, because vLLM's CuMem weight pool cannot reuse freed checkpoint blocks for
differently sized tensors. Block scales below Marlin's flush threshold are zeroed as
in Marlin's preparation, so the products match Marlin's up to accumulation order.
The kernel image is the cubin built with nvcc 13.0 for sm_120a from xmoe.cu,
embedded below and loaded through the CUDA driver API. Layers whose shapes or
settings differ keep vLLM's Marlin path.
"""

import base64
import ctypes
import os
import zlib

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts

logger = init_logger(__name__)

H, INTERMEDIATE_SIZE, TOPK = 2560, 640, 10
TMAX, NSTAGE, KC = 32, 3, 128
A_BYTES, S_BYTES = 256 * KC // 2, 256 * KC // 16
W_BYTES = A_BYTES + S_BYTES
SMEM = NSTAGE * (W_BYTES + TMAX * (KC * 2 + 16)) + 2 * NSTAGE * 8 + NSTAGE * 4
THREADS = 9 * 32
CHUNK_ROWS = 2048  # xmoe_route counts at most 2,048 pairs an expert
MAX_EXPERTS = 512

_CUBIN_SHA256 = "addddc43fc81517be272f360a7594f7337c74de25d9f75f8ca02abc49c0bf476"
_CUBIN_ZB64 = (
    "eNrsvX1wXNd1J/i60QAa4Ovu916/7gbI/niNfg02gEZ344sECRACSYCAFFKgE9uxPinKkiPv6IPT"
    "pJ3GWJWQGJZLXKsshZvdkdbJjIbr2cixJtGw9g+5ktpVabBchtFqVSpXVklcXgWLcjFel4vr8mZZ"
    "s056773vvXvvOf0BQhRtT9Gqosjb7+vec889H/f+zjm/u3j0iN/nOxhUnP/8yv+o+BTx3/ya3/nH"
    "v3D+9teUrnllWulU5pUsvbN0+qnTZ6pnTj5O/uX9vfqM9PeJ0089+0RNKT373JknS89+uXTmn33x"
    "2S88J9qf/5Lb/nKJ/+Pzzz1z6uQZpXTmydqZUu2Z5548UX3uS2ee5Dfh304/dbL65BMtfq0+efrJ"
    "6pfJPyoK+ffTjS8lP55s/FX88MzTp9CXvV/k77Lf4Pu9n07i35548vEv/daJL1RPPuN8vuGHk/AX"
    "SpKTTz/9W9WTp55irVPV5848d2b1lHvtuWdPnzn57JkKpgC64HX78ydPPfNk9fNNydt4zXuK/Yy7"
    "xX6Up07+Ab+78Rp4d/N5aHG18clWXWvCBt5Vl1Nv6b9fYkb3hvYbzzz5TOm5L3zh9JNnKi1WwIkT"
    "z375BL//2OKxE879JyonTj79xZOnG99HGEKxT5z44rNnnqw+e/JpciNpff5LT5w8cfqZ8cqJJ774"
    "5RNfGttzx5eW3IUxqQsTNdaF6rMnnn3uC2f+xYkvTIyfOP30c79NyPrUL2I9SuOEa24ba7TNf0Pk"
    "TweRvvi/fez3robfH2G/Rxp+X2O/6w2/v0L+ZKncd9sB9+//hfyxXjP4795/1937vRu9+/+R/Bkg"
    "39UGnfYNV6n8mo9+V2v47lkfvV9TKv/BufF4t/P799j9gYb7VT/9fUfD7wXyu6FJ4y04f02w38V3"
    "z77lfGeWvcdseM9n2O+xht/r5D9b+jf9j/Tvefp3RD139mxBCdbPnS0E2T+de4ot5tKjTYB+3nk2"
    "EAiKd882+c7ZtbMF/JngC+R/cXpF6/DefZH+Rigqf/8p9F1vTgIW+V8PfeC80+nbHOd1NE5vTgNJ"
    "Mc7C0x9jnC82GeeLYpxJ97fjPvhdj5cY35A/L5M/75Pf7v/svQv3HrQOP1c9pbhWDrN96O17yJ/H"
    "6A+nztSIVDxMhI1FhfUXnz555ovPPWudee65p08XLSIanjx5+klrbKJUKVqfpX+VpqeVQ1/64tNP"
    "WExCsZ+q9P9l5/knq6WJPZPjk3vHJk9UlNGT1c8/ZZ1+5sTYeOWkNfqMtWfS8vpJWeF6B+5nja3d"
    "QDmohMnfT9B/60Gl11uvuvO74t4TIn/n3d+D0j0h798G/Lf3rL/Hp/j9fsXf2al0dPt8ftIPf5dP"
    "CewIKse9+/cGnL4kVNbuoovt/8nydqdigXZA6QNtJgektp+O+obO2z7aY6nN/iP3d5AOdOyaV/xH"
    "fUrHCZ8vMHSc885W/5Hxdd7Kn0DhuHKcEOY18ucj8sciPx4nf943yL+jhDfIn9fIn/fJn7fJMI8T"
    "IfgR+XNqgPQxR3iM/LHIAnqT/JknE/A++eO/j66BoPJamjxPngl0qkxg+ly+C6QdOdfRX1ECvUEm"
    "tc92kH/vCTSld1g5BegToitcoqdKOVhq71A+B9q9dCaldo+yDNpBKu2ldjex++V2l1JB811A822h"
    "+e5D8w3nn8231Ibz/TqZb78z3wt9ymNEBdYpgf5P5+83v+5jfwfGksqbX/MplRd9yvy/8ikvf8On"
    "vP9vfIr2TfL364R/hz59y7zS7L9dZJjN/gQKn1Y+2kl44DCZ2wukX39O/r5J5i/pU94kfPr+GZ/y"
    "GOmT9rJPee2/IbNN+jVP+nX2vyN/k37N/3ufcoP8ee1PfMrxPyXX3yTPkT+MZ/a976+Qn+qUHnPk"
    "Hz4yM267z20fd9tBt/2Y27bc9im3rbnts2674LZfDijO2iftXl+dCcGK1LZ2OW36PR9pv0+E/YPS"
    "9RuHnbZ33fqSQrmFXz97wWl717X/XlEWpOvanztt7/rZTfj9127C75/q9CkPSNdfTjptfn3aB77/"
    "/lEf+P5Hj/rA9z864wPfr/y+Txl02+z7r/iYKeG1P/oD2Lb+Lbr+73zMSvPap/4Ytt9024rXH9KW"
    "6W992wfG/xhpy/R+7dtwfB99G47HegM9/wak12tvoOffgM+//O/h+Of/BI7vBmq/9qewffxNOF7l"
    "P8B2x3QXkYWq8hHhu9e+5VPeJuOn8tDyNZOHZ5g81Jg8PMPloWcz0L//yf37H92/f+b+/YbP29dw"
    "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"
    "VoXxn5lpGM+oq6ieqI7qMzP6CX1VOMLjUZz6i8nBBn9Prg9M9JtXT7h5feAFVB94EdYHnmX0Eest"
    "geqXxlUU/4vrj6oaqKedUqF974xXFforguq7G3D9ZiLweRPZV+3jSwh/GXB/FO4/jVdNtJ8i5adw"
    "4ksSKqhHC/djiD+B6tfj/Ch6CMrTGOA34B869TZTzfgtJOSrCvHoegj6A3B9kvVoWpJ/SekL15th"
    "wPWmIvvBMJrZoxk33orWoy+0X28sXqbI4zPjIVQvWYf6xFl/YrxOfG+IjzcUQvlI4ig+PQztq63m"
    "OxSX/XEy36jerBqG+2OxGNxPTYQMcL8RhuNr7i+GUby+JvR3uH18yjybT1FPVoqfkPy7cIv6vkR/"
    "RSX9RddbVM4/g/fLAD+w/dNpNt+t67/OsvkW9nQi3Cw+P8zlJ5wPL54kwq/HU1Df4XrDqQiOj5bj"
    "yej4oH8Yw/5hBJ5PJIxm/mFE+LMRWX/Ha9FoO3+xcT9l0hwA+hHHvzj+YORW4lEc+z7a3j5MGVye"
    "sfxAUnwK2+8Iy/Y/W8/Dkj8L7L/vO+t5sG19Xse/m295PakOt+cf5q/NtoxneW2b9Xu3G8+SR/Es"
    "9seKZ7EF/gvFB0S3iGeZnNwengLgG28hv/Lt1mfTrbs9PuDU/yzjKW4o15U7iq81fh7xLDaPZ9EQ"
    "v26Fr52a2h6/AnwjzK/8q/psdyb/7VWI/7lNft11O/ja7cez7NmzPf4C+EaYX/lX9dnuTP2/a7D+"
    "3+ad5S/jk41n2bt3e/wF8I4w3/JW9dke/zh42GiutfzK3BXxAR+9C+v/3SZ/RW8HD/tJxbPYHzue"
    "pVLZHr8mk5hfK231bVJJ3l48S/Juj2d5+X+T+fUp5aM7za+fSP3eW7UP2/gzTfl1bGx7/JpKYX4d"
    "a8uvKSV1e/Zh6m63D+c/kPn1z5Xv3Wn9/YnW7x0f3x5/pdOYv8bb2odpJX179mH6brcPX/surP/3"
    "vTvtf3yi9XsnJrbHX5kM5q+JtvZhRsncln2oZ+6sffifQf2//x3W//vwTu/HfCL1e++UfQjil2+B"
    "X2dnMb8W2+rbWSee5+Pbh7N3u3345l/L/PqR8t1fKvuwv0X93lb8avej/W4b2Yc2HVG+Jb+C+OVb"
    "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"
    "/xXLRUW54Pfz5954+pV7IAb/2uX+P7vubJ0="
)


def enabled() -> bool:
    return os.environ.get("XKB_XMOE", "0") == "1"


_drv = None


def _driver():
    global _drv
    if _drv is None:
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
        d.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        d.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _drv = d
    return _drv


def _ck(r: int, what: str) -> None:
    if r != 0:
        s = ctypes.c_char_p()
        _driver().cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"xmoe {what}: CUDA driver error {r} {s.value!r}")


class _Kernels:
    """The cubin's two kernels, loaded into the device's primary context."""

    def __init__(self, device: torch.device):
        with torch.cuda.device(device):
            torch.empty(
                1, device=device
            )  # makes torch's primary context current on this thread
            ctx = ctypes.c_void_p()
            _ck(_driver().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            if not ctx.value:
                raise RuntimeError("xmoe: no current CUDA context")
            self.image = zlib.decompress(base64.b64decode(_CUBIN_ZB64))
            self.mod = ctypes.c_void_p()
            _ck(
                _driver().cuModuleLoadData(ctypes.byref(self.mod), self.image),
                "cuModuleLoadData",
            )
            self.fn = {}
            for name in ("xmoe_route", "xmoe_mlp"):
                f = ctypes.c_void_p()
                _ck(
                    _driver().cuModuleGetFunction(
                        ctypes.byref(f), self.mod, name.encode()
                    ),
                    name,
                )
                self.fn[name] = f
            _ck(
                _driver().cuFuncSetAttribute(self.fn["xmoe_mlp"], 8, SMEM),
                "max dynamic shared memory",
            )
            self.sms = torch.cuda.get_device_properties(device).multi_processor_count

    def _launch(self, name: str, grid: int, block: int, smem: int, args: list) -> None:
        vals = [
            ctypes.c_void_p(a.data_ptr())
            if isinstance(a, torch.Tensor)
            else ctypes.c_int(a)
            for a in args
        ]
        ptrs = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals]
        )
        stream = torch.cuda.current_stream().cuda_stream
        _ck(
            _driver().cuLaunchKernel(
                self.fn[name], grid, 1, 1, block, 1, 1, smem, stream, ptrs, None
            ),
            name,
        )

    def route(self, topk_ids, npairs, E, sorted_pair, groups, ngroups, done) -> None:
        self._launch(
            "xmoe_route",
            1,
            1024,
            0,
            [topk_ids, npairs, E, sorted_pair, groups, ngroups, done],
        )

    def mlp(
        self,
        w13,
        s13,
        osc13,
        w2,
        s2,
        osc2,
        x,
        topk_w,
        sorted_pair,
        groups,
        ngroups,
        act,
        y,
        done,
    ) -> None:
        self._launch(
            "xmoe_mlp",
            self.sms,
            THREADS,
            SMEM,
            [
                w13,
                s13,
                osc13,
                w2,
                s2,
                osc2,
                x,
                sorted_pair,
                groups,
                ngroups,
                topk_w,
                act,
                y,
                done,
                3,
            ],
        )


_kernels: dict[int, _Kernels] = {}


def _get_kernels(device: torch.device) -> _Kernels:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    k = _kernels.get(idx)
    if k is None:
        k = _kernels[idx] = _Kernels(torch.device("cuda", idx))
    return k


# ---- weight repack
# --------------------------------------------------------------------------------------
# One 32-bit word (lane) of an m16 x k16 fragment: nibble n holds W[g + dr][2t + dk]
# with (dr, dk) below.
_NIB = [(8, 8), (0, 8), (8, 0), (0, 0), (8, 9), (0, 9), (8, 1), (0, 1)]


def _index(gated: bool, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat (row * K + k) source index of every nibble of a blob, and of every scale
    byte.
    """
    K = H if gated else INTERMEDIATE_SIZE
    RT = 5 if gated else 10
    NSK = K // KC
    rt = torch.arange(RT, device=dev).view(RT, 1, 1, 1, 1, 1, 1)
    s = torch.arange(NSK, device=dev).view(1, NSK, 1, 1, 1, 1, 1)
    w = torch.arange(8, device=dev).view(1, 1, 8, 1, 1, 1, 1)
    i = torch.arange(4, device=dev).view(1, 1, 1, 4, 1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1, 1)
    wd = torch.arange(4, device=dev).view(1, 1, 1, 1, 1, 4, 1)
    nib = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 1, 8)
    mt = wd % 2
    ks = 2 * i + wd // 2
    if gated:
        base = torch.where(
            mt == 0, rt * 128 + w * 16, INTERMEDIATE_SIZE + rt * 128 + w * 16
        )
    else:
        base = rt * 256 + w * 32 + mt * 16
    g, t = lane // 4, lane % 4
    dr = torch.tensor([d[0] for d in _NIB], device=dev)[nib]
    dk = torch.tensor([d[1] for d in _NIB], device=dev)[nib]
    row = base + g + dr
    k = s * KC + ks * 16 + 2 * t + dk
    aidx = (row * K + k).reshape(RT, NSK, -1)
    # scale bytes: [w 8][i2 2][g 8][ks' 4][mt 2][h 2]
    w = torch.arange(8, device=dev).view(1, 1, 8, 1, 1, 1, 1, 1)
    i2 = torch.arange(2, device=dev).view(1, 1, 1, 2, 1, 1, 1, 1)
    g = torch.arange(8, device=dev).view(1, 1, 1, 1, 8, 1, 1, 1)
    ksp = torch.arange(4, device=dev).view(1, 1, 1, 1, 1, 4, 1, 1)
    mt = torch.arange(2, device=dev).view(1, 1, 1, 1, 1, 1, 2, 1)
    h = torch.arange(2, device=dev).view(1, 1, 1, 1, 1, 1, 1, 2)
    rt = rt.view(RT, 1, 1, 1, 1, 1, 1, 1)
    s = s.view(1, NSK, 1, 1, 1, 1, 1, 1)
    if gated:
        base = torch.where(
            mt == 0, rt * 128 + w * 16, INTERMEDIATE_SIZE + rt * 128 + w * 16
        )
    else:
        base = rt * 256 + w * 32 + mt * 16
    row = base + g + 8 * h
    grp = s * (KC // 16) + i2 * 4 + ksp
    sidx = (row * (K // 16) + grp).reshape(RT, NSK, -1)
    return aidx, sidx


def _scale_bytes(s: torch.Tensor) -> torch.Tensor:
    """Positive block scales (float) -> private E5M3 bytes whose bf16 image (b << 4 |
    0x4000) is s * 2^20; zero stays zero.
    """
    m, e = torch.frexp(s)
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = e - 1 + 19
    b = (e5 << 3) | mant
    b = torch.where(s == 0, torch.zeros_like(b), b)
    if int(b.max()) >= 256 or not bool((s == 0).logical_or(e5 >= 1).all()):
        raise ValueError("xmoe: block scale outside the E5M3 range")
    return b.to(torch.uint8)


@torch.no_grad()
def _scan_scales(
    scales: torch.Tensor, param_dtype: torch.dtype, chunk: int = 16
) -> float:
    """Checks one weight's block scales (no negative or NaN values) and returns the
    power-of-2 factor Marlin's NVFP4 MoE preparation applies to all its experts
    (vLLM's `_nvfp4_compute_scale_factor`), in chunks of experts with one host sync.
    """
    bad = torch.zeros((), dtype=torch.bool, device=scales.device)
    mx = torch.zeros((), dtype=torch.float32, device=scales.device)
    for e0 in range(0, scales.shape[0], chunk):
        f = scales[e0 : e0 + chunk].float()
        bad |= ((f < 0) | f.isnan()).any()
        ws = scales[e0 : e0 + chunk].to(param_dtype).float() * 2**7
        mx = torch.maximum(
            mx, torch.where(ws > 0, ws, 0.0).amax()
        )  # max of the positive scales, 0 if none
        del f, ws
    if bool(bad):
        raise ValueError("xmoe: negative or NaN NVFP4 block scale")
    if 0 < float(mx) < 448 * 2**7:
        return float((448 * 2**7 / mx).log2().floor().exp2())
    return 1.0


@torch.no_grad()
def _repack(codes, scales, gscale, gated, param_dtype, chunk=8):
    """Rewrites codes uint8 [E, N, K/2] (low nibble = even k) and scales e4m3 [E, N,
    K/16] in place into xmoe's layout: codes as [E, RT * NSK, A_BYTES] packed A
    words, scales as [E, RT * NSK, S_BYTES] private scale bytes (same byte counts).
    Returns the oscale fp32 [E]; gscale is not changed. All checks run before the
    first write, and each chunk of experts is read in full before it is overwritten.
    """
    E, N, K2 = codes.shape
    K = K2 * 2
    sf = _scan_scales(scales, param_dtype)
    aidx, sidx = _index(gated, codes.device)
    RT, NSK = aidx.shape[:2]
    if N * K2 != RT * NSK * A_BYTES or N * (K // 16) != RT * NSK * S_BYTES:
        raise ValueError("xmoe: weight sizes do not match the k-chunk layout")
    cw = codes.view(E, RT * NSK, A_BYTES)
    sw = scales.view(torch.uint8).view(E, RT * NSK, S_BYTES)
    for e0 in range(0, E, chunk):
        c = codes[e0 : e0 + chunk]
        n = c.shape[0]
        sc = scales[e0 : e0 + chunk].float()
        sc = torch.where(
            sc * sf * 2**7 < 2, torch.zeros_like(sc), sc
        )  # blocks Marlin flushes to zero
        nib = torch.stack([c & 15, c >> 4], -1).view(n, N, K)
        nib = nib.masked_fill((sc == 0).repeat_interleave(16, -1), 0).view(n, -1)
        a = nib[:, aidx.view(-1)].view(n, RT * NSK, A_BYTES, 2)
        del nib
        packed = a[..., 0] | (a[..., 1] << 4)
        del a
        sb = _scale_bytes(sc).view(n, -1)[:, sidx.view(-1)].view(n, RT * NSK, S_BYTES)
        cw[e0 : e0 + n] = packed
        sw[e0 : e0 + n] = sb
        del packed, sb, sc
    return gscale.float().contiguous() * 2.0**106


def _unsupported(
    layer, w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2, is_act_and_mul
) -> str | None:
    if not torch.cuda.is_available() or w13.device.type != "cuda":
        return "not on CUDA"
    if torch.cuda.get_device_capability(w13.device) != (12, 0):
        return (
            f"device capability {torch.cuda.get_device_capability(w13.device)} "
            "is not (12, 0)"
        )
    cfg = layer.moe_config
    E = w13.shape[0]
    checks = [
        (is_act_and_mul, "not act-and-mul"),
        (
            getattr(cfg, "activation", None) == MoEActivation.SILU,
            f"activation {getattr(cfg, 'activation', None)}",
        ),
        (getattr(cfg, "swiglu_limit", None) is None, "swiglu limit"),
        (not getattr(cfg, "has_bias", False), "bias"),
        (not getattr(cfg, "is_lora_enabled", False), "LoRA"),
        (
            getattr(layer, "top_k", None) == TOPK,
            f"top-k {getattr(layer, 'top_k', None)}",
        ),
        (layer.params_dtype == torch.bfloat16, f"params dtype {layer.params_dtype}"),
        (
            not cfg.moe_parallel_config.use_ep
            and not cfg.moe_parallel_config.enable_eplb,
            "expert parallel or EPLB",
        ),
        (1 <= E <= MAX_EXPERTS, f"{E} experts"),
        (
            w13.dtype == torch.uint8
            and tuple(w13.shape) == (E, 2 * INTERMEDIATE_SIZE, H // 2),
            f"w13 {w13.dtype} {tuple(w13.shape)}",
        ),
        (
            w2.dtype == torch.uint8
            and tuple(w2.shape) == (E, H, INTERMEDIATE_SIZE // 2),
            f"w2 {w2.dtype} {tuple(w2.shape)}",
        ),
        (
            w13_scale.dtype == torch.float8_e4m3fn
            and tuple(w13_scale.shape) == (E, 2 * INTERMEDIATE_SIZE, H // 16),
            f"w13 scale {tuple(w13_scale.shape)}",
        ),
        (
            w2_scale.dtype == torch.float8_e4m3fn
            and tuple(w2_scale.shape) == (E, H, INTERMEDIATE_SIZE // 16),
            f"w2 scale {tuple(w2_scale.shape)}",
        ),
        (w13_scale_2.numel() == E and w2_scale_2.numel() == E, "global scales"),
        (
            all(t.is_contiguous() for t in (w13, w13_scale, w2, w2_scale)),
            "non-contiguous weights (the repack runs in place)",
        ),
    ]
    for ok, why in checks:
        if not ok:
            return why
    return None


def prepare(
    layer, w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2, is_act_and_mul
):
    """Repacks one layer for xmoe and marks it; returns (w13, w13_scale, w13_scale_2,
    w2, w2_scale, w2_scale_2) for convert_to_nvfp4_moe_kernel_format, or None when
    the layer keeps Marlin.
    """
    try:
        why = _unsupported(
            layer, w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2, is_act_and_mul
        )
    except Exception as exc:  # noqa: BLE001
        why = f"check failed: {exc!r}"
    if why is None:
        try:
            _get_kernels(w13.device)
        except Exception as exc:  # noqa: BLE001
            why = f"kernel load failed: {exc!r}"
    if why is not None:
        logger.warning("XKB_XMOE: layer keeps Marlin (%s)", why)
        return None
    # From here on the layer's codes and block scales hold xmoe's layout; a
    # failure raises instead of falling back to Marlin.
    osc13 = _repack(w13, w13_scale, w13_scale_2.reshape(-1), True, layer.params_dtype)
    osc2 = _repack(w2, w2_scale, w2_scale_2.reshape(-1), False, layer.params_dtype)
    layer._xkb_xmoe = True
    logger.info_once(
        "XKB_XMOE: xmoe kernel for NVFP4 MoE layers (cubin sha256 %s)",
        _CUBIN_SHA256[:16],
    )
    return w13, w13_scale, osc13, w2, w2_scale, osc2


def _r(n: int, a: int = 64) -> int:
    return (n + a - 1) // a * a


def _scratch_ints(npairs: int, E: int) -> int:
    maxg = E + npairs // TMAX + 2
    return _r(npairs) + _r(4 * maxg) + 64 + _r(maxg)


class XmoeExperts(MarlinExperts):
    """MarlinExperts that runs layers prepared by `prepare` through xmoe; layers that
    kept Marlin's format run Marlin.
    """

    _xmoe = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        self._xmoe = bool(getattr(layer, "_xkb_xmoe", False))
        if self._xmoe:
            cfg = self.activation_config
            assert cfg.clamp_limit is None, "xmoe: SwiGLU clamp"

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        if not self._xmoe:
            return super().moe_problem_size(a1, w1, w2, topk_ids)
        assert a1.dim() == 2 and topk_ids.dim() == 2 and topk_ids.size(0) == a1.size(0)
        return w1.size(0), a1.size(0), INTERMEDIATE_SIZE, a1.size(-1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        if not self._xmoe:
            return super().workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        npc = min(M, CHUNK_ROWS) * topk
        E = max(global_num_experts, local_num_experts)
        # workspace13: [output alias M * K][act npc * I][int32 scratch];
        # workspace2: per-slot outputs npc * K
        ws13 = (_r(M * K) + _r(npc * INTERMEDIATE_SIZE) + 2 * _scratch_ints(npc, E),)
        return ws13, (npc * K,), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if not self._xmoe:
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        assert expert_map is None and not apply_router_weight_on_input, (
            "xmoe: expert map or router weight on input"
        )
        assert activation == MoEActivation.SILU and self._lora_context is None, (
            "xmoe: activation or LoRA"
        )
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.size(-1) == H, (
            "xmoe: input"
        )
        assert topk_ids.size(1) == TOPK, "xmoe: top-k"
        M = hidden_states.size(0)
        E = w1.size(0)
        x = hidden_states.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        k = _get_kernels(x.device)
        npc = min(M, CHUNK_ROWS) * TOPK
        maxg = E + npc // TMAX + 2
        o = _r(M * H)
        act = workspace13[o : o + npc * INTERMEDIATE_SIZE]
        o += _r(npc * INTERMEDIATE_SIZE)
        ints = workspace13[o : o + 2 * _scratch_ints(npc, E)].view(torch.int32)
        sorted_pair = ints[:npc]
        p = _r(npc)
        groups = ints[p : p + 4 * maxg]
        p += _r(4 * maxg)
        ngroups = ints[p : p + 4]
        done = ints[p + 64 : p + 64 + maxg]
        for m0 in range(0, M, CHUNK_ROWS):
            m1 = min(M, m0 + CHUNK_ROWS)
            n = (m1 - m0) * TOPK
            y = workspace2[: n * H]
            k.route(ids[m0:m1], n, E, sorted_pair, groups, ngroups, done)
            k.mlp(
                w1,
                self.w1_scale,
                self.g1_alphas,
                w2,
                self.w2_scale,
                self.g2_alphas,
                x[m0:m1],
                tw[m0:m1],
                sorted_pair,
                groups,
                ngroups,
                act,
                y,
                done,
            )
            ops.moe_sum(y.view(m1 - m0, TOPK, H), output[m0:m1])
