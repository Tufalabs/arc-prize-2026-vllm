# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN recurrent state in FP8 (``mamba_ssm_cache_dtype="fp8_e4m3"``).

Each state row -- one value channel of one head, its K key channels -- is
K e4m3 values with a float32 scale of its own. A head's rows come first, at
a pitch of K bytes, and their V scales after them, so a head is
``V * (K + SCALE_BYTES)`` bytes and a state keeps the ``[..., V, K +
SCALE_BYTES]`` shape of those bytes (its pages and strides are those of
rows with the scale inline); only a whole head is meaningful. With K a
multiple of 16 every row starts 16-byte aligned, which the scale inline in
the row (a pitch of K + 4) prevented: on a B200 at Flash-Next's decode
shape the RecoverSSM verify took 4.8 ms a step with the scale inline,
against 3.0 with the rows padded to a 16-byte pitch. The scales live in the
page they scale: a page copy (prefix-cache align copies, the host tier)
moves both.

The scale is the row's absolute maximum over the e4m3 maximum, recomputed at
every write. Values round stochastically: the state is rewritten at every
step and many of its heads decay slowly, so round-to-nearest would drop
every change smaller than half a step, the same way at every step; a
stochastic round keeps the expected value exact. The random stream is
seeded from a device counter that every launch advances, so CUDA graph
replays draw fresh numbers.

The kernels that read or write the state call ``load_fp8_state_rows`` and
``store_fp8_state_rows`` with a head's address and the rows they cover, on
whole rows: their tiles hold every key channel of the rows they cover, so a
row's maximum is the tile's own.
"""

import torch

from vllm.triton_utils import tl, triton

FP8_STATE_DTYPE = torch.float8_e4m3fn
#: The float32 scale after each row's K values.
SCALE_BYTES = 4


def is_fp8_state(state: torch.Tensor) -> bool:
    return state.dtype == FP8_STATE_DTYPE


def state_key_dim(state: torch.Tensor) -> int:
    """K of a ``[..., V, K]`` state, or of its FP8 rows ``[..., V, K + 4]``."""
    return state.shape[-1] - SCALE_BYTES if is_fp8_state(state) else state.shape[-1]


def check_fp8_state_layout(state: torch.Tensor) -> None:
    """The kernels address a head's rows and then its scales from the head's
    start: a head's bytes must be contiguous, and the scales after V rows of
    K bytes float32-aligned, so every head starts 4-byte aligned and K is a
    multiple of 4."""
    if state.stride(-1) != 1 or state.stride(-2) != state.shape[-1]:
        raise ValueError("FP8 GDN state heads must be contiguous")
    if (
        state_key_dim(state) % SCALE_BYTES
        or state.data_ptr() % SCALE_BYTES
        or any(stride % SCALE_BYTES for stride in state.stride()[:-2])
    ):
        raise ValueError("FP8 GDN state heads must start 4-byte aligned")


_SEEDS: dict[tuple[str, int], torch.Tensor] = {}


def rounding_seed(device: torch.device) -> torch.Tensor:
    """The device's stochastic-rounding counter, one int32 that each launch
    reads and then advances (``advance_rounding_seed``)."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, index)
    seed = _SEEDS.get(key)
    if seed is None:
        seed = torch.zeros(
            1, dtype=torch.int32, device=torch.device(device.type, index)
        )
        _SEEDS[key] = seed
    return seed


def advance_rounding_seed(seed: torch.Tensor) -> None:
    # A device op, so a captured graph advances it on every replay.
    seed.add_(1)


@triton.jit
def _exp2_int(e):
    """2**e for an integer e in float32's normal range, exactly."""
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def uniform_hash(seed, offsets):
    """Uniform in [0, 1) at 24 bits for each int32 offset under ``seed``: the
    offset advanced by the seed's golden-ratio multiple, then the lowbias32
    integer hash. A stochastic round needs no stronger stream, and one hash
    an element costs a fraction of Philox's ten rounds (``tl.rand``), which
    made the FP8 commit four times slower than the float32 one."""
    x = offsets.to(tl.uint32) + seed.to(tl.uint32) * 1640531527
    x ^= x >> 16
    x *= 569420461
    x ^= x >> 15
    x *= 1935289751
    x ^= x >> 15
    return (x >> 8).to(tl.float32) * 5.9604644775390625e-08


@triton.jit
def stochastic_round_e4m3(y, u):
    """``y`` (float32, within +-448) on the e4m3 grid: down or up to its two
    neighbours with the probabilities that keep its expectation, ``u``
    uniform in [0, 1). The step is a power of two, so scaling by its
    reciprocal is exact."""
    bits = y.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    # Subnormals share the smallest normal binade's step, 2**-9.
    exponent = tl.maximum(exponent, -6)
    rounded = tl.floor(y * _exp2_int(3 - exponent) + u) * _exp2_int(exponent - 3)
    return tl.minimum(tl.maximum(rounded, -448.0), 448.0)


@triton.jit
def _fp8_state_scale_ptrs(head_ptr, offs_v, K: tl.constexpr, V: tl.constexpr):
    """The float32 scales of rows ``offs_v`` of the head at ``head_ptr``,
    after the head's V rows of K bytes."""
    return (head_ptr + V * K + offs_v * 4).to(tl.pointer_type(tl.float32))


@triton.jit
def load_fp8_state_rows(
    head_ptr, offs_v, offs_k, mask_v, mask_k, K: tl.constexpr, V: tl.constexpr
):
    """Float32 ``[BV, BK]`` from rows ``offs_v`` ``[BV]`` of the FP8 head at
    ``head_ptr``."""
    values = tl.load(
        head_ptr + offs_v[:, None] * K + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    scales = tl.load(
        _fp8_state_scale_ptrs(head_ptr, offs_v, K, V), mask=mask_v, other=0.0
    )
    return values * scales[:, None]


@triton.jit
def store_fp8_state_rows(
    head_ptr,
    offs_v,
    offs_k,
    state,
    mask_v,
    mask_k,
    seed,
    rand_offsets,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """Write float32 ``state`` ``[BV, BK]`` as rows ``offs_v`` ``[BV]`` of the
    FP8 head at ``head_ptr``, each with its own scale, rounded stochastically
    (``rand_offsets`` int32 ``[BV, BK]``, distinct within a launch)."""
    mask = mask_v[:, None] & mask_k[None, :]
    amax = tl.max(tl.where(mask, tl.abs(state), 0.0), axis=1)
    scales = tl.where(amax > 0.0, amax / 448.0, 1.0)
    # A reciprocal a row and a product an element, not a division an element.
    scaled = state * tl.where(amax > 0.0, 448.0 / amax, 1.0)[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    scaled = stochastic_round_e4m3(scaled, uniform_hash(seed, rand_offsets))
    tl.store(
        head_ptr + offs_v[:, None] * K + offs_k[None, :],
        scaled.to(head_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(_fp8_state_scale_ptrs(head_ptr, offs_v, K, V), scales, mask=mask_v)


@triton.jit
def _gather_fp8_states_kernel(
    state_ptr,
    out_ptr,
    indices_ptr,
    stride_state_slot,
    stride_state_head,
    stride_out_n,
    stride_out_head,
    stride_out_row,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_v, pid_n, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(indices_ptr + pid_n).to(tl.int64)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    mask_v = (offs_v < V) & (slot >= 0)
    mask_k = offs_k < K
    head_ptr = (
        state_ptr + tl.maximum(slot, 0) * stride_state_slot + pid_h * stride_state_head
    )
    values = load_fp8_state_rows(head_ptr, offs_v, offs_k, mask_v, mask_k, K, V)
    out_ptrs = (
        out_ptr
        + pid_n * stride_out_n
        + pid_h * stride_out_head
        + offs_v[:, None] * stride_out_row
        + offs_k[None, :]
    )
    tl.store(out_ptrs, values, mask=(offs_v < V)[:, None] & mask_k[None, :])


@triton.jit
def _scatter_fp8_states_kernel(
    state_ptr,
    values_ptr,
    indices_ptr,
    seed_ptr,
    stride_state_slot,
    stride_state_head,
    stride_values_n,
    stride_values_head,
    stride_values_row,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_v, pid_n, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(indices_ptr + pid_n).to(tl.int64)
    if slot < 0:
        return
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    mask_v = offs_v < V
    mask_k = offs_k < K
    values = tl.load(
        values_ptr
        + pid_n * stride_values_n
        + pid_h * stride_values_head
        + offs_v[:, None] * stride_values_row
        + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    head_ptr = state_ptr + slot * stride_state_slot + pid_h * stride_state_head
    rand_offsets = (
        ((slot * HV + pid_h) * V + offs_v[:, None]) * K + offs_k[None, :]
    ).to(tl.int32)
    store_fp8_state_rows(
        head_ptr,
        offs_v,
        offs_k,
        values,
        mask_v,
        mask_k,
        tl.load(seed_ptr),
        rand_offsets,
        K,
        V,
    )


def _grid_dims(state: torch.Tensor) -> tuple[int, int, int, int, int]:
    num_heads, value_dim = state.shape[1], state.shape[2]
    key_dim = state_key_dim(state)
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    return num_heads, value_dim, key_dim, block_k, block_v


def gather_fp8_states(state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Float32 ``[n, HV, V, K]`` of the FP8 states at ``indices`` ``[n]``; a
    negative index reads as zeros."""
    check_fp8_state_layout(state)
    num_heads, value_dim, key_dim, block_k, block_v = _grid_dims(state)
    out = torch.empty(
        (indices.shape[0], num_heads, value_dim, key_dim),
        dtype=torch.float32,
        device=state.device,
    )
    if indices.shape[0] == 0:
        return out
    grid = (triton.cdiv(value_dim, block_v), indices.shape[0], num_heads)
    _gather_fp8_states_kernel[grid](
        state,
        out,
        indices,
        state.stride(0),
        state.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
    )
    return out


def scatter_fp8_states_(
    state: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> None:
    """Write ``values`` ``[n, HV, V, K]`` as the FP8 states at ``indices``
    ``[n]``; a negative index is skipped."""
    check_fp8_state_layout(state)
    num_heads, value_dim, key_dim, block_k, block_v = _grid_dims(state)
    if values.shape != (indices.shape[0], num_heads, value_dim, key_dim):
        raise ValueError(
            f"FP8 GDN state values {tuple(values.shape)} do not match "
            f"{indices.shape[0]} states of {(num_heads, value_dim, key_dim)}"
        )
    if values.stride(-1) != 1:
        values = values.contiguous()
    if indices.shape[0] == 0:
        return
    seed = rounding_seed(state.device)
    grid = (triton.cdiv(value_dim, block_v), indices.shape[0], num_heads)
    _scatter_fp8_states_kernel[grid](
        state,
        values,
        indices,
        seed,
        state.stride(0),
        state.stride(1),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        HV=num_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
    )
    advance_rounding_seed(seed)
