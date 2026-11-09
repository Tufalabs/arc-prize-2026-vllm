# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GatedDeltaNet RecoverSSM speculative verify and accepted-state recovery.

The Kimi-K3 KDA RecoverSSM (``vllm.models.kimi_k3.nvidia.ops.recoverssm``)
for the gated delta rule: the verify runs a speculative window from the
unmodified checkpoint, writing no per-token states, and records per position
the correction ``beta * (v - S k)``, the L2-normalised key and the gate. The
commit then replays the accepted positions onto the checkpoint with those
records, ``S <- S * exp(g_t) + c_t k_t^T``, the baseline kernel's own update:
the committed state is the one the baseline would have stored.

GDN differs from KDA in the gate, a scalar a value head
(``-exp(A_log) * softplus(a + dt_bias)``, where KDA's is a vector over the
key dimension), and in grouped keys (``num_v_heads // num_k_heads`` value
heads share one key head). The record keeps the key per key head and the gate
as a float32 scalar, both as the verify computed them, so the commit repeats
no normalisation and no gate arithmetic.

A layer's page holds ``(conv, state, correction, key, gate)``; a conv-only
layer (the Qwen4Exp PLE short convolution) holds ``(conv,)`` and takes part
in the commit's conv-window compaction only.

The verify steps each request's own window, so a maximum window wider than
the bulk of the batch's costs nothing, and computes a position's q/k norm
factors, gate and beta once a program rather than once a position.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops import xkb_xgdn
from vllm.model_executor.layers.mamba.ops.gdn_fp8_state import (
    advance_rounding_seed,
    check_fp8_state_layout,
    is_fp8_state,
    load_fp8_state_rows,
    rounding_seed,
    state_key_dim,
    store_fp8_state_rows,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# Launch shapes, (value-tile rows, warps, stages), from
# benchmarks/kernels/benchmark_gdn_recoverssm.py on a B200 at Flash-Next's
# shapes, 41 requests, k=1 (ms a step, 36 layers): the verify 1.47 at
# (16, 1, 2) against 1.55 at FLA's (32, 4, 2); the commit 1.08 at (8, 1, 2)
# against 1.78 at (32, 4, 2). At k=3 with a float32 state, the verify and the
# layer's gated norm after it: 3.33 at (16, 1, 2), 3.61 at (32, 2, 1); a
# program holding a whole value head, to fuse the norm, 3.52 at best (4 warps).
VERIFY_LAUNCH = (16, 1, 2)
COMMIT_LAUNCH = (8, 1, 2)
# On the 10.x family, read at the B200 cell's own shape (216 requests, k=4, a
# bf16 state): the verify runs 32-row tiles, four a program, and the commit
# takes a whole head a program: 13.42 -> 11.24 and 8.07 -> 6.36 ms a step (36
# layers). Taking tiles in turn shares a head's prologue across them and leaves
# each tile's arithmetic as it was, so the commit stores the same states; the
# verify's wider tile sums each row in another order. Read on a B200 alone, so
# every other card keeps one tile a program at the shapes above, the kernels
# it ran before.
SM100_VERIFY_LAUNCH = (32, 1, 2)
# Value tiles a program takes in turn, sharing a head's prologue across them.
SM100_VERIFY_TILES = 4
SM100_COMMIT_TILES = 16


def _sm100_shapes() -> bool:
    """Whether this card takes the 10.x family's shapes above; off by
    ``VLLM_SM100_KERNEL_SHAPES=0`` on every card."""
    return (
        envs.VLLM_SM100_KERNEL_SHAPES
        and current_platform.is_device_capability_family(100)
    )


def _verify_launch() -> tuple[tuple[int, int, int], int]:
    """The verify's launch shape and value tiles a program on this card."""
    if _sm100_shapes():
        return SM100_VERIFY_LAUNCH, SM100_VERIFY_TILES
    return VERIFY_LAUNCH, 1


def _commit_tiles() -> int:
    """The commit's value tiles a program on this card."""
    return SM100_COMMIT_TILES if _sm100_shapes() else 1


# The fused mode's commit by row (_commit_gdn_state_layers_kernel), on an SM 12.0 card
# with 0, 3 and 7 of 66 rows committing: (8, 2, 1) as fast as (8, 1, 2) within
# 0.005 ms a step, and its 12 unrolled layers compile in about 4 s against 8 s
# (the commit runs outside the warmup, so a cold cache compiles it mid-run).
COMMIT_BY_ROW_LAUNCH = (8, 2, 1)


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    key_cache_ptr,
    gate_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_a_token,
    stride_b_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_key_block,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_gate_block,
    stride_gate_head,
    stride_gate_pos,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    GROUP: tl.constexpr,
    STATE_FP8: tl.constexpr,
    TILES: tl.constexpr,
):
    # A program takes TILES consecutive value tiles of one request's head, in
    # turn, so what a head shares across its tiles -- the window's q/k norms,
    # gate and beta, the request's metadata -- is loaded and computed once.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // GROUP

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_t = tl.arange(0, BT)
    mask_k = offs_k < K
    mask_t = offs_t < query_len
    mask_tk = mask_t[:, None] & mask_k[None, :]
    tokens = bos + offs_t

    if state_idx <= null_block_id:
        for tile in range(TILES):
            offs_v = (pid_v * TILES + tile) * BV + tl.arange(0, BV)
            tl.store(
                out_ptr
                + tokens[:, None] * stride_out_token
                + pid_h * V
                + offs_v[None, :],
                tl.zeros([BT, BV], dtype=tl.float32).to(out_ptr.dtype.element_ty),
                mask=mask_t[:, None] & (offs_v < V)[None, :],
            )
        return

    # Per position, once a program rather than once a position: the q/k
    # L2-norm factors, the gate and beta, with the records no value tile owns
    # (the gate once a value head, the normalised key once a key head). The
    # arithmetic and its order follow
    # fused_sigmoid_gating_delta_rule_update_kernel (beta 1, threshold 20, the
    # q/k L2 norm in the kernel), the path this verify replaces.
    q_window = tl.load(
        q_ptr + tokens[:, None] * stride_q_token + pid_kh * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    k_window = tl.load(
        k_ptr + tokens[:, None] * stride_k_token + pid_kh * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    q_rnorms = tl.rsqrt(tl.sum(q_window * q_window, 1) + 1e-6)
    k_rnorms = tl.rsqrt(tl.sum(k_window * k_window, 1) + 1e-6)
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    raw_a = tl.load(a_ptr + tokens * stride_a_token + pid_h, mask=mask_t, other=0.0)
    raw_b = tl.load(b_ptr + tokens * stride_b_token + pid_h, mask=mask_t, other=0.0)
    x = raw_a.to(tl.float32) + dt_bias
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    gates = neg_A * softplus_x
    betas = tl.sigmoid(raw_b.to(tl.float32))
    if pid_v == 0:
        tl.store(
            gate_cache_ptr
            + state_idx * stride_gate_block
            + pid_h * stride_gate_head
            + offs_t * stride_gate_pos,
            gates,
            mask=mask_t,
        )
        if pid_h % GROUP == 0:
            tl.store(
                key_cache_ptr
                + state_idx * stride_key_block
                + pid_kh * stride_key_head
                + offs_t[:, None] * stride_key_pos
                + offs_k[None, :] * stride_key_dim,
                k_window * k_rnorms[:, None],
                mask=mask_tk,
            )

    for tile in range(TILES):
        offs_v = (pid_v * TILES + tile) * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]
        if STATE_FP8:
            state = load_fp8_state_rows(
                state_ptr + state_idx * stride_state_block + pid_h * stride_state_head,
                offs_v,
                offs_k,
                mask_v,
                mask_k,
                K,
                V,
            )
        else:
            state_ptrs = (
                state_ptr
                + state_idx * stride_state_block
                + pid_h * stride_state_head
                + offs_v[:, None] * stride_state_v
                + offs_k[None, :] * stride_state_k
            )
            state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)

        # Each request's own window, not the widest one.
        for token_offset in range(query_len):
            token = bos + token_offset
            position = offs_t == token_offset
            q = tl.load(
                q_ptr + token * stride_q_token + pid_kh * K + offs_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + token * stride_k_token + pid_kh * K + offs_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + token * stride_v_token + pid_h * V + offs_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            q = q * tl.sum(tl.where(position, q_rnorms, 0.0))
            k = k * tl.sum(tl.where(position, k_rnorms, 0.0))
            q = q * scale
            gate = tl.sum(tl.where(position, gates, 0.0))
            beta = tl.sum(tl.where(position, betas, 0.0))

            decayed = state * tl.exp(gate)
            correction = v - tl.sum(decayed * k[None, :], 1)
            correction *= beta
            state = decayed + correction[:, None] * k[None, :]

            out = tl.sum(state * q[None, :], 1)
            tl.store(
                out_ptr + token * stride_out_token + pid_h * V + offs_v,
                out.to(out_ptr.dtype.element_ty),
                mask=mask_v,
            )
            tl.store(
                correction_cache_ptr
                + state_idx * stride_correction_block
                + pid_h * stride_correction_head
                + token_offset * stride_correction_pos
                + offs_v * stride_correction_dim,
                correction,
                mask=mask_v,
            )


@triton.heuristics(
    {
        "HAS_TAIL": lambda args: args["tail_state_indices_ptr"] is not None,
        "STATE_FP8": lambda args: args["sr_seed_ptr"] is not None,
    }
)
@triton.jit
def _commit_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    key_cache_ref_ptr,
    key_cache_base_addrs_ptr,
    key_cache_block_strides_ptr,
    gate_cache_ref_ptr,
    gate_cache_base_addrs_ptr,
    gate_cache_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_key_cache_head,
    stride_key_cache_pos,
    stride_key_cache_dim,
    stride_gate_cache_head,
    stride_gate_cache_pos,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    tail_state_indices_ptr,
    tail_recovery_lens_ptr,
    HAS_TAIL: tl.constexpr,
    sr_seed_ptr,
    STATE_FP8: tl.constexpr,
    TILES: tl.constexpr,
):
    # TILES consecutive value tiles a program, as the verify takes them.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    pid_kh = pid_h // GROUP

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        source_state_idx * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    key_cache_base_addr = tl.load(key_cache_base_addrs_ptr + pid_l)
    key_cache_block_stride = tl.load(key_cache_block_strides_ptr + pid_l)
    key_cache_ptr = key_cache_base_addr.to(
        tl.pointer_type(key_cache_ref_ptr.dtype.element_ty)
    )
    key_cache_ptr += (
        source_state_idx * key_cache_block_stride + pid_kh * stride_key_cache_head
    )
    gate_cache_base_addr = tl.load(gate_cache_base_addrs_ptr + pid_l)
    gate_cache_block_stride = tl.load(gate_cache_block_strides_ptr + pid_l)
    gate_cache_ptr = gate_cache_base_addr.to(
        tl.pointer_type(gate_cache_ref_ptr.dtype.element_ty)
    )
    gate_cache_ptr += (
        source_state_idx * gate_cache_block_stride + pid_h * stride_gate_cache_head
    )

    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    if STATE_FP8:
        # One launch commits every layer: each draws its own stream.
        seed = tl.load(sr_seed_ptr) + pid_l * 7919
    # The window's stops, in order: where its accepted tokens cross into the
    # next block (align mode) and reach the decode tail (HAS_TAIL), when they
    # do, then its end. The replay runs to each stop in turn and stores the
    # state there, after the per-token loop rather than inside it: stores
    # inside the loop, one per target, put the FP8 store into the loop three
    # times and made the align-mode commit three times slower than with the
    # final store alone. The stops are unrolled: as a runtime loop around the
    # per-token loop the kernel stored wrong states on a B200 at one and two
    # pipeline stages, though the Triton interpreter ran it right. A stop the
    # window does not reach writes nothing; the boundary and the tail may fall
    # on the same token.
    unreached = commit_len + 1
    boundary_len = unreached
    if ALIGN_MODE:
        boundary_len = tl.where(
            boundary_state_idx > null_block_id, boundary_recovery_len, unreached
        )
    tail_state_idx = null_block_id
    tail_len = unreached
    if HAS_TAIL:
        tail_state_idx = tl.load(tail_state_indices_ptr + pid_b).to(tl.int64)
        tail_recovery_len = tl.load(tail_recovery_lens_ptr + pid_b)
        tail_len = tl.where(
            (tail_state_idx > null_block_id) & (tail_recovery_len > 0),
            tail_recovery_len,
            unreached,
        )
    boundary_first = boundary_len <= tail_len
    first_len = tl.minimum(boundary_len, tail_len)
    second_len = tl.maximum(boundary_len, tail_len)
    first_idx = tl.where(boundary_first, boundary_state_idx, tail_state_idx)
    second_idx = tl.where(boundary_first, tail_state_idx, boundary_state_idx)
    # The FP8 stream's salt per target: the final 0, the boundary 1, the tail 2.
    first_salt = tl.where(boundary_first, 1, 2)

    for tile in range(TILES):
        offs_v = (pid_v * TILES + tile) * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]
        if STATE_FP8:
            state = load_fp8_state_rows(
                source_state_ptr, offs_v, offs_k, mask_v, mask_k, K, V
            )
        else:
            state_ptrs = (
                source_state_ptr
                + offs_v[:, None] * stride_state_v
                + offs_k[None, :] * stride_state_k
            )
            state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
        replayed = tl.full([], 0, tl.int32)
        for stop in tl.static_range(3):
            stop_len = tl.where(
                stop == 0, first_len, tl.where(stop == 1, second_len, commit_len)
            )
            target_idx = tl.where(
                stop == 0, first_idx, tl.where(stop == 1, second_idx, final_state_idx)
            )
            reached = tl.minimum(stop_len, commit_len)
            # Forward, one tile: the verify's (and the baseline kernel's) own
            # update, replayed with its recorded correction, key and gate.
            for token_offset in range(replayed, reached):
                k = tl.load(
                    key_cache_ptr
                    + token_offset * stride_key_cache_pos
                    + offs_k * stride_key_cache_dim,
                    mask=mask_k,
                    other=0.0,
                )
                correction = tl.load(
                    correction_cache_ptr
                    + token_offset * stride_correction_cache_pos
                    + offs_v * stride_correction_cache_dim,
                    mask=mask_v,
                    other=0.0,
                )
                gate = tl.load(gate_cache_ptr + token_offset * stride_gate_cache_pos)
                state = state * tl.exp(gate)
                state = state + correction[:, None] * k[None, :]
            replayed = tl.maximum(replayed, reached)
            if stop_len <= commit_len:
                if STATE_FP8:
                    salt = tl.where(
                        stop == 0, first_salt, tl.where(stop == 1, 3 - first_salt, 0)
                    )
                    _store_fp8_commit_rows(
                        state_ptr,
                        target_idx,
                        state_block_stride,
                        pid_h,
                        stride_state_head,
                        offs_v,
                        offs_k,
                        state,
                        mask_v,
                        mask_k,
                        seed + salt,
                        NUM_HEADS,
                        K,
                        V,
                    )
                else:
                    target_ptrs = (
                        state_ptr
                        + target_idx * state_block_stride
                        + pid_h * stride_state_head
                        + offs_v[:, None] * stride_state_v
                        + offs_k[None, :] * stride_state_k
                    )
                    tl.store(target_ptrs, state, mask=mask_state)


@triton.jit
def _store_fp8_commit_rows(
    state_ptr,
    state_idx,
    state_block_stride,
    pid_h,
    stride_state_head,
    offs_v,
    offs_k,
    state,
    mask_v,
    mask_k,
    seed,
    NUM_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    # The block index joins the stream's offsets, so two blocks' rows draw
    # differently within one launch.
    rand_offsets = (
        ((state_idx * NUM_HEADS + pid_h) * V + offs_v[:, None]) * K + offs_k[None, :]
    ).to(tl.int32)
    store_fp8_state_rows(
        state_ptr
        + tl.maximum(state_idx, 0) * state_block_stride
        + pid_h * stride_state_head,
        offs_v,
        offs_k,
        state,
        mask_v,
        mask_k,
        seed,
        rand_offsets,
        K,
        V,
    )


def gdn_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    key_cache: torch.Tensor,
    gate_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float,
    out: torch.Tensor | None = None,
    fused: "GDNFusedCommit | None" = None,
) -> torch.Tensor:
    """Verify a GDN speculative window without modifying its checkpoint.

    ``q``/``k`` are ``[1, tokens, key_heads, K]``, ``v`` is
    ``[1, tokens, value_heads, V]``, ``a``/``b`` are ``[tokens, value_heads]``.
    Returns the output, ``[1, tokens, value_heads, V]``. With ``fused``
    (GDN_FUSED_COMMIT), the previous step's deferred commits are replayed and
    stored first.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("GDN RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_k_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]
    if k.shape != q.shape or v.shape[:2] != (1, total_tokens):
        raise ValueError("GDN RecoverSSM q, k, and v shapes are incompatible")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("GDN RecoverSSM value heads must group the key heads")
    if a.shape != (total_tokens, num_v_heads) or b.shape != a.shape:
        raise ValueError("GDN RecoverSSM a or b shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k)):
        raise ValueError("GDN RecoverSSM q and k heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("GDN RecoverSSM v, a, and b heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    state_fp8 = is_fp8_state(checkpoint_state)
    if checkpoint_state.shape[1:3] != (num_v_heads, value_dim) or (
        state_key_dim(checkpoint_state) != key_dim
    ):
        raise ValueError("GDN RecoverSSM checkpoint shape is incompatible")
    if state_fp8:
        check_fp8_state_layout(checkpoint_state)
    expected_correction_shape = (num_blocks, num_v_heads, spec_query_len, value_dim)
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"GDN RecoverSSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_key_shape = (num_blocks, num_k_heads, spec_query_len, key_dim)
    if key_cache.shape != expected_key_shape:
        raise ValueError(f"GDN RecoverSSM key buffer needs shape {expected_key_shape}")
    expected_gate_shape = (num_blocks, num_v_heads, spec_query_len)
    if gate_cache.shape != expected_gate_shape:
        raise ValueError(
            f"GDN RecoverSSM gate buffer needs shape {expected_gate_shape}"
        )
    if any(
        cache.dtype != torch.float32
        for cache in (correction_cache, key_cache, gate_cache)
    ):
        raise ValueError("GDN RecoverSSM record buffers must use float32")
    if A_log.shape != (num_v_heads,) or dt_bias.shape != (num_v_heads,):
        raise ValueError("GDN RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("GDN RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] < batch + 1:
        raise ValueError("GDN RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "GDN RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("GDN RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("GDN RecoverSSM output heads must be contiguous")
    if total_tokens == 0 or batch == 0:
        return out
    if fused is not None:
        return gdn_recoverssm_fused_verify(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            key_cache,
            gate_cache,
            query_start_loc,
            state_indices,
            spec_query_len,
            scale,
            out,
            fused,
        )

    (block_rows, num_warps, num_stages), tiles = _verify_launch()
    block_v = min(triton.next_power_of_2(value_dim), block_rows)
    block_k = triton.next_power_of_2(key_dim)
    value_tiles = triton.cdiv(value_dim, block_v)
    tiles = min(tiles, value_tiles)
    grid = (triton.cdiv(value_tiles, tiles), batch, num_v_heads)
    _gdn_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        key_cache,
        gate_cache,
        out,
        query_start_loc,
        state_indices,
        scale,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(0),
        b.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        gate_cache.stride(0),
        gate_cache.stride(1),
        gate_cache.stride(2),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        BT=triton.next_power_of_2(spec_query_len),
        GROUP=num_v_heads // num_k_heads,
        STATE_FP8=state_fp8,
        TILES=tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _base_addrs(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device=device
    )


def _block_strides(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [tensor.stride(0) for tensor in tensors], dtype=torch.int64, device=device
    )


@dataclass
class GDNRecoverSSMCommitContext:
    """Commit a step's accepted positions into every layer of one group.

    ``states`` is ``None`` for a conv-only group, whose commit only compacts
    the conv windows.
    """

    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    states: "_GDNCommitStates | None"
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    # The decode tail a window crosses (KVCacheConfig.decode_tail_mode
    # "recover"): its snapshot block and how many accepted tokens reach it.
    tail_state_indices: torch.Tensor
    tail_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "GDNRecoverSSMCommitContext":
        if not layers:
            raise ValueError("GDN RecoverSSM commit requires at least one layer")
        page_lens = {len(layer.kv_cache) for layer in layers}
        if page_lens not in ({1}, {5}):
            raise ValueError(
                "GDN RecoverSSM pages must hold conv, state, correction, key "
                "and gate, or a conv state alone"
            )

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("GDN RecoverSSM conv state must be three-dimensional")
        num_blocks, _, conv_state_len = conv_ref.shape
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("GDN RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != conv_ref.device
            ):
                raise ValueError("GDN RecoverSSM layers need matching conv state")

        device = conv_ref.device
        states = None
        if page_lens == {5}:
            states = _GDNCommitStates.create(
                layers, spec_query_len=spec_query_len, num_blocks=num_blocks
            )

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states, device),
            conv_state_block_strides=_block_strides(conv_states, device),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            states=states,
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            tail_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            tail_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            spec_query_len=spec_query_len,
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        tail_boundaries: torch.Tensor | None = None,
        tail_blocks: torch.Tensor | None = None,
        fused: "GDNFusedCommit | None" = None,
    ) -> None:
        """Fold accepted GDN and convolution inputs into every layer.

        ``tail_boundaries`` and ``tail_blocks`` (per batch row, align mode
        only) name each request's decode-tail target: where its window's
        accepted tokens cross the boundary, the state there is written into
        the block too (KVCacheConfig.decode_tail_mode "recover"). With
        ``fused`` (GDN_FUSED_COMMIT, align mode), the state commits that may
        wait for the next step's verify are deferred to it.
        """
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("GDN RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("GDN RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("GDN RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("GDN RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "GDN RecoverSSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("GDN RecoverSSM block table must be two-dimensional")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = (
            0 if num_computed_tokens is None else num_computed_tokens.stride(0)
        )
        align_mode = block_table is not None
        has_tail = (
            align_mode and tail_boundaries is not None and tail_blocks is not None
        )
        tail_args = (
            (
                tail_boundaries,
                tail_blocks,
                self.tail_state_indices,
                self.tail_recovery_lens,
            )
            if has_tail
            else (None, None, None, None)
        )

        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            *tail_args,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        _compact_conv_state_kernel[
            (triton.cdiv(conv_dim, 256), batch, len(self.conv_states))
        ](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            *tail_args[2:],
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align_mode,
            num_warps=4,
        )
        if self.states is not None:
            state_plan: Any = self
            if fused is not None:
                if not align_mode:
                    raise ValueError("GDN fused commit requires align mode")
                assert num_computed_tokens is not None
                assert mamba_block_size is not None
                state_plan = fused.defer(
                    self,
                    state_indices,
                    request_indices,
                    num_computed_tokens,
                    mamba_block_size,
                    has_tail,
                )
                # Nearly every row was deferred: commit by row, not by layer.
                _commit_gdn_states_by_row(
                    self.states,
                    state_plan,
                    state_indices,
                    align_mode=align_mode,
                    has_tail=has_tail,
                )
                return
            self.states.commit(
                state_plan, state_indices, align_mode=align_mode, has_tail=has_tail
            )


@dataclass
class _GDNCommitStates:
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    key_caches: tuple[torch.Tensor, ...]
    key_cache_base_addrs: torch.Tensor
    key_cache_block_strides: torch.Tensor
    gate_caches: tuple[torch.Tensor, ...]
    gate_cache_base_addrs: torch.Tensor
    gate_cache_block_strides: torch.Tensor

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        num_blocks: int,
    ) -> "_GDNCommitStates":
        checkpoints = [layer.kv_cache[1] for layer in layers]
        correction_caches = [layer.kv_cache[2] for layer in layers]
        key_caches = [layer.kv_cache[3] for layer in layers]
        gate_caches = [layer.kv_cache[4] for layer in layers]

        state_ref = checkpoints[0]
        if state_ref.ndim != 4 or state_ref.shape[0] != num_blocks:
            raise ValueError("GDN RecoverSSM checkpoint must be [blocks, H, V, K]")
        _, num_v_heads, value_dim = state_ref.shape[:3]
        key_dim = state_key_dim(state_ref)
        if is_fp8_state(state_ref):
            for checkpoint in checkpoints:
                check_fp8_state_layout(checkpoint)
        key_ref = key_caches[0]
        num_k_heads = key_ref.shape[1]
        expected = {
            "checkpoint": (checkpoints, state_ref.shape, state_ref.dtype),
            "correction": (
                correction_caches,
                (num_blocks, num_v_heads, spec_query_len, value_dim),
                torch.float32,
            ),
            "key": (
                key_caches,
                (num_blocks, num_k_heads, spec_query_len, key_dim),
                torch.float32,
            ),
            "gate": (
                gate_caches,
                (num_blocks, num_v_heads, spec_query_len),
                torch.float32,
            ),
        }
        for name, (tensors, shape, dtype) in expected.items():
            ref = tensors[0]
            for tensor in tensors:
                if (
                    tensor.shape != shape
                    or tensor.dtype != dtype
                    or tensor.device != state_ref.device
                    or tensor.stride()[1:] != ref.stride()[1:]
                ):
                    raise ValueError(
                        f"GDN RecoverSSM layers need matching {name} buffers "
                        f"of shape {tuple(shape)} and dtype {dtype}"
                    )
        if num_v_heads % num_k_heads != 0:
            raise ValueError("GDN RecoverSSM value heads must group the key heads")

        device = state_ref.device
        return cls(
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints, device),
            state_block_strides=_block_strides(checkpoints, device),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches, device),
            correction_cache_block_strides=_block_strides(correction_caches, device),
            key_caches=tuple(key_caches),
            key_cache_base_addrs=_base_addrs(key_caches, device),
            key_cache_block_strides=_block_strides(key_caches, device),
            gate_caches=tuple(gate_caches),
            gate_cache_base_addrs=_base_addrs(gate_caches, device),
            gate_cache_block_strides=_block_strides(gate_caches, device),
        )

    def commit(
        self,
        plan: GDNRecoverSSMCommitContext,
        state_indices: torch.Tensor,
        *,
        align_mode: bool,
        has_tail: bool = False,
    ) -> None:
        state_ref = self.checkpoints[0]
        _, num_v_heads, value_dim = state_ref.shape[:3]
        key_dim = state_key_dim(state_ref)
        num_k_heads = self.key_caches[0].shape[1]
        sr_seed = rounding_seed(state_ref.device) if is_fp8_state(state_ref) else None
        block_rows, num_warps, num_stages = COMMIT_LAUNCH
        block_k = triton.next_power_of_2(key_dim)
        block_v = min(triton.next_power_of_2(value_dim), block_rows)
        value_tiles = triton.cdiv(value_dim, block_v)
        tiles = min(_commit_tiles(), value_tiles)
        grid = (
            triton.cdiv(value_tiles, tiles),
            state_indices.shape[0],
            len(self.checkpoints) * num_v_heads,
        )
        _commit_gdn_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.key_caches[0],
            self.key_cache_base_addrs,
            self.key_cache_block_strides,
            self.gate_caches[0],
            self.gate_cache_base_addrs,
            self.gate_cache_block_strides,
            state_indices,
            plan.commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.correction_caches[0].stride(3),
            self.key_caches[0].stride(1),
            self.key_caches[0].stride(2),
            self.key_caches[0].stride(3),
            self.gate_caches[0].stride(1),
            self.gate_caches[0].stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_v_heads,
            GROUP=num_v_heads // num_k_heads,
            ALIGN_MODE=align_mode,
            tail_state_indices_ptr=plan.tail_state_indices if has_tail else None,
            tail_recovery_lens_ptr=plan.tail_recovery_lens if has_tail else None,
            sr_seed_ptr=sr_seed,
            TILES=tiles,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if sr_seed is not None:
            advance_rounding_seed(sr_seed)


@triton.jit
def _commit_gdn_state_layers_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    key_cache_ref_ptr,
    key_cache_base_addrs_ptr,
    key_cache_block_strides_ptr,
    gate_cache_ref_ptr,
    gate_cache_base_addrs_ptr,
    gate_cache_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_key_cache_head,
    stride_key_cache_pos,
    stride_key_cache_dim,
    stride_gate_cache_head,
    stride_gate_cache_pos,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    tail_state_indices_ptr,
    tail_recovery_lens_ptr,
    HAS_TAIL: tl.constexpr,
):
    # _commit_gdn_state_kernel's bfloat16 and float32 path, with a program
    # per (value tile, row, head) that commits the group's layers in turn
    # instead of one per layer: under the fused commit nearly every row has
    # nothing left to commit, and the per-layer grid's empty programs (12
    # times as many) cost more than the commits. The replay and its stores
    # are the shipped kernel's, unrolled over the layers.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // GROUP

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    unreached = commit_len + 1
    boundary_len = unreached
    if ALIGN_MODE:
        boundary_len = tl.where(
            boundary_state_idx > null_block_id, boundary_recovery_len, unreached
        )
    tail_state_idx = null_block_id
    tail_len = unreached
    if HAS_TAIL:
        tail_state_idx = tl.load(tail_state_indices_ptr + pid_b).to(tl.int64)
        tail_recovery_len = tl.load(tail_recovery_lens_ptr + pid_b)
        tail_len = tl.where(
            (tail_state_idx > null_block_id) & (tail_recovery_len > 0),
            tail_recovery_len,
            unreached,
        )
    boundary_first = boundary_len <= tail_len
    first_len = tl.minimum(boundary_len, tail_len)
    second_len = tl.maximum(boundary_len, tail_len)
    first_idx = tl.where(boundary_first, boundary_state_idx, tail_state_idx)
    second_idx = tl.where(boundary_first, tail_state_idx, boundary_state_idx)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    for pid_l in tl.static_range(NUM_LAYERS):
        state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
        state_block_stride = tl.load(state_block_strides_ptr + pid_l)
        state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
        source_state_ptr = (
            state_ptr
            + source_state_idx * state_block_stride
            + pid_h * stride_state_head
        )
        correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
        correction_cache_block_stride = tl.load(
            correction_cache_block_strides_ptr + pid_l
        )
        correction_cache_ptr = correction_cache_base_addr.to(
            tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
        )
        correction_cache_ptr += (
            source_state_idx * correction_cache_block_stride
            + pid_h * stride_correction_cache_head
        )
        key_cache_base_addr = tl.load(key_cache_base_addrs_ptr + pid_l)
        key_cache_block_stride = tl.load(key_cache_block_strides_ptr + pid_l)
        key_cache_ptr = key_cache_base_addr.to(
            tl.pointer_type(key_cache_ref_ptr.dtype.element_ty)
        )
        key_cache_ptr += (
            source_state_idx * key_cache_block_stride + pid_kh * stride_key_cache_head
        )
        gate_cache_base_addr = tl.load(gate_cache_base_addrs_ptr + pid_l)
        gate_cache_block_stride = tl.load(gate_cache_block_strides_ptr + pid_l)
        gate_cache_ptr = gate_cache_base_addr.to(
            tl.pointer_type(gate_cache_ref_ptr.dtype.element_ty)
        )
        gate_cache_ptr += (
            source_state_idx * gate_cache_block_stride + pid_h * stride_gate_cache_head
        )
        state_ptrs = (
            source_state_ptr
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)

        replayed = tl.full([], 0, tl.int32)
        for stop in tl.static_range(3):
            stop_len = tl.where(
                stop == 0, first_len, tl.where(stop == 1, second_len, commit_len)
            )
            target_idx = tl.where(
                stop == 0, first_idx, tl.where(stop == 1, second_idx, final_state_idx)
            )
            reached = tl.minimum(stop_len, commit_len)
            for token_offset in range(replayed, reached):
                k = tl.load(
                    key_cache_ptr
                    + token_offset * stride_key_cache_pos
                    + offs_k * stride_key_cache_dim,
                    mask=mask_k,
                    other=0.0,
                )
                correction = tl.load(
                    correction_cache_ptr
                    + token_offset * stride_correction_cache_pos
                    + offs_v * stride_correction_cache_dim,
                    mask=mask_v,
                    other=0.0,
                )
                gate = tl.load(gate_cache_ptr + token_offset * stride_gate_cache_pos)
                state = state * tl.exp(gate)
                state = state + correction[:, None] * k[None, :]
            replayed = tl.maximum(replayed, reached)
            if stop_len <= commit_len:
                target_ptrs = (
                    state_ptr
                    + target_idx * state_block_stride
                    + pid_h * stride_state_head
                    + offs_v[:, None] * stride_state_v
                    + offs_k[None, :] * stride_state_k
                )
                tl.store(target_ptrs, state, mask=mask_state)


def _commit_gdn_states_by_row(
    states: _GDNCommitStates,
    plan: Any,
    state_indices: torch.Tensor,
    *,
    align_mode: bool,
    has_tail: bool,
) -> None:
    """``_GDNCommitStates.commit`` for a bfloat16 or float32 state through
    ``_commit_gdn_state_layers_kernel``: the same commits, a twelfth of the
    programs at 12 layers a group."""
    state_ref = states.checkpoints[0]
    _, num_v_heads, value_dim = state_ref.shape[:3]
    key_dim = state_key_dim(state_ref)
    num_k_heads = states.key_caches[0].shape[1]
    block_rows, num_warps, num_stages = COMMIT_BY_ROW_LAUNCH
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), block_rows)
    grid = (triton.cdiv(value_dim, block_v), state_indices.shape[0], num_v_heads)
    _commit_gdn_state_layers_kernel[grid](
        state_ref,
        states.state_base_addrs,
        states.state_block_strides,
        states.correction_caches[0],
        states.correction_cache_base_addrs,
        states.correction_cache_block_strides,
        states.key_caches[0],
        states.key_cache_base_addrs,
        states.key_cache_block_strides,
        states.gate_caches[0],
        states.gate_cache_base_addrs,
        states.gate_cache_block_strides,
        state_indices,
        plan.commit_lens,
        plan.final_state_indices,
        plan.boundary_state_indices,
        plan.boundary_recovery_lens,
        NULL_BLOCK_ID,
        state_ref.stride(1),
        state_ref.stride(2),
        state_ref.stride(3),
        states.correction_caches[0].stride(1),
        states.correction_caches[0].stride(2),
        states.correction_caches[0].stride(3),
        states.key_caches[0].stride(1),
        states.key_caches[0].stride(2),
        states.key_caches[0].stride(3),
        states.gate_caches[0].stride(1),
        states.gate_caches[0].stride(2),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        GROUP=num_v_heads // num_k_heads,
        NUM_LAYERS=len(states.checkpoints),
        ALIGN_MODE=align_mode,
        tail_state_indices_ptr=plan.tail_state_indices if has_tail else None,
        tail_recovery_lens_ptr=plan.tail_recovery_lens if has_tail else None,
        HAS_TAIL=has_tail,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# The previous step's state commit folded into this step's verify
# (GDN_FUSED_COMMIT=1, align mode, a bfloat16 or float32 state, one pipeline
# stage). A spec row whose accepted tokens leave its state in its block, cross
# no block boundary or decode tail and leave room in the block for the next
# window defers its state commit: the commit records how many tokens it
# accepted, per request slot, and the next step's verify replays them from the
# page's records, stores the committed state and verifies from it, reading the
# state once where the commit and the verify each read and the commit wrote
# it. The conv-window compaction is not deferred.
#
# A request the next step does not verify as a spec row has its pending tokens
# committed before the step's align pre-copy (``GDNFusedCommit.flush``); a
# removed request has them committed at its removal, before any of its blocks
# can be zeroed or written for another request.
#
# The verify writes the new keys and gates into per-row staging, since the
# programs of a request replay the page's records the new ones would
# overwrite, and a copy after it moves them into the page; the corrections a
# program replays are its own and are read before it writes the new ones.


def fused_commit_enabled() -> bool:
    return os.environ.get("GDN_FUSED_COMMIT", "0") == "1"


@triton.jit
def _gdn_recoverssm_fused_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    key_cache_ptr,
    gate_cache_ptr,
    key_stage_ptr,
    gate_stage_ptr,
    row_pending_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_a_token,
    stride_b_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_key_block,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_gate_block,
    stride_gate_head,
    stride_gate_pos,
    stride_key_stage_row,
    stride_key_stage_head,
    stride_key_stage_pos,
    stride_gate_stage_row,
    stride_gate_stage_head,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // GROUP

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_t = tl.arange(0, BT)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_t = offs_t < query_len
    mask_state = mask_v[:, None] & mask_k[None, :]
    mask_tk = mask_t[:, None] & mask_k[None, :]
    tokens = bos + offs_t

    if state_idx <= null_block_id:
        tl.store(
            out_ptr + tokens[:, None] * stride_out_token + pid_h * V + offs_v[None, :],
            tl.zeros([BT, BV], dtype=tl.float32).to(out_ptr.dtype.element_ty),
            mask=mask_t[:, None] & mask_v[None, :],
        )
        return

    # As _gdn_recoverssm_verify_kernel, but the gate and key records go to
    # this row's staging.
    q_window = tl.load(
        q_ptr + tokens[:, None] * stride_q_token + pid_kh * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    k_window = tl.load(
        k_ptr + tokens[:, None] * stride_k_token + pid_kh * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    q_rnorms = tl.rsqrt(tl.sum(q_window * q_window, 1) + 1e-6)
    k_rnorms = tl.rsqrt(tl.sum(k_window * k_window, 1) + 1e-6)
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    raw_a = tl.load(a_ptr + tokens * stride_a_token + pid_h, mask=mask_t, other=0.0)
    raw_b = tl.load(b_ptr + tokens * stride_b_token + pid_h, mask=mask_t, other=0.0)
    x = raw_a.to(tl.float32) + dt_bias
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    gates = neg_A * softplus_x
    betas = tl.sigmoid(raw_b.to(tl.float32))
    if pid_v == 0:
        tl.store(
            gate_stage_ptr
            + pid_b * stride_gate_stage_row
            + pid_h * stride_gate_stage_head
            + offs_t,
            gates,
            mask=mask_t,
        )
        if pid_h % GROUP == 0:
            tl.store(
                key_stage_ptr
                + pid_b * stride_key_stage_row
                + pid_kh * stride_key_stage_head
                + offs_t[:, None] * stride_key_stage_pos
                + offs_k[None, :],
                k_window * k_rnorms[:, None],
                mask=mask_tk,
            )

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)

    # The previous step's deferred commit: its accepted tokens replayed from
    # the page's records as _commit_gdn_state_kernel replays them, which
    # compiles to fma(correction, key, state * exp(gate)); spelled out so this
    # kernel rounds alike. Then the committed state, stored and read back as
    # the verify would read it from the page.
    pending = tl.load(row_pending_ptr + pid_b)
    for token_offset in range(pending):
        k_record = tl.load(
            key_cache_ptr
            + state_idx * stride_key_block
            + pid_kh * stride_key_head
            + token_offset * stride_key_pos
            + offs_k * stride_key_dim,
            mask=mask_k,
            other=0.0,
        )
        correction_record = tl.load(
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
            + offs_v * stride_correction_dim,
            mask=mask_v,
            other=0.0,
        )
        gate_record = tl.load(
            gate_cache_ptr
            + state_idx * stride_gate_block
            + pid_h * stride_gate_head
            + token_offset * stride_gate_pos
        )
        state = tl.fma(
            correction_record[:, None], k_record[None, :], state * tl.exp(gate_record)
        )
    if pending > 0:
        committed = state.to(state_ptr.dtype.element_ty)
        tl.store(state_ptrs, committed, mask=mask_state)
        state = committed.to(tl.float32)

    for token_offset in range(query_len):
        token = bos + token_offset
        position = offs_t == token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_kh * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_kh * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        q = q * tl.sum(tl.where(position, q_rnorms, 0.0))
        k = k * tl.sum(tl.where(position, k_rnorms, 0.0))
        q = q * scale
        gate = tl.sum(tl.where(position, gates, 0.0))
        beta = tl.sum(tl.where(position, betas, 0.0))

        decayed = state * tl.exp(gate)
        correction = v - tl.sum(decayed * k[None, :], 1)
        correction *= beta
        state = decayed + correction[:, None] * k[None, :]

        out = tl.sum(state * q[None, :], 1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out.to(out_ptr.dtype.element_ty),
            mask=mask_v,
        )
        tl.store(
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
            + offs_v * stride_correction_dim,
            correction,
            mask=mask_v,
        )


@triton.jit
def _copy_staged_records_kernel(
    key_stage_ptr,
    gate_stage_ptr,
    key_cache_ptr,
    gate_cache_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    null_block_id,
    stride_key_stage_row,
    stride_key_stage_head,
    stride_key_stage_pos,
    stride_gate_stage_row,
    stride_gate_stage_head,
    stride_key_block,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_gate_block,
    stride_gate_head,
    stride_gate_pos,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    GROUP: tl.constexpr,
    BG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if state_idx <= null_block_id:
        return
    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc)
    query_len = eos - bos
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_g = tl.arange(0, BG)
    mask_t = offs_t < query_len
    mask_tk = mask_t[:, None] & (offs_k[None, :] < K)
    keys = tl.load(
        key_stage_ptr
        + pid_b * stride_key_stage_row
        + pid_kh * stride_key_stage_head
        + offs_t[:, None] * stride_key_stage_pos
        + offs_k[None, :],
        mask=mask_tk,
    )
    tl.store(
        key_cache_ptr
        + state_idx * stride_key_block
        + pid_kh * stride_key_head
        + offs_t[:, None] * stride_key_pos
        + offs_k[None, :] * stride_key_dim,
        keys,
        mask=mask_tk,
    )
    heads = pid_kh * GROUP + offs_g
    mask_gt = (offs_g[:, None] < GROUP) & mask_t[None, :]
    gates = tl.load(
        gate_stage_ptr
        + pid_b * stride_gate_stage_row
        + heads[:, None] * stride_gate_stage_head
        + offs_t[None, :],
        mask=mask_gt,
    )
    tl.store(
        gate_cache_ptr
        + state_idx * stride_gate_block
        + heads[:, None] * stride_gate_head
        + offs_t[None, :] * stride_gate_pos,
        gates,
        mask=mask_gt,
    )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _consume_pending_kernel(
    idx_mapping_ptr,
    request_indices_ptr,
    state_indices_ptr,
    pending_len_ptr,
    pending_block_ptr,
    row_pending_ptr,
    row_req_ptr,
    mismatch_ptr,
    num_rows,
    null_block_id,
    stride_state_indices,
    HAS_REQUEST_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < num_rows
    batch_rows = rows
    if HAS_REQUEST_INDICES:
        batch_rows = tl.load(request_indices_ptr + rows, mask=valid, other=0)
    req = tl.load(idx_mapping_ptr + batch_rows, mask=valid, other=-1)
    valid &= req >= 0
    req = tl.where(valid, req, 0)
    source = tl.load(
        state_indices_ptr + rows * stride_state_indices, mask=valid, other=0
    )
    pending = tl.load(pending_len_ptr + req, mask=valid, other=0)
    pending_block = tl.load(pending_block_ptr + req, mask=valid, other=0)
    # A pending commit lives in the block it was deferred in; a request whose
    # state moved since cannot replay it here (never seen: counted).
    replay = (pending > 0) & (pending_block == source) & (source > null_block_id)
    tl.store(row_pending_ptr + rows, tl.where(replay, pending, 0), mask=valid)
    tl.store(row_req_ptr + rows, tl.where(valid, req, -1), mask=rows < num_rows)
    tl.store(pending_len_ptr + req, tl.zeros_like(pending), mask=valid)
    lost = tl.sum(((pending > 0) & ~replay).to(tl.int32))
    tl.atomic_add(mismatch_ptr, lost)


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "HAS_TAIL": lambda args: args["tail_state_indices_ptr"] is not None,
    }
)
@triton.jit
def _defer_commit_kernel(
    commit_lens_ptr,
    state_indices_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    tail_state_indices_ptr,
    request_indices_ptr,
    num_computed_ptr,
    row_req_ptr,
    pending_len_ptr,
    pending_block_ptr,
    state_commit_lens_ptr,
    row_pending_ptr,
    num_rows,
    null_block_id,
    stride_state_indices,
    stride_num_computed,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < num_rows
    commit_len = tl.load(commit_lens_ptr + rows, mask=valid, other=0)
    source = tl.load(
        state_indices_ptr + rows * stride_state_indices, mask=valid, other=0
    ).to(tl.int64)
    final = tl.load(final_state_indices_ptr + rows, mask=valid, other=0).to(tl.int64)
    boundary = tl.load(boundary_state_indices_ptr + rows, mask=valid, other=0)
    tail = tl.zeros_like(boundary) + null_block_id
    if HAS_TAIL:
        tail = tl.load(tail_state_indices_ptr + rows, mask=valid, other=0)
    batch_rows = rows
    if HAS_REQUEST_INDICES:
        batch_rows = tl.load(request_indices_ptr + rows, mask=valid, other=0)
    num_computed = tl.load(
        num_computed_ptr + batch_rows * stride_num_computed, mask=valid, other=0
    )
    req = tl.load(row_req_ptr + rows, mask=valid, other=-1)
    end = num_computed + commit_len
    # The next window, up to its widest, stays in this block, so the align
    # pre-copy leaves the state where the pending commit lives.
    stays = (end - 1) // MAMBA_BLOCK_SIZE == (end + SPEC_QUERY_LEN - 1) // (
        MAMBA_BLOCK_SIZE
    )
    defer = (
        valid
        & (req >= 0)
        & (commit_len > 0)
        & (source > null_block_id)
        & (final == source)
        & (boundary <= null_block_id)
        & (tail <= null_block_id)
        & stays
    )
    has_req = valid & (req >= 0)
    req = tl.where(has_req, req, 0)
    tl.store(pending_len_ptr + req, tl.where(defer, commit_len, 0), mask=has_req)
    tl.store(pending_block_ptr + req, source.to(tl.int32), mask=has_req)
    tl.store(state_commit_lens_ptr + rows, tl.where(defer, 0, commit_len), mask=valid)
    tl.store(row_pending_ptr + rows, tl.zeros_like(commit_len), mask=valid)


@triton.jit
def _take_pending_kernel(
    slots_ptr,
    pending_len_ptr,
    pending_block_ptr,
    flush_lens_ptr,
    flush_blocks_ptr,
    num_slots,
    null_block_id,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    valid = offs < num_slots
    slots = tl.load(slots_ptr + offs, mask=valid, other=0)
    pending = tl.load(pending_len_ptr + slots, mask=valid, other=0)
    block = tl.load(pending_block_ptr + slots, mask=valid, other=0)
    tl.store(flush_lens_ptr + offs, pending, mask=valid)
    tl.store(
        flush_blocks_ptr + offs,
        tl.where(pending > 0, block, null_block_id),
        mask=valid,
    )
    tl.store(pending_len_ptr + slots, tl.zeros_like(pending), mask=valid)


@dataclass
class _StatePlan:
    """The fields of a commit plan the state kernel reads."""

    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    tail_state_indices: torch.Tensor | None = None
    tail_recovery_lens: torch.Tensor | None = None


class GDNFusedCommit:
    """One GDN group's deferred state commits (GDN_FUSED_COMMIT).

    Per request slot, the tokens a deferred commit still owes and the block it
    owes them to; per spec row of the current step, the tokens its verify
    replays (``row_pending``) and its request slot (``row_req``), and the
    staging the verify writes the new key and gate records into.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_rows: int,
        key_shape: Sequence[int],
        gate_shape: Sequence[int],
        spec_query_len: int,
        device: torch.device,
    ) -> None:
        num_k_heads, key_len, key_dim = key_shape
        num_v_heads, gate_len = gate_shape
        if key_len != spec_query_len or gate_len != spec_query_len:
            raise ValueError("GDN fused commit records need one row a window token")
        self.spec_query_len = spec_query_len
        self.max_num_reqs = max_num_reqs
        self.max_rows = max_rows
        self.pending_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.pending_block = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.row_pending = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.row_req = torch.full((max_rows,), -1, dtype=torch.int32, device=device)
        self.state_commit_lens = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.key_stage = torch.zeros(
            (max_rows, num_k_heads, spec_query_len, key_dim),
            dtype=torch.float32,
            device=device,
        )
        self.gate_stage = torch.zeros(
            (max_rows, num_v_heads, spec_query_len),
            dtype=torch.float32,
            device=device,
        )
        self.flush_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.flush_blocks = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # The flush's boundary stops: none (NULL_BLOCK_ID, length 0).
        self.null_blocks = torch.full(
            (max_num_reqs,), NULL_BLOCK_ID, dtype=torch.int32, device=device
        )
        self.zero_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        #: Pending commits found in another block than the one they were
        #: deferred in, and dropped (never seen; a test asserts zero).
        self.mismatches = torch.zeros(1, dtype=torch.int32, device=device)
        #: The group's commit context, once its builder has made it.
        self.context: GDNRecoverSSMCommitContext | None = None

    def consume(
        self,
        idx_mapping: torch.Tensor,
        request_indices: torch.Tensor | None,
        state_indices: torch.Tensor,
        num_rows: int,
    ) -> None:
        """Hand each spec row of the step its request's pending tokens, for
        its verify to replay, and note the row's request slot for the step's
        commit."""
        if num_rows == 0:
            return
        if num_rows > self.max_rows:
            raise ValueError("GDN fused commit: more spec rows than staged")
        block = 128
        _consume_pending_kernel[(triton.cdiv(num_rows, block),)](
            idx_mapping,
            request_indices,
            state_indices,
            self.pending_len,
            self.pending_block,
            self.row_pending,
            self.row_req,
            self.mismatches,
            num_rows,
            NULL_BLOCK_ID,
            state_indices.stride(0),
            BLOCK=block,
            num_warps=4,
        )

    def defer(
        self,
        plan: "GDNRecoverSSMCommitContext",
        state_indices: torch.Tensor,
        request_indices: torch.Tensor | None,
        num_computed_tokens: torch.Tensor,
        mamba_block_size: int,
        has_tail: bool,
    ) -> _StatePlan:
        """Defer the state commits of the rows that may defer; the plan the
        state kernel runs for the others."""
        batch = state_indices.shape[0]
        block = 128
        _defer_commit_kernel[(triton.cdiv(batch, block),)](
            plan.commit_lens,
            state_indices,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.tail_state_indices if has_tail else None,
            request_indices,
            num_computed_tokens,
            self.row_req,
            self.pending_len,
            self.pending_block,
            self.state_commit_lens,
            self.row_pending,
            batch,
            NULL_BLOCK_ID,
            state_indices.stride(0),
            num_computed_tokens.stride(0),
            MAMBA_BLOCK_SIZE=mamba_block_size,
            SPEC_QUERY_LEN=self.spec_query_len,
            BLOCK=block,
            num_warps=4,
        )
        return _StatePlan(
            commit_lens=self.state_commit_lens,
            final_state_indices=plan.final_state_indices,
            boundary_state_indices=plan.boundary_state_indices,
            boundary_recovery_lens=plan.boundary_recovery_lens,
            tail_state_indices=plan.tail_state_indices,
            tail_recovery_lens=plan.tail_recovery_lens,
        )

    def flush(self, slots: Sequence[int]) -> None:
        """Commit the pending tokens of the request slots ``slots`` into
        their blocks now."""
        if not slots or self.context is None or self.context.states is None:
            return
        num_slots = len(slots)
        if num_slots > self.max_num_reqs:
            raise ValueError("GDN fused commit: more slots than request slots")
        device = self.pending_len.device
        slot_tensor = torch.tensor(slots, dtype=torch.int32).pin_memory()
        slot_tensor = slot_tensor.to(device, non_blocking=True)
        _take_pending_kernel[(1,)](
            slot_tensor,
            self.pending_len,
            self.pending_block,
            self.flush_lens,
            self.flush_blocks,
            num_slots,
            NULL_BLOCK_ID,
            BLOCK=triton.next_power_of_2(num_slots),
            num_warps=4,
        )
        flush_blocks = self.flush_blocks[:num_slots]
        _commit_gdn_states_by_row(
            self.context.states,
            _StatePlan(
                commit_lens=self.flush_lens,
                final_state_indices=self.flush_blocks,
                boundary_state_indices=self.null_blocks,
                boundary_recovery_lens=self.zero_lens,
            ),
            flush_blocks,
            align_mode=False,
            has_tail=False,
        )


def gdn_recoverssm_fused_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    key_cache: torch.Tensor,
    gate_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float,
    out: torch.Tensor,
    fused: GDNFusedCommit,
) -> torch.Tensor:
    """``gdn_recoverssm_verify`` with the previous step's deferred commit
    folded in; the caller has validated the shapes."""
    if is_fp8_state(checkpoint_state):
        raise ValueError("GDN fused commit does not support an FP8 state")
    _, total_tokens, num_k_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]
    batch = state_indices.shape[0]
    if batch > fused.max_rows:
        raise ValueError("GDN fused commit: more spec rows than staged")
    key_stage, gate_stage = fused.key_stage, fused.gate_stage
    group = num_v_heads // num_k_heads
    block_rows, num_warps, num_stages = VERIFY_LAUNCH
    block_v = min(triton.next_power_of_2(value_dim), block_rows)
    block_k = triton.next_power_of_2(key_dim)
    block_t = triton.next_power_of_2(spec_query_len)
    grid = (triton.cdiv(value_dim, block_v), batch, num_v_heads)
    # XKB_XGDN=1: the CUDA port (xkb_xgdn.py) takes these arguments when it serves
    # the shapes.
    xkb_xgdn.launcher(_gdn_recoverssm_fused_verify_kernel[grid], spec_query_len)(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        key_cache,
        gate_cache,
        key_stage,
        gate_stage,
        fused.row_pending,
        out,
        query_start_loc,
        state_indices,
        scale,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(0),
        b.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        gate_cache.stride(0),
        gate_cache.stride(1),
        gate_cache.stride(2),
        key_stage.stride(0),
        key_stage.stride(1),
        key_stage.stride(2),
        gate_stage.stride(0),
        gate_stage.stride(1),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        BT=block_t,
        GROUP=group,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _copy_staged_records_kernel[(batch, num_k_heads)](
        key_stage,
        gate_stage,
        key_cache,
        gate_cache,
        query_start_loc,
        state_indices,
        NULL_BLOCK_ID,
        key_stage.stride(0),
        key_stage.stride(1),
        key_stage.stride(2),
        gate_stage.stride(0),
        gate_stage.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        gate_cache.stride(0),
        gate_cache.stride(1),
        gate_cache.stride(2),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        BK=block_k,
        BT=block_t,
        GROUP=group,
        BG=triton.next_power_of_2(group),
        num_warps=4,
    )
    return out


__all__ = [
    "GDNFusedCommit",
    "GDNRecoverSSMCommitContext",
    "fused_commit_enabled",
    "gdn_recoverssm_verify",
]
