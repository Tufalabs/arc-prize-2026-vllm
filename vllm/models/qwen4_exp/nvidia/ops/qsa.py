# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import round_up

FP8_KV_DTYPE = torch.float8_e4m3fn
_SCALE_BYTES = 4  # one float32 scale per (token, head) side


@dataclass(frozen=True)
class QSAQuantPageLayout:
    """Byte offsets inside one (head, token) cell of a quantized main-KV page.

    The cell holds ``[K | K scale | V | V scale | pad]``: the data in e4m3
    (``data_bytes`` = head_dim) or in int4 packed two to a byte (head_dim / 2),
    each scale a float32 written by the per-token-head cache-update kernel and
    read back by the sparse attention kernel. An int4 scale carries the row's
    zero point in its four low mantissa bits.
    """

    content_bytes: int
    data_bytes: int
    key_offset: int
    key_scale_offset: int
    value_offset: int
    value_scale_offset: int


def qsa_quant_page_layout(
    data_bytes: int, num_kv_heads: int, token_bytes_multiple: int
) -> QSAQuantPageLayout:
    """Lay out the cell for ``data_bytes`` of K and of V so a token's bytes
    across all heads are a multiple of ``token_bytes_multiple``.

    The value data starts on a 16-byte boundary so its tile loads vectorise
    like the key's. The multiple exists because every QSA cache group shares
    one page size and the compressed-key cache, an MLA spec, can only be
    matched by an integer block-size ratio, never by padding; the owner passes
    that cache's bytes per token.
    """
    key_offset = 0
    key_scale_offset = data_bytes
    value_offset = round_up(key_scale_offset + _SCALE_BYTES, 16)
    value_scale_offset = value_offset + data_bytes
    cell = value_scale_offset + _SCALE_BYTES
    # Smallest cell whose per-token bytes (all heads) hit the multiple.
    token_bytes = round_up(num_kv_heads * cell, token_bytes_multiple)
    while token_bytes % num_kv_heads:
        token_bytes += token_bytes_multiple
    return QSAQuantPageLayout(
        content_bytes=token_bytes // num_kv_heads,
        data_bytes=data_bytes,
        key_offset=key_offset,
        key_scale_offset=key_scale_offset,
        value_offset=value_offset,
        value_scale_offset=value_scale_offset,
    )


@dataclass(frozen=True)
class QSAKVCacheViews:
    """The main K/V cache as the kernels address it: ``[blocks, tokens, heads,
    C]`` data views -- bf16 or e4m3 with C = head_dim, or uint8 with C =
    head_dim / 2 for packed int4 -- plus ``[blocks, tokens, heads]`` float32
    scale views when the data is quantized (None when it is bf16)."""

    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None
    value_scale: torch.Tensor | None

    @property
    def is_quantized(self) -> bool:
        return self.key_scale is not None

    @property
    def is_fp8(self) -> bool:
        return self.key.dtype == FP8_KV_DTYPE

    @property
    def is_int4(self) -> bool:
        return self.key.dtype == torch.uint8


def _f32_view(kv_cache: torch.Tensor, byte_offset: int) -> torch.Tensor:
    """View the float32 at ``byte_offset`` of every cell as ``[B, N, H]``."""
    num_blocks, num_heads, num_tokens, _ = kv_cache.shape
    stride_block, stride_head, stride_token, stride_byte = kv_cache.stride()
    assert stride_byte == 1 and kv_cache.element_size() == 1
    offset = kv_cache.storage_offset() + byte_offset
    assert not any(x % 4 for x in (stride_block, stride_head, stride_token, offset))
    base = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
        kv_cache.untyped_storage()
    )
    return torch.as_strided(
        base,
        size=(num_blocks, num_tokens, num_heads),
        stride=(stride_block // 4, stride_token // 4, stride_head // 4),
        storage_offset=offset // 4,
    )


def split_qsa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
    layout: QSAQuantPageLayout | None,
) -> QSAKVCacheViews:
    """Split the owner's bound ``[B, H, N, C]`` cache into kernel views.

    ``layout`` is None for a bf16 cache, whose cell is ``[K | V]``; a
    quantized cache carries its layout from the spec that sized it, and its
    data bytes say which format: one a value is e4m3, half a value packed
    int4, whose views stay bytes.
    """
    cells = kv_cache.transpose(1, 2)  # [B, N, H, C]
    if layout is None:
        key, value = cells.split(head_dim, dim=-1)
        return QSAKVCacheViews(key, value, None, None)
    if kv_cache.dtype != torch.uint8 or cells.shape[-1] != layout.content_bytes:
        raise ValueError("QSA quantized cache does not match its page layout")
    key = cells[..., layout.key_offset : layout.key_offset + layout.data_bytes]
    value = cells[..., layout.value_offset : layout.value_offset + layout.data_bytes]
    if layout.data_bytes == head_dim:
        key, value = key.view(FP8_KV_DTYPE), value.view(FP8_KV_DTYPE)
    elif 2 * layout.data_bytes != head_dim:
        raise ValueError("QSA quantized cache rows are neither FP8 nor packed INT4")
    return QSAKVCacheViews(
        key,
        value,
        _f32_view(kv_cache, layout.key_scale_offset),
        _f32_view(kv_cache, layout.value_scale_offset),
    )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_FP8: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if KV_FP8:
            # Dequantize by folding each token's scale into the score column
            # and the probability column rather than into the tile itself;
            # the data only needs the cast to the query's dtype for the dot.
            keys = keys.to(query.dtype)
            values = values.to(query.dtype)
            key_scales = tl.load(
                k_scale_ptr
                + safe_page * stride_ks_block
                + page_offset * stride_ks_token
                + kv_head * stride_ks_head,
                mask=valid,
                other=1.0,
            )
            value_scales = tl.load(
                v_scale_ptr
                + safe_page * stride_vs_block
                + page_offset * stride_vs_token
                + kv_head * stride_vs_head,
                mask=valid,
                other=1.0,
            )
            scores = tl.dot(query, keys) * (softmax_scale_log2 * key_scales)[None, :]
        else:
            scores = tl.dot(query, keys)
            # Scaling scores avoids re-quantizing a scaled query to BF16.
            scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if KV_FP8:  # noqa: SIM108 -- a ternary would trace the bf16 arm too
            weighted = probabilities * value_scales[None, :]
        else:
            weighted = probabilities
        accumulator = tl.dot(
            weighted.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_int4_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """The split-K kernel over a packed INT4 cache, ungated: the caller
    rotates the query and the output and applies the gate after.

    A byte holds element 2i in its low nibble and 2i + 1 in its high one, so
    each gathered row is loaded once, split into its two halves and dotted
    against the query's even and odd elements. The zero points come off once
    per column and per row rather than per element:
    q . (k - z) = q . k - z * sum(q), and sum_j w_j (v_j - z_j) = w . v -
    sum_j w_j z_j. The raw nibbles are exact in the query's dtype.
    """
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # The packed selection buffer's trailing count column, as in
    # _qsa_sparse_paged_gqa_splitk_kernel.
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HEAD_DIM // 2)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    query_rows = (
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
    )
    query_even = tl.load(
        query_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0
    )
    query_odd = tl.load(
        query_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0
    )
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(
        query_odd.to(tl.float32), axis=1
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_bytes = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_bytes = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        # Each float32 scale carries its row's zero point in the four low
        # mantissa bits (the INT4 write kernel's steganography).
        key_bits = tl.load(
            k_scale_ptr
            + safe_page * stride_ks_block
            + page_offset * stride_ks_token
            + kv_head * stride_ks_head,
            mask=valid,
            other=1.0,
        ).to(tl.int32, bitcast=True)
        value_bits = tl.load(
            v_scale_ptr
            + safe_page * stride_vs_block
            + page_offset * stride_vs_token
            + kv_head * stride_vs_head,
            mask=valid,
            other=1.0,
        ).to(tl.int32, bitcast=True)
        key_zero_points = (key_bits & 0xF).to(tl.float32)
        value_zero_points = (value_bits & 0xF).to(tl.float32)
        key_scales = (key_bits & -16).to(tl.float32, bitcast=True)
        value_scales = (value_bits & -16).to(tl.float32, bitcast=True)

        raw_scores = tl.dot(
            query_even, (key_bytes & 0xF).to(query_even.dtype)
        ) + tl.dot(query_odd, (key_bytes >> 4).to(query_even.dtype))
        scores = (raw_scores - query_sum[:, None] * key_zero_points[None, :]) * (
            softmax_scale_log2 * key_scales
        )[None, :]
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weighted = (probabilities * value_scales[None, :]).to(query_even.dtype)
        zero_correction = tl.sum(
            weighted.to(tl.float32) * value_zero_points[None, :], axis=1
        )
        accumulator_even = (
            tl.dot(
                weighted,
                (value_bytes & 0xF).to(query_even.dtype),
                acc=accumulator_even * alpha[:, None],
            )
            - zero_correction[:, None]
        )
        accumulator_odd = (
            tl.dot(
                weighted,
                (value_bytes >> 4).to(query_even.dtype),
                acc=accumulator_odd * alpha[:, None],
            )
            - zero_correction[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    output_even = tl.where(has_values[:, None], accumulator_even / denominator, 0.0)
    output_odd = tl.where(has_values[:, None], accumulator_odd / denominator, 0.0)
    if NUM_SPLITS == 1:
        output_rows = (
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
        )
        tl.store(
            output_rows + 2 * half_offsets[None, :],
            output_even.to(output_ptr.dtype.element_ty),
            mask=head_mask,
        )
        tl.store(
            output_rows + 2 * half_offsets[None, :] + 1,
            output_odd.to(output_ptr.dtype.element_ty),
            mask=head_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        partial_rows = (
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
        )
        tl.store(partial_rows + 2 * half_offsets[None, :], output_even, mask=head_mask)
        tl.store(
            partial_rows + 2 * half_offsets[None, :] + 1, output_odd, mask=head_mask
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    if APPLY_GATE:
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + head * stride_output_gate_head
            + dim_offsets
        ).to(tl.float32)
        merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_int4_rotate_kernel(
    x_ptr,
    matrix_ptr,
    gate_ptr,
    output_ptr,
    stride_x_token,
    stride_x_head,
    stride_gate_token,
    stride_gate_head,
    stride_output_token,
    stride_output_head,
    num_rows,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    """``output = x @ matrix`` over (token, head) rows of HEAD_DIM, in one
    pass: accumulated in FP32 and rounded to the output dtype once. With the
    gate, that rounded row times sigmoid(gate) in FP32, rounded again -- the
    gated kernels' order (_qsa_merge_splitk_kernel)."""
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = rows < num_rows
    token = (rows // NUM_HEADS).to(tl.int64)
    head = rows % NUM_HEADS
    x_rows = x_ptr + token * stride_x_token + head * stride_x_head
    accumulator = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
    for k in tl.static_range(0, HEAD_DIM, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_rows[:, None] + k_offsets[None, :], mask=row_mask[:, None], other=0.0
        )
        matrix = tl.load(matrix_ptr + k_offsets[:, None] * HEAD_DIM + cols[None, :])
        accumulator = tl.dot(x, matrix, accumulator)
    rotated = accumulator.to(output_ptr.dtype.element_ty)
    if APPLY_GATE:
        gate = tl.load(
            gate_ptr
            + token[:, None] * stride_gate_token
            + head[:, None] * stride_gate_head
            + cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rotated = (rotated.to(tl.float32) * tl.sigmoid(gate)).to(
            output_ptr.dtype.element_ty
        )
    tl.store(
        output_ptr
        + token[:, None] * stride_output_token
        + head[:, None] * stride_output_head
        + cols[None, :],
        rotated,
        mask=row_mask[:, None],
    )


# The rotation kernel's tiles: rows of (token, head), output columns, and the
# reduction step; a head_dim below a tile takes the head_dim.
_ROTATE_BLOCK_ROWS = 64
_ROTATE_BLOCK_COLS = 128
_ROTATE_BLOCK_K = 64
_INT4_ROTATIONS: dict[tuple[str, int, torch.dtype, str], torch.Tensor] = {}


def _int4_rotation(
    kind: str, head_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The INT4 cache's randomized Hadamard transform
    (``int4_per_token_head.single_rht``, H D1 x) as a matrix a row multiplies
    on the right: "cache" rotates a K or V row as the cache write does,
    unnormalized (D1 H); "query" rotates a query into the cache's basis and
    "output" brings an attended row back (H D1), each over head_dim, so the
    scores and the output keep their scale. Every entry is +-1 or
    +-1/head_dim, exact in BF16."""
    key = (kind, head_dim, dtype, str(device))
    matrix = _INT4_ROTATIONS.get(key)
    if matrix is None:
        from vllm.v1.attention.ops.int4_per_token_head import (
            _get_hadamard_matrix,
            _get_rht_signs,
        )

        hadamard = _get_hadamard_matrix(head_dim, torch.float32, device)
        signs = _get_rht_signs(head_dim, 0, device).to(torch.float32)
        if kind == "cache":
            matrix = signs[:, None] * hadamard
        elif kind == "query":
            matrix = signs[:, None] * hadamard / head_dim
        elif kind == "output":
            matrix = hadamard * signs[None, :] / head_dim
        else:
            raise ValueError(f"unknown INT4 rotation {kind!r}")
        matrix = matrix.to(dtype).contiguous()
        _INT4_ROTATIONS[key] = matrix
    return matrix


def _rotate_config(head_dim: int) -> tuple[int, int]:
    """The rotation kernel's column and reduction tiles for ``head_dim``."""
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(
            f"INT4 rotation needs a power-of-two head_dim >= 16, got {head_dim}"
        )
    return min(head_dim, _ROTATE_BLOCK_COLS), min(head_dim, _ROTATE_BLOCK_K)


def qsa_int4_rotate(
    x: torch.Tensor,
    kind: str,
    *,
    out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate ``[tokens, heads, head_dim]`` rows for the INT4 cache in one
    kernel (_int4_rotation names the kinds), into ``out`` or a new contiguous
    tensor; with ``gate`` (``kind="output"`` only), the rotated rows times
    sigmoid(gate). Replaces the unfused chain -- casts, signs, the Hadamard
    transform, the scale and the gate as separate kernels -- that
    ``single_rht`` and the gate cost per call."""
    num_tokens, num_heads, head_dim = x.shape
    if x.dtype not in (torch.bfloat16, torch.float16) or x.stride(2) != 1:
        raise ValueError("INT4 rotation takes 16-bit rows contiguous in head_dim")
    if gate is not None and kind != "output":
        raise ValueError("only the output rotation takes the gate")
    block_cols, block_k = _rotate_config(head_dim)
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    assert out.shape == x.shape and out.dtype == x.dtype and out.stride(2) == 1
    num_rows = num_tokens * num_heads
    if not num_rows:
        return out
    if gate is not None:
        assert gate.shape == x.shape and gate.stride(2) == 1
        gate_arg, gate_strides = gate, (gate.stride(0), gate.stride(1))
    else:
        gate_arg, gate_strides = x, (0, 0)
    _qsa_int4_rotate_kernel[
        (triton.cdiv(num_rows, _ROTATE_BLOCK_ROWS), head_dim // block_cols)
    ](
        x,
        _int4_rotation(kind, head_dim, x.dtype, x.device),
        gate_arg,
        out,
        x.stride(0),
        x.stride(1),
        *gate_strides,
        out.stride(0),
        out.stride(1),
        num_rows,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_ROWS=_ROTATE_BLOCK_ROWS,
        BLOCK_COLS=block_cols,
        BLOCK_K=block_k,
        APPLY_GATE=gate is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def qsa_store_quantized_rows(
    key: torch.Tensor,
    value: torch.Tensor,
    views: QSAKVCacheViews,
    slot_mapping: torch.Tensor,
) -> None:
    """Write K/V rows into a quantized main-KV cache: FP8 through the shared
    per-token-head kernel; INT4 rotated by qsa_int4_rotate, then packed by
    the shared INT4 kernel, which is what ``reshape_and_cache_int4`` does
    with the rotation unfused."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    assert views.key_scale is not None and views.value_scale is not None
    if views.is_int4:
        from vllm.v1.attention.ops.int4_per_token_head import (
            _INT4_PACKING_FACTOR,
            _reshape_cache_int4_kernel,
            _run_reshape_kernel,
        )

        _run_reshape_kernel(
            _reshape_cache_int4_kernel,
            key=qsa_int4_rotate(key, "cache"),
            value=qsa_int4_rotate(value, "cache"),
            key_cache=views.key,
            value_cache=views.value,
            k_scale_cache=views.key_scale,
            v_scale_cache=views.value_scale,
            slot_mapping=slot_mapping,
            packing_factor=_INT4_PACKING_FACTOR,
        )
        return
    assert views.is_fp8
    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        views.key,
        views.value,
        views.key_scale,
        views.value_scale,
        slot_mapping,
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _int4_configs(kv_int4: bool) -> bool:
    """Whether an INT4 cache takes its kernel's own decode configs on this card:
    read on a B200 alone, so the 10.x family, and every other card keeps the
    table's; off by ``VLLM_SM100_KERNEL_SHAPES=0`` on every card."""
    return (
        kv_int4
        and envs.VLLM_SM100_KERNEL_SHAPES
        and current_platform.is_device_capability_family(100)
    )


def _select_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    int4_configs: bool = False,
) -> tuple[int, int, int, int]:
    """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.

    Tuned on GB300 for the Qwen3.8-Flash-Next TP1/TP2/TP4 shapes, keyed on
    base_programs = num_rows * num_kv_heads. The bp > 2048 region splits on
    use_prefill_config (capture-stable: at FULL-graph capture max_query_len is the
    uniform decode/verify length). ``int4_configs`` takes the INT4 kernel's own
    decode rows, read on a B200 at a k=4 verify of 216 requests and the
    drafter's single-row passes over them: wider column tiles and four warps,
    where its packed rows are half the bytes the table's bf16 and FP8 rows were
    tuned for (``_int4_configs`` says where they apply).
    """
    base_programs = num_rows * num_kv_heads
    if base_programs > 2048:
        if use_prefill_config:
            BLOCK_N, target_splits, num_warps = 32, 1, 1
        elif int4_configs:
            BLOCK_N, target_splits, num_warps = 128, 1, 4
        else:
            BLOCK_N, target_splits, num_warps = 64, 1, 2
    elif base_programs <= 24:
        BLOCK_N, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 32:
        BLOCK_N, target_splits, num_warps = 32, 16, 1
    elif base_programs <= 64:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 128:
        BLOCK_N, target_splits, num_warps = 32, 4, 1
    elif base_programs <= 256:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 512:
        BLOCK_N, target_splits, num_warps = (128, 2, 4) if int4_configs else (64, 4, 2)
    else:
        BLOCK_N, target_splits, num_warps = 64, 1, 2
    num_tiles = triton.cdiv(num_columns, BLOCK_N)
    # Never more splits than tiles, never empty.
    num_splits = min(target_splits, num_tiles)
    return BLOCK_N, num_warps, num_tiles, num_splits


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    *,
    output_gate: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged K/V caches, bf16, FP8 or INT4.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.

    A quantized cache comes with per-(token, head) float32 scale caches shaped
    ``[blocks, page, heads]``; a bf16 cache comes with none. An INT4 cache is
    uint8, two values a byte, each row stored after the per-token-head INT4
    write's randomized Hadamard rotation.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    kv_fp8 = k_cache.dtype == FP8_KV_DTYPE
    kv_int4 = k_cache.dtype == torch.uint8
    quantized = kv_fp8 or kv_int4
    if quantized != (k_scale_cache is not None) or quantized != (
        v_scale_cache is not None
    ):
        raise ValueError(
            "QSA sparse attention needs scale caches iff the cache is quantized"
        )
    if quantized:
        assert k_scale_cache is not None and v_scale_cache is not None
        if (
            k_scale_cache.shape != k_cache.shape[:3]
            or v_scale_cache.shape != k_cache.shape[:3]
        ):
            raise ValueError("QSA scale caches must be [blocks, page, heads]")
        assert k_scale_cache.dtype == v_scale_cache.dtype == torch.float32
        assert k_scale_cache.device == v_scale_cache.device == q.device
        scale_strides = (*k_scale_cache.stride(), *v_scale_cache.stride())
    else:
        scale_strides = (None,) * 6
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    values_per_byte = 2 if kv_int4 else 1
    if (
        q.shape[2] != values_per_byte * k_cache.shape[3]
        or q.shape[1] % k_cache.shape[2]
    ):
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype in (torch.bfloat16, FP8_KV_DTYPE, torch.uint8)
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    if kv_int4:
        # The cache holds each row rotated by the unnormalized randomized
        # Hadamard transform, which grows a norm by sqrt(head_dim). Rotating
        # the query the same way and dividing by head_dim keeps every score;
        # the output, a mix of rotated values, is rotated back below, and the
        # gate applied after that, so the kernel leaves it out.
        query = qsa_int4_rotate(q, "query")
        attended = torch.empty_like(q)
    else:
        query, attended = q, out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = _select_config(
        q.shape[0],
        k_cache.shape[2],
        use_prefill_config,
        selection_width,
        _int4_configs(kv_int4),
    )

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = attended
        partial_lse = attended
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    common = dict(
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if kv_int4:
        _qsa_sparse_paged_gqa_splitk_int4_kernel[partial_grid](
            query,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            attended,
            query.stride(0),
            query.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            *scale_strides,
            logical_indices.stride(0),
            block_table.stride(0),
            attended.stride(0),
            attended.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            **common,
        )
    else:
        _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            out,
            output_gate_view,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            *scale_strides,
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            KV_FP8=kv_fp8,
            **common,
        )
    if num_splits > 1:
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output,
            partial_lse,
            attended,
            output_gate_view,
            attended.stride(0),
            attended.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            q.shape[0],
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            APPLY_GATE=not kv_int4,
            num_warps=2,
            num_stages=1,
        )
    if kv_int4:
        # Rounded to BF16 before the gate in FP32, as the kernel does.
        qsa_int4_rotate(attended, "output", out=out, gate=output_gate_view)
    return out


def warmup_qsa_sparse_paged_attention(
    views: QSAKVCacheViews,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every production-reachable split-K/merge specialization."""

    key_cache, value_cache = views.key, views.value
    # A packed INT4 row holds two values a byte.
    head_dim = key_cache.shape[-1] * (2 if views.is_int4 else 1)
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    # Every config the dispatch can pick for this group size.
    profiles = {
        _select_config(
            num_rows,
            num_kv_heads,
            use_prefill_config,
            selection_width,
            _int4_configs(views.is_int4),
        )
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    # Scalars constant per deployment get their real values (their divisibility
    # specialization is wanted); the batch-varying ones are do_not_specialize'd
    # on the kernels, so any value here compiles the only variant.
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        key_cache.dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        value_cache.dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    if views.is_quantized:
        assert views.key_scale is not None and views.value_scale is not None
        k_scale_ptr = TritonWarmupTensor(
            torch.float32,
            shape=tuple(views.key_scale.shape),
            strides=tuple(views.key_scale.stride()),
        )
        v_scale_ptr = TritonWarmupTensor(
            torch.float32,
            shape=tuple(views.value_scale.shape),
            strides=tuple(views.value_scale.stride()),
        )
        scale_strides = (*views.key_scale.stride(), *views.value_scale.stride())
    else:
        k_scale_ptr = None
        v_scale_ptr = None
        scale_strides = (None,) * 6
    # +1: the packed buffer's trailing count column.
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    # The output gate is mandatory at runtime; warm the specialization that
    # applies it, or for INT4 the one leaving it to the caller.
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(key_cache.shape[0])

    warmed = []
    for block_n, warps, num_tiles, num_splits in sorted(profiles):
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        common = dict(
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=2,
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if views.is_int4:
            _qsa_sparse_paged_gqa_splitk_int4_kernel.warmup(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                k_scale_ptr,
                v_scale_ptr,
                indices_ptr,
                block_table_ptr,
                token_to_req_ptr,
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                row_stride,
                head_stride,
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                *scale_strides,
                selection_width + 1,
                block_table.stride(0),
                row_stride,
                head_stride,
                num_rows,
                num_cache_blocks,
                num_requests,
                **common,
            )
        else:
            _qsa_sparse_paged_gqa_splitk_kernel.warmup(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                k_scale_ptr,
                v_scale_ptr,
                indices_ptr,
                block_table_ptr,
                token_to_req_ptr,
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                *scale_strides,
                selection_width + 1,
                block_table.stride(0),
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                num_cache_blocks,
                num_requests,
                KV_FP8=views.is_fp8,
                **common,
            )
        if num_splits > 1:
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                APPLY_GATE=not views.is_int4,
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
            )
        warmed.append((block_n, num_splits, warps))
    if views.is_int4:
        # The rotations around the INT4 kernel and in its cache write: the
        # query and the gated output over the query heads, K and V over the
        # KV heads.
        block_cols, block_k = _rotate_config(head_dim)
        # The matrices are built here, before the graphs are captured: one
        # first built inside a capture would hold its values only once the
        # graph replays.
        for kind in ("cache", "query", "output"):
            _int4_rotation(kind, head_dim, torch.bfloat16, key_cache.device)
        matrix_ptr = TritonWarmupTensor(torch.bfloat16, shape=(head_dim, head_dim))
        for num_heads, gated in (
            (num_query_heads, False),
            (num_query_heads, True),
            (num_kv_heads, False),
        ):
            rows_ptr = TritonWarmupTensor(
                torch.bfloat16, shape=(num_rows, num_heads, head_dim)
            )
            token_stride = num_heads * head_dim
            _qsa_int4_rotate_kernel.warmup(
                rows_ptr,
                matrix_ptr,
                rows_ptr,
                rows_ptr,
                token_stride,
                head_dim,
                token_stride if gated else 0,
                head_dim if gated else 0,
                token_stride,
                head_dim,
                num_rows * num_heads,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                BLOCK_ROWS=_ROTATE_BLOCK_ROWS,
                BLOCK_COLS=block_cols,
                BLOCK_K=block_k,
                APPLY_GATE=gated,
                num_warps=4,
                num_stages=2,
                grid=(
                    triton.cdiv(num_rows * num_heads, _ROTATE_BLOCK_ROWS),
                    head_dim // block_cols,
                ),
            )
    return tuple(warmed)


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_int4_rotate",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "qsa_store_quantized_rows",
    "warmup_qsa_sparse_paged_attention",
]
