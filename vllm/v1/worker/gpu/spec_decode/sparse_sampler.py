# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse speculative sampling (XKB_SPARSE_SAMPLER).

The stock chain samples each draft over the full vocabulary although the draft
head has a reduced one (EAGLE-3 d2t), caches full-vocabulary draft rows, and
verifies against full-vocabulary target rows although top-k leaves at most k
tokens. Here:
  - the drafter samples over its own vocabulary with the noise keyed by target
    token id, so it draws the same token as the stock full-vocabulary sampler
    (equal values, equal noise, ties to the smallest target id), and caches the
    compact row plus its logsumexp after temperature;
  - the target sampler's top-k fold lists the kept tokens and their values
    instead of writing the masked row (apply_top_k_top_p_triton's cand);
  - one kernel per request runs the ratio test and samples the bonus or
    residual token over the kept tokens only, with the stock kernels' formulas.
Token for token this equals the stock chain given the same logsumexps; the
draft logsumexp, and the target's of rows whose list came from the fold's
outlier buffer (cand_fast), are summed in a different order than the stock
8192-wide tiles, so they can differ in their last bits.
"""

import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    gumbel_noised_argmax,
    tl_rand32,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    VOCAB_BLOCK_SIZE,
    _compute_global_logsumexp,
    _compute_global_target_argmax,
)

# A jit-accessed global, so a tl.constexpr where Triton is available; the
# CPU platform's placeholder has no constexpr, and the rejection sampler
# imports this module whatever XKB_SPARSE_SAMPLER says (as gumbel.py's).
_INT32_MAX = tl.constexpr(2**31 - 1) if HAS_TRITON else 2**31 - 1


@triton.jit
def _target_lse(
    cand_val_ptr,
    cand_stride,
    cand_cnt_ptr,
    local_max_ptr,
    local_sumexp_ptr,
    stats_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    C: tl.constexpr,
    TARGET_STATS: tl.constexpr,
):
    """The target row's logsumexp: from the fold's tile stats, or (not
    TARGET_STATS) from the kept-token list when the row has at most C."""
    if TARGET_STATS:
        lse = _compute_global_logsumexp(
            local_max_ptr,
            stats_stride,
            local_sumexp_ptr,
            stats_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
    else:
        cnt = tl.load(cand_cnt_ptr + logit_idx)
        if cnt <= C:
            offs = tl.arange(0, C)
            vals = tl.load(
                cand_val_ptr + logit_idx * cand_stride + offs,
                mask=offs < cnt,
                other=float("-inf"),
            )
            m = tl.max(vals, axis=0)
            lse = m + tl.log(tl.sum(tl.exp(vals - m), axis=0))
        else:
            lse = _compute_global_logsumexp(
                local_max_ptr,
                stats_stride,
                local_sumexp_ptr,
                stats_stride,
                logit_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    return lse


@triton.jit
def _target_argmax(
    cand_idx_ptr,
    cand_val_ptr,
    cand_stride,
    cand_cnt_ptr,
    local_argmax_ptr,
    local_max_ptr,
    stats_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    C: tl.constexpr,
    TARGET_STATS: tl.constexpr,
):
    """The target row's argmax (smallest position among ties), as _target_lse."""
    if TARGET_STATS:
        am = _compute_global_target_argmax(
            local_max_ptr,
            stats_stride,
            local_argmax_ptr,
            stats_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        ).to(tl.int64)
    else:
        cnt = tl.load(cand_cnt_ptr + logit_idx)
        if cnt <= C:
            offs = tl.arange(0, C)
            ids = tl.load(
                cand_idx_ptr + logit_idx * cand_stride + offs,
                mask=offs < cnt,
                other=_INT32_MAX,
            )
            vals = tl.load(
                cand_val_ptr + logit_idx * cand_stride + offs,
                mask=offs < cnt,
                other=float("-inf"),
            )
            m = tl.max(vals, axis=0)
            am = tl.min(tl.where(vals == m, ids, _INT32_MAX), axis=0).to(tl.int64)
            am = tl.where(m > -float("inf"), am, 0)
        else:
            am = _compute_global_target_argmax(
                local_max_ptr,
                stats_stride,
                local_argmax_ptr,
                stats_stride,
                logit_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            ).to(tl.int64)
    return am


@triton.jit
def _draft_sample_compact_kernel(
    # [num_tokens]
    sampled_ptr,
    # [max_num_reqs, num_cols, draft_vocab_size]
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    cache_col_ptr,
    # [max_num_reqs, num_cols]
    lse_ptr,
    lse_stride_0,
    # [num_tokens, draft_vocab_size]
    logits_ptr,
    logits_stride,
    # [draft_vocab_size] int32 target token ids
    targets_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    draft_vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    if PER_TOKEN_COL:
        col = tl.load(cache_col_ptr + token_idx)
    else:
        col = tl.load(cache_col_ptr)
    cache_row = cache_ptr + req_state_idx * cache_stride_0 + col * cache_stride_1

    best_value = -float("inf")
    best_token = tl.zeros((), dtype=tl.int32)
    best_set = tl.zeros((), dtype=tl.int32)
    run_max = -float("inf")
    run_sum = 0.0
    for start in range(0, draft_vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < draft_vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        # The cache holds the logits before temperature, as gumbel_sample's.
        tl.store(cache_row + offs, logits, mask=mask & is_valid_req)
        keys = tl.load(targets_ptr + offs, mask=mask, other=0)
        # gumbel_noised_argmax(IS_DRAFTING, APPLY_TEMPERATURE, fp32), keyed by
        # target token id; the per-element math is the same expressions.
        if temp != 0.0:
            logits = logits / temp
            gumbel_seed = tl.randint(seed, pos + _DRAFT_NOISE_SALT)
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
            noised = tl.where(mask, logits + gumbel_noise, float("-inf"))
            # Logsumexp of the scaled row, for the rejection sampler.
            blk_max = tl.max(tl.where(mask, logits, float("-inf")), axis=0)
            new_max = tl.maximum(run_max, blk_max)
            if new_max > -float("inf"):
                run_sum = run_sum * tl.exp(run_max - new_max) + tl.sum(
                    tl.where(mask, tl.exp(logits - new_max), 0.0)
                )
                run_max = new_max
        else:
            noised = logits
        value = tl.max(noised, axis=0)
        # Ties go to the smallest target id, as the stock sampler's
        # first-index argmax over the full vocabulary.
        token = tl.min(tl.where(mask & (noised == value), keys, _INT32_MAX), axis=0)
        better = (best_set == 0) | (value > best_value)
        same = (best_set != 0) & (value == best_value)
        best_token = tl.where(
            better, token, tl.where(same, tl.minimum(best_token, token), best_token)
        )
        best_value = tl.where(better, value, best_value)
        best_set = tl.full((), 1, tl.int32)
    tl.store(sampled_ptr + token_idx, best_token.to(tl.int64))
    tl.store(
        lse_ptr + req_state_idx * lse_stride_0 + col,
        run_max + tl.log(run_sum),
        mask=is_valid_req & (temp != 0.0),
    )


def draft_sample_compact(
    logits: torch.Tensor,  # [num_tokens, draft_vocab_size]
    targets: torch.Tensor,  # [draft_vocab_size] int32 target ids
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    cache: torch.Tensor,  # [max_num_reqs, num_cols, draft_vocab_size]
    cache_lse: torch.Tensor,  # [max_num_reqs, num_cols] fp32
    cache_col: torch.Tensor,  # scalar or [num_tokens]
    block_size: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    """gumbel_sample(apply_temperature=True, is_drafting=True) for a draft head
    whose logits cover targets only (every other token at -inf), without
    building the full-vocabulary row."""
    num_tokens, draft_vocab_size = logits.shape
    assert logits.stride(1) == 1 and cache.stride(2) == 1
    assert targets.dtype == torch.int32 and targets.shape == (draft_vocab_size,)
    assert cache.size(-1) == draft_vocab_size
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    cache_col = cache_col.contiguous()
    sampled = logits.new_empty(num_tokens, dtype=torch.int64)
    _draft_sample_compact_kernel[(num_tokens,)](
        sampled,
        cache,
        cache.stride(0),
        cache.stride(1),
        cache_col,
        cache_lse,
        cache_lse.stride(0),
        logits,
        logits.stride(0),
        targets,
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        draft_vocab_size,
        BLOCK_SIZE=block_size,
        PER_TOKEN_COL=cache_col.dim() > 0,
        num_warps=num_warps,
    )
    return sampled


@triton.jit
def _sparse_verify_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, C], [num_logits, C], [num_logits]
    cand_idx_ptr,
    cand_val_ptr,
    cand_stride,
    cand_cnt_ptr,
    # [num_logits, V]: the masked row, written only where cnt > C
    dense_ptr,
    dense_stride,
    # [num_logits, num_blocks] target tile statistics from the fold
    target_local_argmax_ptr,
    target_local_max_ptr,
    target_local_sumexp_ptr,
    target_stats_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, draft_vocab_size]
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    # [max_num_reqs, num_speculative_steps]
    cache_lse_ptr,
    cache_lse_stride_0,
    # [V] int32 draft index of each target token, -1 outside the draft vocabulary
    t2d_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    vocab_num_blocks,
    C: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    TARGET_STATS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    cache_req = cache_ptr + req_state_idx * cache_stride_0
    offs = tl.arange(0, C)

    # _rejection_kernel's standard path (no synthetic, block or relaxed mode).
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
            if is_greedy:
                target_argmax = _target_argmax(
                    cand_idx_ptr,
                    cand_val_ptr,
                    cand_stride,
                    cand_cnt_ptr,
                    target_local_argmax_ptr,
                    target_local_max_ptr,
                    target_stats_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    C,
                    TARGET_STATS,
                )
                accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                cnt = tl.load(cand_cnt_ptr + logit_idx)
                if cnt <= C:
                    ids = tl.load(
                        cand_idx_ptr + logit_idx * cand_stride + offs,
                        mask=offs < cnt,
                        other=-1,
                    )
                    vals = tl.load(
                        cand_val_ptr + logit_idx * cand_stride + offs,
                        mask=offs < cnt,
                        other=float("-inf"),
                    )
                    target_logit = tl.max(
                        tl.where(ids == draft_sampled, vals, float("-inf")), axis=0
                    )
                else:
                    target_logit = tl.load(
                        dense_ptr + logit_idx * dense_stride + draft_sampled
                    ).to(tl.float32)
                target_lse = _target_lse(
                    cand_val_ptr,
                    cand_stride,
                    cand_cnt_ptr,
                    target_local_max_ptr,
                    target_local_sumexp_ptr,
                    target_stats_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    C,
                    TARGET_STATS,
                )
                target_logprob = target_logit - target_lse
                d = tl.load(t2d_ptr + draft_sampled)
                draft_logit = (
                    tl.load(
                        cache_req + i * cache_stride_1 + d,
                        mask=d >= 0,
                        other=float("-inf"),
                    ).to(tl.float32)
                    / temp
                )
                draft_lse = tl.load(
                    cache_lse_ptr + req_state_idx * cache_lse_stride_0 + i
                )
                draft_logprob = draft_logit - draft_lse
                accepted = target_logprob > tl.log(u) + draft_logprob
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))

    # _resample_kernel + _insert_resampled_kernel over the kept tokens.
    resample_idx = accepted_length
    resample_token_idx = start_idx + resample_idx
    is_bonus = resample_token_idx == end_idx - 1
    if not is_greedy or is_bonus:
        rejected_draft_token = tl.load(
            draft_sampled_ptr + resample_token_idx + 1, mask=not is_bonus, other=0
        )
        use_target = is_bonus | (rejected_draft_token < 0)
        target_lse = _target_lse(
            cand_val_ptr,
            cand_stride,
            cand_cnt_ptr,
            target_local_max_ptr,
            target_local_sumexp_ptr,
            target_stats_stride,
            resample_token_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
            C,
            TARGET_STATS,
        )
        draft_lse = tl.load(
            cache_lse_ptr + req_state_idx * cache_lse_stride_0 + resample_idx,
            mask=use_target == 0,
            other=0.0,
        )
        draft_row = cache_req + resample_idx * cache_stride_1
        pos = tl.load(pos_ptr + resample_token_idx)
        cnt = tl.load(cand_cnt_ptr + resample_token_idx)
        if cnt <= C:
            mask = offs < cnt
            ids = tl.load(
                cand_idx_ptr + resample_token_idx * cand_stride + offs,
                mask=mask,
                other=0,
            )
            target_logits = tl.load(
                cand_val_ptr + resample_token_idx * cand_stride + offs,
                mask=mask,
                other=float("-inf"),
            )
            if use_target:
                residual_logits = target_logits
            else:
                d = tl.load(t2d_ptr + ids, mask=mask, other=-1)
                draft_logits = (
                    tl.load(
                        draft_row + d, mask=mask & (d >= 0), other=float("-inf")
                    ).to(tl.float32)
                    / temp
                )
                target_log_probs = target_logits - target_lse
                draft_log_probs = draft_logits - draft_lse
                ratio = tl.exp(draft_log_probs - target_log_probs)
                residual_logits = tl.where(
                    ratio < 1.0,
                    target_log_probs + tldevice.log1p(-ratio),
                    float("-inf"),
                ).to(tl.float32)
            value, idx = gumbel_noised_argmax(
                residual_logits,
                ids,
                mask,
                seed,
                pos,
                temp,
                IS_DRAFTING=False,
                USE_FP64=False,
                APPLY_TEMPERATURE=False,
            )
            token = tl.sum(tl.where(offs == idx, ids, 0), axis=0).to(tl.int64)
            # Every residual at -inf: the stock kernels return token 0.
            token = tl.where(value > -float("inf"), token, 0)
        else:
            # More kept tokens than candidates: the stock scan of the masked row.
            best_value = -float("inf")
            token = tl.zeros((), dtype=tl.int64)
            for b in range(0, vocab_size, DENSE_BLOCK):
                block = b + tl.arange(0, DENSE_BLOCK)
                mask = block < vocab_size
                target_logits = tl.load(
                    dense_ptr + resample_token_idx * dense_stride + block,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                if use_target:
                    residual_logits = target_logits
                else:
                    d = tl.load(t2d_ptr + block, mask=mask, other=-1)
                    draft_logits = (
                        tl.load(
                            draft_row + d, mask=mask & (d >= 0), other=float("-inf")
                        ).to(tl.float32)
                        / temp
                    )
                    target_log_probs = target_logits - target_lse
                    draft_log_probs = draft_logits - draft_lse
                    ratio = tl.exp(draft_log_probs - target_log_probs)
                    residual_logits = tl.where(
                        ratio < 1.0,
                        target_log_probs + tldevice.log1p(-ratio),
                        float("-inf"),
                    ).to(tl.float32)
                value, idx = gumbel_noised_argmax(
                    residual_logits,
                    block,
                    mask,
                    seed,
                    pos,
                    temp,
                    IS_DRAFTING=False,
                    USE_FP64=False,
                    APPLY_TEMPERATURE=False,
                )
                take = value > best_value
                token = tl.where(take, (b + idx).to(tl.int64), token)
                best_value = tl.where(take, value, best_value)
        tl.store(sampled_ptr + req_idx * sampled_stride + resample_idx, token)


def sparse_rejection_sample(
    cand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dense: torch.Tensor,  # [num_logits, V] fp32, read only where cnt > C
    target_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cache: torch.Tensor,  # [max_num_reqs, num_speculative_steps, draft_vocab_size]
    cache_lse: torch.Tensor,  # [max_num_reqs, num_speculative_steps]
    t2d: torch.Tensor,  # [V] int32
    draft_sampled: torch.Tensor,  # [num_logits]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    pos: torch.Tensor,  # [num_logits]
    idx_mapping: torch.Tensor,  # [num_reqs]
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    num_warps: int = 4,
    target_stats_complete: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rejection_sample's standard path (temperature, top-k, probabilistic
    drafts) over the fold's kept-token lists and the compact draft cache.
    target_stats_complete False (the fold's cand_fast) takes the logsumexp
    and greedy argmax of rows with at most C kept tokens from their lists."""
    cand_idx, cand_val, cand_cnt = cand
    target_local_argmax, target_local_max, target_local_sumexp = target_stats
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, C = cand_idx.shape
    vocab_size = dense.shape[1]
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    assert target_local_max.shape == (num_logits, vocab_num_blocks)
    stride = target_local_max.stride(0)
    assert target_local_argmax.stride(0) == target_local_sumexp.stride(0) == stride
    assert dense.stride(1) == 1 and cache.stride(2) == 1
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _sparse_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        cand_idx,
        cand_val,
        cand_idx.stride(0),
        cand_cnt,
        dense,
        dense.stride(0),
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        stride,
        draft_sampled,
        cache,
        cache.stride(0),
        cache.stride(1),
        cache_lse,
        cache_lse.stride(0),
        t2d,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        vocab_size,
        vocab_num_blocks,
        C=C,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(vocab_num_blocks),
        DENSE_BLOCK=1024,
        TARGET_STATS=target_stats_complete,
        num_warps=num_warps,
    )
    return sampled, num_sampled


def draft_vocab_maps(
    d2t_targets: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(targets int32 [draft_vocab_size], t2d int32 [vocab_size]) from the
    draft head's target ids (base + d2t)."""
    targets = d2t_targets.to(torch.int32).contiguous()
    t2d = torch.full((vocab_size,), -1, dtype=torch.int32, device=targets.device)
    t2d[targets.long()] = torch.arange(
        targets.numel(), dtype=torch.int32, device=targets.device
    )
    return targets, t2d


def expand_draft_cache(
    cache: torch.Tensor, targets: torch.Tensor, full: torch.Tensor
) -> torch.Tensor:
    """The stock full-vocabulary cache from the compact one; full's columns
    outside targets must already hold -inf."""
    full[:, :, targets.long()] = cache
    return full


# Kept-token list size of the target rows; rows keeping more get the masked
# row (top-k above 256, or no top-k).
SPARSE_LIST_SIZE = 256


def sparse_sampler_requested() -> bool:
    return os.environ.get("XKB_SPARSE_SAMPLER", "0") == "1"


class SparseDraftState:
    """The drafter's compact cache (XKB_SPARSE_SAMPLER) beside the stock
    full-vocabulary one, which expand() rebuilds for a batch the sparse
    verification does not cover (logprobs, other rejection modes)."""

    def __init__(
        self, targets: torch.Tensor, vocab_size: int, full_cache: torch.Tensor
    ):
        self.targets, self.t2d = draft_vocab_maps(targets, vocab_size)
        self.targets_long = self.targets.long()
        self.full = full_cache
        max_num_reqs, num_steps, _ = full_cache.shape
        self.cache = torch.zeros(
            (max_num_reqs, num_steps, self.targets.numel()),
            dtype=full_cache.dtype,
            device=full_cache.device,
        )
        self.cache_lse = torch.zeros(
            (max_num_reqs, num_steps), dtype=torch.float32, device=full_cache.device
        )
        # The stock cache holds -inf outside the draft vocabulary once a row
        # is written; expand() writes only the draft vocabulary.
        full_cache.fill_(float("-inf"))
        self.list_size = SPARSE_LIST_SIZE

    def expand(self) -> None:
        self.full[:, :, self.targets_long] = self.cache


def make_sparse_draft_state(
    targets: torch.Tensor | None, vocab_size: int, full_cache: torch.Tensor | None
) -> tuple["SparseDraftState | None", str]:
    """The state, or None and the reason the sparse sampler stays off."""
    if full_cache is None:
        return None, "drafts are not probabilistic"
    if targets is None:
        return None, "the draft head has no reduced vocabulary"
    if int(os.environ.get("XKB_SAMPLER_FOLD", "0") or 0) < 2:
        return None, "XKB_SAMPLER_FOLD is below 2"
    if full_cache.shape[-1] != vocab_size:
        return None, "the draft cache is not the target vocabulary"
    t = targets.long()
    if (
        t.numel() == 0
        or int(t.min()) < 0
        or int(t.max()) >= vocab_size
        or t.unique().numel() != t.numel()
    ):
        return (
            None,
            "the draft vocabulary's target ids are not distinct "
            "ids below the vocabulary size",
        )
    return SparseDraftState(targets, vocab_size, full_cache), ""
