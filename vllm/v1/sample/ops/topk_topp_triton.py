# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Combined Top-K and Top-P Triton kernels.

Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
using Pivot-based Truncation and Selection" By Park et al.
(https://arxiv.org/abs/2602.01518)

"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units

_TRITON_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_TRITON_BUFFER_CACHE: dict[
    tuple[torch.device, torch.dtype, int] | tuple[torch.device, str], torch.Tensor
] = {}

# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503, 
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373, 
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198, 
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113, 
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970, 
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844, 
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733, 
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595, 
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485, 
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364, 
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250, 
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147, 
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016, 
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897, 
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751, 
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603, 
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422, 
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163, 
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806, 
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813 
]

_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659, 
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280, 
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030, 
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837, 
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675, 
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521, 
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383, 
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247, 
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124, 
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003, 
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122, 
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248, 
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382, 
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521, 
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673, 
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, 
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, 
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, 
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658, 
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813 
]
# fmt: on


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  → replace both
      - tile min == running min → accumulate count
      - tile min > running min  → keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    # The fold (FOLD): the masked row is written to OUT, scaled by the row's
    # temperature (TEMP[IDX_MAP[row]]) as _temperature_kernel would, and LOGITS
    # may be bf16 and is left untouched; FUSE_STATS also writes each
    # BLOCK_SIZE tile's argmax, max and sum of exp(x - max) of the written row
    # to STATS_* [rows, tiles], what the rejection sampler's
    # _compute_local_logits_stats_kernel computes for the target.
    OUT,
    OUT_STRIDE_0,
    TEMP,
    IDX_MAP,
    STATS_ARGMAX,
    STATS_MAX,
    STATS_SUMEXP,
    STATS_STRIDE_0,
    # CAND (XKB_SPARSE_SAMPLER): instead of the masked row, the kept tokens'
    # positions (ascending) and written values go to CAND_IDX/CAND_VAL
    # [rows, CAND_MAX] and their count to CAND_CNT [rows]; a row keeping more
    # than CAND_MAX tokens also gets its masked row written to OUT.
    CAND_IDX,
    CAND_VAL,
    CAND_CNT,
    CAND_STRIDE_0,
    # CAND_FAST: pass 1 also records the outliers' positions in BUFFER_IDX
    # [programs, IDX_CAP]; a top-k row whose pivot came from the outlier
    # search (so every kept token is an outlier) and has at most IDX_CAP
    # outliers emits its candidates from the buffer and skips the full final
    # read and its stats; such a row keeping more than CAND_MAX tokens gets
    # the masked row and stats written by the full read.
    BUFFER_IDX,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
    FOLD: tl.constexpr,
    FUSE_STATS: tl.constexpr,
    CAND: tl.constexpr = False,
    CAND_MAX: tl.constexpr = 256,
    CAND_FAST: tl.constexpr = False,
    IDX_CAP: tl.constexpr = 32768,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        # The program's scratch row (BUFFER_ROW, BUFFER_IDX_ROW) passes between
        # warps: a gather stores at cumsum positions that other warps' tiles
        # then read. Every handoff has a barrier, this one against the
        # previous row's reads; the conditions around each are per-row
        # scalars, so every warp reaches it.
        tl.debug_barrier()
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE
        BUFFER_IDX_ROW = BUFFER_IDX + pid * IDX_CAP
        used_outliers = tl.zeros((), dtype=tl.int32)
        outlier_pivot = float("inf")
        num_outliers = tl.zeros((), dtype=tl.uint32)
        if FOLD:
            OUT_ROW = OUT + row_id.to(tl.int64) * OUT_STRIDE_0
            req_state_idx = tl.load(IDX_MAP + row_id)
            temp = tl.load(TEMP + req_state_idx).to(tl.float32)
        else:
            OUT_ROW = LOGITS_ROW

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                ).to(tl.float32)
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )

                # Calculate outlier pivot t for Gaussian sigma-truncation
                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                percentile = tl.minimum(percentile, 199)
                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                sigma = sigma + tl.abs(sigma) * -0.15
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # First pass: compute max and min logits and gather outliers
                num_finite_total = tl.zeros((), dtype=tl.uint32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    ).to(tl.float32)

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk_mask = logits_blk > -float("inf")
                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                    num_finite_total += tl.sum(finite_blk_mask & mask_n)

                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)
                    if CAND_FAST:
                        tl.store(
                            BUFFER_IDX_ROW + write_pos,
                            offs_n,
                            mask=outlier_mask & (write_pos < IDX_CAP),
                        )
                # Every warp's outlier stores before the search reads them.
                tl.debug_barrier()

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Second passes: Ternary search for pivot
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.uint32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                if num_outliers > k:
                    used_outliers = tl.full((), 1, dtype=tl.int32)
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute k_pivots_num,
                        # min_larger, and num_min_larger together to avoid
                        # a second data scan. See _update_min_larger_stats
                        # for the tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                else:
                    # If top-k outlier gathering failed, search whole logit space
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass over full vocab (same approach
                        # as the buffer path above).
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            ).to(tl.float32)

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Third pass: Calculate exp logits and sum, gather outliers
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling for Top-k
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = (
                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
                                    )
                                    duplicate_count = (
                                        tl.cumsum(duplicate_mask) + num_kept
                                    )
                                    duplicate_keep_mask = (
                                        duplicate_count <= num_keep
                                    ) & duplicate_mask
                                    duplicate_remove_mask = (
                                        duplicate_mask & ~duplicate_keep_mask
                                    )
                                    outlier_mask = outlier_mask & (
                                        ~duplicate_remove_mask
                                    )
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed,
                            # retry gathering using top-k pivot
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                ).to(tl.float32)

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling for Top-k
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                    tl.int32,
                                )
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(
                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                                )
                            tl.debug_barrier()

                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        tl.debug_barrier()

                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0

                        # Fifth passes: Search for p_pivot
                        found_pivot = 0
                        while found_pivot == 0:
                            p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                            # Single fused pass: compute p_pivots_sum,
                            # min_larger, and num_min_larger together.
                            # See _update_min_larger_stats for the
                            # tile-level merge logic.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                above_0 = probs_blk > p_pivot_0
                                p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                                min_larger_0, num_min_larger_0 = (
                                    _update_min_larger_stats(
                                        probs_blk,
                                        above_0,
                                        min_larger_0,
                                        num_min_larger_0,
                                        1.0,
                                    )
                                )

                            # Check if the pivot satisfies termination condition
                            if p_pivots_sum_0 >= p and (
                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                            ):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # Update range
                            if p_pivots_sum_0 > p:
                                min_range = p_pivot_0
                            elif p_pivots_sum_0 < p:
                                max_range = p_pivot_0

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                p_pivot = (max_range + min_range) / 2.0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                        duplicate_logit = (
                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        )
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast(
                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
                        )
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                ).to(tl.float32)
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    ).to(tl.float32)
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    ).to(tl.float32)
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)
                tl.debug_barrier()

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        ).to(tl.float32)
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)
                    tl.debug_barrier()

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        # The fold writes every row, a pivot of -inf keeping all of it.
        if FOLD or final_pivot != -float("inf"):
            num_kept_0 = num_kept
            num_cand = tl.zeros((), dtype=tl.int32)
            CAND_ROW = row_id.to(tl.int64) * CAND_STRIDE_0
            fast = tl.zeros((), dtype=tl.int32)
            if CAND_FAST and not TOPP_ENABLED:  # noqa: SIM102 - Triton compile-time branch
                # Every kept token is above final_pivot >= outlier_pivot, and a
                # duplicate of duplicate_logit (> final_pivot) at least 1e-9
                # above outlier_pivot cannot match a non-outlier, so the buffer
                # (outliers in token order) yields the same list.
                if (  # noqa: SIM102 - Triton compile-time branch
                    (used_outliers != 0)
                    & (num_outliers <= IDX_CAP)
                    & (final_pivot > -float("inf"))
                ):
                    if (num_keep >= num_duplicate_logit) | (
                        duplicate_logit - outlier_pivot >= 1e-9
                    ):
                        fast = tl.full((), 1, dtype=tl.int32)
            if CAND_FAST:  # noqa: SIM102 - Triton compile-time branch
                if fast != 0:
                    n_out = tl.cast(num_outliers, tl.int32)
                    for i in range(
                        0, (n_out + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC
                    ):
                        offs_b = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_b = offs_b < n_out
                        vals = tl.load(
                            BUFFER_ROW + offs_b, mask=mask_b, other=-float("inf")
                        ).to(tl.float32)
                        ids = tl.load(BUFFER_IDX_ROW + offs_b, mask=mask_b, other=0)
                        keep_mask = (vals > final_pivot) & mask_b
                        if num_keep < num_duplicate_logit:
                            duplicate_mask = (
                                tl.abs(vals - duplicate_logit) < 1e-9
                            ) & mask_b
                            duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                            duplicate_keep_mask = (
                                duplicate_count <= num_keep
                            ) & duplicate_mask
                            duplicate_remove_mask = (
                                duplicate_mask & ~duplicate_keep_mask
                            )
                            num_kept += tl.sum(duplicate_keep_mask)
                            keep_mask = keep_mask & (~duplicate_remove_mask)
                        if temp != 0.0 and temp != 1.0:
                            vals = vals / temp
                        keep_i = keep_mask.to(tl.int32)
                        tile_keep = tl.sum(keep_i)
                        if tile_keep > 0:
                            slot = tl.cumsum(keep_i) - 1 + num_cand
                            cand_mask = keep_mask & (slot < CAND_MAX)
                            tl.store(CAND_IDX + CAND_ROW + slot, ids, mask=cand_mask)
                            tl.store(CAND_VAL + CAND_ROW + slot, vals, mask=cand_mask)
                        num_cand += tile_keep
            if CAND_FAST:  # noqa: SIM108 - Triton compile-time branch
                n_tiles = tl.where(fast == 0, NUM_TILES, 0)
            else:
                n_tiles = NUM_TILES
            for i in range(0, n_tiles):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                ).to(tl.float32)
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                if FOLD:  # noqa: SIM102 - Triton compile-time branch
                    # The same division as _temperature_kernel, which skips
                    # temperatures 0 and 1.
                    if temp != 0.0 and temp != 1.0:
                        logits_blk = logits_blk / temp
                if CAND:
                    keep_i = keep_mask.to(tl.int32)
                    tile_keep = tl.sum(keep_i)
                    if tile_keep > 0:
                        slot = tl.cumsum(keep_i) - 1 + num_cand
                        cand_mask = keep_mask & (slot < CAND_MAX)
                        tl.store(CAND_IDX + CAND_ROW + slot, offs_n, mask=cand_mask)
                        tl.store(CAND_VAL + CAND_ROW + slot, logits_blk, mask=cand_mask)
                    num_cand += tile_keep
                else:
                    tl.store(OUT_ROW + offs_n, logits_blk, mask=mask_n)
                if FUSE_STATS:
                    tile_max, tile_idx = tl.max(logits_blk, axis=0, return_indices=True)
                    tile_sumexp = tl.where(
                        tile_max > -float("inf"),
                        tl.sum(tl.exp(logits_blk - tile_max)),
                        0.0,
                    )
                    stats_off = row_id.to(tl.int64) * STATS_STRIDE_0 + i
                    tl.store(STATS_ARGMAX + stats_off, i * BLOCK_SIZE + tile_idx)
                    tl.store(STATS_MAX + stats_off, tile_max)
                    tl.store(STATS_SUMEXP + stats_off, tile_sumexp)
            if CAND:
                tl.store(CAND_CNT + row_id, num_cand)
                if num_cand > CAND_MAX:
                    # Too many kept tokens for the candidate list: the same
                    # pass again, writing the masked row.
                    num_kept = num_kept_0
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE
                        logits_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        ).to(tl.float32)
                        keep_mask = (logits_blk > final_pivot) & mask_n
                        if num_keep < num_duplicate_logit:
                            duplicate_mask = (
                                tl.abs(logits_blk - duplicate_logit) < 1e-9
                            ) & mask_n
                            duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                            duplicate_keep_mask = (
                                duplicate_count <= num_keep
                            ) & duplicate_mask
                            duplicate_remove_mask = (
                                duplicate_mask & ~duplicate_keep_mask
                            )
                            num_kept += tl.sum(duplicate_keep_mask)
                            keep_mask = keep_mask & (~duplicate_remove_mask)
                        logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                        if temp != 0.0 and temp != 1.0:
                            logits_blk = logits_blk / temp
                        tl.store(OUT_ROW + offs_n, logits_blk, mask=mask_n)
                        if FUSE_STATS and CAND_FAST:  # noqa: SIM102 - Triton compile-time branch
                            if fast != 0:
                                tile_max, tile_idx = tl.max(
                                    logits_blk, axis=0, return_indices=True
                                )
                                tile_sumexp = tl.where(
                                    tile_max > -float("inf"),
                                    tl.sum(tl.exp(logits_blk - tile_max)),
                                    0.0,
                                )
                                stats_off = row_id.to(tl.int64) * STATS_STRIDE_0 + i
                                tl.store(
                                    STATS_ARGMAX + stats_off, i * BLOCK_SIZE + tile_idx
                                )
                                tl.store(STATS_MAX + stats_off, tile_max)
                                tl.store(STATS_SUMEXP + stats_off, tile_sumexp)


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
    fold: tuple | None = None,
    cand: tuple | None = None,
    cand_fast: bool = False,
) -> torch.Tensor:
    """
    Apply combined top-k and top-p masking using Triton.

    fold = (out, temperature, idx_mapping, stats) runs top-k only (p None),
    reads logits of any float dtype without touching them and writes the masked
    row divided by temperature[idx_mapping[row]] (as _temperature_kernel does,
    skipping 0 and 1) into out, an fp32 tensor of the same shape; stats, None or
    (argmax, max, sumexp) tensors of shape [batch, ceil(vocab / 8192)], receives
    each 8192-wide tile's statistics of the written row for the rejection
    sampler. Returns out.

    cand = (idx, val, cnt), with fold only, replaces the masked row by the
    kept tokens: idx [batch, C] int32 receives their positions in ascending
    order, val [batch, C] fp32 the values the row would hold there, and
    cnt [batch] int32 how many were kept. A row keeping more than C tokens
    has only its first C listed and its masked row written to out.
    cand_fast (with cand) lets a top-k row whose cutoff came from the outlier
    buffer emit its list from that buffer, skipping the final full read; such
    rows get no stats unless they keep more than C tokens, so the consumer
    must take the target's logsumexp and argmax from the list.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [batch_size, vocab_size] float32 tensor. The returned tensor
            may alias this input or be a new contiguous tensor for unsupported
            layouts.
        k: [batch_size] int32 tensor of top-k values per row, or None to disable top-k
        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p
        mask_value: Value for masked positions (default: -inf)

    Returns:
        The masked logits tensor. It may or may not be modified in-place.
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32 or fold is not None
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None
    if fold is None:
        out, temperature, idx_mapping, stats = logits, logits, logits, None
    else:
        out, temperature, idx_mapping, stats = fold
        assert topk_enabled and not topp_enabled, "the fold covers top-k rows only"
        assert out.shape == logits.shape and out.dtype == torch.float32
        assert out.stride(1) == 1 and logits.stride(1) == 1
        assert idx_mapping.shape == (batch_size,)

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    # The Triton kernel supports arbitrary row strides, but it still assumes
    # the vocab dimension is laid out contiguously within each row.
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    num_sm = num_compute_units(logits.device.index)
    NUM_PROGRAMS = min(num_sm, batch_size)

    # Cache per-Triton Program buffer on each device.
    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    # Cache lookup table entries on each device.
    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        with gpu_sync_allowed():
            normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
            percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
            _TRITON_TABLE_CACHE[logits.device] = (
                normal_cdf_to_sigma_table,
                percentile_to_std_table,
            )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    # Smaller tiles compile and run faster on CPU; GPU benefits from larger tiles.
    # On XPU, large BLOCK_SIZE causes precision loss in the single-pass pivot
    # approximation; use smaller tiles for accurate top-p results.
    launch_kwargs = {}
    if logits.device.type == "cpu":
        block_size, block_size_trunc = 256, 128
    elif logits.device.type == "xpu":
        block_size, block_size_trunc = 4096, 2048
    else:
        block_size, block_size_trunc = 8192, 4096
        # Each program serially sweeps the vocab row in BLOCK_SIZE tiles, so
        # per-tile latency bounds kernel latency, and Triton's default of 4
        # warps leaves an 8192-wide tile at 16 elements per lane. 8 warps is
        # faster on every arch measured (SM90, SM100, SM120, gfx950); 16 is not.
        launch_kwargs["num_warps"] = 8

    if cand is None:
        cand_idx = cand_val = cand_cnt = logits
        cand_stride, cand_max = 0, 256
    else:
        assert fold is not None, "candidates come from the fold only"
        cand_idx, cand_val, cand_cnt = cand
        cand_max = cand_idx.shape[1]
        assert cand_max & (cand_max - 1) == 0
        assert cand_idx.dtype == cand_cnt.dtype == torch.int32
        assert cand_val.dtype == torch.float32 and cand_val.shape == cand_idx.shape
        assert cand_idx.shape[0] == batch_size and cand_cnt.shape == (batch_size,)
        assert cand_idx.stride(1) == cand_val.stride(1) == 1
        assert cand_idx.stride(0) == cand_val.stride(0)
        cand_stride = cand_idx.stride(0)
    cand_fast = cand_fast and cand is not None and not topp_enabled
    idx_cap = 32768
    if cand_fast:
        idx_key = (logits.device, "cand_idx")
        buffer_idx = _TRITON_BUFFER_CACHE.get(idx_key)
        if buffer_idx is None or buffer_idx.shape[0] < NUM_PROGRAMS:
            buffer_idx = torch.empty(
                (num_sm, idx_cap), dtype=torch.int32, device=logits.device
            )
            _TRITON_BUFFER_CACHE[idx_key] = buffer_idx
    else:
        buffer_idx = logits

    if stats is None:
        stats_argmax = stats_max = stats_sumexp = logits
        stats_stride = 0
    else:
        stats_argmax, stats_max, stats_sumexp = stats
        num_tiles = (vocab_size + block_size - 1) // block_size
        for t in stats:
            assert t.shape == (batch_size, num_tiles) and t.stride(1) == 1, (
                t.shape,
                batch_size,
                num_tiles,
            )
        assert stats_argmax.dtype == torch.int64
        assert stats_max.dtype == stats_sumexp.dtype == torch.float32
        stats_stride = stats_max.stride(0)
        assert stats_argmax.stride(0) == stats_sumexp.stride(0) == stats_stride

    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        out,
        out.stride(0),
        temperature,
        idx_mapping,
        stats_argmax,
        stats_max,
        stats_sumexp,
        stats_stride,
        cand_idx,
        cand_val,
        cand_cnt,
        cand_stride,
        buffer_idx,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
        FOLD=fold is not None,
        FUSE_STATS=stats is not None,
        CAND=cand is not None,
        CAND_MAX=cand_max,
        CAND_FAST=cand_fast,
        IDX_CAP=idx_cap,
        **launch_kwargs,
    )

    return out


def reset_buffer_cache():
    _TRITON_BUFFER_CACHE.clear()
    _TRITON_TABLE_CACHE.clear()
    torch.accelerator.empty_cache()
