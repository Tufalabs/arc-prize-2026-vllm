# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""speculative_config.draft_top_k_top_p: the drafter's distribution cut to the
request's own top_k and top_p before a draft token is drawn.

Exactness does not depend on the cut -- rejection sampling is exact for any
proposal -- only on the proposal the rejection sampler reads being the one the
draft was drawn from. So: the draft is drawn from the cut logits, and the
logits cached for the rejection sampler are the cut ones. What the cut buys is
acceptance, which needs it to be the target sampler's own: temperature first,
then top-k and top-p on the scaled distribution.
"""

import math
import types

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for draft top-k/top-p tests", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

DEVICE = "cuda"
VOCAB_SIZE = 32_768
NEG_INF = float("-inf")


def _mask(top_k, top_p, logits, idx_mapping, temperature):
    """The speculator's cut, with only the two buffers it reads."""
    speculator = types.SimpleNamespace(top_k=top_k, top_p=top_p)
    return DraftModelSpeculator._mask_top_k_top_p(
        speculator, logits, idx_mapping, temperature
    )


def _logits(num_rows: int, seed: int, scale: float = 3.0) -> torch.Tensor:
    """Spread logits with a quarter of the vocabulary at -inf, as a reduced
    draft head scatters them."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = scale * torch.randn(num_rows, VOCAB_SIZE, generator=gen, device=DEVICE)
    outside = torch.rand(VOCAB_SIZE, generator=gen, device=DEVICE) < 0.25
    return logits.masked_fill(outside, NEG_INF)


def test_the_cut_is_the_target_samplers():
    # Request states 0-5; rows point at them, one padded row at -1.
    top_k = torch.tensor([VOCAB_SIZE, 194, 50, VOCAB_SIZE, 194, 1], dtype=torch.int32)
    top_p = torch.tensor([0.8899, 1.0, 0.5, 1.0, 0.8899, 0.9])
    temperature = torch.tensor([0.6, 1.0, 0.3, 0.6, 0.0, 1.2])
    idx_mapping = torch.tensor([3, 0, 5, 1, 4, 2, -1], dtype=torch.int32)
    top_k, top_p, temperature, idx_mapping = (
        t.to(DEVICE) for t in (top_k, top_p, temperature, idx_mapping)
    )
    logits = _logits(len(idx_mapping), seed=7)

    cut = _mask(top_k, top_p, logits, idx_mapping, temperature)

    for row, state in enumerate(idx_mapping.tolist()):
        if state < 0:
            continue  # never sampled
        temp = temperature[state].item()
        scaled = logits[row : row + 1] / (temp if temp > 0 else 1.0)
        expected = apply_top_k_top_p_pytorch(
            scaled.clone(), top_k[state : state + 1], top_p[state : state + 1]
        )
        kept = expected[0] > NEG_INF
        assert torch.equal(cut[row] > NEG_INF, kept), f"row {row}"
        # Kept tokens keep their pre-temperature logits: the rejection sampler
        # applies the temperature itself.
        assert torch.equal(cut[row][kept], logits[row][kept]), f"row {row}"
    # The greedy request still has its argmax.
    greedy_row = idx_mapping.tolist().index(4)
    assert cut[greedy_row].argmax() == logits[greedy_row].argmax()


def test_a_request_without_top_k_or_top_p_is_untouched():
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    top_k = torch.full((2,), VOCAB_SIZE, dtype=torch.int32, device=DEVICE)
    top_p = torch.ones(2, device=DEVICE)
    temperature = torch.tensor([0.6, 1.0], device=DEVICE)
    logits = _logits(2, seed=11)
    assert torch.equal(_mask(top_k, top_p, logits, idx_mapping, temperature), logits)


def test_drafts_are_drawn_from_the_cut_and_cached_as_it():
    """One request at ARC2 main's play profile (temperature 0.6, top_k 194,
    top_p 0.8899): every draft lands inside the cut, their frequencies are
    the cut distribution's, and the cached row the rejection sampler reads
    as the proposal is the cut row."""
    temp, k, p = 0.6, 194, 0.8899
    idx_one = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([temp], device=DEVICE)
    # Flat enough that both cuts bind (top-k first, then top-p over what it
    # kept, as the target sampler cuts) and every kept token is drawn often
    # enough to test its frequency.
    logits = _logits(1, seed=13, scale=0.3)
    cut = _mask(
        torch.tensor([k], dtype=torch.int32, device=DEVICE),
        torch.tensor([p], device=DEVICE),
        logits,
        idx_one,
        temperature,
    )
    seed = torch.tensor([0xD7AF7], dtype=torch.int64, device=DEVICE)

    # The draft step: one row, drawn and cached.
    cache = torch.zeros(1, 1, VOCAB_SIZE, device=DEVICE)
    gumbel_sample(
        cut,
        idx_one,
        temperature,
        seed,
        torch.zeros(1, dtype=torch.int64, device=DEVICE),
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=torch.zeros((), dtype=torch.int64, device=DEVICE),
    )
    assert torch.equal(cache[0, 0], cut[0].to(cache.dtype))

    # Many independent draws from the same row.
    num_samples = 400_000
    draws = gumbel_sample(
        cut.expand(num_samples, VOCAB_SIZE),
        torch.zeros(num_samples, dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        torch.arange(num_samples, dtype=torch.int64, device=DEVICE),
        apply_temperature=True,
        is_drafting=True,
    )
    kept = cut[0] > NEG_INF
    num_kept = kept.sum().item()
    assert 0 < num_kept < k
    assert kept[draws].all(), "a draft outside the cut"
    probs = torch.softmax(cut[0].double() / temp, dim=0)
    counts = torch.bincount(draws, minlength=VOCAB_SIZE).double()
    expected = probs * num_samples
    tested = expected >= 25
    z = (counts[tested] - expected[tested]) / expected[tested].sqrt()
    assert tested.sum().item() == num_kept
    assert z.abs().max().item() < 6.0, f"max |z| {z.abs().max().item():.2f}"
    # The cut is what raises acceptance: the uncut distribution puts mass
    # outside it that the target, cut the same way, can never accept.
    uncut = torch.softmax(logits[0].double() / temp, dim=0)
    assert math.isclose(probs[kept].sum().item(), 1.0, rel_tol=1e-9)
    assert uncut[~kept].sum().item() > 0.0
