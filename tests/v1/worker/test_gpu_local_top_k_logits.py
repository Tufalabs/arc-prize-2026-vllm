# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling from each tensor-parallel rank's own top-k logits, gathered,
instead of the vocabulary all-gather (VLLM_TP_LOCAL_TOPK_LOGITS): when the
model runner takes that path, and that the logits the sampler's top-k keeps
are the ones it keeps from the all-gather."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.model_executor.layers import logits_processor as lp
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu import model_runner as mr
from vllm.v1.worker.gpu.sample.sampler import Sampler

VOCAB = 4096


def _sampler(top_k, **flags):
    n = len(top_k)
    sampler = Mock(spec=Sampler)
    sampler.compute_nans = flags.get("compute_nans", False)
    sampler.trace_replay_state = None
    sampler.sampling_states = SimpleNamespace(
        top_k=SimpleNamespace(np=np.array(top_k, dtype=np.int32))
    )
    sampler.logit_bias_state = SimpleNamespace(
        num_allowed_token_ids=SimpleNamespace(np=np.full(n, flags.get("allowed", 0))),
        raises_logits=np.full(n, flags.get("raises", False)),
        num_logit_bias=SimpleNamespace(np=np.full(n, flags.get("biased", 0))),
        num_stop_token_ids=SimpleNamespace(np=np.full(n, flags.get("stop_tokens", 0))),
    )
    sampler.penalties_state = SimpleNamespace(
        use_penalty=np.full(n, flags.get("penalty", False))
    )
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.full(n, flags.get("bad_words", 0)))
    )
    sampler.thinking_budget_state = SimpleNamespace(
        enabled=flags.get("thinking_budget", False),
        use_thinking_budget=np.full(n, flags.get("thinking_budget", False)),
    )
    sampler.get_logprobs_dims.return_value = flags.get("logprobs")
    return sampler


def _runner(top_k, tp=2, **flags):
    runner = object.__new__(mr.GPUModelRunner)
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=tp)
    runner.sampler = _sampler(top_k, **flags)
    runner.model = SimpleNamespace(compute_logits_top_k=None)
    runner.vocab_size = VOCAB
    batch = SimpleNamespace(
        num_reqs=len(top_k), idx_mapping_np=np.arange(len(top_k), dtype=np.int32)
    )
    return runner, batch


@pytest.mark.parametrize(
    "flags",
    [{}, {"biased": 3}, {"stop_tokens": 2}, {"biased": 100, "stop_tokens": 28}],
    ids=["plain", "a ban", "min_tokens", "lowering up to the margin"],
)
def test_a_top_k_batch_takes_the_local_top_k(monkeypatch, flags):
    """Logit bias that only lowers (a ban, min_tokens holding the stop tokens
    back, as the traffic replay sends) stays on the path up to the margin."""
    monkeypatch.setenv("VLLM_TP_LOCAL_TOPK_LOGITS", "1")
    runner, batch = _runner([194, 20, 194], **flags)
    assert runner._local_top_k(batch, None) == 194 + mr._LOCAL_TOP_K_MARGIN


@pytest.mark.parametrize(
    "why",
    [
        "switch off",
        "one rank",
        "grammar",
        "a request without top-k",
        "allowed tokens",
        "raising logit bias",
        "lowering past the margin",
        "penalties",
        "bad words",
        "thinking budget",
        "logprobs",
        "nan count",
        "custom sampler",
        "model without the hook",
    ],
)
def test_the_vocabulary_all_gather_when_the_top_k_alone_is_not_exact(monkeypatch, why):
    monkeypatch.setenv("VLLM_TP_LOCAL_TOPK_LOGITS", "0" if why == "switch off" else "1")
    cases: dict[str, dict[str, Any]] = {
        "allowed tokens": {"allowed": 5},
        "raising logit bias": {"biased": 1, "raises": True},
        "lowering past the margin": {"biased": 100, "stop_tokens": 29},
        "penalties": {"penalty": True},
        "bad words": {"bad_words": 1},
        "thinking budget": {"thinking_budget": True},
        "logprobs": {"logprobs": (20, 0)},
        "nan count": {"compute_nans": True},
    }
    flags = cases.get(why, {})
    top_k = [194, VOCAB] if why == "a request without top-k" else [194, 194]
    runner, batch = _runner(top_k, tp=1 if why == "one rank" else 2, **flags)
    if why == "custom sampler":
        runner.sampler = object()
    if why == "model without the hook":
        runner.model = SimpleNamespace()
    grammar = Mock() if why == "grammar" else None
    assert runner._local_top_k(batch, grammar) is None


def _two_rank_top_k(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """LogitsProcessor.get_top_k_tokens on rank 0 of two, the vocabulary split
    in halves; the all-gather hands back both ranks' own top k."""
    half = logits.shape[1] // 2
    shards = (logits[:, :half], logits[:, half:])
    proc = object.__new__(lp.LogitsProcessor)
    proc.scale, proc.soft_cap, proc.head_dtype = 1.0, None, None

    def head(rank):
        return SimpleNamespace(
            tp_size=2,
            quant_method=SimpleNamespace(apply=lambda *a, **kw: shards[rank].clone()),
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=0, org_vocab_start_index=rank * half
            ),
        )

    # What rank 1 contributes to each gather: its own top k, by value then id.
    v1, i1 = torch.topk(shards[1], k, dim=-1)
    peer = iter((v1, i1 + half))

    def all_gather(t, dim=-1):
        return torch.cat([t, next(peer).to(t.dtype)], dim=dim)

    orig = lp.tensor_model_parallel_all_gather
    lp.tensor_model_parallel_all_gather = all_gather
    try:
        return proc.get_top_k_tokens(head(0), torch.zeros(logits.shape[0], 1), k)
    finally:
        lp.tensor_model_parallel_all_gather = orig


def test_the_sampler_reads_the_same_logits_from_the_two_ranks_top_k():
    """The top-k mask over the rebuilt logits equals the one over the
    gathered vocabulary, ties at the k-th included; top-p after it keeps the
    same logits a row. Which members of a tie top-p's cut keeps is the
    sort's choice on either path (it is not stable), so the kept values are
    compared, not the token ids."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, VOCAB, generator=g).to(torch.bfloat16).float()
    # A tie at row 0's 20th largest, across both halves of the vocabulary.
    kth = logits[0].topk(20).values[-1].item()
    logits[0, [5, VOCAB // 2 + 7, VOCAB - 3]] = kth
    top_k = torch.tensor([20, 194, 1, 50, 194, 7])
    top_p = torch.tensor([0.89, 0.89, 1.0, 0.5, 0.99, 0.89])
    k = int(top_k.max()) + mr._LOCAL_TOP_K_MARGIN

    runner = object.__new__(mr.GPUModelRunner)
    runner.vocab_size = VOCAB
    runner.model = SimpleNamespace(
        compute_logits_top_k=lambda h, kk: _two_rank_top_k(logits, kk)
    )
    rebuilt = runner._top_k_logits(torch.zeros(6, 1), k)

    want = apply_top_k_top_p_pytorch(logits.clone(), top_k, None)
    got = apply_top_k_top_p_pytorch(rebuilt.clone(), top_k, None)
    assert torch.equal(want, got)
    # Lowering-only bias (min_tokens masks the stop tokens): row 1's five
    # largest masked on both, the top-k after it still among the gathered.
    masked = logits[1].topk(5).indices
    lowered, lowered_rebuilt = logits.clone(), rebuilt.clone()
    lowered[1, masked] = lowered_rebuilt[1, masked] = float("-inf")
    assert torch.equal(
        apply_top_k_top_p_pytorch(lowered, top_k, None),
        apply_top_k_top_p_pytorch(lowered_rebuilt, top_k, None),
    )
    # The tie at row 0's k-th is kept whole, both halves' members.
    assert torch.isfinite(got[0, [5, VOCAB // 2 + 7, VOCAB - 3]]).all()

    want = apply_top_k_top_p_pytorch(logits.clone(), top_k, top_p)
    got = apply_top_k_top_p_pytorch(rebuilt.clone(), top_k, top_p)
    for row_want, row_got in zip(want, got):
        assert torch.equal(
            row_want[torch.isfinite(row_want)].sort().values,
            row_got[torch.isfinite(row_got)].sort().values,
        )


@pytest.mark.parametrize(
    "case", ["lowering bias spends the margin", "a tie wider than the margin"]
)
def test_a_tie_past_the_gathered_logits_keeps_the_all_gathers_values(case):
    """More tokens tie at the k-th logit than the gathered ones hold: which of
    them the top-k keeps can differ from the all-gather, as it does between
    the top-k kernels, but the k values it keeps are the all-gather's."""
    ids = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(1))
    logits = torch.full((1, VOCAB), -10.0)
    lowered = ids[:0]
    if case == "lowering bias spends the margin":
        # 128 tokens lowered by bias, then 21 tied at the 20th logit.
        lowered = ids[:128]
        logits[0, lowered] = 10.0
        logits[0, ids[128:149]] = 5.0
    else:
        logits[0, ids[:170]] = 3.0
    top_k = torch.tensor([20])
    k = int(top_k.max()) + mr._LOCAL_TOP_K_MARGIN

    runner = object.__new__(mr.GPUModelRunner)
    runner.vocab_size = VOCAB
    runner.model = SimpleNamespace(
        compute_logits_top_k=lambda h, kk: _two_rank_top_k(logits, kk)
    )
    rebuilt = runner._top_k_logits(torch.zeros(1, 1), k)

    full, local = logits.clone(), rebuilt.clone()
    full[0, lowered] -= 100.0
    local[0, lowered] -= 100.0
    want = apply_top_k_top_p_pytorch(full, top_k, None)
    got = apply_top_k_top_p_pytorch(local, top_k, None)
    assert torch.equal(want.topk(20).values, got.topk(20).values)
