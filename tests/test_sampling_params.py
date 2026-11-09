# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


# zero_recurrent_state: what construction accepts and refuses. The position is
# what the scheduler cuts a chunk on and what the worker arms, so a wrong one
# blanks somewhere else and an unreachable one blanks nowhere without saying so.


def test_zero_recurrent_takes_an_absolute_position():
    params = SamplingParams(zero_recurrent_state=True, zero_recurrent_at=1568)
    assert params.zero_recurrent_at == 1568


def test_zero_recurrent_position_without_the_switch_is_rejected():
    """A position with the feature off would blank nothing; refusing beats
    accepting a request whose stated intent cannot happen."""
    with pytest.raises(ValueError, match="zero_recurrent_state is False"):
        SamplingParams(zero_recurrent_at=1568)


def test_zero_recurrent_switch_without_a_position_is_rejected():
    """There is no default position: it decides which state the model keeps,
    and a silently-chosen one is a different perturbation."""
    with pytest.raises(ValueError, match="without zero_recurrent_at"):
        SamplingParams(zero_recurrent_state=True)


def test_zero_recurrent_negative_position_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        SamplingParams(zero_recurrent_state=True, zero_recurrent_at=-1)


def test_zero_recurrent_position_is_not_checked_against_the_prompt_here():
    """Construction cannot see the prompt, so the range check lives where the
    tokenized length is known (`InputProcessor.process_inputs`, which rejects
    a position the prompt does not reach). This pins the split, so a reader
    does not look for the check on the wrong side of it."""
    SamplingParams(zero_recurrent_state=True, zero_recurrent_at=4000)
