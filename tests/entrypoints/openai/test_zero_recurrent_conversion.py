# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""zero_recurrent_state over the OpenAI-compatible protocols.

The endpoints build their own `SamplingParams` from their own request
models, so a field the models do not declare never reaches the engine.
`OpenAIBaseModel` allows unknown extras, which means an undeclared key is
accepted, logged as ignored, and dropped -- a request that looks served
and blanks nothing. These tests pin the declaration and the forwarding,
which is the only thing standing between a server client and that.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

pytestmark = pytest.mark.cpu_test

MESSAGES = [{"role": "user", "content": "hi"}]


def test_completion_request_forwards_the_blank():
    request = CompletionRequest(
        model="m", prompt="hi", zero_recurrent_state=True, zero_recurrent_at=1568
    )
    params = request.to_sampling_params(max_tokens=16)
    assert params.zero_recurrent_state is True
    assert params.zero_recurrent_at == 1568


def test_chat_request_forwards_the_blank():
    request = ChatCompletionRequest(
        model="m", messages=MESSAGES, zero_recurrent_state=True, zero_recurrent_at=1568
    )
    params = request.to_sampling_params(max_tokens=16, default_sampling_params={})
    assert params.zero_recurrent_state is True
    assert params.zero_recurrent_at == 1568


@pytest.mark.parametrize(
    "request_obj",
    [
        CompletionRequest(model="m", prompt="hi"),
        ChatCompletionRequest(model="m", messages=MESSAGES),
    ],
)
def test_requests_are_unblanked_by_default(request_obj):
    """An untouched request must not acquire the perturbation."""
    if isinstance(request_obj, ChatCompletionRequest):
        params = request_obj.to_sampling_params(
            max_tokens=16, default_sampling_params={}
        )
    else:
        params = request_obj.to_sampling_params(max_tokens=16)
    assert params.zero_recurrent_state is False
    assert params.zero_recurrent_at is None


def test_a_position_without_the_switch_is_refused_by_sampling_params():
    """The protocol layer defers to SamplingParams' own rule rather than
    duplicating it, so an inconsistent pair fails at conversion."""
    request = CompletionRequest(model="m", prompt="hi", zero_recurrent_at=1568)
    with pytest.raises(ValueError, match="zero_recurrent_state is False"):
        request.to_sampling_params(max_tokens=16)


def test_responses_surface_rejects_the_blank():
    """`/v1/responses` does not forward the fields, and unknown extras are
    allowed — so without an explicit refusal the request would be accepted
    and served unblanked. Rejecting is the honest answer for a surface that
    cannot honour them."""
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    with pytest.raises(Exception, match="not supported on /v1/responses"):
        ResponsesRequest(
            model="m", input="hi", zero_recurrent_state=True, zero_recurrent_at=1568
        )
    # Untouched requests on that surface are unaffected.
    ResponsesRequest(model="m", input="hi")
