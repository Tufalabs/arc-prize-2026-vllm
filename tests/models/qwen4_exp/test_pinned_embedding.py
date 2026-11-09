# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp's input embedding in pinned host memory (``--cpu-offload-params
embed_tokens``): selected only when named, and its row gather is the table's
own rows."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model


def _config(params: set[str]) -> SimpleNamespace:
    return SimpleNamespace(
        offload_config=SimpleNamespace(uva=SimpleNamespace(cpu_offload_params=params))
    )


def test_selected_only_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen4_exp_model, "is_uva_available", lambda: True)
    assert qwen4_exp_model.select_input_embedding_method(_config(set())) is None
    assert qwen4_exp_model.select_input_embedding_method(_config({"experts"})) is None
    method = qwen4_exp_model.select_input_embedding_method(_config({"embed_tokens"}))
    assert isinstance(method, qwen4_exp_model.Qwen4ExpPinnedHostEmbeddingMethod)
    # The post-load move to the device would orphan the UVA view.
    assert method.requires_device_loading is False


def test_refused_without_uva(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen4_exp_model, "is_uva_available", lambda: False)
    with pytest.raises(RuntimeError, match="UVA"):
        qwen4_exp_model.select_input_embedding_method(_config({"embed_tokens"}))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the gather reads pinned memory through UVA"
)
def test_gather_returns_the_tables_rows() -> None:
    vocab, dim = 1000, 2560
    layer = nn.Module()
    layer.embedding_dim = dim
    method = qwen4_exp_model.Qwen4ExpPinnedHostEmbeddingMethod()
    method.create_weights(layer, dim, [vocab], dim, vocab, torch.bfloat16)
    assert layer.weight.device.type == "cpu" and layer.weight.is_pinned()
    layer.weight.data.copy_(torch.randn(vocab, dim).to(torch.bfloat16))
    ids = torch.tensor([[0, 7, 999], [42, 42, 3]], device="cuda")
    out = method.embedding(layer, ids)
    assert out.device.type == "cuda" and out.shape == (2, 3, dim)
    assert torch.equal(out.cpu(), layer.weight[ids.cpu()])
    assert method.embedding(layer, ids[:, :0]).shape == (2, 0, dim)
