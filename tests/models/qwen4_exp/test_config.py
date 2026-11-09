# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def test_qwen4_exp_mtp_returns_sample_and_multi_streams() -> None:
    from vllm.models.qwen4_exp.nvidia.mtp import (
        Qwen4ExpMultiTokenPredictor,
    )

    model = object.__new__(Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model._xkb_keep_rows = [None]
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.layers = [
        lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["hidden_states"],
            torch.zeros(kwargs["hidden_states"].shape[0], 2),
        ),
    ]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    with patch(
        "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
        return_value=pp_group,
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(2),
            intermediate_tensors={"hidden_states": multi_hidden},
        )

    torch.testing.assert_close(
        sample_hidden,
        multi_hidden.unflatten(-1, (2, 4)).mean(dim=-2),
    )
    assert returned_multi_hidden is multi_hidden


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


@pytest.mark.parametrize(
    "ple_layer_ids, pipeline_parallel_size, refused",
    [
        ([1], 1, False),
        ([1], 2, False),
        ([2], 2, True),
        ([1, 2], 2, True),
        ([], 2, False),
    ],
)
def test_qwen4_exp_pipelines_when_every_ple_layer_is_on_the_first_stage(
    ple_layer_ids, pipeline_parallel_size, refused
) -> None:
    """A PLE layer gathers by the raw input_ids, which only the first pipeline
    stage receives; the rest of the architecture is PP-capable. So the refusal
    is exactly a PLE layer placed after that stage -- two layers over two
    stages put layer id 1 on the first and id 2 on the second -- and it lands
    before the engine spends time loading weights."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(ple_layer_ids=ple_layer_ids),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        if refused:
            with pytest.raises(NotImplementedError, match="first pipeline stage"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config
                )
        else:
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
        # Dummy runs establish the addresses used during CUDA graph capture.
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(second["query_start_loc"], expected_query_start_loc)
    torch.testing.assert_close(
        second["ngram_context"], torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr


def test_mtp_block_keeps_its_own_expert_count() -> None:
    """A pruned checkpoint that left its MTP block whole declares the block's
    count as mtp_num_experts; the draft config takes it, a config without
    the key is untouched."""
    from vllm.models.qwen4_exp.nvidia.mtp import _apply_mtp_num_experts

    pruned = Qwen4ExpTextConfig(num_experts=307, mtp_num_experts=512)
    _apply_mtp_num_experts(pruned)
    assert pruned.num_experts == 512

    whole = Qwen4ExpTextConfig(num_experts=512)
    _apply_mtp_num_experts(whole)
    assert whole.num_experts == 512


def test_layer_num_experts_reads_a_per_layer_count() -> None:
    """A checkpoint cut to a different number of experts per layer declares
    one count per decoder layer; a config without the list gives None, which
    leaves every layer at num_experts."""
    from vllm.models.qwen4_exp.nvidia.model import layer_num_experts

    uniform = Qwen4ExpTextConfig(num_experts=205, num_hidden_layers=3)
    assert layer_num_experts(uniform, 1) is None

    uneven = Qwen4ExpTextConfig(
        num_experts=270,
        num_experts_per_tok=10,
        num_hidden_layers=3,
        num_experts_per_layer=[232, 117, 270],
    )
    assert [layer_num_experts(uneven, i) for i in range(3)] == [232, 117, 270]

    short = Qwen4ExpTextConfig(
        num_experts=270, num_hidden_layers=3, num_experts_per_layer=[232, 117]
    )
    with pytest.raises(ValueError, match="names 2 layers"):
        layer_num_experts(short, 0)

    for bad in (9, 271):
        config = Qwen4ExpTextConfig(
            num_experts=270,
            num_experts_per_tok=10,
            num_hidden_layers=3,
            num_experts_per_layer=[232, bad, 270],
        )
        with pytest.raises(ValueError, match=f"has {bad} experts"):
            layer_num_experts(config, 1)


def test_mtp_block_drops_the_decoder_per_layer_counts() -> None:
    """The MTP block is whole: the draft config takes mtp_num_experts and
    none of the decoder's per-layer counts."""
    from vllm.models.qwen4_exp.nvidia.model import layer_num_experts
    from vllm.models.qwen4_exp.nvidia.mtp import _apply_mtp_num_experts

    draft = Qwen4ExpTextConfig(
        num_experts=270,
        mtp_num_experts=512,
        num_hidden_layers=3,
        num_experts_per_layer=[232, 117, 270],
    )
    _apply_mtp_num_experts(draft)
    assert draft.num_experts == 512
    assert layer_num_experts(draft, 3) is None
