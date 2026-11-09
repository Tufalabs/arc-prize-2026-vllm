# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each pipeline stage builds only the vocabulary tables and the vision tower it
runs: the embedding and the tower on the first stage, the LM head on the last.
The decoder layers, the tables and the tower are stand-ins here; what the test
pins is which stage builds which of them."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.models.qwen4_exp.test_config import _text_config
from vllm.config import CompilationMode
from vllm.model_executor.models.utils import PPMissingLayer, StageMissingLayer
from vllm.models.qwen4_exp.config import Qwen4ExpConfig
from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model


class _Weights(nn.Module):
    """A module holding one parameter, so where it was built can be read."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))


def _build(is_first_rank: bool, is_last_rank: bool, use_v2_model_runner: bool = True):
    config = Qwen4ExpConfig(
        text_config=_text_config(ple_layer_ids=[]),
        vision_config={"deepstack_visual_indexes": []},
    )
    multimodal_config = SimpleNamespace(
        language_model_only=False,
        mm_encoder_only=False,
        mm_encoder_tp_mode="weights",
        video_pruning_rate=0.0,
        get_video_pruning_spec=lambda: None,
        is_multimodal_pruning_enabled=lambda: False,
        get_limit_per_prompt=lambda modality: 1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            hf_text_config=config.text_config,
            multimodal_config=multimodal_config,
            get_multimodal_config=lambda: multimodal_config,
            dtype=torch.bfloat16,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        speculative_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        # The input embedding stays on the card unless named here.
        offload_config=SimpleNamespace(uva=SimpleNamespace(cpu_offload_params=set())),
        use_v2_model_runner=use_v2_model_runner,
    )
    stage = SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank)
    with (
        patch.object(qwen4_exp_model, "get_pp_group", return_value=stage),
        patch.object(
            qwen4_exp_model,
            "make_layers",
            return_value=(0, 0, nn.ModuleList()),
        ),
        patch.object(qwen4_exp_model, "VocabParallelEmbedding", _Weights),
        patch.object(qwen4_exp_model, "ParallelLMHead", _Weights),
        patch.object(qwen4_exp_model, "GatedResidual", _Weights),
        patch.object(qwen4_exp_model, "LogitsProcessor", _Weights),
        patch.object(qwen4_exp_model, "Qwen3_VisionTransformer", _Weights),
        patch.object(qwen4_exp_model, "cached_tokenizer_from_config"),
        patch.object(qwen4_exp_model, "enable_qwen4_exp_low_latency_gemm"),
        patch.object(qwen4_exp_model.Qwen4ExpMixtureOfExperts, "set_moe_parameters"),
    ):
        return qwen4_exp_model.Qwen4ExpForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize(
    "is_first_rank, is_last_rank",
    [(True, True), (True, False), (False, False), (False, True)],
)
def test_a_stage_builds_only_the_tables_and_tower_it_runs(
    is_first_rank: bool, is_last_rank: bool
) -> None:
    model = _build(is_first_rank, is_last_rank)
    language_model = model.language_model

    assert isinstance(language_model.model.embed_tokens, _Weights) == is_first_rank
    assert isinstance(language_model.lm_head, _Weights) == is_last_rank
    if not is_first_rank:
        assert isinstance(language_model.model.embed_tokens, PPMissingLayer)
    if not is_last_rank:
        assert isinstance(language_model.lm_head, PPMissingLayer)

    if is_first_rank:
        assert isinstance(model.visual, _Weights)
        assert model.visual.weight.device.type == "cpu"
    else:
        # The tower a later stage never runs is a placeholder over meta weights.
        assert isinstance(model.visual, StageMissingLayer)
        assert model.visual.module.weight.device.type == "meta"
        assert not any(
            name.startswith("visual.") for name, _ in model.named_parameters()
        )


def test_a_later_stage_without_the_v2_runner_is_refused() -> None:
    """V1's profile_run calls the encoder on every stage, which the placeholder
    a later stage holds refuses at run time. The refusal lands at build."""
    with pytest.raises(NotImplementedError, match="V2 model runner"):
        _build(is_first_rank=False, is_last_rank=True, use_v2_model_runner=False)
    # The first stage holds a real tower either way.
    assert isinstance(
        _build(is_first_rank=True, is_last_rank=True, use_v2_model_runner=False).visual,
        _Weights,
    )
