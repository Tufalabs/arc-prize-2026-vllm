# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_decode_tail_targets_reach_each_groups_builders() -> None:
    """Each recurrent group keeps its own decode-tail targets per request
    slot, and its builders get them per batch row: a group with no target for
    a request passes none for it, and a cleared target passes none."""
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state.max_num_reqs = 4
    state.device = torch.device("cuda")
    state.recoverssm = RecoverSSMState()
    state._tail_tables = {}
    slots = {"a": 3, "b": 0}
    state.set_decode_tail_targets(
        {
            "a": {1: (72, 10), 2: (72, 11)},
            "b": {1: (136, 12)},
            "finished": {1: (8, 13)},
        },
        slots,
    )
    builders = {
        group_id: SimpleNamespace(decode_tail_boundaries=None, decode_tail_blocks=None)
        for group_id in (1, 2, 3)
    }
    attn_groups: list[list[SimpleNamespace]] = [[]] + [
        [SimpleNamespace(get_metadata_builder=lambda _, b=builders[g]: b)]
        for g in (1, 2, 3)
    ]
    # Batch rows: b, a, and a padding row.
    input_batch = SimpleNamespace(
        num_reqs=2, idx_mapping=torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    )

    def handed(for_capture: bool = False) -> dict:
        state._set_decode_tail_builders(
            input_batch, 3, [1, 2, 3], attn_groups, for_capture=for_capture
        )
        return {
            group_id: None
            if b.decode_tail_boundaries is None
            else (b.decode_tail_boundaries.tolist(), b.decode_tail_blocks.tolist())
            for group_id, b in builders.items()
        }

    null = NULL_BLOCK_ID
    assert handed() == {
        1: ([136, 72, 0], [12, 10, null]),
        2: ([0, 72, 0], [null, 11, null]),
        3: None,
    }
    state.set_decode_tail_targets({"a": {1: (0, 0)}}, slots)
    assert handed()[1] == ([136, 0, 0], [12, null, null])
    assert handed(for_capture=True) == {1: None, 2: None, 3: None}
