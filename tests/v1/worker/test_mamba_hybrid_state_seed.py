# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The align-mode state seed is counted in the mamba group's block size.

On a hybrid the mamba group's block is a multiple of the attention block
the cache config carries. A request resuming over a cached prefix has its
running state column seeded from its computed length, and that column only
means anything in the mamba unit: seeded in the attention unit it lands
past the request's own blocks, and the fused pre-copy reads from there.
"""

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 800


def _state(monkeypatch, *, mamba_spec) -> MambaHybridModelState:
    """A hybrid state with only what ``add_request`` touches, on CPU."""
    monkeypatch.setattr(
        mamba_hybrid.DefaultModelState, "add_request", lambda self, i, r: None
    )
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state._align_mode = True
    state.cache_config = SimpleNamespace(block_size=ATTN_BLOCK_SIZE)
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    state._zero_recurrent_target_gpu = torch.full((4,), -1, dtype=torch.int32)
    state._zero_recurrent = {}
    state._mamba_spec = mamba_spec
    state._tail_tables = {}
    state._fused_commit = False
    return state


def _resume(num_computed_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(req_id="r", num_computed_tokens=num_computed_tokens)


def test_a_resume_is_seeded_in_the_mamba_units(monkeypatch) -> None:
    state = _state(monkeypatch, mamba_spec=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE))
    computed = 3 * MAMBA_BLOCK_SIZE  # a cached prefix, three mamba blocks long
    state.add_request(0, _resume(computed))
    assert state._mamba_state_idx_gpu[0].item() == (computed - 1) // MAMBA_BLOCK_SIZE
    assert state._mamba_state_idx_gpu[0].item() != (computed - 1) // ATTN_BLOCK_SIZE


def test_a_fresh_request_seeds_the_same_in_either_unit(monkeypatch) -> None:
    """Before any batch has run the spec is unset and the attention unit is
    all there is; a fresh request has nothing computed, so the seed agrees."""
    for spec in (None, SimpleNamespace(block_size=MAMBA_BLOCK_SIZE)):
        state = _state(monkeypatch, mamba_spec=spec)
        state.add_request(0, _resume(0))
        assert state._mamba_state_idx_gpu[0].item() == -1
