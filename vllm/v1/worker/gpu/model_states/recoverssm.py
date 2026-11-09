# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata
from vllm.v1.worker.utils import AttentionGroup


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing."""

    def __init__(self) -> None:
        self._step: tuple[RecoverSSMMetadata, ...] | None = None

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
    ) -> None:
        if for_capture:
            self._step = None
            return

        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        self._step = tuple(step)

    def take_deferred_step(self) -> tuple[RecoverSSMMetadata, ...] | None:
        """The step just recorded, detached for a commit that runs later.

        A pipeline stage before the last commits a step only when the last
        stage's broadcast of its acceptance is consumed, ``pp_size`` steps on
        (``PPHandler``); the step travels with that broadcast's entry, since
        the steps in between record their own and rebuild the buffers this
        one's metadata views. The per-position records the commit replays live
        in each request's own state page, so they wait unharmed.
        """
        step = self._step
        self._step = None
        if step is None:
            return None
        deferred = tuple(
            commit
            for commit in (metadata.deferred_commit() for metadata in step)
            if commit is not None
        )
        return deferred or None

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        deferred_step: tuple[RecoverSSMMetadata, ...] | None = None,
    ) -> None:
        """Commit this stage's step, or ``deferred_step`` where a stage before
        the last consumes a step's acceptance late (``take_deferred_step``)."""
        step: tuple[RecoverSSMMetadata, ...] | None
        if deferred_step is not None:
            step = deferred_step
        else:
            step = self._step
            self._step = None
        if isinstance(num_sampled, int) or step is None:
            return
        if deferred_step is not None:
            # The consume drops rows (-1) whose request was freed since the
            # step or that sampled nothing; their slots may belong to another
            # request by now, so they commit no position.
            num_sampled = torch.where(
                idx_mapping >= 0, num_sampled, torch.zeros_like(num_sampled)
            )

        for metadata in step:
            postprocess_meta = metadata.commit_recoverssm_state(num_sampled)
            if postprocess_meta is None:
                continue
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
            _postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                postprocess_meta.request_indices,
                postprocess_meta.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
                BLOCK_TABLE_WIDTH=postprocess_meta.block_table.shape[1],
            )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    # The column of the last computed token, where the commit put the state
    # (ceil(n / block) - 1, the align pre-copy's own convention).
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(
            tl.maximum(num_computed + num_sampled - 1, 0) // MAMBA_BLOCK_SIZE,
            BLOCK_TABLE_WIDTH - 1,
        ),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)
