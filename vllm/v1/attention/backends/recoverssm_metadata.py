# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass, fields, replace
from typing import Any

import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError

    def deferred_commit(self) -> "RecoverSSMMetadata | None":
        """This step's commit, detached from the builder's per-step buffers.

        A pipeline stage before the last learns a step's acceptance only
        ``pp_size`` steps on, when the last stage's broadcast is consumed; by
        then the builders have rebuilt the buffers this step's metadata views
        for the steps in between. None when the step has nothing to commit.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot defer its RecoverSSM commit, which a "
            "pipeline stage before the last needs"
        )


@dataclass
class RecoverSSMAlignMetadata:
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int
    # Per batch row: each request's decode-tail boundary (0 for none) and this
    # group's snapshot block for it (KVCacheConfig.decode_tail_mode
    # "recover"). None when no request of the batch keeps one.
    tail_boundaries: torch.Tensor | None = None
    tail_blocks: torch.Tensor | None = None


@dataclass
class RecoverSSMCommitMetadata:
    """One group's step as its commit needs it (the GDN and PLE groups; the
    Kimi-K3 KDA builder keeps its own copy of this record)."""

    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    align: RecoverSSMAlignMetadata | None

    def detached(self) -> "RecoverSSMCommitMetadata":
        """A copy owning its tensors: the builders' views are rewritten by the
        next step's metadata. Every tensor field is cloned, the align record's
        included (its decode-tail targets among them)."""
        copy = _clone_tensor_fields(self)
        if copy.align is not None:
            copy.align = _clone_tensor_fields(copy.align)
        return copy


def _clone_tensor_fields(record: Any) -> Any:
    """The dataclass ``record`` with each of its tensor fields cloned."""
    return replace(
        record,
        **{
            field.name: value.clone()
            for field in fields(record)
            if isinstance(value := getattr(record, field.name), torch.Tensor)
        },
    )


@dataclass
class DeferredRecoverSSMCommit(RecoverSSMMetadata):
    """One group's step, detached for a commit that runs later
    (``RecoverSSMMetadata.deferred_commit``)."""

    commit: RecoverSSMCommitMetadata
    context: Any
    num_spec_decodes: int

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        return commit_recoverssm_step(
            self.commit, self.context, self.num_spec_decodes, num_accepted_tokens
        )

    def deferred_commit(self) -> "RecoverSSMMetadata | None":
        return self


def defer_recoverssm_commit(
    commit: RecoverSSMCommitMetadata | None,
    context: Any,
    num_spec_decodes: int,
) -> RecoverSSMMetadata | None:
    """``deferred_commit`` for a group whose step rides a
    ``RecoverSSMCommitMetadata`` (the GDN and PLE groups)."""
    if commit is None:
        return None
    return DeferredRecoverSSMCommit(commit.detached(), context, num_spec_decodes)


def commit_recoverssm_step(
    commit: RecoverSSMCommitMetadata | None,
    context: Any,
    num_spec_decodes: int,
    num_accepted_tokens: torch.Tensor,
    fused: Any = None,
) -> RecoverSSMPostprocessMetadata | None:
    """Run ``context.commit`` for one step, as ``KimiK3KDAMetadata`` does.
    ``fused``: the group's deferred state commits (GDN_FUSED_COMMIT)."""
    if commit is None:
        return None
    assert context is not None
    align = commit.align
    state_indices = commit.state_indices[:num_spec_decodes]
    if state_indices.ndim == 2:
        # GDN keeps a slot column per request; PLE a single slot per request.
        state_indices = state_indices[:, 0]
    context.commit(
        num_accepted_tokens,
        state_indices,
        commit.query_start_loc[: num_spec_decodes + 1],
        request_indices=commit.request_indices,
        block_table=align.block_table if align is not None else None,
        num_computed_tokens=align.num_computed_tokens if align is not None else None,
        mamba_block_size=align.block_size if align is not None else None,
        tail_boundaries=align.tail_boundaries if align is not None else None,
        tail_blocks=align.tail_blocks if align is not None else None,
        **({} if fused is None else {"fused": fused}),
    )
    if align is None:
        return None
    return RecoverSSMPostprocessMetadata(
        num_spec_decodes=num_spec_decodes,
        request_indices=commit.request_indices,
        block_table=align.block_table,
        num_computed_tokens=align.num_computed_tokens,
        block_size=align.block_size,
    )
