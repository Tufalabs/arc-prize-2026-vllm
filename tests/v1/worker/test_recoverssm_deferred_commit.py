# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM under pipeline parallelism: a stage before the last commits a
step only when the last stage's acceptance for it arrives, pp_size steps on,
so the step's commit record must survive the steps recorded in between; and
every model state takes the postprocess hook the deferred step rides."""

import ast
import inspect
import pathlib
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import numpy as np
import torch

import vllm
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMAlignMetadata,
    RecoverSSMCommitMetadata,
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    commit_recoverssm_step,
    defer_recoverssm_commit,
)
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


class _Context:
    """Stands in for a group's commit context: records what each commit saw."""

    def __init__(self) -> None:
        self.commits: list[dict[str, list[int]]] = []

    def commit(self, num_accepted_tokens, state_indices, query_start_loc, **_: Any):
        self.commits.append(
            {
                "num_accepted": num_accepted_tokens.tolist(),
                "state_indices": state_indices.tolist(),
                "query_start_loc": query_start_loc.tolist(),
            }
        )


@dataclass
class _GroupMetadata(RecoverSSMMetadata):
    """A group's step metadata as the GDN builder makes it: views into buffers
    the next step's build rewrites."""

    recoverssm_commit: RecoverSSMCommitMetadata | None
    recoverssm_context: Any
    num_spec_decodes: int
    extra: dict = field(default_factory=dict)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        return commit_recoverssm_step(
            self.recoverssm_commit,
            self.recoverssm_context,
            self.num_spec_decodes,
            num_accepted_tokens,
        )

    def deferred_commit(self) -> RecoverSSMMetadata | None:
        return defer_recoverssm_commit(
            self.recoverssm_commit, self.recoverssm_context, self.num_spec_decodes
        )


class _Builder:
    """Persistent buffers, rebuilt in place each step, as the builders do."""

    def __init__(self, context: _Context) -> None:
        self.state_indices = torch.zeros(8, dtype=torch.int32)
        self.query_start_loc = torch.zeros(9, dtype=torch.int32)
        self.context = context

    def build(self, slots: list[int], window: int) -> dict[str, _GroupMetadata]:
        n = len(slots)
        self.state_indices[:n] = torch.tensor(slots, dtype=torch.int32)
        self.query_start_loc[: n + 1] = torch.arange(n + 1, dtype=torch.int32) * window
        metadata = _GroupMetadata(
            recoverssm_commit=RecoverSSMCommitMetadata(
                state_indices=self.state_indices[:n],
                query_start_loc=self.query_start_loc[: n + 1],
                request_indices=None,
                align=None,
            ),
            recoverssm_context=self.context,
            num_spec_decodes=n,
        )
        return {"layer.0": metadata}


def _groups() -> list[list[Any]]:
    return [[Mock(layer_names=["layer.0"])]]


def _record(state: RecoverSSMState, builder: _Builder, slots: list[int]) -> None:
    state.record_step(builder.build(slots, window=4), _groups(), for_capture=False)


def test_deferred_step_commits_its_own_slots_after_later_steps_rebuild():
    """Micro-batch A records, the stage hands A's step to its receive, B
    records over the same buffers, and A's acceptance arriving afterwards
    commits A's slots, not B's."""
    context = _Context()
    builder = _Builder(context)
    state = RecoverSSMState()

    _record(state, builder, slots=[11, 12, 13])  # micro-batch A
    deferred_a = state.take_deferred_step()
    assert deferred_a is not None
    _record(state, builder, slots=[21, 22])  # micro-batch B rewrites the buffers
    deferred_b = state.take_deferred_step()

    idx_a = torch.tensor([0, 1, 2], dtype=torch.int32)
    state.commit_step(
        torch.tensor([2, 4, 1], dtype=torch.int32),
        idx_a,
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
        deferred_step=deferred_a,
    )
    assert context.commits[-1] == {
        "num_accepted": [2, 4, 1],
        "state_indices": [11, 12, 13],
        "query_start_loc": [0, 4, 8, 12],
    }

    state.commit_step(
        torch.tensor([3, 1], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
        deferred_step=deferred_b,
    )
    assert context.commits[-1]["state_indices"] == [21, 22]


def test_rows_dropped_by_the_consume_commit_nothing():
    """A request freed between its step and the step's consume (idx -1) may
    have handed its slot on; its row commits no position."""
    context = _Context()
    builder = _Builder(context)
    state = RecoverSSMState()
    _record(state, builder, slots=[11, 12, 13])
    deferred = state.take_deferred_step()

    state.commit_step(
        torch.tensor([2, 4, 3], dtype=torch.int32),
        torch.tensor([0, -1, 2], dtype=torch.int32),
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
        deferred_step=deferred,
    )
    assert context.commits[-1]["num_accepted"] == [2, 0, 3]


def test_the_last_stage_still_commits_the_step_it_just_recorded():
    context = _Context()
    builder = _Builder(context)
    state = RecoverSSMState()
    _record(state, builder, slots=[5, 6])

    state.commit_step(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
    )
    assert context.commits[-1]["state_indices"] == [5, 6]
    assert state.take_deferred_step() is None


def test_a_step_without_speculative_decodes_defers_nothing():
    state = RecoverSSMState()
    metadata = _GroupMetadata(
        recoverssm_commit=None, recoverssm_context=None, num_spec_decodes=0
    )
    state.record_step({"layer.0": metadata}, _groups(), for_capture=False)
    assert state.take_deferred_step() is None


def test_a_detached_record_owns_every_tensor_the_commit_reads():
    """The align record's decode-tail targets ride the deferred commit too."""
    align = RecoverSSMAlignMetadata(
        block_table=torch.tensor([[3, 4]], dtype=torch.int32),
        num_computed_tokens=torch.tensor([17], dtype=torch.int32),
        block_size=16,
        tail_boundaries=torch.tensor([32], dtype=torch.int32),
        tail_blocks=torch.tensor([9], dtype=torch.int32),
    )
    commit = RecoverSSMCommitMetadata(
        state_indices=torch.tensor([3], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        request_indices=None,
        align=align,
    )
    detached = commit.detached()
    assert detached.align is not None
    for name in (
        "block_table",
        "num_computed_tokens",
        "tail_boundaries",
        "tail_blocks",
    ):
        original, copy = getattr(align, name), getattr(detached.align, name)
        assert torch.equal(original, copy)
        assert copy.data_ptr() != original.data_ptr(), name
    assert detached.state_indices.data_ptr() != commit.state_indices.data_ptr()


def test_the_deferred_postprocess_rides_the_pp_entry_to_its_consume():
    """PPHandler hands back what the step deferred with its sampled outputs."""
    handler = object.__new__(pp_utils.PPHandler)
    handler.main_stream = Mock()
    handler.device = torch.device("cpu")
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    payload = object()
    handler.queue = pp_utils.deque(
        [
            pp_utils.PendingRecv(
                event=Mock(),
                sampled_tokens=torch.zeros(2, 4, dtype=torch.int64),
                num_sampled=torch.tensor([2, 1], dtype=torch.int32),
                num_rejected=torch.tensor([2, 3], dtype=torch.int32),
                idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
                idx_mapping_np=np.array([0, 1], dtype=np.int32),
                need_sampled_mask=np.array([True, True]),
                gen_at_receive_np=np.zeros(2, dtype=np.int32),
                deferred_postprocess=payload,
            )
        ]
    )
    outputs = handler.get_prev_sampled_outputs()
    assert outputs is not None
    assert outputs["deferred_postprocess"] is payload


def test_the_consume_drops_the_drafts_of_rows_freed_since_the_receive(monkeypatch):
    """A row whose request slot was freed since the receive leaves that slot's
    drafts alone; the kept rows' drafts land at their slots."""
    monkeypatch.setattr(
        pp_utils,
        "async_copy_to_gpu",
        lambda x, out=None, device=None: torch.as_tensor(x).clone(),
    )
    handler = object.__new__(pp_utils.PPHandler)
    handler.main_stream = Mock()
    handler.device = torch.device("cpu")
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    handler.queue = pp_utils.deque(
        [
            pp_utils.PendingRecv(
                event=Mock(),
                sampled_tokens=torch.zeros(3, 4, dtype=torch.int64),
                num_sampled=torch.tensor([1, 1, 1], dtype=torch.int32),
                num_rejected=torch.tensor([0, 0, 0], dtype=torch.int32),
                idx_mapping=torch.tensor([1, 2, 3], dtype=torch.int32),
                idx_mapping_np=np.array([1, 2, 3], dtype=np.int32),
                need_sampled_mask=np.array([True, True, True]),
                gen_at_receive_np=np.zeros(3, dtype=np.int32),
                draft_tokens=torch.tensor([[11, 12], [21, 22], [31, 32]]),
            )
        ]
    )
    handler.req_idx_gen_np[2] += 1  # slot 2 freed since the receive
    drafts = torch.zeros(4, 2, dtype=torch.int64)
    outputs = handler.get_prev_sampled_outputs(drafts)
    assert outputs is not None
    assert drafts.tolist() == [[0, 0], [11, 12], [0, 0], [31, 32]]
    assert outputs["idx_mapping"].tolist() == [1, -1, 3]


def test_every_model_state_override_takes_the_postprocess_hook_keywords():
    """The runner calls ``postprocess_state`` with the interface's keywords on
    every sampled step, ``deferred_postprocess`` included, so a model state
    whose override drops one fails at its first step. Read from the source,
    so a model state no test imports is covered too."""
    wanted = [
        name
        for name in inspect.signature(ModelState.postprocess_state).parameters
        if name != "self"
    ]
    root = pathlib.Path(vllm.__file__).parent
    lacking: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        source = path.read_text()
        if "def postprocess_state" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef) or not any(
                "ModelState" in ast.unparse(base) for base in node.bases
            ):
                continue
            for method in node.body:
                if (
                    not isinstance(method, ast.FunctionDef)
                    or method.name != "postprocess_state"
                    or method.args.kwarg is not None
                ):
                    continue
                args = method.args
                params = (*args.posonlyargs, *args.args, *args.kwonlyargs)
                names = {param.arg for param in params}
                missing = [name for name in wanted if name not in names]
                if missing:
                    lacking[f"{path.relative_to(root)}::{node.name}"] = missing
    assert not lacking
