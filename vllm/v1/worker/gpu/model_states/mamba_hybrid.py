# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_group_ids,
    get_mamba_groups,
    preprocess_mamba_align_fused_kernel,
    validate_mamba_state_copy_funcs,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _spec_decode_rows(input_batch: InputBatch) -> np.ndarray:
    """The batch rows the GDN builder takes as RecoverSSM spec decodes:
    a decode with drafts, or a one-token decode (GDNAttentionMetadataBuilder.
    build, from the inputs prepare_attn hands it)."""
    num_reqs = input_batch.num_reqs
    rows = np.zeros(num_reqs, dtype=np.bool_)
    drafts = input_batch.num_draft_tokens_per_req
    if drafts is not None:
        drafts = drafts[:num_reqs]
        rows |= (drafts > 0) & (
            input_batch.num_scheduled_tokens[:num_reqs] == drafts + 1
        )
    query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
    rows |= ~input_batch.is_prefilling_np[:num_reqs] & (query_lens == 1)
    return rows


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                ShortConvAttentionMetadataBuilder,
                PleShortConvAttentionMetadataBuilder,
            ),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        self.recoverssm = (
            RecoverSSMState()
            if self.cache_config.use_kda_recoverssm
            or self.cache_config.use_gdn_recoverssm
            else None
        )
        #: req_id -> absolute blank position, for
        #: SamplingParams.zero_recurrent_state. The caller resolves the
        #: absolute position; nothing here resolves or subtracts.
        #: Kept here rather than threaded through the scheduler: the blank is a
        #: worker-side edit to state the worker already owns, and routing it via
        #: the scheduler would drag in the connector path this feature exists to
        #: reproduce WITHOUT.
        self._zero_recurrent: dict[str, int] = {}
        # GDN_FUSED_COMMIT (ops/gdn_recoverssm.py, GDNFusedCommit): each GDN
        # group's deferred state commits; the request slots (and their
        # requests) of the last step's spec rows, the only ones that can owe
        # one; and the request held in each slot.
        from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (
            fused_commit_enabled,
        )

        self._fused_commit = (
            fused_commit_enabled()
            and self._align_mode
            and self.cache_config.use_gdn_recoverssm
            and vllm_config.num_speculative_tokens > 0
        )
        if self._fused_commit:
            logger.info("GDN_FUSED_COMMIT: on, state commits fold into the next verify")
        elif fused_commit_enabled():
            logger.warning(
                "GDN_FUSED_COMMIT is set but unused: it needs mamba_cache_mode "
                "align, gdn_recoverssm and speculative decoding"
            )
        self._fused_groups: list[Any] | None = None
        self._fused_candidates: dict[int, str] = {}
        self._fused_slot_req: dict[int, str] = {}
        self._fused_req_slot: dict[str, int] = {}
        self._fused_armed = False
        self._fused_steps = 0
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            # zero_recurrent_state, GPU-resident: -1 is "not armed", and the
            # align preprocess kernel writes -1 back to disarm a slot once it
            # has fired (or decided not to). The host writes a target here
            # exactly once per request, when the request is added.
            self._zero_recurrent_target_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._zero_recurrent_blank_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            # Decode-tail targets (SchedulerOutput.decode_tail_targets): per
            # kv cache group that has had one, per request slot, the boundary
            # (0: none) and the snapshot block the group's RecoverSSM commit
            # writes it into. Reset when a slot takes a new request.
            self._tail_tables: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # The state column is counted in the MAMBA group's block size. On a
            # hybrid that differs from the attention block size the cache
            # config carries, and a resume over a cached prefix seeded in the
            # wrong unit points the fused pre-copy at a block the request does
            # not own. The spec is populated by the first batch's preprocess,
            # which every resume follows; a fresh request seeds -1 in either
            # unit.
            mamba_block_size = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )
            # Written for EVERY request, not only armed ones: request slots are
            # reused, and a target left behind by a slot's previous occupant
            # would blank a request that never asked for one.
            self._zero_recurrent_target_gpu[req_index].fill_(
                self._zero_recurrent.get(new_req_data.req_id, -1)
            )
            # Likewise decode-tail targets.
            for boundaries, _ in self._tail_tables.values():
                boundaries[req_index].fill_(0)
        if self._fused_commit:
            self._fused_slot_req[req_index] = new_req_data.req_id
            self._fused_req_slot[new_req_data.req_id] = req_index

    def set_decode_tail_targets(
        self,
        targets: dict[str, dict[int, tuple[int, int]]],
        req_id_to_index: dict[str, int],
    ) -> None:
        if not self._align_mode:
            return
        per_group: dict[int, tuple[list[int], list[int], list[int]]] = {}
        for req_id, groups in targets.items():
            slot = req_id_to_index.get(req_id)
            if slot is None:
                continue  # finished since; its slot is reset when reused
            for group_id, (boundary, block_id) in groups.items():
                slots, boundaries, blocks = per_group.setdefault(group_id, ([], [], []))
                slots.append(slot)
                boundaries.append(boundary)
                blocks.append(block_id if boundary else NULL_BLOCK_ID)
        for group_id, (slots, boundaries, blocks) in per_group.items():
            tables = self._tail_tables.get(group_id)
            if tables is None:
                tables = (
                    torch.zeros(
                        self.max_num_reqs, dtype=torch.int32, device=self.device
                    ),
                    torch.full(
                        (self.max_num_reqs,),
                        NULL_BLOCK_ID,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                )
                self._tail_tables[group_id] = tables
            # Pinned and non-blocking: a copy from a list's pageable memory
            # waits for everything queued on the GPU, which holds the host --
            # and the next step's launches -- on every step whose targets
            # change, nearly every step at a full batch.
            index = async_copy_to_gpu(
                torch.tensor(slots, dtype=torch.int64), device=self.device
            )
            for table, values in zip(tables, (boundaries, blocks)):
                table[index] = async_copy_to_gpu(
                    torch.tensor(values, dtype=torch.int32), device=self.device
                )

    def _set_decode_tail_builders(
        self,
        input_batch: InputBatch,
        num_reqs: int,
        mamba_group_ids: list[int],
        attn_groups: list[list[AttentionGroup]],
        for_capture: bool,
    ) -> None:
        """Hand the batch's decode-tail targets, per batch row, to the GDN and
        PLE builders of each recurrent group, for their RecoverSSM commit; or
        None for a group that has had none (and always under capture)."""
        for group_id in mamba_group_ids:
            boundaries = blocks = None
            tables = self._tail_tables.get(group_id)
            if tables is not None and not for_capture and self.recoverssm is not None:
                boundaries = torch.zeros(
                    num_reqs, dtype=torch.int32, device=self.device
                )
                blocks = torch.full(
                    (num_reqs,), NULL_BLOCK_ID, dtype=torch.int32, device=self.device
                )
                boundaries[: input_batch.num_reqs] = tables[0][input_batch.idx_mapping]
                blocks[: input_batch.num_reqs] = tables[1][input_batch.idx_mapping]
            for group in attn_groups[group_id]:
                builder = group.get_metadata_builder(0)
                if hasattr(builder, "decode_tail_boundaries") and hasattr(
                    builder, "decode_tail_blocks"
                ):
                    builder.decode_tail_boundaries = boundaries
                    builder.decode_tail_blocks = blocks

    def remove_request(self, req_id: str) -> None:
        self.drop_zero_recurrent(req_id)
        if self._fused_commit:
            self._fused_remove(req_id)
        return super().remove_request(req_id)

    def _fused_remove(self, req_id: str) -> None:
        """Commit a removed request's pending tokens now: its blocks may be
        zeroed or written for another request before the next step's flush,
        and a request re-added at once (a streaming input) keeps them."""
        slot = self._fused_req_slot.pop(req_id, None)
        if slot is None:
            return
        if self._fused_slot_req.get(slot) == req_id:
            del self._fused_slot_req[slot]
        if self._fused_candidates.get(slot) == req_id:
            del self._fused_candidates[slot]
            for fused in self._fused_groups or ():
                fused.flush([slot])

    def _fused_flush_unkept(self, input_batch: InputBatch) -> None:
        """Before the align pre-copy: commit the pending tokens of each
        request this step does not verify as a spec row, which reads or moves
        its state, and of every request when a zero_recurrent_state blank is
        armed, which overwrites it."""
        candidates = self._fused_candidates
        self._fused_candidates = {}
        if not candidates or not self._fused_groups:
            return
        keep: set[int] = set()
        if not self._zero_recurrent:
            rows = _spec_decode_rows(input_batch)
            keep = set(
                input_batch.idx_mapping_np[: input_batch.num_reqs][rows].tolist()
            )
        flush = sorted(
            slot
            for slot, req_id in candidates.items()
            if slot not in keep or self._fused_slot_req.get(slot) != req_id
        )
        if flush:
            for fused in self._fused_groups:
                fused.flush(flush)

    def _fused_consume(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        mamba_group_ids: list[int],
    ) -> None:
        """After the build: hand each spec row its request's pending tokens,
        for the verify, and keep the step's spec rows as the next step's
        candidates."""
        groups: list[tuple[Any, Any]] = []
        for group_id in mamba_group_ids:
            for group in attn_groups[group_id]:
                fused = getattr(group.get_metadata_builder(0), "fused_commit", None)
                if fused is not None:
                    groups.append((fused, attn_metadata[group.layer_names[0]]))
        if self._fused_groups is None:
            self._fused_groups = [fused for fused, _ in groups]
        num_reqs = input_batch.num_reqs
        rows = _spec_decode_rows(input_batch)
        for fused, metadata in groups:
            num_spec = metadata.num_spec_decodes
            masks = metadata.spec_sequence_masks_cpu
            builder_rows = (
                np.zeros(num_reqs, dtype=np.bool_)
                if num_spec == 0 or masks is None
                else masks[:num_reqs].numpy()
            )
            if not np.array_equal(builder_rows, rows):
                raise RuntimeError(
                    "GDN_FUSED_COMMIT: the spec rows the model state predicted "
                    "differ from the builder's"
                )
            commit = metadata.recoverssm_commit
            if num_spec == 0 or commit is None:
                continue
            if metadata.recoverssm_fused is not fused:
                raise RuntimeError("GDN_FUSED_COMMIT: metadata of another group")
            fused.consume(
                input_batch.idx_mapping,
                commit.request_indices,
                commit.state_indices[:num_spec, 0],
                num_spec,
            )
        slots = input_batch.idx_mapping_np[:num_reqs][rows].tolist()
        req_ids = [input_batch.req_ids[i] for i in np.flatnonzero(rows).tolist()]
        self._fused_candidates = dict(zip(slots, req_ids))
        self._fused_steps += 1
        if self._fused_steps % 2048 == 0 and self._fused_groups:
            lost = sum(int(fused.mismatches.item()) for fused in self._fused_groups)
            if lost:
                logger.warning(
                    "GDN_FUSED_COMMIT: %d deferred commits found off their block",
                    lost,
                )

    def register_zero_recurrent(self, req_id: str, target: int) -> None:
        """Arm `zero_recurrent_state` for one request (see SamplingParams).

        `target` is the ABSOLUTE prompt position, the only spelling the
        sampling parameters accept. The blank is applied at the step
        whose running state stands exactly at the target: the first step for
        a request resuming from a cache hit at that position, or the
        matching prefill step for one arriving cold -- either way the same
        position, so cached and cold service of the same request produce
        the same state. In align mode prefill chunks stop on recurrent
        block boundaries, so a block-aligned target is hit exactly (the
        scheduler also cuts a chunk that would stride across the target).

        Arming is all the host does. Whether the step that arrives is AT the
        target (blank), past it (inherit the state the cache already holds
        there), or short of it (wait) is decided on the device, inside
        ``preprocess_mamba_align_fused_kernel`` -- so nothing here syncs, and
        nothing here can report what happened. A blank that never lands looks
        exactly like a blank that landed and did nothing, so the check that
        this works is external: the offline replay must show the logprob
        collapse AND the sweep shape (minimum at the requested position,
        degrading either side), not a single matching number.
        """
        if not self._align_mode:
            raise ValueError(
                "zero_recurrent_state requires mamba_cache_mode='align'; this "
                "engine is running a different recurrent cache mode."
            )
        self._zero_recurrent[req_id] = target

    def drop_zero_recurrent(self, req_id: str) -> None:
        self._zero_recurrent.pop(req_id, None)

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_spec = next(iter(mamba_groups))
            assert all(
                spec.block_size == mamba_spec.block_size
                and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
                and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
                for spec in mamba_groups
            ), "all mamba groups must share cache scheduling parameters"
            self._mamba_group_ids = get_mamba_group_ids(mamba_groups)
            self._mamba_spec = mamba_spec
        return self._mamba_group_ids, self._mamba_spec

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        if self._mamba_state_copy_funcs is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_types = {spec.mamba_type for spec in mamba_groups}
            copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
            validate_mamba_state_copy_funcs(mamba_groups, copy_funcs, self.cache_config)
            self._mamba_state_copy_funcs = copy_funcs
        copy_funcs = self._mamba_state_copy_funcs
        if self._mamba_ctx is None:
            # Both SD and DS conv layouts support a >0 spec-decode shift: the
            # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
            # ``token_bias = num_accepted - 1`` window shift per conv layout
            # (SD: contiguous slice; DS: per-dim-row strided slice), matching
            # the V1 ``get_conv_copy_spec`` semantics.
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                copy_funcs=copy_funcs,
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(
                    n, dtype=dtype, device=self.device
                ),
            )
        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            # block_tables are batch-order slices of the persistent
            # input_block_tables (stable data_ptr), so the metadata is captured
            # once here and reused across steps.
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                copy_funcs,
                [block_tables[gid] for gid in mamba_group_ids],
            )
        return ctx

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
        if self._fused_commit:
            self._fused_flush_unkept(input_batch)
            self._fused_armed = True

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        # Compiled out entirely while nothing is armed, so an engine that
        # never asks for a blank runs the kernel it ran before this feature.
        has_zero_recurrent = bool(self._zero_recurrent)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            self._zero_recurrent_target_gpu,
            self._zero_recurrent_blank_col_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
            HAS_ZERO_RECURRENT=has_zero_recurrent,
        )
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

        # After the pre-copy, so the blank lands on the column the align
        # machinery actually selected rather than one recomputed here -- and
        # overwrites the state the pre-copy just migrated into it.
        if has_zero_recurrent:
            ctx.run_fused_blank(
                num_reqs,
                self._zero_recurrent_blank_col_gpu,
                input_batch.idx_mapping,
            )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # A row is a spec-decode row only when its whole prompt is already
                # computed, i.e. exactly one non-draft (decode) token is scheduled.
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            self._set_decode_tail_builders(
                input_batch, num_reqs, mamba_group_ids, attn_groups, for_capture
            )
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                ctx = self._ensure_align_ctx(
                    kv_cache_config, mamba_group_ids, block_tables
                )
                all_group_indices = ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self._fused_armed and not for_capture:
            self._fused_armed = False
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            self._fused_consume(
                input_batch, attn_metadata, attn_groups, mamba_group_ids
            )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
            )
        return attn_metadata

    def take_deferred_postprocess(self) -> Any:
        if self.recoverssm is None:
            return None
        return self.recoverssm.take_deferred_step()

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
        deferred_postprocess: Any = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            if not isinstance(num_sampled, int):
                # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
                # kernel skips them rather than scattering with a host-side gather.
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    num_sampled,
                    self.num_accepted_tokens_gpu,
                )
            else:
                # Fill with single value.
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    max(num_sampled, 1),
                )

        if self.recoverssm is not None:
            self.recoverssm.commit_step(
                num_sampled,
                idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
                deferred_step=deferred_postprocess,
            )

        if not num_reqs:
            return

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
