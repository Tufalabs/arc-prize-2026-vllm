# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMAlignMetadata,
    RecoverSSMCommitMetadata,
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    commit_recoverssm_step,
    defer_recoverssm_commit,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (
        GDNRecoverSSMCommitContext,
    )


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # XKB_GDN_MIX: in a batch of RecoverSSM verifies and prefill chunks whose
    # verify tokens are the batch's first ones, their count; else -1.
    xkb_spec_prefix: int = -1


# XKB_GDN_MIX=1 or 2 sets xkb_spec_prefix for qwen_gdn_linear_attn's sliced mixed path.
XKB_GDN_MIX = int(os.environ.get("XKB_GDN_MIX", "0") or 0)
# Mixed RecoverSSM builds with the switch on: verify tokens leading, or not.
XKB_GDN_MIX_BUILDS = {"prefix": 0, "fallback": 0}


@dataclass
class GDNRecoverSSMAttentionMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    """GDN metadata under RecoverSSM: the step's commit rides along."""

    recoverssm_commit: RecoverSSMCommitMetadata | None = None
    recoverssm_context: "GDNRecoverSSMCommitContext | None" = field(
        default=None, repr=False, compare=False
    )
    # The group's deferred state commits (GDN_FUSED_COMMIT), or None.
    recoverssm_fused: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        return commit_recoverssm_step(
            self.recoverssm_commit,
            self.recoverssm_context,
            self.num_spec_decodes,
            num_accepted_tokens,
            fused=self.recoverssm_fused,
        )

    def deferred_commit(self) -> RecoverSSMMetadata | None:
        return defer_recoverssm_commit(
            self.recoverssm_commit, self.recoverssm_context, self.num_spec_decodes
        )


def get_recoverssm_commit_context(
    vllm_config: VllmConfig, layer_names: list[str]
) -> "GDNRecoverSSMCommitContext":
    """Build a group's RecoverSSM commit context from its bound layers."""
    from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (
        GDNRecoverSSMCommitContext,
    )

    forward_context = vllm_config.compilation_config.static_forward_context
    return GDNRecoverSSMCommitContext.create(
        [forward_context[layer_name] for layer_name in layer_names],
        spec_query_len=1 + vllm_config.num_speculative_tokens,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
    )


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1
    # Set by the model state before each build: the batch's decode-tail
    # targets for this group (RecoverSSMAlignMetadata), or None.
    decode_tail_boundaries: torch.Tensor | None = None
    decode_tail_blocks: torch.Tensor | None = None

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # RecoverSSM verifies off the checkpoint slot alone and commits the
        # accepted state into it, so the kernels read it with an accepted
        # count of one.
        self.use_recoverssm: bool = vllm_config.cache_config.use_gdn_recoverssm
        self.spec_state_slots = 1 if self.use_recoverssm else self.num_spec + 1
        self.recoverssm_num_accepted_tokens: torch.Tensor | None = None
        self.recoverssm_context: GDNRecoverSSMCommitContext | None = None
        if self.use_recoverssm:
            self.recoverssm_num_accepted_tokens = torch.ones(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )

        # GDN_FUSED_COMMIT: the previous step's state commit folded into the
        # next step's verify (ops/gdn_recoverssm.py, GDNFusedCommit).
        self.fused_commit: Any = None
        if self.use_recoverssm:
            from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (
                GDNFusedCommit,
                fused_commit_enabled,
            )

            if fused_commit_enabled():
                if kv_cache_spec.mamba_cache_mode != "align":
                    raise ValueError("GDN_FUSED_COMMIT requires mamba_cache_mode align")
                if vllm_config.parallel_config.pipeline_parallel_size != 1:
                    raise ValueError("GDN_FUSED_COMMIT requires one pipeline stage")
                if len(kv_cache_spec.shapes) != 5:
                    raise ValueError("GDN_FUSED_COMMIT needs the RecoverSSM records")
                if kv_cache_spec.dtypes[1] not in (torch.bfloat16, torch.float32):
                    raise ValueError(
                        "GDN_FUSED_COMMIT requires a bfloat16 or float32 state"
                    )
                max_num_seqs = vllm_config.scheduler_config.max_num_seqs
                self.fused_commit = GDNFusedCommit(
                    max_num_reqs=max_num_seqs,
                    max_rows=max(self.decode_cudagraph_max_bs, max_num_seqs),
                    key_shape=kv_cache_spec.shapes[3],
                    gate_shape=kv_cache_spec.shapes[4],
                    spec_query_len=self.num_spec + 1,
                    device=device,
                )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.spec_state_slots),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if aligned_state_indices is not None:
            block_table_tensor = aligned_state_indices[: m.num_reqs]
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        xkb_spec_prefix = -1
        # Batch rows of the spec requests, for the RecoverSSM commit; None
        # when they are the batch's leading rows.
        spec_request_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            if self.use_recoverssm:
                # Every decode goes through the verify and its commit, a
                # draft-less one included, as in the KDA builder: the commit
                # is what keeps the conv windows and the align column in step.
                # A draft-less decode is one token; KDA's `> 0` would also take
                # the many-token rows of a piecewise-capture dummy batch, which
                # is not marked prefilling.
                assert m.is_prefilling is not None
                assert m.is_prefilling.device.type == "cpu"
                spec_sequence_masks_cpu |= (~m.is_prefilling) & (
                    query_start_loc_cpu.diff() == 1
                )
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0 or (
                not self.use_recoverssm
                and num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            if (
                self.use_recoverssm
                and torch.any(
                    query_lens_cpu[spec_sequence_masks_cpu] > self.num_spec + 1
                ).item()
            ):
                raise ValueError(
                    "GDN RecoverSSM speculative decode query length exceeds "
                    f"its activation capacity ({self.num_spec + 1})"
                )

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.spec_state_slots
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]
                # The batch sort puts the drafted and the draft-less decodes
                # before the prefill chunks, so the verify tokens are usually
                # [0, num_spec_decode_tokens) and the rest follow them up to the
                # last scheduled token; checked on CPU tensors.
                if (
                    XKB_GDN_MIX
                    and self.use_recoverssm
                    and num_spec_decodes > 0
                    and bool(spec_sequence_masks_cpu[:num_spec_decodes].all())
                    and int(query_start_loc_cpu[num_spec_decodes])
                    == num_spec_decode_tokens
                    and int(query_start_loc_cpu[-1])
                    == num_spec_decode_tokens + num_non_spec_tokens
                    and int(query_start_loc_cpu[-1]) <= m.num_actual_tokens
                ):
                    xkb_spec_prefix = num_spec_decode_tokens
                    XKB_GDN_MIX_BUILDS["prefix"] += 1
                elif XKB_GDN_MIX and self.use_recoverssm:
                    XKB_GDN_MIX_BUILDS["fallback"] += 1

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.spec_state_slots
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]
                if self.use_recoverssm:
                    spec_request_indices = async_tensor_h2d(
                        spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
            if self.use_recoverssm:
                assert self.recoverssm_num_accepted_tokens is not None
                num_accepted_tokens = self.recoverssm_num_accepted_tokens[
                    :num_spec_decodes
                ]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        recoverssm_kwargs = {}
        if self.use_recoverssm:
            recoverssm_commit = None
            if num_spec_decodes > 0:
                assert spec_state_indices_tensor is not None
                assert spec_query_start_loc is not None
                align = None
                if self.kv_cache_spec.mamba_cache_mode == "align":
                    align = RecoverSSMAlignMetadata(
                        block_table=m.block_table_tensor,
                        num_computed_tokens=m.compute_num_computed_tokens(),
                        block_size=self.kv_cache_spec.block_size,
                        tail_boundaries=self.decode_tail_boundaries,
                        tail_blocks=self.decode_tail_blocks,
                    )
                recoverssm_commit = RecoverSSMCommitMetadata(
                    state_indices=spec_state_indices_tensor,
                    query_start_loc=spec_query_start_loc,
                    request_indices=spec_request_indices,
                    align=align,
                )
                if self.recoverssm_context is None:
                    self.recoverssm_context = get_recoverssm_commit_context(
                        self.vllm_config, self.layer_names
                    )
                if self.fused_commit is not None:
                    self.fused_commit.context = self.recoverssm_context
            recoverssm_kwargs = dict(
                recoverssm_commit=recoverssm_commit,
                recoverssm_context=(
                    self.recoverssm_context if recoverssm_commit is not None else None
                ),
                recoverssm_fused=(
                    self.fused_commit if recoverssm_commit is not None else None
                ),
            )
        metadata_cls = (
            GDNRecoverSSMAttentionMetadata
            if self.use_recoverssm
            else GDNAttentionMetadata
        )
        attn_metadata = metadata_cls(
            **recoverssm_kwargs,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            xkb_spec_prefix=xkb_spec_prefix,
        )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``, except RecoverSSM's
        commit and context, which name this group's state slots, block table
        and layers and are rebuilt as ``build`` builds them."""
        m = metadata
        group_block_table = blk_table
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            assert self.mamba_aligned_state_indices is not None
            blk_table = self.mamba_aligned_state_indices
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        else:
            spec_indices = blk_table[masks, : self.num_spec + 1]
            if m.num_prefills > 0:
                non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        recoverssm_kwargs = {}
        commit = getattr(m, "recoverssm_commit", None)
        if self.use_recoverssm and commit is not None:
            assert spec_indices is not None
            align = commit.align
            if align is not None:
                align = replace(
                    align,
                    block_table=group_block_table,
                    tail_boundaries=self.decode_tail_boundaries,
                    tail_blocks=self.decode_tail_blocks,
                )
            if self.recoverssm_context is None:
                self.recoverssm_context = get_recoverssm_commit_context(
                    self.vllm_config, self.layer_names
                )
            if self.fused_commit is not None:
                self.fused_commit.context = self.recoverssm_context
            recoverssm_kwargs = dict(
                recoverssm_commit=replace(
                    commit, state_indices=spec_indices, align=align
                ),
                recoverssm_context=self.recoverssm_context,
                recoverssm_fused=self.fused_commit,
            )
        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            **recoverssm_kwargs,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
