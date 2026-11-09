# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

import os
from collections.abc import Iterable
from typing import Any, cast

import regex as re
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from ..common.qsa_cache import QSAForwardMetadata
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
)
from .ops.qsa import qsa_sparse_paged_attention
from .qsa import Qwen4ExpQSAAttention, Qwen4ExpQSAFlashAttentionImpl


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    _apply_mtp_num_experts(draft_vllm_config.model_config.hf_text_config)
    return draft_vllm_config


def _apply_mtp_num_experts(text_config: Any) -> None:
    """Let the MTP block keep an expert count of its own.

    A checkpoint whose decoder experts were pruned but whose MTP block was
    left whole declares the block's count as ``mtp_num_experts``; the draft
    config is a fresh parse of the same file, so the override reaches the
    draft's layers and nothing of the target's.
    """
    mtp_num_experts = getattr(text_config, "mtp_num_experts", None)
    if mtp_num_experts:
        text_config.num_experts = int(mtp_num_experts)
    # The decoder's per-layer counts are not the MTP block's.
    if getattr(text_config, "num_experts_per_layer", None) is not None:
        text_config.num_experts_per_layer = None


def _install_draft_prune(
    mlp_hc: GatedResidual,
    keep_rows: list[torch.Tensor | None],
    attn_pruned: bool = False,
) -> None:
    """XKB_DRAFT_PRUNE: wrap this HC module's combine_and_mix so that, while
    keep_rows[0] is set, its three inputs are first reduced to those rows.
    With attn_pruned the attention output (prev_block_output) already has
    only those rows, in keep_rows order (_install_attn_prune)."""
    inner = mlp_hc.combine_and_mix

    def combine_and_mix(hidden_states, prev_block_output, prev_injection):
        rows = keep_rows[0]
        if rows is not None:
            hidden_states = hidden_states.index_select(0, rows)
            if not attn_pruned:
                prev_block_output = prev_block_output.index_select(0, rows)
            if prev_injection is not None:
                prev_injection = prev_injection.index_select(0, rows)
        return inner(hidden_states, prev_block_output, prev_injection)

    mlp_hc.combine_and_mix = combine_and_mix  # type: ignore[method-assign]
    mlp_hc._xkb_prune = True  # type: ignore[attr-defined]


@eager_break_during_capture
def _run_qsa_kept(
    attn: Qwen4ExpQSAAttention,
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """XKB_DRAFT_PRUNE=2: Qwen4ExpQSAAttention._run_qsa, except that the
    sparse attention runs for the query rows in ``rows`` only and writes them
    to ``output`` in that order. The indexer (side caches and selections) and
    the K/V cache write still cover every row, so the draft caches and the
    step-0 selections that later draft steps reuse are those of the stock
    path. The kept rows use the decode kernel configuration (one query row
    per request); an eager break like _run_qsa, so ``rows`` is an argument."""
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        output.zero_()
        return
    main_metadata = cast(FlashAttentionMetadata, metadata[attn.layer_name])
    if attn.kv_cache.numel() == 0:
        raise RuntimeError("QSA main K/V cache is not bound")
    num_tokens = main_metadata.num_actual_tokens
    side_metadata = cast(
        QSAForwardMetadata, metadata[attn.indexer.raw_key_cache.prefix]
    )
    if side_metadata.num_actual_tokens != num_tokens:
        raise RuntimeError("QSA main and side metadata token counts disagree")
    attn.indexer(projected_qk, positions, attn.topk_indices_buffer[:num_tokens])
    impl = cast(Qwen4ExpQSAFlashAttentionImpl, attn.impl)
    impl.do_kv_cache_update(attn, key, value, attn.kv_cache, main_metadata.slot_mapping)
    output.zero_()
    if num_tokens == 0:
        return
    views = impl.cache_views(attn.kv_cache)
    qsa_sparse_paged_attention(
        query.index_select(0, rows),
        views.key,
        views.value,
        attn.topk_indices_buffer.index_select(0, rows),
        main_metadata.block_table,
        side_metadata.token_to_req.index_select(0, rows),
        False,
        output,
        output_gate=output_gate.index_select(0, rows),
        k_scale_cache=views.key_scale,
        v_scale_cache=views.value_scale,
    )


def _install_attn_prune(attn: nn.Module, keep_rows: list[torch.Tensor | None]) -> None:
    """XKB_DRAFT_PRUNE=2: while keep_rows[0] is set, this QSA attention
    computes Q/K/V, the indexer and the cache writes for every row, and the
    sparse attention and o_proj for the kept rows only; it returns those rows
    in keep_rows order."""
    inner = attn.forward

    def forward(positions: torch.Tensor, hidden_states: torch.Tensor):
        rows = keep_rows[0]
        if rows is None:
            return inner(positions=positions, hidden_states=hidden_states)
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v, gate = attn._project_qkv_gate(qkv, positions)
        assert gate is not None
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, attn.num_heads, attn.head_dim)
        key = k.view(num_tokens, attn.num_kv_heads, attn.head_dim)
        value = v.view(num_tokens, attn.num_kv_heads, attn.head_dim)
        attn_output = query.new_empty(rows.numel(), attn.num_heads, attn.head_dim)
        projected_qk, _ = attn.indexer.index_qk_proj(hidden_states)
        _run_qsa_kept(
            attn, projected_qk, positions, query, key, value, attn_output, gate, rows
        )
        output, _ = attn.o_proj(attn_output.view(rows.numel(), -1))
        return output

    attn.forward = forward  # type: ignore[method-assign]


class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        # The draft's own config from here on: its expert count may differ
        # from the target's (see _apply_mtp_num_experts), and load_weights
        # sizes the shared-expert fusion by it.
        config = draft_vllm_config.model_config.hf_text_config
        self.config = config
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )
        # XKB_DRAFT_PRUNE=1: forward(keep_rows=...) runs everything after the
        # attention (MLP HC combine and mix, MoE, final mixer) on keep_rows
        # only. The speculator passes each request's last-token row: the
        # other rows' outputs are discarded, while attention still runs on
        # every row so the draft KV cache is written for all positions.
        # XKB_DRAFT_PRUNE=2 also runs the QSA sparse attention and o_proj on
        # those rows only (_install_attn_prune).
        self._xkb_keep_rows: list[torch.Tensor | None] = [None]
        prune_level = int(os.environ.get("XKB_DRAFT_PRUNE", "0") or "0")
        self._xkb_prune_level = prune_level
        for layer in self.layers if prune_level >= 1 else ():
            attn = getattr(layer, "self_attn", None)
            attn_pruned = prune_level >= 2 and isinstance(attn, Qwen4ExpQSAAttention)
            if attn_pruned:
                assert isinstance(attn, Qwen4ExpQSAAttention)
                _install_attn_prune(attn, self._xkb_keep_rows)
            _install_draft_prune(
                cast(GatedResidual, layer.mlp_hyper_connection),
                self._xkb_keep_rows,
                attn_pruned,
            )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        keep_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size
        prev_block_output: torch.Tensor | None = None

        # Branch on the inputs, not the rank: the drafter is built entirely on
        # the last PP stage, where `is_first_rank` is False.
        if intermediate_tensors is None:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            prev_block_output = inputs_embeds
        else:
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        if keep_rows is not None and get_pp_group().is_last_rank:
            # XKB_DRAFT_PRUNE: the wrapped mlp_hyper_connection.combine_and_mix
            # (_install_draft_prune) keeps these rows; outputs have
            # keep_rows.numel() rows, in keep_rows order.
            assert getattr(layer.mlp_hyper_connection, "_xkb_prune", False)
            self._xkb_keep_rows[0] = keep_rows
        try:
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=prev_block_output,
                prev_injection=None,
                positions=positions,
                input_ids=None,
                query_start_loc=None,
                ngram_context=None,
            )
        finally:
            self._xkb_keep_rows[0] = None
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        # A reduced draft vocabulary, as EAGLE-3's draft_vocab_size and d2t: the
        # head's own LM head over its ids (so it is not swapped for the
        # target's), its logits scattered into the target vocabulary, where
        # the draft is sampled and verified. Read from the head's own config:
        # this model is built with the target's.
        assert vllm_config.speculative_config is not None
        draft_text_config = (
            vllm_config.speculative_config.draft_model_config.hf_text_config
        )
        draft_vocab_size = getattr(draft_text_config, "draft_vocab_size", None)
        if draft_vocab_size == config.vocab_size:
            draft_vocab_size = None
        self.draft_vocab_size = draft_vocab_size
        self.has_own_lm_head = draft_vocab_size is not None
        head_vocab_size = draft_vocab_size or config.vocab_size

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings and draft_vocab_size is None:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    head_vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(head_vocab_size)
        self.draft_id_to_target_id = (
            nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False
            )
            if draft_vocab_size is not None
            else None
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        keep_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        if keep_rows is not None:
            return self.model(
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=spec_step_idx,
                keep_rows=keep_rows,
            )
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    @property
    def supports_draft_prune(self) -> bool:
        """XKB_DRAFT_PRUNE was on when the drafter was built."""
        return all(
            getattr(layer.mlp_hyper_connection, "_xkb_prune", False)
            for layer in self.model.layers
        )

    @property
    def draft_prune_level(self) -> int:
        """XKB_DRAFT_PRUNE level the drafter was built with."""
        return self.model._xkb_prune_level

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None or logits is None:
            return logits
        base = torch.arange(self.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size), float("-inf")
        )
        logits_new[:, targets] = logits
        return logits_new

    def compute_draft_vocab_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """The head's logits over its own vocabulary, before compute_logits
        scatters them into the target's (XKB_SPARSE_SAMPLER)."""
        return self.logits_processor(self.lm_head, hidden_states)

    def draft_vocab_targets(self) -> torch.Tensor | None:
        """The target token id of each draft vocabulary entry, as compute_logits
        scatters them; None without a reduced draft vocabulary."""
        if self.draft_id_to_target_id is None:
            return None
        base = torch.arange(
            self.draft_vocab_size, device=self.draft_id_to_target_id.device
        )
        return base + self.draft_id_to_target_id

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                if name == "d2t":
                    if self.draft_id_to_target_id is not None:
                        yield "draft_id_to_target_id", weight
                    continue
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)
        if (
            self.draft_id_to_target_id is not None
            and "draft_id_to_target_id" not in loaded
        ):
            raise ValueError(
                f"draft_vocab_size={self.draft_vocab_size} needs the head's d2t "
                "(draft-to-target id offsets) in its checkpoint."
            )
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
