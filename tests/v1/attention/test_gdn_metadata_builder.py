# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # No speculative config, decode alongside prefill
    "regular_decode_with_prefill": GDNBuildTestCase(
        seq_lens=[40, 100],
        query_lens=[1, 50],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    recoverssm: bool = False,
    layer_name: str = "layer.0",
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config; with
    ``recoverssm``, in align mode with GDN RecoverSSM on."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if recoverssm:
        vllm_config.cache_config.use_gdn_recoverssm = True
        vllm_config.cache_config.mamba_cache_mode = "align"
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align" if recoverssm else "none",
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    block_table: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if block_table is not None:
        common = common.replace(block_table_tensor=block_table)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_update_block_table_matches_build(
    test_case: GDNBuildTestCase, full_cuda_graph: bool, mamba_cache_mode: str
):
    """update_block_table() on another group's metadata matches build()."""
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    src, dst, ref = (
        _create_gdn_builder(test_case.num_speculative_tokens, full_cuda_graph)
        for _ in range(3)
    )
    for builder in (src, dst, ref):
        builder.vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    if mamba_cache_mode == "align":
        # MRV2 precomputes these for every Mamba group each step.
        dst.mamba_aligned_state_indices = ref.mamba_aligned_state_indices = (
            mamba_get_block_table_tensor(
                common.block_table_tensor, common.seq_lens, ref.kv_cache_spec, "align"
            )
        )
    draft_tokens = test_case.num_decode_draft_tokens
    expected = _build(ref, batch, draft_tokens, common.block_table_tensor)
    source = _build(src, batch, draft_tokens)
    fields = (
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
    )
    source_indices = [getattr(source, f) for f in fields]
    source_indices = [t if t is None else t.clone() for t in source_indices]
    meta = dst.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )

    for field, source_index in zip(fields, source_indices):
        actual = getattr(meta, field)
        torch.testing.assert_close(actual, getattr(expected, field))
        # The source group's indices are untouched.
        torch.testing.assert_close(getattr(source, field), source_index)
        # FULL graph state indices land in this group's own buffers.
        if full_cuda_graph and meta.num_prefills == 0 and actual is not None:
            assert actual.data_ptr() == getattr(dst, field).data_ptr()


@pytest.mark.parametrize(
    "test_case",
    [
        GDN_BUILD_TEST_CASES[k]
        for k in ("pure_spec_decode", "spec_decode_with_real_prefill")
    ],
    ids=["pure_spec_decode", "spec_decode_with_real_prefill"],
)
def test_update_block_table_rebuilds_recoverssm_commit(
    test_case: GDNBuildTestCase, monkeypatch: pytest.MonkeyPatch
):
    """Under RecoverSSM, update_block_table() gives a group its own commit:
    its spec state slots, its block table and decode-tail targets, and the
    context of its own layers, as build() would; the source group's commit
    is untouched."""
    from vllm.v1.attention.backends import gdn_attn

    monkeypatch.setattr(
        gdn_attn,
        "get_recoverssm_commit_context",
        lambda vllm_config, layer_names: ("context", tuple(layer_names)),
    )
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    src = _create_gdn_builder(test_case.num_speculative_tokens, recoverssm=True)
    dst, ref = (
        _create_gdn_builder(
            test_case.num_speculative_tokens, recoverssm=True, layer_name="layer.1"
        )
        for _ in range(2)
    )
    draft_tokens = test_case.num_decode_draft_tokens
    assert draft_tokens is not None
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    is_prefilling = torch.tensor([n < 0 for n in draft_tokens], dtype=torch.bool)
    common = common.replace(is_prefilling=is_prefilling)
    # The destination group's own slots, block table and decode-tail targets.
    dst_table = common.block_table_tensor + 1000
    for builder, table, offset in (
        (src, common.block_table_tensor, 0),
        (dst, dst_table, 7),
        (ref, dst_table, 7),
    ):
        builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
            table, common.seq_lens, builder.kv_cache_spec, "align"
        )
        builder.decode_tail_boundaries = torch.full(
            (batch.batch_size,), 16 + offset, dtype=torch.int32
        )
        builder.decode_tail_blocks = torch.full(
            (batch.batch_size,), 3 + offset, dtype=torch.int32
        )
    kwargs = dict(
        num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        num_accepted_tokens=torch.ones(batch.batch_size, dtype=torch.int32),
    )
    source = src.build(0, common, **kwargs)
    expected = ref.build(0, common.replace(block_table_tensor=dst_table), **kwargs)
    source_slots = source.recoverssm_commit.state_indices.clone()
    meta = dst.update_block_table(source, dst_table, common.slot_mapping)

    commit, want = meta.recoverssm_commit, expected.recoverssm_commit
    torch.testing.assert_close(commit.state_indices, want.state_indices)
    torch.testing.assert_close(commit.query_start_loc, want.query_start_loc)
    if want.request_indices is None:
        assert commit.request_indices is None
    else:
        torch.testing.assert_close(commit.request_indices, want.request_indices)
    torch.testing.assert_close(commit.align.block_table, want.align.block_table)
    torch.testing.assert_close(commit.align.tail_boundaries, want.align.tail_boundaries)
    torch.testing.assert_close(commit.align.tail_blocks, want.align.tail_blocks)
    assert meta.recoverssm_context == ("context", ("layer.1",))
    assert source.recoverssm_context == ("context", ("layer.0",))
    torch.testing.assert_close(source.recoverssm_commit.state_indices, source_slots)
    assert not torch.equal(commit.state_indices, source_slots)


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)
