# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the B200's kernels select, on any card: the CuteDSL NvFP4 MoE's W4A16
mode behind its variable and for BF16 alone, and the GDN RecoverSSM tiles and
the INT4 QSA decode configs on the 10.x family alone, unless
VLLM_SM100_KERNEL_SHAPES turns them off. The kernels themselves need a GPU;
runs are their evidence."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.mamba.ops import gdn_recoverssm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

Backend = nvfp4_oracle.NvFp4MoeBackend
# Capability families: the B200's, the RTX PRO 6000's, the L4's.
B200, RTX, L4 = 10, 12, 8


def _card(family: int) -> SimpleNamespace:
    """A platform stand-in for a card of capability ``family``.x."""
    return SimpleNamespace(
        is_device_capability_family=lambda capability, device_id=0: (
            capability // 10 == family
        )
    )


@pytest.mark.parametrize("forced", [False, True])
def test_the_cutedsl_moe_takes_bf16_activations_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, forced: bool
) -> None:
    monkeypatch.setattr(nvfp4_oracle.envs, "VLLM_CUTEDSL_MOE_FP4_FORCE_A16", forced)
    assert nvfp4_oracle._use_a16(Backend.FLASHINFER_CUTEDSL, False) == forced
    # The variable is the CuteDSL backend's alone.
    for backend in (
        Backend.FLASHINFER_TRTLLM,
        Backend.FLASHINFER_CUTLASS,
        Backend.FLASHINFER_CUTEDSL_BATCHED,
    ):
        assert not nvfp4_oracle._use_a16(backend, False)


def test_the_w4a16_config_takes_the_weights_global_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BF16 activations quantize nothing, so the alphas are the weights' own
    global scales, with no activation scale folded in."""
    monkeypatch.setattr(nvfp4_oracle.envs, "VLLM_CUTEDSL_MOE_FP4_FORCE_A16", True)
    # The MMA view is FlashInfer's, and checks its own aliasing.
    monkeypatch.setattr(
        nvfp4_oracle, "nvfp4_swizzled_scale_to_cutedsl_mma_view", lambda scale: scale
    )
    experts = 4
    w13_scale_2 = torch.full((experts,), 0.5)
    w2_scale_2 = torch.full((experts,), 0.25)
    config = nvfp4_oracle.make_nvfp4_moe_quant_config(
        Backend.FLASHINFER_CUTEDSL,
        w13_scale=torch.ones(experts, 128, 8),
        w2_scale=torch.ones(experts, 128, 8),
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        a13_scale=torch.full((experts,), 2.0),
        a2_scale=torch.full((experts,), 3.0),
    )
    assert config.quant_dtype is None
    assert config.g1_alphas is w13_scale_2
    assert config.g2_alphas is w2_scale_2


def test_the_int4_decode_configs_are_the_10x_familys_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for family, expected in ((B200, True), (RTX, False), (L4, False)):
        monkeypatch.setattr(qsa_ops, "current_platform", _card(family))
        assert qsa_ops._int4_configs(True) is expected
        assert qsa_ops._int4_configs(False) is False


def test_the_int4_decode_configs_change_only_their_rows() -> None:
    """Over every row count the dispatch meets, the INT4 configs differ from
    the table's in two regions alone: decode above 2,048 base programs (a
    drafted verify) and 257-512 (the drafter's single-row passes), where they
    take 128-column tiles and four warps."""
    heads, columns = 2, 2048
    for num_rows in range(1, 8193):
        base_programs = num_rows * heads
        for prefill in (False, True):
            table = qsa_ops._select_config(num_rows, heads, prefill, columns)
            int4 = qsa_ops._select_config(
                num_rows, heads, prefill, columns, int4_configs=True
            )
            own = (base_programs > 2048 and not prefill) or (256 < base_programs <= 512)
            if own:
                block_n, num_warps = int4[0], int4[1]
                assert (block_n, num_warps) == (128, 4), (num_rows, prefill)
            else:
                assert int4 == table, (num_rows, prefill)


def test_the_gdn_tiles_are_the_10x_familys_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_recoverssm, "current_platform", _card(B200))
    assert gdn_recoverssm._verify_launch() == (
        gdn_recoverssm.SM100_VERIFY_LAUNCH,
        gdn_recoverssm.SM100_VERIFY_TILES,
    )
    assert gdn_recoverssm._commit_tiles() == gdn_recoverssm.SM100_COMMIT_TILES
    for family in (RTX, L4):
        monkeypatch.setattr(gdn_recoverssm, "current_platform", _card(family))
        # One tile a program at the base's launch shapes: the kernels these
        # cards ran before.
        assert gdn_recoverssm._verify_launch() == (gdn_recoverssm.VERIFY_LAUNCH, 1)
        assert gdn_recoverssm._commit_tiles() == 1


def _moe_config(in_dtype: torch.dtype) -> FusedMoEConfig:
    """A Flash-Next MoE layer's config at this activation dtype."""
    return FusedMoEConfig(
        num_experts=307,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=307,
        num_logical_experts=307,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=in_dtype,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


@pytest.mark.parametrize(
    "forced,dtype,expected",
    [
        (False, torch.bfloat16, False),
        (True, torch.bfloat16, True),
        (True, torch.half, False),
    ],
)
def test_the_cutedsl_moe_claims_bf16_activations_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, forced: bool, dtype: torch.dtype, expected: bool
) -> None:
    """A native W4A16 checkpoint (no activation key) reaches the CuteDSL
    kernel only under the variable and at BF16, so ``auto`` keeps falling
    through to Marlin otherwise; the W4A4 scheme is unchanged."""
    monkeypatch.setattr(envs, "VLLM_CUTEDSL_MOE_FP4_FORCE_A16", forced)
    monkeypatch.setattr(
        FlashInferCuteDSLExperts, "_supports_current_device", staticmethod(lambda: True)
    )
    experts = FlashInferCuteDSLExperts
    standard = mk.FusedMoEActivationFormat.Standard
    supported, _ = experts.is_supported_config(
        experts, _moe_config(dtype), kNvfp4Static, None, standard
    )
    assert supported is expected
    w4a4, _ = experts.is_supported_config(
        experts, _moe_config(dtype), kNvfp4Static, kNvfp4Dynamic, standard
    )
    assert w4a4


def test_the_sm100_shapes_turn_off_on_the_10x_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM_SM100_KERNEL_SHAPES=0 puts a B200 on the shapes every other card
    runs: one GDN tile a program at the base launch, the table's QSA configs."""
    monkeypatch.setattr(envs, "VLLM_SM100_KERNEL_SHAPES", False)
    monkeypatch.setattr(gdn_recoverssm, "current_platform", _card(B200))
    monkeypatch.setattr(qsa_ops, "current_platform", _card(B200))
    assert gdn_recoverssm._verify_launch() == (gdn_recoverssm.VERIFY_LAUNCH, 1)
    assert gdn_recoverssm._commit_tiles() == 1
    assert qsa_ops._int4_configs(True) is False


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sm120_gdn_image_is_never_loaded_on_another_architecture(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """An enabled optimization must retain the Triton path on B200 and CPU."""
    from vllm.model_executor.layers.mamba.ops import xkb_xgdn

    monkeypatch.setattr(xkb_xgdn, "ENABLED", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    calls = []

    def baseline(*args, **kwargs):
        calls.append((args, kwargs))
        return "baseline"

    def forbidden(*args, **kwargs):
        raise AssertionError("attempted to load an SM120 image")

    monkeypatch.setattr(xkb_xgdn, "_module", forbidden)
    tensor = SimpleNamespace(device=torch.device(device))
    result = xkb_xgdn.launcher(baseline, 4)(tensor, *([None] * 18), K=128)
    assert result == "baseline"
    assert len(calls) == 1
