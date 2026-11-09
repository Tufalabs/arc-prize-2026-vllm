# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cute_dsl_fused_moe_nvfp4,
    has_flashinfer_cutedsl_moe_nvfp4,
)


class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    """
    CuteDSL NvFP4 MoE experts using the FlashInfer functional API.

    Uses Standard activation format (non-batched). The kernel handles
    routing, expert computation, and reduction internally.
    Supports expert parallelism natively.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        # BF16 activations against the NvFP4 weights (W4A16) when the quant
        # config quantizes no activation (VLLM_CUTEDSL_MOE_FP4_FORCE_A16).
        self.use_a16 = quant_config.quant_dtype is None
        assert quant_config.quant_dtype == "nvfp4" or (
            self.use_a16 and quant_config.weight_quant_dtype == "nvfp4"
        ), "Only nvfp4 quantization is currently supported."
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts
        self.gemm1_alpha = quant_config.gemm1_alpha
        self.gemm1_beta = quant_config.gemm1_beta
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self._w4a16 = self._make_w4a16_wrapper(moe_config) if self.use_a16 else None

    def _make_w4a16_wrapper(self, moe_config: FusedMoEConfig):
        """FlashInfer's CuteDSL MoE in its W4A16 mode, one per layer: its
        kernels are compiled for the layer's expert count."""
        from flashinfer.fused_moe.cute_dsl import CuteDslMoEWrapper

        activation = moe_config.activation
        swiglu_params: dict[str, float] = {}
        if activation == MoEActivation.SILU:
            if self.gemm1_clamp_limit is not None:
                swiglu_params = {"swiglu_limit": self.gemm1_clamp_limit}
        elif activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            swiglu_params = {
                k: v
                for k, v in {
                    "swiglu_alpha": self.gemm1_alpha,
                    "swiglu_beta": self.gemm1_beta,
                    "swiglu_limit": self.gemm1_clamp_limit,
                }.items()
                if v is not None
            }
        return CuteDslMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            output_dtype=self.out_dtype,
            activation_type=activation_to_flashinfer_int(activation),
            quant_mode="w4a16",
            use_fused_finalize=envs.VLLM_CUTEDSL_MOE_W4A16_FUSED_FINALIZE,
            **swiglu_params,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_a16:
            # The alphas stay the weights' global scales: no activation is
            # quantized, so there is no input scale to fold in.
            return
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe_nvfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A: list[tuple[QuantKey, QuantKey | None]] = [
            (kNvfp4Static, kNvfp4Dynamic)
        ]
        # BF16 activations only when asked for (VLLM_CUTEDSL_MOE_FP4_FORCE_A16):
        # without the variable a native W4A16 checkpoint keeps the backends it
        # reached before, Marlin among them, which serves FP16 too.
        if envs.VLLM_CUTEDSL_MOE_FP4_FORCE_A16:
            SUPPORTED_W_A.append((kNvfp4Static, None))
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        # FlashInfer's W4A16 mode writes BF16 output alone.
        if (
            is_supported
            and activation_key is None
            and moe_config.in_dtype != torch.bfloat16
        ):
            return False, (
                "kernel's W4A16 mode serves BF16 output alone, got "
                f"{moe_config.in_dtype}"
            )
        return is_supported, reason

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        # K is packed (K//2 for uint8) under W4A4 and BF16 under W4A16, so
        # output uses hidden_dim.
        assert self.hidden_dim == (K if self.use_a16 else K * 2)
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        if self._w4a16 is not None:
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            output.copy_(
                self._w4a16.run(
                    hidden_states,
                    None,
                    topk_ids.to(torch.int32),
                    topk_weights.float(),
                    w1,
                    self.w1_scale,
                    self.g1_alphas,
                    None,
                    w2,
                    self.w2_scale,
                    self.g2_alphas,
                )
            )
            return
        assert self.quant_dtype == "nvfp4"
        assert a1q_scale is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        # a1q_scale is (M, K//16) float8_e4m3fn from fp4_quantize.
        # The functional API expects x_sf with trailing dim: (M, K//16, 1).
        x_sf = a1q_scale.unsqueeze(-1)

        # The kernel defaults swiglu_{alpha,beta,limit} to the plain-SwiGLU
        # values, so only forward the ones the model actually sets.
        swiglu_params: dict[str, float | None] = {}
        if activation == MoEActivation.SILU:
            swiglu_params = {"swiglu_limit": self.gemm1_clamp_limit}
        elif activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            swiglu_params = {
                "swiglu_alpha": self.gemm1_alpha,
                "swiglu_beta": self.gemm1_beta,
                "swiglu_limit": self.gemm1_clamp_limit,
            }
        swiglu_kwargs = {k: v for k, v in swiglu_params.items() if v is not None}

        flashinfer_cute_dsl_fused_moe_nvfp4(
            x=hidden_states,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.float(),
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self.a2_gscale,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self.g2_alphas,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=activation_to_flashinfer_int(activation),
            **swiglu_kwargs,
        )
