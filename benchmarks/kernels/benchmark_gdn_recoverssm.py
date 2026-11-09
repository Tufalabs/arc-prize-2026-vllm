# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM kernels at Qwen3.8-Flash-Next's shapes, per decode step.

36 GDN layers (three groups of twelve), 48 value heads over 16 key heads,
K = V = 128; the requests (``BENCH_REQUESTS``, default 41), the window
(``BENCH_P``, default 4: k = 3) and the state dtype (``BENCH_STATE_DTYPE``,
default the checkpoint's float32) from the environment. The verify once a
layer with the layer's gated norm after it (and the copy of its gate), and the
commit once a group, against the baseline Triton kernel writing a state per
draft token; ``BENCH_OTHER_OPS`` (a path to another version of the ops module)
times its verify alike. Tries the launch shapes and prints ms per step.
"""

import os
import sys
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.ops import gdn_recoverssm as ops
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    layer_norm_fwd,
)

DEVICE = "cuda"
LAYERS, GROUPS = 36, 3
KEY_HEADS, VALUE_HEADS, DIM = 16, 48, 128
REQUESTS = int(os.environ.get("BENCH_REQUESTS", "41"))
P = int(os.environ.get("BENCH_P", "4"))
STATE_DTYPE = getattr(torch, os.environ.get("BENCH_STATE_DTYPE", "float32"))
CONV_DIM = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main():
    torch.manual_seed(0)
    tokens = REQUESTS * P
    num_blocks = 1 + REQUESTS * P
    per_group = LAYERS // GROUPS
    q = torch.randn(1, tokens, KEY_HEADS, DIM, device=DEVICE).to(torch.bfloat16)
    k = torch.randn(1, tokens, KEY_HEADS, DIM, device=DEVICE).to(torch.bfloat16)
    v = torch.randn(1, tokens, VALUE_HEADS, DIM, device=DEVICE).to(torch.bfloat16)
    a = torch.randn(tokens, VALUE_HEADS, device=DEVICE).to(torch.bfloat16)
    b = torch.randn(tokens, VALUE_HEADS, device=DEVICE).to(torch.bfloat16)
    A_log = (-2 + 0.5 * torch.randn(VALUE_HEADS, device=DEVICE)).float()
    dt_bias = (0.5 * torch.randn(VALUE_HEADS, device=DEVICE)).to(torch.bfloat16)
    qsl = torch.arange(0, tokens + 1, P, dtype=torch.int32, device=DEVICE)
    slots = torch.arange(1, num_blocks, dtype=torch.int32, device=DEVICE).view(
        REQUESTS, P
    )
    sources = slots[:, 0].contiguous()
    accepted = torch.randint(1, P + 1, (REQUESTS,), dtype=torch.int32, device=DEVICE)
    ones = torch.ones(REQUESTS, dtype=torch.int32, device=DEVICE)

    groups = []
    for _ in range(GROUPS):
        layers = []
        for _ in range(per_group):
            state = (
                0.1 * torch.randn(num_blocks, VALUE_HEADS, DIM, DIM, device=DEVICE)
            ).to(STATE_DTYPE)
            conv = torch.randn(num_blocks, CONV_DIM, 3 + P - 1, device=DEVICE).to(
                torch.bfloat16
            )
            records = (
                torch.zeros(num_blocks, VALUE_HEADS, P, DIM, device=DEVICE),
                torch.zeros(num_blocks, KEY_HEADS, P, DIM, device=DEVICE),
                torch.zeros(num_blocks, VALUE_HEADS, P, device=DEVICE),
            )
            layers.append(SimpleNamespace(kv_cache=(conv, state, *records)))
        groups.append(layers)
    ops.is_conv_state_dim_first = lambda: True
    contexts = [
        ops.GDNRecoverSSMCommitContext.create(
            layers, spec_query_len=P, max_num_reqs=REQUESTS
        )
        for layers in groups
    ]
    all_layers = [layer for layers in groups for layer in layers]

    def baseline():
        for layer in all_layers:
            fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                initial_state=layer.kv_cache[1],
                inplace_final_state=True,
                cu_seqlens=qsl,
                ssm_state_indices=slots,
                num_accepted_tokens=ones,
                use_qk_l2norm_in_kernel=True,
            )

    # The layer's z, a row-strided slice of its packed projection, and the
    # norm weight; the output in place, as a verify-only batch writes it.
    packed = torch.randn(tokens, CONV_DIM + VALUE_HEADS * DIM, device=DEVICE).to(
        torch.bfloat16
    )
    output_gate = packed[:, CONV_DIM:].view(tokens, VALUE_HEADS, DIM)
    norm_weight = torch.ones(DIM, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(v)

    def verify():
        for layer in all_layers:
            ops.gdn_recoverssm_verify(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                checkpoint_state=layer.kv_cache[1],
                correction_cache=layer.kv_cache[2],
                key_cache=layer.kv_cache[3],
                gate_cache=layer.kv_cache[4],
                query_start_loc=qsl,
                state_indices=sources,
                spec_query_len=P,
                scale=DIM**-0.5,
                out=out,
            )
            # The layer's _rms_norm_gated_cuda: z reshaped (a copy), the norm.
            layer_norm_fwd(
                out.view(-1, DIM),
                norm_weight,
                None,
                1e-6,
                z=output_gate.reshape(-1, DIM),
                out=out.view(-1, DIM),
                group_size=DIM,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="silu",
            )

    def commit():
        for context in contexts:
            context.commit(accepted, sources, qsl)

    state_bytes = LAYERS * REQUESTS * VALUE_HEADS * DIM * DIM * STATE_DTYPE.itemsize
    print(
        f"P={P}, {STATE_DTYPE}: one state a layer and request, all layers: "
        f"{state_bytes / 1e9:.2f} GB"
    )
    print(f"baseline (triton, writes {P} states): {timed(baseline):.3f} ms/step")

    def shapes(name, default):
        if os.environ.get(name):
            return [
                tuple(int(x) for x in shape.split(","))
                for shape in os.environ[name].split(";")
            ]
        return default

    # Another version of the ops module (a file path), timed alike first.
    if os.environ.get("BENCH_OTHER_OPS"):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "other_ops", os.environ["BENCH_OTHER_OPS"]
        )
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        # Its dataclasses look their module up by name.
        sys.modules["other_ops"] = other
        spec.loader.exec_module(other)

        def other_verify():
            for layer in all_layers:
                other.gdn_recoverssm_verify(
                    q=q,
                    k=k,
                    v=v,
                    a=a,
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    checkpoint_state=layer.kv_cache[1],
                    correction_cache=layer.kv_cache[2],
                    key_cache=layer.kv_cache[3],
                    gate_cache=layer.kv_cache[4],
                    query_start_loc=qsl,
                    state_indices=sources,
                    spec_query_len=P,
                    scale=DIM**-0.5,
                    out=out,
                )
                layer_norm_fwd(
                    out.view(-1, DIM),
                    norm_weight,
                    None,
                    1e-6,
                    z=output_gate.reshape(-1, DIM),
                    out=out.view(-1, DIM),
                    group_size=DIM,
                    norm_before_gate=True,
                    is_rms_norm=True,
                    activation="silu",
                )

        print(
            f"other ops, verify + norm {other.VERIFY_LAUNCH}: "
            f"{timed(other_verify):.3f} ms/step"
        )
    # The constant this card's verify reads its launch shape from.
    verify_launch = (
        "SM100_VERIFY_LAUNCH"
        if current_platform.is_device_capability_family(100)
        else "VERIFY_LAUNCH"
    )
    for shape in shapes("BENCH_SHAPES", [(16, 1, 2), (16, 1, 1), (32, 2, 1)]):
        setattr(ops, verify_launch, shape)
        try:
            print(f"verify + norm {shape}: {timed(verify):.3f} ms/step")
        except Exception as error:  # noqa: BLE001
            print(f"verify {shape}: failed {type(error).__name__}: {error}"[:200])
    for shape in shapes("BENCH_COMMIT_SHAPES", [(8, 1, 2)]):
        ops.COMMIT_LAUNCH = shape
        try:
            print(f"commit {shape}: {timed(commit):.3f} ms/step")
        except Exception as error:  # noqa: BLE001
            print(f"commit {shape}: failed {type(error).__name__}: {error}"[:200])


if __name__ == "__main__":
    main()
