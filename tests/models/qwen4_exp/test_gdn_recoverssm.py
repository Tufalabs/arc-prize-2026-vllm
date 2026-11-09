# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM against the path it replaces.

The baseline verifies a speculative window with the sigmoid-gating recurrent
kernel, writing the state after every position into its own slot, and the
next step reads the accepted slot and the conv window at the accepted offset.
RecoverSSM verifies from the checkpoint, commits the accepted state into it
and compacts the conv windows, so the next step reads slot 0 at offset 0.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.ops import gdn_recoverssm as recoverssm_ops
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (
    GDNRecoverSSMCommitContext,
    gdn_recoverssm_verify,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)

DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="The GDN kernels need CUDA."
)

# Grouped as Qwen4Exp's 16 key heads to 48 value heads, three to one.
KEY_HEADS, VALUE_HEADS, DIM = 4, 12, 128
SCALE = DIM**-0.5


def _params(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    A_log = (-2.0 + 0.5 * torch.randn(VALUE_HEADS, generator=gen, device=DEVICE)).to(
        torch.float32
    )
    dt_bias = (0.5 * torch.randn(VALUE_HEADS, generator=gen, device=DEVICE)).to(
        torch.bfloat16
    )
    return A_log.contiguous(), dt_bias.contiguous()


def _qkvab(num_tokens: int, seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=DEVICE).to(torch.bfloat16)

    q = randn(1, num_tokens, KEY_HEADS, DIM)
    k = randn(1, num_tokens, KEY_HEADS, DIM)
    v = randn(1, num_tokens, VALUE_HEADS, DIM)
    a = randn(num_tokens, VALUE_HEADS)
    b = randn(num_tokens, VALUE_HEADS)
    return q, k, v, a, b


def _records(num_blocks: int, spec_query_len: int):
    return (
        torch.full(
            (num_blocks, VALUE_HEADS, spec_query_len, DIM),
            float("nan"),
            device=DEVICE,
        ),
        torch.full(
            (num_blocks, KEY_HEADS, spec_query_len, DIM), float("nan"), device=DEVICE
        ),
        torch.full(
            (num_blocks, VALUE_HEADS, spec_query_len), float("nan"), device=DEVICE
        ),
    )


def _native_window(q, k, v, a, b, A_log, dt_bias, pool, qsl, slots, num_accepted):
    """The baseline: per-position states into ``slots``, read from slot a-1."""
    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=pool,
        inplace_final_state=True,
        cu_seqlens=qsl,
        ssm_state_indices=slots,
        num_accepted_tokens=num_accepted,
        use_qk_l2norm_in_kernel=True,
    )
    return out


def _assert_states_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # One float32 recurrence rounded once to bf16 on both sides, summed in a
    # different order: at most an ulp apart.
    torch.testing.assert_close(actual.float(), expected.float(), atol=8e-3, rtol=8e-3)


@pytest.mark.parametrize("spec_query_len", [2, 4])
@pytest.mark.parametrize(
    ("align_mode", "use_request_indices", "conv_state_dim_first"),
    [
        pytest.param(False, False, True, id="none"),
        pytest.param(False, True, False, id="none-request-indexed-SD"),
        pytest.param(True, False, False, id="align-SD"),
        pytest.param(True, True, True, id="align-request-indexed"),
    ],
)
@torch.inference_mode()
def test_verify_matches_the_baseline_and_commit_recovers_the_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
    spec_query_len: int,
    align_mode: bool,
    use_request_indices: bool,
    conv_state_dim_first: bool,
):
    monkeypatch.setattr(
        recoverssm_ops, "is_conv_state_dim_first", lambda: conv_state_dim_first
    )
    P = spec_query_len
    block_size = 4
    # Full windows and a draft-less decode. Aligned, they cover a state
    # landing back in the old column, a boundary crossed into the next one,
    # and a window accepted exactly up to a block's end (the state stays in
    # that block: the next column is not allocated yet).
    query_lens = [P, P, P, 1]
    # Decode-tail targets (aligned only), as tokens past num_computed: one
    # beyond the accepted tokens, one crossed inside the window and one at
    # its accepted end, and a stale one at the window's start (not crossed).
    if P == 4:
        num_computed = [5, 6, 4, 8]
        accepted = [2, 4, 4, 1]
        tail_offsets = [4, 4, 1, 0]
    else:
        num_computed = [3, 5, 7, 8]
        accepted = [1, 2, 2, 1]
        tail_offsets = [2, 1, 2, 0]
    num_seqs = len(query_lens)
    qsl = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    total_tokens = sum(query_lens)

    batch_rows = [1, 3, 4, 6] if use_request_indices else list(range(num_seqs))
    batch_size = max(batch_rows) + 1
    request_indices = (
        torch.tensor(batch_rows, dtype=torch.int32, device=DEVICE)
        if use_request_indices
        else None
    )
    global_accepted = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    global_accepted[batch_rows] = torch.tensor(
        accepted, dtype=torch.int32, device=DEVICE
    )

    # Block ids: a block-table row of four columns per request, then the
    # baseline's scratch slots. Block 0 is the null block.
    width = 4
    block_table = torch.zeros(batch_size, width, dtype=torch.int32, device=DEVICE)
    next_block = 1
    for row in batch_rows:
        block_table[row] = torch.arange(
            next_block, next_block + width, dtype=torch.int32
        )
        next_block += width
    scratch_start = next_block
    tail_start = scratch_start + num_seqs * P
    num_blocks = tail_start + num_seqs

    sources, finals, boundaries = [], [], []
    for i, row in enumerate(batch_rows):
        nc, q_len, a = num_computed[i], query_lens[i], accepted[i]
        if align_mode:
            source = int(block_table[row, (nc + q_len - 1) // block_size])
            final = int(block_table[row, (nc + a - 1) // block_size])
            next_boundary = (nc // block_size + 1) * block_size
            boundary = (
                (int(block_table[row, nc // block_size]), next_boundary - nc)
                if nc + a >= next_boundary
                else None
            )
        else:
            source = int(block_table[row, 0])
            final, boundary = source, None
        sources.append(source)
        finals.append(final)
        boundaries.append(boundary)
    tail_boundaries = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    tail_blocks = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    tails = []
    for i, row in enumerate(batch_rows):
        tail_boundaries[row] = num_computed[i] + tail_offsets[i]
        tail_blocks[row] = tail_start + i
        crossed = align_mode and 0 < tail_offsets[i] <= accepted[i]
        tails.append((tail_start + i, tail_offsets[i]) if crossed else None)
    if align_mode:
        assert any(t is not None and t[1] < a for t, a in zip(tails, accepted))
        assert any(t is not None and t[1] == a for t, a in zip(tails, accepted))
        assert any(t is None for t in tails)
        assert any(f != s for f, s in zip(finals, sources))
        assert any(b is not None and b[0] != f for b, f in zip(boundaries, finals))
        assert any(b is not None and b[0] == f for b, f in zip(boundaries, finals))
    state_indices = torch.tensor(sources, dtype=torch.int32, device=DEVICE)

    # The baseline's slots: the checkpoint, then scratch.
    native_slots = torch.zeros(num_seqs, P, dtype=torch.int32, device=DEVICE)
    for i in range(num_seqs):
        native_slots[i, 0] = sources[i]
        native_slots[i, 1:] = torch.arange(
            scratch_start + i * P + 1, scratch_start + (i + 1) * P, dtype=torch.int32
        )

    history_len, conv_dim = 3, 40
    layers, expected = [], []
    for layer_idx in range(2):
        A_log, dt_bias = _params(100 + layer_idx)
        q, k, v, a, b = _qkvab(total_tokens, 200 + layer_idx)
        gen = torch.Generator(device=DEVICE).manual_seed(300 + layer_idx)
        pool = (
            0.1
            * torch.randn(
                num_blocks, VALUE_HEADS, DIM, DIM, generator=gen, device=DEVICE
            )
        ).to(torch.bfloat16)
        pool[0] = 0
        conv_shape = (
            (num_blocks, conv_dim, history_len + P - 1)
            if conv_state_dim_first
            else (num_blocks, history_len + P - 1, conv_dim)
        )
        conv_state = torch.randn(conv_shape, generator=gen, device=DEVICE).to(
            torch.bfloat16
        )

        native_pool = pool.clone()
        native_out = _native_window(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            native_pool,
            qsl,
            native_slots,
            torch.ones(num_seqs, dtype=torch.int32, device=DEVICE),
        )

        records = _records(num_blocks, P)
        recover_pool = pool.clone()
        out = gdn_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=recover_pool,
            correction_cache=records[0],
            key_cache=records[1],
            gate_cache=records[2],
            query_start_loc=qsl,
            state_indices=state_indices,
            spec_query_len=P,
            scale=SCALE,
        )
        torch.testing.assert_close(
            out.float(), native_out.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(recover_pool, pool, atol=0, rtol=0)

        layers.append(
            SimpleNamespace(
                kv_cache=(conv_state, recover_pool, *records),
                initial_pool=pool,
                initial_conv=conv_state.clone(),
            )
        )
        expected.append(native_pool)

    context = GDNRecoverSSMCommitContext.create(
        layers, spec_query_len=P, max_num_reqs=batch_size
    )
    context.commit(
        global_accepted,
        state_indices,
        qsl,
        request_indices=request_indices,
        block_table=block_table if align_mode else None,
        num_computed_tokens=(
            torch.tensor(
                [
                    num_computed[batch_rows.index(r)] if r in batch_rows else 0
                    for r in range(batch_size)
                ],
                dtype=torch.int32,
                device=DEVICE,
            )
            if align_mode
            else None
        ),
        mamba_block_size=block_size if align_mode else None,
        tail_boundaries=tail_boundaries if align_mode else None,
        tail_blocks=tail_blocks if align_mode else None,
    )

    def conv_window(conv: torch.Tensor, block: int, start: int) -> torch.Tensor:
        view = conv if conv_state_dim_first else conv.transpose(-1, -2)
        return view[block, :, start : start + history_len]

    for layer, native_pool in zip(layers, expected):
        pool = layer.kv_cache[1]
        conv = layer.kv_cache[0]
        written = set()
        for i in range(num_seqs):
            a = accepted[i]
            _assert_states_close(pool[finals[i]], native_pool[native_slots[i, a - 1]])
            torch.testing.assert_close(
                conv_window(conv, finals[i], 0),
                conv_window(layer.initial_conv, sources[i], a - 1),
            )
            written.add(finals[i])
            if (boundary := boundaries[i]) is not None:
                block, recovery_len = boundary
                _assert_states_close(
                    pool[block], native_pool[native_slots[i, recovery_len - 1]]
                )
                torch.testing.assert_close(
                    conv_window(conv, block, 0),
                    conv_window(layer.initial_conv, sources[i], recovery_len - 1),
                )
                written.add(block)
            if (tail := tails[i]) is not None:
                block, recovery_len = tail
                _assert_states_close(
                    pool[block], native_pool[native_slots[i, recovery_len - 1]]
                )
                torch.testing.assert_close(
                    conv_window(conv, block, 0),
                    conv_window(layer.initial_conv, sources[i], recovery_len - 1),
                )
                written.add(block)
        untouched = [n for n in range(num_blocks) if n not in written]
        torch.testing.assert_close(
            pool[untouched], layer.initial_pool[untouched], atol=0, rtol=0
        )
        torch.testing.assert_close(
            conv[untouched], layer.initial_conv[untouched], atol=0, rtol=0
        )


@pytest.mark.parametrize("spec_query_len", [2, 4])
@torch.inference_mode()
def test_decode_steps_match_the_baseline(
    monkeypatch: pytest.MonkeyPatch, spec_query_len: int
):
    """Conv plus recurrence over several steps with random acceptance: the
    baseline's per-draft slots and accepted offsets against RecoverSSM's
    commit and compaction, output for output."""
    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    P = spec_query_len
    num_seqs, width = 3, 4
    conv_dim = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM
    num_blocks = 1 + num_seqs * P
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    A_log, dt_bias = _params(11)
    conv_weight = (0.3 * torch.randn(conv_dim, width, generator=gen, device=DEVICE)).to(
        torch.bfloat16
    )
    pool = (
        0.1
        * torch.randn(num_blocks, VALUE_HEADS, DIM, DIM, generator=gen, device=DEVICE)
    ).to(torch.bfloat16)
    conv = torch.randn(
        num_blocks, conv_dim, width - 1 + P - 1, generator=gen, device=DEVICE
    ).to(torch.bfloat16)

    native_slots = torch.arange(1, num_blocks, dtype=torch.int32, device=DEVICE).view(
        num_seqs, P
    )
    native_pool, native_conv = pool.clone(), conv.clone()
    native_accepted = torch.ones(num_seqs, dtype=torch.int32, device=DEVICE)

    sources = native_slots[:, 0].contiguous()
    recover_pool, recover_conv = pool.clone(), conv.clone()
    records = _records(num_blocks, P)
    layer = SimpleNamespace(kv_cache=(recover_conv, recover_pool, *records))
    context = GDNRecoverSSMCommitContext.create(
        [layer], spec_query_len=P, max_num_reqs=num_seqs
    )
    ones = torch.ones(num_seqs, dtype=torch.int32, device=DEVICE)
    qsl = torch.arange(0, num_seqs * P + 1, P, dtype=torch.int32, device=DEVICE)

    def split(mixed: torch.Tensor):
        tokens = mixed.shape[0]
        q, k, v = mixed.split(
            [KEY_HEADS * DIM, KEY_HEADS * DIM, VALUE_HEADS * DIM], dim=-1
        )
        return (
            q.contiguous().view(1, tokens, KEY_HEADS, DIM),
            k.contiguous().view(1, tokens, KEY_HEADS, DIM),
            v.contiguous().view(1, tokens, VALUE_HEADS, DIM),
        )

    for step in range(5):
        step_gen = torch.Generator(device=DEVICE).manual_seed(1000 + step)
        mixed = torch.randn(
            num_seqs * P, conv_dim, generator=step_gen, device=DEVICE
        ).to(torch.bfloat16)
        a = torch.randn(
            num_seqs * P, VALUE_HEADS, generator=step_gen, device=DEVICE
        ).to(torch.bfloat16)
        b = torch.randn(
            num_seqs * P, VALUE_HEADS, generator=step_gen, device=DEVICE
        ).to(torch.bfloat16)

        native_mixed = causal_conv1d_update(
            mixed.clone(),
            native_conv,
            conv_weight,
            None,
            "silu",
            conv_state_indices=native_slots[:, 0],
            num_accepted_tokens=native_accepted,
            query_start_loc=qsl,
            max_query_len=P,
        )
        recover_mixed = causal_conv1d_update(
            mixed.clone(),
            recover_conv,
            conv_weight,
            None,
            "silu",
            conv_state_indices=sources,
            num_accepted_tokens=ones,
            query_start_loc=qsl,
            max_query_len=P,
        )
        torch.testing.assert_close(recover_mixed, native_mixed, atol=0, rtol=0)

        q, k, v = split(native_mixed)
        native_out = _native_window(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            native_pool,
            qsl,
            native_slots,
            native_accepted,
        )
        q, k, v = split(recover_mixed)
        recover_out = gdn_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=recover_pool,
            correction_cache=records[0],
            key_cache=records[1],
            gate_cache=records[2],
            query_start_loc=qsl,
            state_indices=sources,
            spec_query_len=P,
            scale=SCALE,
        )
        torch.testing.assert_close(
            recover_out.float(), native_out.float(), atol=2e-2, rtol=2e-2
        )

        accepted = torch.randint(
            1, P + 1, (num_seqs,), generator=step_gen, device=DEVICE
        ).to(torch.int32)
        native_accepted = accepted
        context.commit(accepted, sources, qsl)
        for i in range(num_seqs):
            _assert_states_close(
                recover_pool[sources[i]],
                native_pool[native_slots[i, accepted[i] - 1]],
            )


@pytest.mark.parametrize("spec_query_len", [2, 4])
@torch.inference_mode()
def test_ple_conv_steps_match_the_baseline(
    monkeypatch: pytest.MonkeyPatch, spec_query_len: int
):
    """The conv-only PLE group: accepted offsets against compaction."""
    from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv

    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    P = spec_query_len
    num_seqs, channels, kernel_size, dilation = 3, 320, 4, 2
    state_len = (kernel_size - 1) * dilation
    num_blocks = 1 + num_seqs
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    weights = (
        0.3 * torch.randn(channels, kernel_size, generator=gen, device=DEVICE)
    ).to(torch.bfloat16)
    conv = torch.randn(
        num_blocks, channels, state_len + P - 1, generator=gen, device=DEVICE
    ).to(torch.bfloat16)
    native_conv, recover_conv = conv.clone(), conv.clone()
    slots = torch.arange(1, num_blocks, dtype=torch.int32, device=DEVICE)
    qsl = torch.arange(0, num_seqs * P + 1, P, dtype=torch.int32, device=DEVICE)
    ones = torch.ones(num_seqs, dtype=torch.int32, device=DEVICE)
    native_accepted = ones.clone()
    context = GDNRecoverSSMCommitContext.create(
        [SimpleNamespace(kv_cache=(recover_conv,))],
        spec_query_len=P,
        max_num_reqs=num_seqs,
    )

    for step in range(5):
        step_gen = torch.Generator(device=DEVICE).manual_seed(2000 + step)
        inputs = torch.randn(
            num_seqs * P, channels, generator=step_gen, device=DEVICE
        ).to(torch.bfloat16)
        native_residual = torch.zeros_like(inputs)
        recover_residual = torch.zeros_like(inputs)
        # ple_conv adds the layer's outer residual too (#55309); zero on both.
        outer_residual = torch.zeros_like(inputs)
        ple_conv(
            inputs,
            native_residual,
            native_conv,
            weights,
            slots,
            outer_residual,
            mode="spec",
            dilation=dilation,
            query_start_loc=qsl,
            num_accepted_tokens=native_accepted,
            spec_query_len=P,
        )
        ple_conv(
            inputs,
            recover_residual,
            recover_conv,
            weights,
            slots,
            outer_residual,
            mode="spec",
            dilation=dilation,
            query_start_loc=qsl,
            num_accepted_tokens=ones,
            spec_query_len=P,
        )
        torch.testing.assert_close(recover_residual, native_residual, atol=0, rtol=0)
        accepted = torch.randint(
            1, P + 1, (num_seqs,), generator=step_gen, device=DEVICE
        ).to(torch.int32)
        native_accepted = accepted
        context.commit(accepted, slots, qsl)


@pytest.mark.parametrize("spec_query_len", [2, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_verify_steps_each_window_as_the_recurrence(
    spec_query_len: int, state_dtype: torch.dtype
):
    """Windows of every length and a request on the null block: the output
    and the records against the gated delta rule stepped in float32, and the
    record positions past a window left alone."""
    P = spec_query_len
    query_lens = [P, 1, P, max(P - 1, 1), P]
    num_seqs = len(query_lens)
    qsl = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    total_tokens = sum(query_lens)
    A_log, dt_bias = _params(11)
    q, k, v, a, b = _qkvab(total_tokens, 12)
    gen = torch.Generator(device=DEVICE).manual_seed(13)
    num_blocks = num_seqs + 1
    pool = (
        0.1
        * torch.randn(num_blocks, VALUE_HEADS, DIM, DIM, generator=gen, device=DEVICE)
    ).to(state_dtype)
    pool[0] = 0
    # Block 0 is the null block: the third request has no state.
    blocks = [1, 2, 0, 3, 4]
    records = _records(num_blocks, P)
    out = gdn_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=pool,
        correction_cache=records[0],
        key_cache=records[1],
        gate_cache=records[2],
        query_start_loc=qsl,
        state_indices=torch.tensor(blocks, dtype=torch.int32, device=DEVICE),
        spec_query_len=P,
        scale=SCALE,
    )

    group = VALUE_HEADS // KEY_HEADS
    for request, (block, start) in enumerate(zip(blocks, qsl.tolist())):
        rows = slice(start, start + query_lens[request])
        if block == 0:
            assert torch.all(out[0, rows] == 0)
            continue
        state = pool[block].float()
        for position in range(query_lens[request]):
            token = start + position
            kt = k[0, token].float()
            kt = kt * torch.rsqrt(kt.pow(2).sum(-1, keepdim=True) + 1e-6)
            qt = q[0, token].float()
            qt = qt * torch.rsqrt(qt.pow(2).sum(-1, keepdim=True) + 1e-6) * SCALE
            gate = -A_log.exp() * torch.nn.functional.softplus(
                a[token].float() + dt_bias.float()
            )
            beta = torch.sigmoid(b[token].float())
            k_heads = kt.repeat_interleave(group, dim=0)
            q_heads = qt.repeat_interleave(group, dim=0)
            state = state * gate.exp()[:, None, None]
            correction = beta[:, None] * (
                v[0, token].float() - torch.einsum("hvk,hk->hv", state, k_heads)
            )
            state = state + correction[:, :, None] * k_heads[:, None, :]
            expected = torch.einsum("hvk,hk->hv", state, q_heads)
            torch.testing.assert_close(
                out[0, token].float(), expected, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                records[0][block, :, position], correction, atol=1e-3, rtol=1e-3
            )
            torch.testing.assert_close(
                records[1][block, :, position], kt, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                records[2][block, :, position], gate, atol=1e-5, rtol=1e-5
            )
        # Positions past the window keep what they held.
        for record in records:
            assert torch.isnan(record[block, :, query_lens[request] :]).all()


def test_record_shapes():
    shapes = mamba_utils.MambaStateShapeCalculator.append_gdn_recoverssm_record(
        ((10240, 4), (48, 128, 128)),
        tp_world_size=1,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        spec_query_len=2,
    )
    assert shapes[2:] == ((48, 2, 128), (16, 2, 128), (48, 2))
    dtypes = mamba_utils.MambaStateDtypeCalculator.append_gdn_recoverssm_record(
        (torch.bfloat16, torch.bfloat16)
    )
    assert dtypes == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
    )


@pytest.mark.parametrize("window", [4, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_deferred_commits_match_immediate_at_tail_boundary_and_removal(
    monkeypatch: pytest.MonkeyPatch, window: int, dtype: torch.dtype
) -> None:
    """Consecutive deferrals must survive row remapping, tail writes and flush."""
    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    blocks, batch = 12, 2
    generator = torch.Generator(device=DEVICE).manual_seed(731)
    initial = (
        torch.randn(
            blocks, VALUE_HEADS, DIM, DIM, device=DEVICE, generator=generator
        ).to(dtype)
        * 0.1
    )
    conv = torch.randn(blocks, 32, 3 + window, device=DEVICE, generator=generator)
    layers = []
    contexts = []
    for _ in range(2):
        layer = SimpleNamespace(
            kv_cache=(conv.clone(), initial.clone(), *_records(blocks, window))
        )
        layers.append(layer)
        contexts.append(
            GDNRecoverSSMCommitContext.create(
                [layer], spec_query_len=window, max_num_reqs=3
            )
        )
    fused = recoverssm_ops.GDNFusedCommit(
        max_num_reqs=3,
        max_rows=batch,
        key_shape=(KEY_HEADS, window, DIM),
        gate_shape=(VALUE_HEADS, window),
        spec_query_len=window,
        device=torch.device(DEVICE),
    )
    fused.context = contexts[1]
    mapping = torch.tensor([2, 0], dtype=torch.int32, device=DEVICE)
    states = torch.tensor([[1], [5]], dtype=torch.int32, device=DEVICE)
    table = torch.tensor([[1, 2], [5, 6]], dtype=torch.int32, device=DEVICE)
    computed = torch.tensor([16, 20], dtype=torch.int32, device=DEVICE)
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * window
    A_log, dt_bias = _params(31)
    for step in range(6):
        fused.consume(mapping, None, states[:, 0], batch)
        q, k, v, a, b = _qkvab(batch * window, 300 + step)
        outputs = []
        for index, layer in enumerate(layers):
            _, pool, corr, key, gate = layer.kv_cache
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                checkpoint_state=pool,
                correction_cache=corr,
                key_cache=key,
                gate_cache=gate,
                query_start_loc=qsl,
                state_indices=states[:, 0],
                spec_query_len=window,
                scale=SCALE,
            )
            if index:
                outputs.append(
                    recoverssm_ops.gdn_recoverssm_fused_verify(
                        **kwargs, out=torch.empty_like(v), fused=fused
                    )
                )
            else:
                outputs.append(gdn_recoverssm_verify(**kwargs))
        torch.testing.assert_close(outputs[1], outputs[0], atol=1e-2, rtol=1e-2)
        accepted = torch.tensor(
            [1 + step % window, window], dtype=torch.int32, device=DEVICE
        )
        tail = step == 3
        # A final step crosses a block boundary; earlier steps defer in place.
        if step == 5:
            computed.fill_(62)
        for index, context in enumerate(contexts):
            context.commit(
                accepted,
                states,
                qsl,
                block_table=table,
                num_computed_tokens=computed,
                mamba_block_size=64,
                tail_boundaries=computed + 1 if tail else None,
                tail_blocks=torch.tensor([9, 10], dtype=torch.int32, device=DEVICE)
                if tail
                else None,
                fused=fused if index else None,
            )
        computed += accepted
        if step == 2:
            # Removal/parking must make both states current before reuse.
            fused.flush([2, 0])
            torch.testing.assert_close(
                layers[1].kv_cache[1], layers[0].kv_cache[1], atol=8e-3, rtol=8e-3
            )
    fused.flush([2, 0])
    assert fused.mismatches.item() == 0
    assert fused.pending_len.sum().item() == 0
    torch.testing.assert_close(
        layers[1].kv_cache[1], layers[0].kv_cache[1], atol=8e-3, rtol=8e-3
    )
    torch.testing.assert_close(
        layers[1].kv_cache[0], layers[0].kv_cache[0], atol=0, rtol=0
    )
