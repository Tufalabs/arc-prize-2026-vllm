# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN recurrent state in FP8 rows (``gdn_fp8_state``): the shape and
dtype the state calculators give it, and the models that refuse it.

The kernels that read and write the rows need a GPU, and the fork carries no
GPU tests of its own (``TUFA.md``).
"""

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.ops.gdn_fp8_state import (
    FP8_STATE_DTYPE,
    SCALE_BYTES,
)


def test_shapes_and_dtypes():
    conv, state = mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape(
        2, 16, 48, 128, 128, 4, num_spec=4, fp8_state=True
    )
    assert state == (24, 128, 128 + SCALE_BYTES)
    dtypes = mamba_utils.MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.bfloat16, "auto", "fp8_e4m3", allow_fp8_state=True
    )
    assert dtypes == (torch.bfloat16, FP8_STATE_DTYPE)
    # A model whose kernels do not read FP8 rows refuses the option.
    with pytest.raises(ValueError, match="fp8_e4m3"):
        mamba_utils.MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            torch.bfloat16, "auto", "fp8_e4m3"
        )
    with pytest.raises(ValueError, match="fp8_e4m3"):
        mamba_utils.MambaStateDtypeCalculator.mamba2_state_dtype(
            torch.bfloat16, "auto", "fp8_e4m3"
        )
