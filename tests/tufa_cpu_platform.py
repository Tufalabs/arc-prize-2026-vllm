# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin vLLM's platform to CPU on a host where none resolves.

The fork's CI installs the precompiled CUDA wheel on a GPU-less runner, and
a cluster login node has the same shape: the CUDA plugin finds no device
and the CPU plugin accepts only a ``+cpu`` build, so ``VllmConfig`` raises
"Failed to infer device type" and the pure-Python scheduler tests cannot
build one. Loaded with ``-p tests.tufa_cpu_platform``; a no-op wherever a
platform resolves on its own.

The cached ``is_pin_memory_available()`` and ``torch_utils.PIN_MEMORY`` were
answered by the platform this replaces, which offers pinned memory that needs
a driver; both are asked again of the CPU platform before the modules that
import ``PIN_MEMORY`` by name are collected.
"""

import vllm.platforms as platforms

if not platforms.current_platform.device_type:
    import vllm.utils.platform_utils as platform_utils
    import vllm.utils.torch_utils as torch_utils
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    platform_utils.is_pin_memory_available.cache_clear()
    torch_utils.PIN_MEMORY = platform_utils.is_pin_memory_available()
