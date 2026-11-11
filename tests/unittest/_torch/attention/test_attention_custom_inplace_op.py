# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which attention calls route through the metadata-registry custom op."""

import weakref
from functools import partial
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import enable_breakable_cuda_graph
from tensorrt_llm._torch.utils import model_extra_attrs


class _Metadata:
    pass


def _layer(backend="FLASHINFER", register_to_config=True):
    layer = SimpleNamespace(
        register_to_config=register_to_config, attn_backend=backend, is_marlin_enabled=False
    )
    layer.use_custom_op = partial(Attention._use_custom_inplace_op, layer)
    return layer


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["FLASHINFER", "TRTLLM"])
def test_breakable_graph_routes_only_the_registered_metadata(backend):
    """Under a breakable CUDA graph, only calls with the forward's registered metadata use the custom op.

    The custom op resolves the registered metadata, so a call carrying other metadata (a
    speculative draft step on its draft view) must run the backend with its own metadata.
    """
    target, draft = _Metadata(), _Metadata()
    layer = _layer(backend)
    with model_extra_attrs({"attention_metadata": weakref.ref(target)}):
        assert not layer.use_custom_op(target)
        with enable_breakable_cuda_graph():
            assert layer.use_custom_op(target)
            assert not layer.use_custom_op(draft)
    with enable_breakable_cuda_graph():
        assert not layer.use_custom_op(target)


@pytest.mark.cpu_only
def test_custom_op_requires_a_registered_supported_layer():
    target = _Metadata()
    with model_extra_attrs({"attention_metadata": weakref.ref(target)}):
        with enable_breakable_cuda_graph():
            assert not _layer(register_to_config=False).use_custom_op(target)
            assert not _layer(backend="VANILLA").use_custom_op(target)
