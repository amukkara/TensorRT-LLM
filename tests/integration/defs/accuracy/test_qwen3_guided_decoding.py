# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

from ..conftest import skip_pre_hopper
from .accuracy_core import JsonModeEval, LlmapiAccuracyTestHarness

models_root = "/home/amukkara/scratch/datasets/trt-llm/hf_models"
_TARGET_MODEL_DIR = f"{models_root}/Qwen/Qwen3-4B-FP8"
_EAGLE3_MODEL_DIR = f"{models_root}/Qwen/Qwen3-4B_eagle3"


class TestQwen3_4BGuidedDecoding(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3-4B"
    MODEL_PATH = _TARGET_MODEL_DIR

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
    cuda_graph_config = CudaGraphConfig(enable_padding=True)
    max_batch_size = 64

    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mocker):
        """Accuracy test for guided decoding (JSON mode) without speculative decoding."""
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        with LLM(
            self.MODEL_PATH,
            guided_decoding_backend=backend,
            kv_cache_config=self.kv_cache_config,
            cuda_graph_config=self.cuda_graph_config,
            enable_chunked_prefill=True,
            max_batch_size=self.max_batch_size,
        ) as llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_with_eagle3(self, backend: str, mocker):
        """Accuracy test for guided decoding (JSON mode) combined with Eagle3 one-model speculative decoding."""
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        spec_config = Eagle3DecodingConfig(
            max_draft_len=3, speculative_model=_EAGLE3_MODEL_DIR, eagle3_one_model=True
        )
        with LLM(
            self.MODEL_PATH,
            guided_decoding_backend=backend,
            kv_cache_config=self.kv_cache_config,
            cuda_graph_config=self.cuda_graph_config,
            enable_chunked_prefill=True,
            max_batch_size=self.max_batch_size,
            speculative_config=spec_config,
        ) as llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)
