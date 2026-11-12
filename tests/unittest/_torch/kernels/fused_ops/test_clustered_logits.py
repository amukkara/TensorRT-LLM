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

import pytest
import torch

from tensorrt_llm._torch.kernels.fused_ops.clustered_logits import clustered_logits, topk_indices


@pytest.mark.parametrize("rows", [0, 1, 8, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_indices(rows, dtype):
    torch.manual_seed(0)
    # 2048 clusters, top 32: the Gemma4-E4B assistant's clustered head.
    scores = torch.randn((rows, 2048), device="cuda").to(dtype)
    idx = topk_indices(scores, 32)
    assert idx.shape == (rows, 32) and idx.dtype == torch.int32
    ref_vals, _ = torch.topk(scores.float(), 32, dim=-1)
    got_vals = scores.float().gather(1, idx.long())
    # Same selected scores, largest first (ties may pick other indices than torch.topk).
    torch.testing.assert_close(got_vals, ref_vals, atol=0, rtol=0)
    assert all(len(set(r.tolist())) == 32 for r in idx)
    if dtype == torch.float32:
        torch.testing.assert_close(idx.long(), torch.topk(scores, 32, dim=-1).indices)


def _reference(hidden, weight, ordering, top, csize, fill):
    positions = ordering.view(-1, csize)[top.long()].flatten(1)
    selected = weight[positions.reshape(-1)].view(
        hidden.shape[0], positions.shape[1], weight.shape[1]
    )
    logits = torch.bmm(hidden.unsqueeze(1), selected.transpose(1, 2)).squeeze(1)
    out = torch.full((hidden.shape[0], weight.shape[0]), fill, dtype=hidden.dtype, device="cuda")
    return out.scatter_(1, positions, logits).float()


@pytest.mark.parametrize("rows", [0, 1, 8, 32])
def test_clustered_logits_matches_gather_bmm_scatter(rows):
    torch.manual_seed(1)
    vocab, hidden, clusters, k = 262144, 256, 2048, 32
    csize = vocab // clusters
    h = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((vocab, hidden), dtype=torch.bfloat16, device="cuda") * 0.05
    ordering = torch.randperm(vocab, device="cuda")
    top = topk_indices(torch.randn((rows, clusters), device="cuda"), k)
    fill = torch.finfo(torch.bfloat16).min
    got = clustered_logits(h, w, ordering, top, csize, fill)
    ref = _reference(h, w, ordering, top, csize, fill)
    assert got.dtype == torch.float32 and got.shape == (rows, vocab)
    # fp32 accumulation in another order than bmm: at most one bf16 ulp apart.
    torch.testing.assert_close(got, ref, atol=0, rtol=2**-7)
    selected = ref > fill
    assert torch.equal(got > fill, selected)
    if rows:
        assert (got.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.95
