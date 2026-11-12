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
"""Clustered-vocabulary logits (Triton): logits only for the tokens of the top-scoring clusters.

Replaces ``topk`` + gather + ``bmm`` + ``scatter_``: one kernel picks the top-``K`` clusters per
row, a second writes each selected token's logit to its vocabulary position (others get a fill
value). A logit can differ from the bf16 ``bmm`` by one ulp; ties at the ``K``-th score go to the
lower cluster index.
"""

import torch
import triton
import triton.language as tl

# Tokens per program of the logits kernel; with 4 warps it reads a 32 x H tile of embedding rows.
_BLOCK_TOKENS = 32


@triton.jit
def _topk_indices_kernel(
    scores_ptr,  # [R, C] float, row stride SS
    out_ptr,  # [R, K] int32
    SS,
    C: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, C)
    s = tl.load(scores_ptr + row * SS + cols).to(tl.float32)
    # Order-preserving int32 image of the fp32 score (negative values reverse their bits).
    bits = s.to(tl.int32, bitcast=True)
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    # Pack the score above the reversed index: a descending sort ranks equal scores by lower index.
    packed = (key << 32) | (C - 1 - cols).to(tl.int64)
    ranked = tl.sort(packed, descending=True)
    idx = (C - 1) - (ranked & 0xFFFFFFFF).to(tl.int32)
    tl.store(out_ptr + row * K + cols, idx, mask=cols < K)


@triton.jit
def _clustered_logits_kernel(
    hidden_ptr,  # [R, H], row stride SH
    weight_ptr,  # [V, H], row stride SW
    ordering_ptr,  # [V] int64: token ids, cluster by cluster
    top_ptr,  # [R, K] int32 selected clusters
    out_ptr,  # [R, V] fp32, row stride SO
    SH,
    SW,
    SO,
    K: tl.constexpr,
    CSIZE: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    cluster = tl.load(top_ptr + row * K + slot).to(tl.int64)
    j = tl.program_id(2) * BT + tl.arange(0, BT)
    tokens = tl.load(ordering_ptr + cluster * CSIZE + j)
    d = tl.arange(0, H)
    h = tl.load(hidden_ptr + row * SH + d).to(tl.float32)
    w = tl.load(weight_ptr + tokens[:, None] * SW + d[None, :]).to(tl.float32)
    logit = tl.sum(w * h[None, :], axis=1)
    logit = logit.to(weight_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row.to(tl.int64) * SO + tokens, logit)


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest entries of each row of ``scores`` [R, C] (C a power of two),
    largest first, ties to the lower index. Returns [R, k] int32."""
    assert scores.dim() == 2 and scores.stride(-1) == 1
    rows, c = scores.shape
    assert c & (c - 1) == 0 and 0 < k <= c
    out = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
    if rows:
        _topk_indices_kernel[(rows,)](scores, out, scores.stride(0), C=c, K=k, num_warps=8)
    return out


def clustered_logits(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    token_ordering: torch.Tensor,
    top_clusters: torch.Tensor,
    cluster_size: int,
    fill_value: float,
) -> torch.Tensor:
    """Full-vocabulary fp32 logits where only the tokens of the selected clusters are computed.

    Args:
        hidden_states: [R, H] (H a power of two), the head input.
        weight: [V, H] output embedding, same dtype as ``hidden_states``.
        token_ordering: [V] int64 token ids listed cluster by cluster.
        top_clusters: [R, K] int32 selected cluster indices.
        cluster_size: tokens per cluster (a power of two).
        fill_value: logit of every token outside the selected clusters.

    Returns:
        [R, V] float32: ``hidden_states[r] . weight[t]`` (rounded to the embedding dtype) for each
        token ``t`` of row ``r``'s selected clusters, ``fill_value`` elsewhere.
    """
    rows, hidden = hidden_states.shape
    vocab = weight.shape[0]
    k = top_clusters.shape[1]
    assert weight.shape[1] == hidden and hidden & (hidden - 1) == 0
    assert cluster_size & (cluster_size - 1) == 0 and vocab % cluster_size == 0
    assert hidden_states.stride(-1) == 1 and weight.stride(-1) == 1
    assert token_ordering.dtype == torch.int64 and token_ordering.numel() == vocab
    out = torch.full((rows, vocab), fill_value, dtype=torch.float32, device=weight.device)
    if rows:
        bt = min(cluster_size, _BLOCK_TOKENS)
        _clustered_logits_kernel[(rows, k, cluster_size // bt)](
            hidden_states,
            weight,
            token_ordering,
            top_clusters,
            out,
            hidden_states.stride(0),
            weight.stride(0),
            out.stride(0),
            K=k,
            CSIZE=cluster_size,
            H=hidden,
            BT=bt,
            num_warps=4,
        )
    return out
