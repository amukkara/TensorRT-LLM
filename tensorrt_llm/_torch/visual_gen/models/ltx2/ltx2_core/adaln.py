# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Optional, Tuple

import torch

from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings

# Per-block AdaLN slot count. Base: shift/scale/gate for self-attn + FF (6);
# text_cross_attn_adaln adds shift/scale/gate for the text cross-attn norm (+3).
ADALN_NUM_BASE_PARAMS = 6
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(text_cross_attn_adaln: bool) -> int:
    """Total number of AdaLN modulation slots per transformer block."""
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if text_cross_attn_adaln else 0)


class AdaLayerNormSingle(torch.nn.Module):
    """Adaptive layer norm (adaLN-single) from PixArt-Alpha.

    Produces scale/shift/gate modulation parameters from timestep embeddings.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6, make_linear=None):
        super().__init__()
        if make_linear is None:
            make_linear = torch.nn.Linear
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            make_linear=make_linear,
        )
        self.silu = torch.nn.SiLU()
        self.linear = make_linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
