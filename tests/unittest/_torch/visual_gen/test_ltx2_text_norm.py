# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Numerical tests for the config-driven LTX-2 / LTX-2.3 text-embedding norm.

``LTX2Pipeline._pack_text_embeds`` normalizes the stacked Gemma hidden states
before the feature-extractor projection. LTX-2 uses ``min_max`` (mean-center +
masked min-max range, x8); LTX-2.3 declares ``per_token_rms`` (per-token
RMSNorm over the hidden dim within each layer). LTX-2.3 also rescales the
per-token-RMS features by ``sqrt(out_features / embedding_dim)`` inside each
per-modality projection (``GemmaFeaturesExtractorProjLinear``).

These tests pin both norm branches and the rescale against standalone reference
implementations. CPU-only, tiny dims — no GPU or checkpoint required.
"""

import math
import unittest

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.connector import (
    GemmaFeaturesExtractorProjLinear,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline


def _ref_per_token_rms(encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Reference per-token RMSNorm over the hidden dim, padding zeroed."""
    b, t, d, n_layers = encoded.shape
    variance = torch.mean(encoded**2, dim=2, keepdim=True)
    normed = encoded * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(b, t, d * n_layers)
    mask = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask, normed, torch.zeros_like(normed))


def _ref_min_max(encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Reference masked mean-center + min-max range scaling (x8), padding zeroed."""
    b, _, d, n_layers = encoded.shape
    eps = 1e-6
    seq_len = attention_mask.sum(-1)
    mask = attention_mask.bool()[:, :, None, None]
    masked = encoded.masked_fill(~mask, 0.0)
    denom = (seq_len * d).view(b, 1, 1, 1)
    mean = masked.sum((1, 2), keepdim=True) / (denom + eps)
    x_min = encoded.masked_fill(~mask, float("inf")).amin((1, 2), keepdim=True)
    x_max = encoded.masked_fill(~mask, float("-inf")).amax((1, 2), keepdim=True)
    normed = 8 * (encoded - mean) / ((x_max - x_min) + eps)
    normed = normed.reshape(b, -1, d * n_layers)
    mask_flat = mask.squeeze(-1).expand(-1, -1, d * n_layers)
    return normed.masked_fill(~mask_flat, 0.0)


def _mask_to_lengths(attention_mask: torch.Tensor):
    return attention_mask.sum(-1)


class TestPackTextEmbedsNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.D, self.L = 3, 7, 5, 4
        self.x = torch.randn(self.B, self.T, self.D, self.L, dtype=torch.float32)

    def _run(self, attention_mask, padding_side, norm_type):
        return LTX2Pipeline._pack_text_embeds(
            self.x,
            _mask_to_lengths(attention_mask),
            device="cpu",
            padding_side=padding_side,
            norm_type=norm_type,
        )

    def test_per_token_rms_matches_reference_right_pad(self):
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        attn[0, 5:] = 0  # right padding
        attn[1, 4:] = 0
        out = self._run(attn, "right", "per_token_rms")
        self.assertTrue(torch.equal(out, _ref_per_token_rms(self.x, attn)))

    def test_per_token_rms_matches_reference_left_pad(self):
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        attn[0, :2] = 0  # left padding
        attn[1, :3] = 0
        out = self._run(attn, "left", "per_token_rms")
        self.assertTrue(torch.equal(out, _ref_per_token_rms(self.x, attn)))

    def test_min_max_matches_reference_right_pad(self):
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        attn[0, 5:] = 0
        attn[1, 4:] = 0
        out = self._run(attn, "right", "min_max")
        self.assertTrue(torch.allclose(out, _ref_min_max(self.x, attn), atol=1e-6))

    def test_padding_positions_are_zeroed(self):
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        attn[0, 5:] = 0
        for norm_type in ("per_token_rms", "min_max"):
            out = self._run(attn, "right", norm_type)
            self.assertTrue(torch.all(out[0, 5:] == 0.0), f"{norm_type}: padded rows must be zero")
            self.assertTrue(
                torch.any(out[0, :5] != 0.0), f"{norm_type}: valid rows must be non-zero"
            )

    def test_unknown_norm_type_raises(self):
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        with self.assertRaises(ValueError):
            self._run(attn, "right", "bogus")

    def test_norm_types_differ(self):
        # The two schemes are genuinely different transforms.
        attn = torch.ones(self.B, self.T, dtype=torch.long)
        rms = self._run(attn, "right", "per_token_rms")
        mm = self._run(attn, "right", "min_max")
        self.assertFalse(torch.allclose(rms, mm))


class _TinyFeatureExtractor(GemmaFeaturesExtractorProjLinear):
    # Small fan-in / embedding dim so the split Linears are cheap to allocate.
    EMBEDDING_DIM = 4
    IN_FEATURES = 8  # EMBEDDING_DIM * 2 "layers"


class TestFeatureExtractorRescale(unittest.TestCase):
    def test_rescale_factor(self):
        with torch.device("meta"):
            fe = GemmaFeaturesExtractorProjLinear.from_config(
                {"transformer": {"caption_proj_before_connector": True}}
            )
        x = torch.randn(2, 3, 7)
        for out_features in (4096, 2048):
            expected = x * math.sqrt(out_features / fe.EMBEDDING_DIM)
            self.assertTrue(torch.equal(fe._rescale(x, out_features), expected))

    def test_split_forward_folds_rescale_before_projection(self):
        torch.manual_seed(0)
        fe = _TinyFeatureExtractor(split=True, video_dim=6, audio_dim=3).eval()
        x = torch.randn(2, 4, fe.IN_FEATURES)
        with torch.no_grad():
            video, audio = fe(x)
            expected_video = fe.video_aggregate_embed(x * math.sqrt(6 / fe.EMBEDDING_DIM))
            expected_audio = fe.audio_aggregate_embed(x * math.sqrt(3 / fe.EMBEDDING_DIM))
        self.assertTrue(torch.allclose(video, expected_video, atol=1e-6))
        self.assertTrue(torch.allclose(audio, expected_audio, atol=1e-6))

    def test_single_shared_forward_has_no_rescale(self):
        # LTX-2 (split=False) projects the raw features with no rescale.
        torch.manual_seed(0)
        fe = _TinyFeatureExtractor(split=False).eval()
        x = torch.randn(2, 4, fe.IN_FEATURES)
        with torch.no_grad():
            out = fe(x)
            expected = fe.aggregate_embed(x)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
