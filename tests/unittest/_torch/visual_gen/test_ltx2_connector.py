# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the config-driven LTX-2 text-embedding front-end.

Covers the generalization that lets one code path build both the LTX-2 and
LTX-2.3 connector layouts from the embedded checkpoint config:

- ``Embeddings1DConnectorConfigurator.from_config`` sizing (per-modality: video
  reads ``connector_*``, audio reads ``audio_connector_*`` with fallback).
- ``GemmaFeaturesExtractorProjLinear`` single-shared vs split video/audio layout,
  selected by ``caption_proj_before_connector``.

Modules are built on the ``meta`` device so full-size (188160-fan-in) Linears
allocate no memory — the tests assert only structure/shapes. No CUDA required.
"""

import unittest

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.connector import (
    Embeddings1DConnectorConfigurator,
    GemmaFeaturesExtractorProjLinear,
)

# Representative embedded-config "transformer" sub-dicts, matching the values in
# the real bundled checkpoints (ltx-2-19b-dev / ltx-2.3-22b-dev).
LTX2_CONFIG = {
    "transformer": {
        "rope_type": "split",
        "connector_num_layers": 2,
        "connector_num_attention_heads": 30,
        "connector_attention_head_dim": 128,
        "connector_num_learnable_registers": 128,
        "connector_positional_embedding_max_pos": [4096],
        # No connector_apply_gated_attention, audio_connector_*, or
        # caption_proj_before_connector keys (as in the LTX-2 checkpoint).
    }
}

LTX23_CONFIG = {
    "transformer": {
        "rope_type": "split",
        "connector_num_layers": 8,
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
        "connector_apply_gated_attention": True,
        "connector_num_learnable_registers": 128,
        "connector_positional_embedding_max_pos": [4096],
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
    }
}


def _gate(connector):
    """Return block-0 self-attn gate module (None if gating disabled)."""
    return connector.transformer_1d_blocks[0].attn1.to_gate_logits


class TestConnectorConfigurator(unittest.TestCase):
    def test_ltx2_video_and_audio_identical_3840(self):
        with torch.device("meta"):
            video = Embeddings1DConnectorConfigurator.from_config(LTX2_CONFIG, modality="video")
            audio = Embeddings1DConnectorConfigurator.from_config(LTX2_CONFIG, modality="audio")

        for name, c in (("video", video), ("audio", audio)):
            self.assertEqual(c.inner_dim, 3840, name)  # 30 * 128
            self.assertEqual(len(c.transformer_1d_blocks), 2, name)
            self.assertIsNone(_gate(c), f"{name}: LTX-2 has no gated attention")
            self.assertEqual(tuple(c.learnable_registers.shape), (128, 3840), name)
            # to_q sized to inner_dim -> matches checkpoint [3840, 3840].
            self.assertEqual(
                tuple(c.transformer_1d_blocks[0].attn1.to_q.weight.shape), (3840, 3840), name
            )

    def test_ltx23_video_4096_audio_2048_gated_8layers(self):
        with torch.device("meta"):
            video = Embeddings1DConnectorConfigurator.from_config(LTX23_CONFIG, modality="video")
            audio = Embeddings1DConnectorConfigurator.from_config(LTX23_CONFIG, modality="audio")

        # Video: 32 * 128 = 4096, 8 layers, gated.
        self.assertEqual(video.inner_dim, 4096)
        self.assertEqual(len(video.transformer_1d_blocks), 8)
        self.assertIsNotNone(_gate(video))
        self.assertEqual(tuple(video.learnable_registers.shape), (128, 4096))
        self.assertEqual(
            tuple(video.transformer_1d_blocks[0].attn1.to_q.weight.shape), (4096, 4096)
        )

        # Audio: 32 * 64 = 2048 (audio_connector_* keys), 8 layers (fallback to
        # shared connector_num_layers), gated.
        self.assertEqual(audio.inner_dim, 2048)
        self.assertEqual(len(audio.transformer_1d_blocks), 8)
        self.assertIsNotNone(_gate(audio))
        self.assertEqual(tuple(audio.learnable_registers.shape), (128, 2048))
        self.assertEqual(
            tuple(audio.transformer_1d_blocks[0].attn1.to_q.weight.shape), (2048, 2048)
        )


class TestFeatureExtractor(unittest.TestCase):
    def test_ltx2_single_shared_no_bias(self):
        with torch.device("meta"):
            fe = GemmaFeaturesExtractorProjLinear.from_config(LTX2_CONFIG)
        self.assertFalse(fe.split)
        self.assertTrue(hasattr(fe, "aggregate_embed"))
        self.assertFalse(hasattr(fe, "video_aggregate_embed"))
        self.assertEqual(tuple(fe.aggregate_embed.weight.shape), (3840, 3840 * 49))
        self.assertIsNone(fe.aggregate_embed.bias)

    def test_ltx23_split_video_audio_with_bias(self):
        with torch.device("meta"):
            fe = GemmaFeaturesExtractorProjLinear.from_config(LTX23_CONFIG)
        self.assertTrue(fe.split)
        self.assertFalse(hasattr(fe, "aggregate_embed"))
        # Output dims equal the connector inner dims (4096 video / 2048 audio).
        self.assertEqual(tuple(fe.video_aggregate_embed.weight.shape), (4096, 3840 * 49))
        self.assertEqual(tuple(fe.audio_aggregate_embed.weight.shape), (2048, 3840 * 49))
        self.assertIsNotNone(fe.video_aggregate_embed.bias)
        self.assertIsNotNone(fe.audio_aggregate_embed.bias)
        self.assertEqual(tuple(fe.video_aggregate_embed.bias.shape), (4096,))
        self.assertEqual(tuple(fe.audio_aggregate_embed.bias.shape), (2048,))

    def test_forward_return_shape_contract(self):
        # split=False -> single tensor; split=True -> (video, audio) tuple.
        with torch.device("meta"):
            shared = GemmaFeaturesExtractorProjLinear.from_config(LTX2_CONFIG)
            split = GemmaFeaturesExtractorProjLinear.from_config(LTX23_CONFIG)
            x = torch.empty(1, 8, 3840 * 49)
            out_shared = shared(x)
            out_split = split(x)
        self.assertIsInstance(out_shared, torch.Tensor)
        self.assertEqual(tuple(out_shared.shape), (1, 8, 3840))
        self.assertIsInstance(out_split, tuple)
        self.assertEqual(tuple(out_split[0].shape), (1, 8, 4096))
        self.assertEqual(tuple(out_split[1].shape), (1, 8, 2048))


if __name__ == "__main__":
    unittest.main()
