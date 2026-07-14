# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the config-driven LTX-2 / LTX-2.3 video VAE decoder.

Guards three decoder-sizing fixes needed for the LTX-2.3 ``vae`` recipe, which
uses ``compress_space``/``compress_time`` blocks (LTX-2 only uses
``compress_all``):

- conv_in width must include the compress_space/compress_time multipliers.
- compress_space/compress_time must reduce out-channels by their multiplier.
- the padding-mode key is ``spatial_padding_mode`` (LTX-2.3 sets it to ``zeros``).

Decoders are built on the ``meta`` device (no memory / no CUDA) and only shapes
are asserted. Recipes mirror the real bundled checkpoints.
"""

import unittest

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.model_configurator import (
    VideoDecoderConfigurator,
)

# Exact LTX-2.3 vae recipe (from ltx-2.3-22b-dev.safetensors config metadata).
LTX23_VAE = {
    "vae": {
        "dims": 3,
        "latent_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "spatial_padding_mode": "zeros",
        "decoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
        ],
    }
}

# LTX-2-style recipe: only compress_all (no padding key -> default reflect).
LTX2_VAE = {
    "vae": {
        "dims": 3,
        "latent_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "decoder_blocks": [
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
        ],
    }
}


def _conv_padding_mode(module):
    for m in module.modules():
        if hasattr(m, "padding_mode") and isinstance(m.padding_mode, str):
            return m.padding_mode
    return None


class TestVideoDecoderSizing(unittest.TestCase):
    def _build(self, cfg):
        with torch.device("meta"):
            return VideoDecoderConfigurator.from_config(cfg)

    def test_ltx23_conv_in_width_includes_all_compress_multipliers(self):
        # Bug A: 128 * (compress_space 2 * compress_time 2 * compress_all 1 * compress_all 2) = 1024.
        dec = self._build(LTX23_VAE)
        self.assertEqual(dec.conv_in.conv.weight.shape[0], 1024)

    def test_ltx2_conv_in_width_unchanged(self):
        # LTX-2: 128 * (2 * 2 * 2) = 1024 (regression guard; only compress_all).
        dec = self._build(LTX2_VAE)
        self.assertEqual(dec.conv_in.conv.weight.shape[0], 1024)

    def test_ltx23_compress_blocks_reduce_channels(self):
        # Bug B: compress_time/compress_space convs reduce channels by the multiplier
        # (DepthToSpaceUpsample conv out = prod(stride) * in // multiplier), not blow up.
        dec = self._build(LTX23_VAE)
        conv_shapes = [
            b.conv.weight.shape
            for b in dec.up_blocks
            if hasattr(b, "conv") and hasattr(b.conv, "weight")
        ]
        # compress_time (stride prod 2, in 512, mult 2) -> [512, 512]
        # compress_space (stride prod 4, in 256, mult 2) -> [512, 256]
        self.assertIn(torch.Size([512, 512, 3, 3, 3]), conv_shapes)
        self.assertIn(torch.Size([512, 256, 3, 3, 3]), conv_shapes)

    def test_padding_mode_key(self):
        # Bug C: read `spatial_padding_mode`; LTX-2.3 -> zeros, LTX-2 default -> reflect.
        self.assertEqual(_conv_padding_mode(self._build(LTX23_VAE)), "zeros")
        self.assertEqual(_conv_padding_mode(self._build(LTX2_VAE)), "reflect")


if __name__ == "__main__":
    unittest.main()
