# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the config-driven LTX-2 / LTX-2.3 vocoder.

LTX-2 uses a flat ``vocoder`` config (HiFi-GAN, resblock "1") -> plain
``Vocoder``. LTX-2.3 nests ``vocoder`` + ``bwe`` sub-dicts (BigVGAN-v2 AMP1 +
bandwidth extension) -> ``VocoderWithBWE``, which upsamples the waveform to a
higher rate. Uses small channel counts so both paths run on CPU quickly.
"""

import unittest
import warnings

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.model_configurator import (
    VocoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.resnet import ResBlock1
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.vocoder import (
    AMPBlock1,
    Vocoder,
    VocoderWithBWE,
)

# Small flat HiFi-GAN config (LTX-2 shape, reduced channels).
LTX2_VOCODER = {
    "vocoder": {
        "resblock": "1",
        "stereo": True,
        "upsample_initial_channel": 32,
        "upsample_rates": [2, 2],
        "upsample_kernel_sizes": [4, 4],
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
    }
}

# Small nested AMP1 + BWE config (LTX-2.3 shape, reduced channels).
_AMP = {
    "resblock": "AMP1",
    "stereo": True,
    "activation": "snakebeta",
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "upsample_initial_channel": 32,
    "upsample_rates": [2, 2],
    "upsample_kernel_sizes": [4, 4],
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
}
# The BWE residual (bwe_generator output) and the sinc-resampled skip only align
# when prod(bwe_upsample_rates) == (out_sr/in_sr) * hop_length (real config:
# 240 == 3 * 80). Keep that invariant with small numbers: 12 == 3 * 4.
LTX23_VOCODER = {
    "vocoder": {
        "vocoder": dict(_AMP),
        "bwe": dict(
            _AMP,
            upsample_rates=[6, 2],
            upsample_kernel_sizes=[12, 4],
            apply_final_activation=False,
            input_sampling_rate=16000,
            output_sampling_rate=48000,
            hop_length=4,
            n_fft=8,
            num_mels=64,
        ),
    }
}


class TestVocoderConfig(unittest.TestCase):
    def test_ltx2_builds_plain_hifigan(self):
        voc = VocoderConfigurator.from_config(LTX2_VOCODER)
        self.assertIsInstance(voc, Vocoder)
        self.assertFalse(voc.is_amp)
        self.assertIsInstance(voc.resblocks[0], ResBlock1)
        self.assertFalse(hasattr(voc, "bwe_generator"))

    def test_ltx23_builds_vocoder_with_bwe(self):
        voc = VocoderConfigurator.from_config(LTX23_VOCODER)
        self.assertIsInstance(voc, VocoderWithBWE)
        self.assertTrue(voc.vocoder.is_amp)
        self.assertTrue(voc.bwe_generator.is_amp)
        self.assertIsInstance(voc.vocoder.resblocks[0], AMPBlock1)
        self.assertTrue(hasattr(voc, "mel_stft"))
        self.assertEqual(voc.output_sampling_rate, 48000)
        # bwe_generator has no final activation (predicts a residual).
        self.assertFalse(voc.bwe_generator.apply_final_activation)


class TestVocoderForward(unittest.TestCase):
    def _run(self, voc, T=16, mel=64):
        x = torch.randn(1, 2, T, mel) * 0.1
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # CPU autocast-fp32 is a no-op warning
            return voc(x)

    def test_ltx2_forward_shape(self):
        voc = VocoderConfigurator.from_config(LTX2_VOCODER).eval()
        out = self._run(voc)
        # stereo out, length = frames * prod(upsample_rates) = 16 * 4
        self.assertEqual(out.shape[:2], (1, 2))
        self.assertEqual(out.shape[2], 16 * 4)

    def test_ltx23_bwe_upsamples_rate(self):
        voc = VocoderConfigurator.from_config(LTX23_VOCODER).eval()
        base = VocoderConfigurator.from_config(
            {"vocoder": LTX23_VOCODER["vocoder"]["vocoder"]}
        ).eval()
        base_len = self._run(base).shape[2]
        out = self._run(voc)
        # BWE resamples 16 kHz -> 48 kHz, i.e. 3x the base length.
        self.assertEqual(out.shape[:2], (1, 2))
        self.assertEqual(out.shape[2], base_len * 48000 // 16000)


if __name__ == "__main__":
    unittest.main()
