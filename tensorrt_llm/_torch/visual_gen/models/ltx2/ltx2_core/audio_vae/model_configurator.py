# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from ..normalization import NormType
from .audio_vae import AudioDecoder
from .causality_axis import CausalityAxis
from .vocoder import MelSTFT, Vocoder, VocoderWithBWE


def _vocoder_from_config(
    cfg: dict,
    apply_final_activation: bool = True,
    output_sample_rate: int | None = None,
) -> Vocoder:
    """Build a plain Vocoder from a flat config sub-dict."""
    return Vocoder(
        resblock_kernel_sizes=cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        upsample_rates=cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=cfg.get(
            "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        ),
        upsample_initial_channel=cfg.get("upsample_initial_channel", 1024),
        stereo=cfg.get("stereo", True),
        resblock=cfg.get("resblock", "1"),
        output_sample_rate=(
            output_sample_rate
            if output_sample_rate is not None
            else cfg.get("output_sampling_rate", cfg.get("output_sample_rate", 24000))
        ),
        activation=cfg.get("activation", "snake"),
        use_tanh_at_final=cfg.get("use_tanh_at_final", True),
        apply_final_activation=apply_final_activation,
        use_bias_at_final=cfg.get("use_bias_at_final", True),
    )


class VocoderConfigurator:
    """Build the vocoder, auto-detecting the checkpoint format.

    LTX-2 has a flat ``vocoder`` config (HiFi-GAN) -> plain ``Vocoder``.
    LTX-2.3 nests ``vocoder`` + ``bwe`` sub-dicts (BigVGAN-v2 + bandwidth
    extension) -> ``VocoderWithBWE``.
    """

    @classmethod
    def from_config(cls, config: dict) -> Vocoder | VocoderWithBWE:
        cfg = config.get("vocoder", {})
        if "bwe" not in cfg:
            return _vocoder_from_config(cfg)

        vocoder_cfg = cfg.get("vocoder", {})
        bwe_cfg = cfg["bwe"]
        vocoder = _vocoder_from_config(
            vocoder_cfg,
            output_sample_rate=bwe_cfg["input_sampling_rate"],
        )
        bwe_generator = _vocoder_from_config(
            bwe_cfg,
            apply_final_activation=False,
            output_sample_rate=bwe_cfg["output_sampling_rate"],
        )
        mel_stft = MelSTFT(
            filter_length=bwe_cfg["n_fft"],
            hop_length=bwe_cfg["hop_length"],
            win_length=bwe_cfg["n_fft"],
            n_mel_channels=bwe_cfg["num_mels"],
        )
        return VocoderWithBWE(
            vocoder=vocoder,
            bwe_generator=bwe_generator,
            mel_stft=mel_stft,
            input_sampling_rate=bwe_cfg["input_sampling_rate"],
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
            hop_length=bwe_cfg["hop_length"],
        )


class AudioDecoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> AudioDecoder:
        audio_vae_cfg = config.get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {})
        model_params = model_cfg.get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = (
            ddconfig.get("mel_bins")
            or mel_cfg.get("n_mel_channels")
            or variables_cfg.get("mel_bins")
        )

        return AudioDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=ddconfig.get("attn_resolutions", {8, 16, 32}),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            dropout=ddconfig.get("dropout", 0.0),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )
