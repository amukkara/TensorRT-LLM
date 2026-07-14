# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end weight-load + forward test for LTX-2 and LTX-2.3 checkpoints.

Loads the diffusion transformer + text connectors + feature extractor from a
bundled ``.safetensors`` checkpoint (skipping the text encoder, VAE, audio VAE
and vocoder), asserts the transformer / connector / text-projection weights load
with **no** missing or unexpected keys, then runs a forward pass through the real
feature-extractor -> connector -> transformer path.

This is the regression guard that both the LTX-2 and LTX-2.3 text-conditioning
front-ends (config-driven connector + AdaLN sizing) load and run without error.

Requires a GPU and the bundled checkpoints; skipped otherwise:
- LTX-2:   ``<LLM_MODELS_ROOT>/LTX-2/ltx-2-19b-dev.safetensors``
- LTX-2.3: env ``LTX2_3_MODEL_PATH`` (no standard models-root location yet)
"""

import logging
import os

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

# Keep transformer + connectors + feature extractor; skip everything else.
# (LTX-2.3's VAE/audio-VAE/vocoder are not ported yet, so they must be skipped.)
_SKIP = [
    PipelineComponent.TEXT_ENCODER,
    PipelineComponent.TOKENIZER,
    PipelineComponent.VAE,
    PipelineComponent.SCHEDULER,
    "audio_vae",
    "vocoder",
    "video_encoder",
]

# Prefixes whose weights must load 1:1 (no missing / unexpected keys).
_KEY_PREFIXES = ("diffusion_model", "text_embedding_projection", "embeddings_connector")

_IN_FEATURES = 3840 * 49  # packed Gemma feature width the feature extractor consumes


def _ltx2_bf16_path():
    try:
        from test_common.llm_data import llm_models_root

        return os.path.join(str(llm_models_root(check=True)), "LTX-2", "ltx-2-19b-dev.safetensors")
    except Exception:
        return None


def _checkpoints():
    out = []
    p2 = _ltx2_bf16_path()
    if p2 and os.path.exists(p2):
        out.append(pytest.param(p2, id="ltx-2"))
    p23 = os.environ.get("LTX2_3_MODEL_PATH")
    if p23 and os.path.exists(p23):
        out.append(pytest.param(p23, id="ltx-2.3"))
    return out


class _KeyWarningGrabber(logging.Handler):
    """Collects weight-load 'missing'/'unexpected' key warnings."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "missing" in msg.lower() or "unexpected" in msg.lower():
            self.messages.append(msg)


def _make_positions(batch, grid, device):
    """1-based start/end position grid of shape (B, n_dims, T, 2)."""
    n_dims = len(grid)
    n = 1
    for g in grid:
        n *= g
    pos = torch.zeros(batch, n_dims, n, 2, device=device)
    coords = [0] * n_dims
    for t in range(n):
        for d in range(n_dims):
            pos[:, d, t, :] = torch.tensor([coords[d], coords[d] + 1.0])
        # increment mixed-radix counter
        for d in reversed(range(n_dims)):
            coords[d] += 1
            if coords[d] < grid[d]:
                break
            coords[d] = 0
    return pos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "checkpoint",
    _checkpoints()
    or [pytest.param(None, marks=pytest.mark.skip(reason="no LTX-2/LTX-2.3 checkpoint available"))],
)
def test_ltx2_load_and_forward(checkpoint):
    device, dtype = "cuda", torch.bfloat16

    grabber = _KeyWarningGrabber()
    root_logger = logging.getLogger()
    root_logger.addHandler(grabber)
    try:
        args = VisualGenArgs(model=checkpoint, pipeline_config={"text_encoder_path": None})
        pipeline = PipelineLoader(args).load(skip_warmup=True, skip_components=_SKIP)
    finally:
        root_logger.removeHandler(grabber)

    # (1) Weights loaded correctly: no missing/unexpected keys for our prefixes.
    offending = [m for m in grabber.messages if any(p in m for p in _KEY_PREFIXES)]
    assert not offending, "weight-load key mismatch:\n" + "\n".join(offending)

    transformer = pipeline.transformer

    # (2) Forward via the real feature-extractor -> connector -> transformer path.
    batch, text_len = 1, 128  # text_len must be a multiple of the 128 learnable registers
    prompt_embeds = torch.randn(batch, text_len, _IN_FEATURES, device=device, dtype=dtype) * 0.02
    attn_mask = torch.ones(batch, text_len, device=device, dtype=dtype)
    video_ctx, audio_ctx, _ = pipeline._process_connectors(prompt_embeds, attn_mask)

    cfg = getattr(transformer, "_transformer_config", {})
    in_ch = cfg.get("in_channels", 128)
    a_in_ch = cfg.get("audio_in_channels", 128)
    v_grid, a_patches = (1, 4, 4), 8
    v_patches = v_grid[0] * v_grid[1] * v_grid[2]
    v_pos = _make_positions(batch, v_grid, device)
    a_pos = _make_positions(batch, (a_patches,), device)

    video = Modality(
        latent=torch.randn(batch, v_patches, in_ch, device=device, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=v_pos,
        context=video_ctx,
    )
    audio = Modality(
        latent=torch.randn(batch, a_patches, a_in_ch, device=device, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=a_pos,
        context=audio_ctx,
    )
    text_cache = transformer.prepare_text_cache(
        video_context=video_ctx,
        video_positions=v_pos,
        audio_context=audio_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )
    with torch.no_grad():
        video_out, audio_out = transformer(video=video, audio=audio, text_cache=text_cache)

    out_ch = cfg.get("out_channels", 128)
    a_out_ch = cfg.get("audio_out_channels", 128)
    assert tuple(video_out.shape) == (batch, v_patches, out_ch)
    assert tuple(audio_out.shape) == (batch, a_patches, a_out_ch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
