#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Profile Wan 2.1 text-to-video generation with torch.profiler, nsys, or ncu.

Examples:
    # 1. Torch profiler (writes a Chrome trace JSON):
    python visual_gen_wan_t2v_profile.py \\
        --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
        --prompt "A red panda dancing in the snow" \\
        --profiler torch --trace_path wan21_torch_trace.json

    # 2. Nsight Systems / Compute (script emits NVTX + cudaProfilerStart/Stop;
    #    works for both nsys and ncu via --capture-range=cudaProfilerApi):
    nsys profile -o wan21_nsys --capture-range=cudaProfilerApi \\
        --trace=cuda,nvtx,osrt,cudnn,cublas --force-overwrite=true \\
        python visual_gen_wan_t2v_profile.py \\
        --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
        --prompt "A red panda dancing in the snow" \\
        --profiler nsight

    ncu --target-processes all --capture-range cudaProfilerApi \\
        --kernel-name regex:'nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT' \\
        --launch-skip 100 --launch-count 1 --set full --import-source yes \\
        -o nvjet_prof --force-overwrite \\
        python visual_gen_wan_t2v_profile.py --profiler nsight
"""

import argparse
import os
import time

import torch
import torch.cuda.profiler as cuda_profiler

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger

logger.set_level("info")


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile Wan T2V generation (torch profiler, nsys, or ncu)."
    )

    p.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument(
        "--prompt", type=str, default="A red panda dancing in the snow, cinematic lighting"
    )
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--output_path", type=str, default="wan21_profile_output.mp4")

    # Keep these small by default to make profile traces manageable.
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # Attention / compile knobs.
    p.add_argument(
        "--attention_backend", type=str, default="TRTLLM", choices=["VANILLA", "TRTLLM", "FA4"]
    )
    p.add_argument("--disable_torch_compile", action="store_true")
    p.add_argument("--enable_cudagraph", action="store_true")
    p.add_argument("--disable_autotune", action="store_true")

    # Profiler selection.
    p.add_argument(
        "--profiler",
        type=str,
        default="torch",
        choices=["none", "torch", "nsight"],
        help="'nsight' emits NVTX + cudaProfilerStart/Stop for both nsys and ncu.",
    )
    p.add_argument(
        "--trace_path",
        type=str,
        default="wan21_torch_trace.json",
        help="Chrome trace output path for --profiler torch",
    )
    p.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip the warmup generation (profile will include compile/autotune cost).",
    )

    return p.parse_args()


def build_visual_gen(args) -> VisualGen:
    # Worker subprocess reads these env vars in serve_forever to attach
    # torch.profiler. The env propagates through mp.spawn since we set it
    # before constructing VisualGen.
    if args.profiler == "torch":
        os.environ["TLLM_VISUAL_GEN_TORCH_PROFILE_TRACE"] = args.trace_path
        # Skip the first request (= our warmup) so the trace captures the
        # second request without compile/autotune overhead.
        os.environ["TLLM_VISUAL_GEN_TORCH_PROFILE_SKIP"] = "0" if args.skip_warmup else "1"
    visual_gen_args = VisualGenArgs(
        attention_config={"backend": args.attention_backend},
        parallel_config={
            "cfg_size": 1,
            "ulysses_size": 1,
            "ring_size": 1,
            "attn2d_size": (1, 1),
            "parallel_vae_size": 1,
        },
        torch_compile_config={
            "enable": not args.disable_torch_compile,
            "enable_fullgraph": False,
            "enable_autotune": not args.disable_autotune,
        },
        cuda_graph_config={"enable": args.enable_cudagraph},
        # Marker pass adds NVTX ranges per layer -- very useful for nsys/ncu.
        enable_layerwise_nvtx_marker=(args.profiler == "nsight"),
    )
    return VisualGen(model=args.model_path, args=visual_gen_args)


def run_generate(visual_gen: VisualGen, args) -> object:
    return visual_gen.generate(
        inputs=args.prompt,
        params=VisualGenParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_frames=args.num_frames,
            negative_prompt=args.negative_prompt,
        ),
    )


def profile_torch(visual_gen: VisualGen, args):
    # The actual capture happens in the worker subprocess (see executor.py
    # serve_forever, gated by TLLM_VISUAL_GEN_TORCH_PROFILE_TRACE). Parent
    # just issues the generate call; worker exports the Chrome trace.
    logger.info(f"torch.profiler: worker will capture into {args.trace_path}")
    output = run_generate(visual_gen, args)
    torch.cuda.synchronize()
    return output


def profile_nsight(visual_gen: VisualGen, args):
    logger.info("nsight mode: emitting cudaProfilerStart + NVTX range 'wan_t2v_generate'")
    torch.cuda.synchronize()
    cuda_profiler.start()
    torch.cuda.nvtx.range_push("wan_t2v_generate")
    try:
        output = run_generate(visual_gen, args)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        cuda_profiler.stop()
    return output


def main():
    args = parse_args()
    visual_gen = build_visual_gen(args)
    try:
        if not args.skip_warmup:
            logger.info("Warmup generation (compile + autotune; not profiled)")
            t0 = time.time()
            run_generate(visual_gen, args)
            torch.cuda.synchronize()
            logger.info(f"Warmup done in {time.time() - t0:.2f}s")

        logger.info(f"Profiled generation ({args.profiler})")
        t0 = time.time()
        if args.profiler == "torch":
            output = profile_torch(visual_gen, args)
        elif args.profiler == "nsight":
            output = profile_nsight(visual_gen, args)
        else:
            output = run_generate(visual_gen, args)
            torch.cuda.synchronize()
        logger.info(f"Generation in {time.time() - t0:.2f}s")

        output.save(args.output_path)
        logger.info(f"Saved output to {args.output_path}")
    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
