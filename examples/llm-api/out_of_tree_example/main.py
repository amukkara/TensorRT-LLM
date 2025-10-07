import asyncio
import random
import time
from concurrent.futures import ProcessPoolExecutor

import modeling_qwen  # noqa

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import TorchCompileConfig


def warmup(llm, prompt_ids, sampling_params):
    for _ in range(3):
        llm.generate(prompt_ids, sampling_params)


def print_time(exp, iters, num_prompts, start, end):
    ms = 1000 * (end - start) / iters
    print(f"[{exp}] Avg. time for {iters=} and {num_prompts=}: {ms} ms")


def test_generate(llm, prompt_ids, sampling_params, iters):
    start = time.time()
    for _ in range(iters):
        llm.generate(prompt_ids, sampling_params)
    end = time.time()
    print_time("generate", iters, len(prompt_ids), start, end)


def test_generate_async(llm, batch_prompt_ids, sampling_params, iters):
    start = time.time()
    for _ in range(iters):
        tasks = [
            llm.generate_async(prompt_ids, sampling_params)
            for prompt_ids in batch_prompt_ids
        ]
        [task.result() for task in tasks]
    end = time.time()
    print_time("generate_async", iters, len(batch_prompt_ids), start, end)


async def submit_prompt(llm, prompt_ids, sampling_params):
    await llm.generate_async(prompt_ids, sampling_params)


def test_parallel_submit(llm, batch_prompt_ids, sampling_params, iters):

    async def submit_loop():
        tasks = [
            submit_prompt(llm, prompt_ids, sampling_params)
            for prompt_ids in batch_prompt_ids
        ]
        await asyncio.gather(*tasks)

    start = time.time()
    for _ in range(iters):
        asyncio.run(submit_loop())
    end = time.time()
    print_time("parallel_submit", iters, len(batch_prompt_ids), start, end)
    return

    with ProcessPoolExecutor(max_workers=2) as pool:
        start = time.time()
        for _ in range(iters):
            futures = []
            for prompt_ids in batch_prompt_ids:
                future = pool.submit(submit_prompt, llm, prompt_ids,
                                     sampling_params)
                futures.append(future)

            for future in futures:
                output = future.result()
                output = output.result()

        end = time.time()
        print_time("parallel_submit", iters, len(batch_prompt_ids), start, end)


def main():
    prompt_len = 512
    batch_size = 32

    llm = LLM(
        model=
        '/home/amukkara/scratch/datasets/trt-llm/hf_models/Qwen/Qwen2.5-0.5B-Instruct-FP8',
        print_iter_log=True,
        disable_overlap_scheduler=True,
        max_batch_size=batch_size,
        max_num_tokens=prompt_len * batch_size,
        torch_compile_config=TorchCompileConfig(
            enable_fullgraph=True,
            enable_inductor=True,
            enable_piecewise_cuda_graph=False),
        batch_wait_timeout_ms=40,
    )

    sampling_params = SamplingParams(max_tokens=1, return_context_logits=True)

    prompt_ids = []
    for bi in range(batch_size):
        prompt_ids.append(list(range(bi * prompt_len, (bi + 1) * prompt_len)))
        random.shuffle(prompt_ids[-1])

    outputs = llm.generate(prompt_ids, sampling_params)

    scores = outputs[0].context_logits
    print("Scores", scores.shape)
    assert scores.shape == (1, 256)
    assert not outputs[0].outputs[0].text

    warmup(llm, prompt_ids, sampling_params)

    iters = 5
    test_generate(llm, prompt_ids, sampling_params, iters)


if __name__ == '__main__':
    main()
