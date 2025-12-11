import random

from tensorrt_llm import LLM, SamplingParams


def main():
    prompt_len = 512
    batch_size = 4

    llm = LLM(
        model="/home/amukkara/scratch/datasets/trt-llm/hf_models/Qwen/Qwen3-Embedding-0.6B",
        disable_overlap_scheduler=True,
        max_batch_size=batch_size,
        max_num_tokens=prompt_len * batch_size,
    )

    sampling_params = SamplingParams(max_tokens=1, return_context_logits=True)

    prompt_ids = []
    for bi in range(batch_size):
        prompt_ids.append(list(range(bi * prompt_len, (bi + 1) * prompt_len)))
        random.shuffle(prompt_ids[-1])

    outputs = llm.generate(prompt_ids, sampling_params)

    scores = outputs[0].context_logits
    print(f"{scores.shape=}")
    hidden_size = llm._hf_model_config.hidden_size
    assert scores.shape == (1, hidden_size)
    assert not outputs[0].outputs[0].text


if __name__ == "__main__":
    main()
