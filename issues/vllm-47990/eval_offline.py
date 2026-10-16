"""GSM8K accuracy and generation throughput through vLLM's offline engine, so a variant's kernel routing (its patches
and VLLM_DISABLED_KERNELS) is measured on the whole model, not only on one GEMM.

GSM8K is 5-shot, greedy, over the full 1319-question test set, with the prompt format of SGLang's few_shot_gsm8k,
scored on the last number in the completion. Throughput uses random token prompts with ignore_eos, so every variant
decodes exactly the same number of tokens, and reports the median of 3 runs per shape.
Usage: python3 eval_offline.py <model> <revision> <gsm8k_dir> <out.json>
"""

import json
import random
import re
import statistics
import sys
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

SHOTS = 5


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def last_number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k(llm, data_dir):
    train = load_jsonl(f"{data_dir}/train.jsonl")[:SHOTS]
    test = load_jsonl(f"{data_dir}/test.jsonl")
    shots = "".join(f"Question: {q['question']}\nAnswer: {q['answer']}\n\n" for q in train)
    prompts = [f"{shots}Question: {q['question']}\nAnswer:" for q in test]
    params = SamplingParams(temperature=0, max_tokens=512, stop=["Question:", "\n\n\n"])
    start = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.perf_counter() - start
    correct = invalid = 0
    for q, out in zip(test, outs):
        got = last_number(out.outputs[0].text)
        want = last_number(q["answer"].split("####")[-1])
        invalid += got is None
        correct += got is not None and want is not None and abs(got - want) < 1e-6
    return {"questions": len(test), "accuracy": correct / len(test), "invalid": invalid, "seconds": seconds}


def throughput(llm, batch, input_len, output_len, repeats=3):
    # Seeded per shape so every variant gets the same prompts. Ids under 30000 are ordinary tokens in DeepSeek's
    # 102400 vocab, away from its special tokens at 100000 and up.
    rng = random.Random(batch * 100003 + input_len)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(input_len)])
               for _ in range(batch)]
    params = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    rates = []
    for _ in range(repeats):
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        rates.append(sum(len(o.outputs[0].token_ids) for o in outs) / seconds)
    return {"batch": batch, "input_len": input_len, "output_len": output_len, "output_tok_s": statistics.median(rates),
            "runs_tok_s": rates}


def main():
    model, revision, data_dir, out_path = sys.argv[1:5]
    # Prefix caching off: with it, repeated throughput prompts would skip prefill and the runs would measure the cache.
    llm = LLM(model=model, revision=revision, max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=False)
    result = {"model": model, "revision": revision, "gsm8k": gsm8k(llm, data_dir), "throughput": [
        throughput(llm, 1, 512, 512),
        throughput(llm, 64, 1024, 256),
        throughput(llm, 256, 1024, 256),
    ]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
