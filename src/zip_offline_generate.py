#!/usr/bin/env python3
"""
Stage 1: Unpruned offline generation for ZIP.

- No logits processors, no pruning.
- Generates repeated batches of reasoning / non-reasoning samples per prompt.
- Writes a single Parquet (plus shards) with full token streams and positions
  so later stages can reconstruct per-token predictions.
"""

from __future__ import annotations
import argparse, os, sys, time, traceback, random
from multiprocessing import get_context
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

DEFAULTS = {
    "model": "Qwen/Qwen3-0.6B",
    "out": "data/offline_generations.parquet",
    "max_prompts": 131072,
    "max_model_len": 32768,
    "max_tokens": 32768,
    "split": "test",
    "temperature": 0.6,
    "min_p": 0.05,
}

CLEAN_ENV_KEYS = (
    "VLLM_DP_RANK","VLLM_DP_SIZE","VLLM_DP_MASTER_IP","VLLM_DP_MASTER_PORT","VLLM_DP_LOCAL_RANK",
    "RANK","WORLD_SIZE","LOCAL_RANK","MASTER_ADDR","MASTER_PORT",
)

def _clear_distributed_env():
    for k in CLEAN_ENV_KEYS:
        os.environ.pop(k, None)

def _resolve_parent_visible_gpus() -> List[int]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        return [int(x.strip()) for x in mask.split(",") if x.strip() != ""]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = int(os.environ.get("NUM_CUDA_DEVICES", "0"))
    return list(range(n))

def load_benchmark(name: str) -> Tuple[List[str], List[str]]:
    # Mirrors your data.py/inference.py to keep parity
    benchmarks = {
        "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
        "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "solution"),
        "amc2023": ("zwhe99/amc23", None, "test", "question", "answer"),
        "aime2025": ("math-ai/aime25", None, "test", "problem", "answer"),
    }
    if name in benchmarks:
        dataset_name, config, split, q_col, a_col = benchmarks[name]
        ds = load_dataset(dataset_name, config, split=split) if config else load_dataset(dataset_name, split=split)
        return ds[q_col], ds[a_col]
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        prompts, answers, rnd = [], [], random.Random(42)
        for row in ds:
            pool = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            idxs = list(range(4)); rnd.shuffle(idxs)
            options = [f"{L}) {pool[i]}" for L, i in zip("ABCD", idxs)]
            correct_letter = "ABCD"[idxs.index(0)]
            options_block = "\n".join(options)
            prompt = (f"Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.\n"
                      f"Problem: {row['Question']}\nOptions:\n{options_block}\nAnswer:")
            prompts.append(prompt); answers.append(correct_letter)
        return prompts, answers
    elif name in ("truthfulqa_mc", "truthfulqa"):
        # EleutherAI/truthful_qa_mc — split: validation
        ds = load_dataset("EleutherAI/truthful_qa_mc", split="validation")
        prompts, answers, rnd = [], [], random.Random(42)
        for row in ds:
            # Dataset schema: question:str, choices:List[str] (len=4), label:int (0-based index of correct choice)
            choices = list(row["choices"])
            idxs = list(range(len(choices))); rnd.shuffle(idxs)
            options = [f"{L}) {choices[i]}" for L, i in zip("ABCD", idxs)]
            correct_letter = "ABCD"[idxs.index(int(row["label"]))]
            options_block = "\n".join(options)
            prompt = (f"Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.\n"
                      f"Problem: {row['question']}\nOptions:\n{options_block}\nAnswer:")
            prompts.append(prompt); answers.append(correct_letter)
        return prompts, answers
    else:
        ds = load_dataset("Maxwell-Jia/AIME_2024")["train"]
        return ds["Problem"], ds["Solution"]

def _build_chat_str(tokenizer, prompt: str, reasoning: bool) -> str:
    msgs = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=bool(reasoning), **kwargs)
    except TypeError:
        if reasoning:
            msgs = [{"role": "system", "content": "Think step by step before answering."}] + msgs
        return tokenizer.apply_chat_template(msgs, **kwargs)

def _safe_max_new_tokens(max_model_len: int) -> int:
    return max(128, max_model_len - 1024)

def worker(
    rank: int,
    dp_size: int,
    assigned_physical_gpus: List[int],
    model_id: str,
    out_path: str,
    prompts: List[str],
    answers: List[str],
    num_thinking_samples: int,
    num_nonthinking_samples: int,
    repeat_factor: int,
    temperature: float,
    min_p: float,
    top_p: float | None,
    top_k: int | None,
    max_model_len: int,
    max_tokens: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_physical_gpus)
    _clear_distributed_env()
    os.environ["VLLM_USE_RAY"] = os.environ.get("VLLM_USE_RAY", "0")

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    prompts_sub = prompts[rank::dp_size]
    answers_sub = answers[rank::dp_size]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, max_model_len=max_model_len,
              tensor_parallel_size=len(assigned_physical_gpus))
    sp = SamplingParams(
        max_tokens=min(max_tokens, _safe_max_new_tokens(max_model_len)),
        temperature=temperature, min_p=min_p, top_p=top_p, top_k=top_k, detokenize=True
    )

    rows = []
    t0 = time.perf_counter()

    for i_local, prompt in enumerate(prompts_sub):
        if prompt is None: continue
        global_idx = rank + i_local * dp_size
        ans = answers_sub[i_local] if i_local < len(answers_sub) else None

        # Build N samples per prompt per repeat (reasoning first, then non-reasoning)
        inputs = []
        metas = []  # (global_prompt_idx, reasoning_enabled, sample_id)
        sample_id = 0
        for _ in range(repeat_factor):
            for _ in range(num_thinking_samples):
                inputs.append(_build_chat_str(tok, prompt, True))
                metas.append((global_idx, True, sample_id)); sample_id += 1
            for _ in range(num_nonthinking_samples):
                inputs.append(_build_chat_str(tok, prompt, False))
                metas.append((global_idx, False, sample_id)); sample_id += 1

        if not inputs:
            continue

        # NEW: generate in small, memory-safe chunks to avoid per-GPU OOM/driver errors
        B = int(os.environ.get("ZIP_MAX_CONCURRENT", "8"))
        gens = []
        for i in range(0, len(inputs), B):
            gens.extend(llm.generate(inputs[i:i+B], sp))

        for gen, meta in zip(gens, metas):
            out = gen.outputs[0]
            prompt_ids = gen.prompt_token_ids
            out_ids = list(out.token_ids)
            input_ids = prompt_ids + out_ids
            eos_id = tok.eos_token_id
            response_text = out.text if hasattr(out, "text") else tok.decode(out_ids, skip_special_tokens=True).strip()

            rows.append({
                "prompt_idx": int(meta[0]),
                "prompt": prompt,
                "answer": ans,
                "response": response_text,
                "length": len(out_ids),
                "finished": bool(eos_id is not None and len(out_ids) > 0 and out_ids[-1] == eos_id),
                "pruned": False,  # set later in stage 3
                "expected_reward": None,
                "expected_tokens": None,
                "expected_reward_history": [],
                "reward_values": None,  # filled in stage 3
                "prompt_token_ids": prompt_ids,
                "output_token_ids": out_ids,
                "input_ids": input_ids,
                "label_positions": list(range(len(prompt_ids), len(input_ids))),  # predict positions for generated tokens
                "reasoning_enabled": bool(meta[1]),
                "model_id": model_id,
                "temperature": float(temperature),
                "min_p": float(min_p),
                "max_model_len": int(max_model_len),
                "sample_id": int(meta[2]),
            })

    shard_path = f"{out_path}.part{rank}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), shard_path)
    print(f"[rank {rank}] wrote {len(rows)} rows to {shard_path} in {time.perf_counter()-t0:.2f}s", flush=True)

def merge_shards(out_path: str, dp_size: int) -> None:
    tabs = []
    for r in range(dp_size):
        p = f"{out_path}.part{r}"
        if os.path.exists(p):
            tabs.append(pq.read_table(p))
    if not tabs:
        print("✗ No shards found."); sys.exit(1)
    table = pa.concat_tables(tabs, promote=True)
    pq.write_table(table, out_path)
    for r in range(dp_size):
        p = f"{out_path}.part{r}"
        if os.path.exists(p): os.remove(p)
    print(f"✓ Wrote {out_path} ({table.num_rows} rows)")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--benchmark", choices=["aime2024", "gsm8k", "amc2023", "aime2025", "math500", "gpqa", "truthfulqa_mc", "truthfulqa"], default="math500")
    p.add_argument("--max-num-prompts", type=int, default=DEFAULTS["max_prompts"])
    p.add_argument("--repeat-factor", type=int, default=32)
    p.add_argument("--num-thinking-samples", type=int, default=8)
    p.add_argument("--num-nonthinking-samples", type=int, default=0)
    p.add_argument("--dp-size", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"])
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--min-p", type=float, default=DEFAULTS["min_p"])
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    prompts, answers = load_benchmark(args.benchmark)
    prompts = prompts[:args.max_num_prompts]
    answers = answers[:len(prompts)]
    physical_ids = _resolve_parent_visible_gpus()
    total_needed = args.dp_size * max(1, args.tp_size)
    if total_needed > len(physical_ids):
        raise RuntimeError(f"Requested dp({args.dp_size})×tp({args.tp_size})={total_needed} GPUs but only {len(physical_ids)} visible.")

    assignments: List[List[int]] = []
    for r in range(args.dp_size):
        start, end = r * args.tp_size, r * args.tp_size + args.tp_size
        assignments.append(physical_ids[start:end])

    print(f"Generation plan: prompts={len(prompts)}  repeat={args.repeat_factor}  "
          f"samples/prompt={args.num_thinking_samples + args.num_nonthinking_samples}  "
          f"dp={args.dp_size} tp/worker={args.tp_size} gpus={assignments}")

    ctx = get_context("spawn")
    procs = []
    for rank in range(args.dp_size):
        p = ctx.Process(target=worker, args=(
            rank, args.dp_size, assignments[rank],
            args.model, args.out, prompts, answers,
            args.num_thinking_samples, args.num_nonthinking_samples, args.repeat_factor,
            args.temperature, args.min_p, args.top_p, args.top_k,
            args.max_model_len, args.max_tokens))
        p.start(); procs.append(p)

    any_fail = False
    for r, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            any_fail = True
            print(f"[main] worker {r} exit {p.exitcode}")
    if any_fail: sys.exit(1)
    merge_shards(args.out, args.dp_size)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)