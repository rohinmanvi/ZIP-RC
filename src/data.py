#!/usr/bin/env python3
"""
Generate training data for ZIP (Zero-overhead Inference-time Prediction) by sampling
a base model against a prompt set and writing the results to a single Parquet file.

Design (robust version):
- No vLLM data-parallel (DP) at all. Each Python process is independent and owns
  its assigned GPU(s). Optional intra-process tensor parallel (TP) only.
- Safe CUDA multiprocessing via 'spawn'.
- Strict shard merge by default (no accidental partial datasets).

Example:
    python src/data.py \
        --model Qwen/Qwen3-0.6B \
        --dataset allenai/llama-3.1-tulu-3-8b-preference-mixture \
        --split train \
        --prompt-column prompt \
        --thinking-samples 0 \
        --non-thinking-samples 1 \
        --out data/tulu_prompts_only.parquet \
        --max-num-prompts 100 \
        --max-model-len 4096 \
        --dp-size 8 --tp-size 1
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
    "dataset": "agentica-org/DeepScaleR-Preview-Dataset",
    "output": "data/zip_training_data.parquet",
    "max_prompts": 131_072,
    "thinking_samples": 2,
    "non_thinking_samples": 2,
    "max_model_len": 32_768,
    "split": "train",
    "prompt_column": "problem",
    "answer_column": "answer",
}

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

CLEAN_ENV_KEYS = (
    # vLLM DP env
    "VLLM_DP_RANK", "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT",
    "VLLM_DP_LOCAL_RANK",
    # torch/SLURM-ish env that can trigger torch.distributed defaults
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
)

def _clear_distributed_env():
    """Remove env that can cause vLLM or torch.distributed to initialize DP/PG."""
    for k in CLEAN_ENV_KEYS:
        os.environ.pop(k, None)

def _resolve_parent_visible_gpus() -> List[int]:
    """
    Return the *physical* GPU IDs that are visible to this job at the parent level.
    Respects a pre-set CUDA_VISIBLE_DEVICES (e.g., '4,6,7').
    """
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        # Respect parent mapping exactly.
        ids = [int(x.strip()) for x in mask.split(",") if x.strip() != ""]
        return ids
    # No mask → assume all GPUs [0 .. N-1] are visible.
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = int(os.environ.get("NUM_CUDA_DEVICES", "0"))  # last-resort fallback
    return list(range(n))

def load_benchmark(name: str) -> Tuple[List[str], List[str]]:
    """Load prompts and answers for the specified benchmark.

    Mirrors the support in inference.py so we can generate rollouts for the same
    suites (AIME, AMC, GSM8K, MATH-500, GPQA, etc.).
    """
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

    if name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        prompts, answers = [], []
        rnd = random.Random(42)

        for row in ds:
            answer_pool = [row["Correct Answer"], row["Incorrect Answer 1"],
                           row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            indices = list(range(4))
            rnd.shuffle(indices)

            options = [f"{letter}) {answer_pool[idx]}" for letter, idx in zip("ABCD", indices)]
            correct_letter = "ABCD"[indices.index(0)]

            prompt = (f"Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.\n"
                      f"Problem: {row['Question']}\nOptions: {', '.join(options)}\nAnswer:")

            prompts.append(prompt)
            answers.append(correct_letter)

        return prompts, answers

    # Default to AIME 2024
    ds = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    return ds["Problem"], ds["Solution"]

def _build_chat_str(tokenizer, prompt: str, reasoning: bool) -> str:
    """
    Build an input string using the tokenizer's chat template.
    If the tokenizer doesn't accept 'enable_thinking', fall back to a system nudge.
    """
    msgs = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)

    try:
        # Some tokenizers (Qwen reasoning variants) accept 'enable_thinking'.
        return tokenizer.apply_chat_template(
            msgs, enable_thinking=bool(reasoning), **kwargs
        )
    except TypeError:
        # Fallback: prepend a light system hint when reasoning=True.
        if reasoning:
            msgs = [{"role": "system", "content": "Think step by step before answering."}] + msgs
        return tokenizer.apply_chat_template(msgs, **kwargs)

def _safe_max_new_tokens(max_model_len: int) -> int:
    """
    Conservative cap to avoid negative budgets when users pass small model lens.
    """
    return max(128, max_model_len - 1024)

# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def worker(
    rank: int,
    dp_size: int,
    assigned_physical_gpus: List[int],
    model_id: str,
    out_path: str,
    prompts: List[str],
    answers: List[str],
    temperature: float,
    min_p: float,
    max_model_len: int,
    thinking_samples: int,
    non_thinking_samples: int,
    max_num_seqs: int | None,
) -> None:
    """
    Independent worker (no vLLM DP). It owns 'assigned_physical_gpus' exclusively.
    """
    # 1) Pin CUDA devices *before* any torch.cuda activity.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_physical_gpus)

    # 2) Scrub distributed env so vLLM does not try to form DP groups.
    _clear_distributed_env()
    # Also ensure Ray/distributed executor is not roped in implicitly
    os.environ["VLLM_USE_RAY"] = os.environ.get("VLLM_USE_RAY", "0")

    # 3) Imports that may query CUDA are now safe.
    import torch  # local import to avoid early CUDA touch in parent
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # index within this worker's private view

    # 4) Data slice for this rank (striding) and tokenizer/model init.
    prompts_sub = prompts[rank::dp_size]
    answers_sub = answers[rank::dp_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        tensor_parallel_size=len(assigned_physical_gpus),
        max_num_seqs=max_num_seqs,
    )

    sampling = SamplingParams(
        max_tokens=_safe_max_new_tokens(max_model_len),
        temperature=temperature,
        min_p=min_p,
        detokenize=True,
    )

    # 5) Prepare inputs (reasoning + non-reasoning)
    inputs: List[str] = []
    sample_meta: List[tuple[int, bool]] = []  # (global_prompt_idx, is_reasoning)

    for i_local, prompt in enumerate(prompts_sub):
        if prompt is None:
            continue
        global_idx = rank + i_local * dp_size  # fix: global index across shards

        for _ in range(thinking_samples):
            inputs.append(_build_chat_str(tokenizer, prompt, reasoning=True))
            sample_meta.append((global_idx, True))

        for _ in range(non_thinking_samples):
            inputs.append(_build_chat_str(tokenizer, prompt, reasoning=False))
            sample_meta.append((global_idx, False))

    # 6) Generate
    start = time.perf_counter()
    generations = list(llm.generate(inputs, sampling))
    elapsed = time.perf_counter() - start

    # 7) Build rows
    rows = []
    for gen_idx, gen in enumerate(generations):
        global_idx, is_reasoning = sample_meta[gen_idx]
        # Map back to local sub-array to pick the right prompt/answer
        i_local = (global_idx - rank) // dp_size
        prompt_text = prompts_sub[i_local]
        answer_text = answers_sub[i_local] if i_local < len(answers_sub) else None

        out = gen.outputs[0]
        input_ids = gen.prompt_token_ids + list(out.token_ids)
        eos_id = tokenizer.eos_token_id

        # Resolve response text (vLLM provides .text when detokenize=True)
        try:
            response_text = out.text
        except Exception:
            response_text = tokenizer.decode(out.token_ids, skip_special_tokens=True).strip()

        rows.append({
            "prompt_idx": int(global_idx),
            "prompt": prompt_text,
            "answer": answer_text,
            "prompt_token_ids": gen.prompt_token_ids,
            "output_token_ids": list(out.token_ids),
            "input_ids": input_ids,
            "label_positions": list(range(len(gen.prompt_token_ids), len(input_ids))),
            "response": response_text,
            "length": len(out.token_ids),
            "finished": (eos_id is not None and input_ids and input_ids[-1] == eos_id),
            "reasoning_enabled": bool(is_reasoning),
            "model_id": model_id,
            "temperature": float(temperature),
            "min_p": float(min_p),
            "max_model_len": int(max_model_len),
        })

    # 8) Write shard
    shard_path = f"{out_path}.part{rank}"
    df = pd.DataFrame(rows)
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(shard_path, engine="pyarrow", index=False)

    print(f"[rank {rank}] gpus={assigned_physical_gpus} wrote {shard_path} "
          f"({len(rows)} rows) in {elapsed:.2f}s", flush=True)


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def merge_shards(out_path: str, dp_size: int, allow_partial: bool) -> tuple[bool, int, int]:
    """
    Merge worker shards into a single Parquet file.
    Returns (merged, found, expected).
    - merged=False when shards missing and allow_partial=False.
    """
    part_tables = []
    found = 0
    for r in range(dp_size):
        part_path = f"{out_path}.part{r}"
        if os.path.exists(part_path):
            part_tables.append(pq.read_table(part_path))
            found += 1

    if found == 0:
        return (False, 0, dp_size)

    if found != dp_size and not allow_partial:
        # Do NOT merge; let the user inspect shards.
        return (False, found, dp_size)

    table = pa.concat_tables(part_tables, promote=True)
    pq.write_table(table, out_path)
    # Remove shards only if we wrote the final file successfully.
    for r in range(dp_size):
        part_path = f"{out_path}.part{r}"
        if os.path.exists(part_path):
            os.remove(part_path)
    return (True, found, dp_size)


# --------------------------------------------------------------------------- #
# CLI / Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--out", default=DEFAULTS["output"])
    # Benchmark mode (optional). If set, overrides --dataset/--prompt-column/--answer-column.
    p.add_argument("--benchmark", choices=["aime2024", "gsm8k", "amc2023", "aime2025", "math500", "gpqa"], default=None,
                   help="Named benchmark to load (overrides --dataset mode when set)")
    p.add_argument("--repeat-factor", type=int, default=1,
                   help="Repeat prompts this many times (with shuffle) for robustness like inference sweeps")
    p.add_argument("--repeat-seed", type=int, default=123, help="Shuffle seed when applying repeat-factor")
    p.add_argument("--dp-size", type=int, default=8,  # safer default than 8
                   help="Number of independent workers (no vLLM DP).")
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor-parallel GPUs per worker (contiguous assignment).")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"])
    p.add_argument("--max-num-prompts", type=int, default=DEFAULTS["max_prompts"])
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="Cap concurrent in-flight sequences per worker to reduce KV cache pressure.")
    p.add_argument("--thinking-samples", type=int, default=DEFAULTS["thinking_samples"],
                   help="Samples with 'reasoning' enabled per prompt.")
    p.add_argument("--non-thinking-samples", type=int, default=DEFAULTS["non_thinking_samples"],
                   help="Samples without reasoning per prompt.")
    p.add_argument("--dataset", default=DEFAULTS["dataset"],
                   help="Hugging Face dataset to load.")
    p.add_argument("--split", default=DEFAULTS["split"])
    p.add_argument("--prompt-column", default=DEFAULTS["prompt_column"])
    p.add_argument("--answer-column", default=DEFAULTS["answer_column"])
    p.add_argument("--allow-partial-merge", action="store_true",
                   help="If set, merge whatever shards exist; otherwise require all dp-size shards.")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Guard: samples per prompt
    total_per_prompt = args.thinking_samples + args.non_thinking_samples
    if total_per_prompt == 0:
        print("No samples requested (--thinking-samples + --non-thinking-samples == 0). Exiting.")
        return

    # Determine parent-visible physical GPU IDs and plan assignments
    physical_ids = _resolve_parent_visible_gpus()
    total_needed = args.dp_size * max(1, args.tp_size)

    if total_needed > len(physical_ids):
        raise RuntimeError(
            f"Requested dp_size({args.dp_size}) * tp_size({args.tp_size}) = {total_needed} GPUs, "
            f"but only {len(physical_ids)} are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')})."
        )

    # Load prompts/answers either from a named benchmark or from a generic dataset
    if args.benchmark:
        prompts, answers = load_benchmark(args.benchmark)
        # Cap to max_num_prompts
        prompts = prompts[: args.max_num_prompts]
        answers = answers[: len(prompts)]
    else:
        ds = load_dataset(args.dataset, split=args.split).shuffle(seed=42)
        if args.prompt_column not in ds.column_names:
            raise ValueError(f"Prompt column '{args.prompt_column}' not in dataset columns: {ds.column_names}")
        prompts = ds[args.prompt_column][:args.max_num_prompts]
        if args.answer_column and args.answer_column in ds.column_names:
            answers = ds[args.answer_column][:args.max_num_prompts]
        else:
            answers = [None] * len(prompts)

    # Optional repeat for robustness (shuffle applied)
    if args.repeat_factor and args.repeat_factor > 1:
        prompts = prompts * args.repeat_factor
        answers = answers * args.repeat_factor
        indices = list(range(len(prompts)))
        random.Random(args.repeat_seed).shuffle(indices)
        prompts = [prompts[i] for i in indices]
        answers = [answers[i] for i in indices]

    # Log plan
    print(f"Using model: {args.model}")
    if args.benchmark:
        print(f"Benchmark: {args.benchmark}  |  Prompts: {len(prompts)} (max {args.max_num_prompts})  |  repeat={args.repeat_factor}")
    else:
        print(f"Prompts: {len(prompts)} (max {args.max_num_prompts})  |  split: {args.split}")
    print(f"Samples / prompt: {total_per_prompt}  "
          f"(reasoning={args.thinking_samples}, non_reasoning={args.non_thinking_samples})")
    print(f"Workers (dp-size): {args.dp_size}  |  TP per worker: {args.tp_size}")
    print(f"Visible GPUs: {physical_ids}")

    # Plan per-worker GPU slices (contiguous)
    assignments: List[List[int]] = []
    for r in range(args.dp_size):
        start = r * args.tp_size
        end = start + args.tp_size
        assignments.append(physical_ids[start:end])

    # Safer CUDA multiprocessing
    ctx = get_context("spawn")

    # Launch workers
    procs = []
    for rank in range(args.dp_size):
        p = ctx.Process(
            target=worker,
            args=(
                rank,
                args.dp_size,
                assignments[rank],
                args.model,
                args.out,
                prompts,
                answers,
                args.temperature,
                args.min_p,
                args.max_model_len,
                args.thinking_samples,
                args.non_thinking_samples,
                args.max_num_seqs,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait and report failures explicitly
    any_fail = False
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            any_fail = True
            print(f"[main] worker rank {rank} (pid={p.pid}) exited with code {p.exitcode}", flush=True)

    # Merge shards (strict by default)
    merged, found, expected = merge_shards(args.out, args.dp_size, allow_partial=args.allow_partial_merge)
    if not merged:
        if found == 0:
            print("✗ No shards were produced; check worker errors above.")
        else:
            print(f"✗ Shards missing ({found}/{expected}). "
                  f"Re-run or pass --allow-partial-merge to write a partial dataset.")
        # Surface failure if any worker failed or merge did not happen
        if any_fail or found != expected:
            sys.exit(1)
        return

    # Success
    suffix = "" if found == expected else f" (PARTIAL: {found}/{expected} shards)"
    print(f"✓ Generated data written to {args.out}{suffix}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
