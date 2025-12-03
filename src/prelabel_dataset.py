#!/usr/bin/env python3
"""
Prelabel a ZIP dataset with correctness and extracted answers.

- Reads a labeled parquet (Stage 2 output; contains prompt/answer/response/...).
- For FINISHED rows:
    * Use the grader model to label `correct` (Yes/No).
    * Use an extraction prompt to compute `extracted_answer` (string).
- Writes the updated parquet with the new columns.

Run once per dataset, then `zip_offline_prune.py` can reuse these cached labels
to compute all metrics without calling the grader again.

Example:
  python src/prelabel_dataset.py \
      --in data/amc2023_stage2.parquet \
      --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
      --out data/amc2023_stage2_labeled.parquet
"""

from __future__ import annotations
import argparse, os, random, time, warnings
from typing import List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["VLLM_USE_V1"] = "0"

DEFAULTS = {
    "eval_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "thinking_token": "</think>",
    "thinking_token_id": 151667,
    "max_model_len": 32768,
    "tp_size": 8,
    "gpu_mem_util": 0.90,
    "max_num_seqs": 8,
    "correct_phrase": "Yes",
}

def apply_hf_chat(tokenizer, user_content: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return user_content

def extract_response(response: str, token: str) -> str:
    return response[response.rfind(token) + len(token):].strip() if token in response else response.strip()

def get_eval_prompt(prompt: str, response: str, answer: str, thinking_token: str) -> str:
    return (
        "Your task is to compare the proposed solution with the verified solution. "
        "Start by identifying exactly what the question is asking. "
        "Next, determine the correctness of the proposed solution based on the verified solution. "
        "Are the two answers equivalent? "
        'Respond with ONLY the sentence "Yes." or "No."\n\n'
        f'Question:\n\n"\n{prompt}\n"\n\n'
        f'Verified Solution:\n\n"\n{answer}\n"\n\n'
        f'Proposed Solution:\n\n"\n{extract_response(response, thinking_token)}\n"'
    )

def get_hallucination_eval_prompt(prompt: str, response: str, thinking_token: str) -> str:
    return (
        "Your task is to determine whether the proposed response contains hallucinations or unsupported claims. "
        "Start by identifying what the question asks. "
        "Then, assess if the response introduces facts, references, or details that are not grounded in the question or likely fabricated. "
        'Respond with ONLY the sentence "Yes." if the response is grounded (no hallucination), or "No." if it contains hallucination.\n\n'
        f'Question:\n\n"\n{prompt}\n"\n\n'
        f'Proposed Response:\n\n"\n{extract_response(response, thinking_token)}\n"'
    )

def get_answer_extraction_prompt(question: str, response: str, thinking_token: str) -> str:
    return (
        "Your task is to extract the *final answer* from the proposed solution. "
        "Start by identifying exactly what the question is asking. "
        "Next, determine the final answer from the proposed solution. "
        "Finally, extract the final answer from the proposed solution. "
        'Respond with ONLY that answer (e.g. "42", "A").\n\n'
        f'Question:\n\n"\n{question}\n"\n\n'
        f'Proposed Solution:\n\n"\n{extract_response(response, thinking_token)}\n"'
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Stage-2 labeled parquet (input)")
    p.add_argument("--out", default=None, help="Output parquet path (default: overwrite --in)")
    p.add_argument("--model", default=DEFAULTS["eval_model"], help="Grader model ID")
    p.add_argument("--task", choices=["correctness", "hallucination"], default="correctness",
                   help="Evaluation task for correctness flag")
    p.add_argument("--thinking-token", default=DEFAULTS["thinking_token"]) 
    p.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"]) 
    p.add_argument("--tensor-parallel-size", type=int, default=DEFAULTS["tp_size"]) 
    p.add_argument("--gpu-memory-utilization", type=float, default=DEFAULTS["gpu_mem_util"]) 
    p.add_argument("--max-num-seqs", type=int, default=DEFAULTS["max_num_seqs"]) 
    p.add_argument("--enforce-eager", action="store_true", default=False)
    return p.parse_args()

def main():
    args = parse_args()
    out_path = args.out or args.inp

    df = pq.read_table(args.inp).to_pandas()
    # Only grade FINISHED rows. This is stable under later pruning (unfinished rows are irrelevant).
    df["correct"] = df.get("correct", False)
    df["extracted_answer"] = df.get("extracted_answer", None)

    llm = LLM(model=args.model,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              dtype=torch.bfloat16,
              trust_remote_code=True,
              enforce_eager=args.enforce_eager,
              max_num_seqs=args.max_num_seqs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ----------------- correctness grading -----------------
    finished = df[df["finished"]].copy()
    inputs, idxs = [], []
    for idx, row in finished.iterrows():
        if args.task == "hallucination":
            prompt_text = get_hallucination_eval_prompt(row["prompt"], row["response"], args.thinking_token)
        else:
            # if no gold `answer`, skip correctness
            if "answer" not in row or pd.isna(row["answer"]):
                continue
            prompt_text = get_eval_prompt(row["prompt"], row["response"], row["answer"], args.thinking_token)
        chat_input = apply_hf_chat(tokenizer, prompt_text)
        if len(tokenizer(chat_input).input_ids) > args.max_model_len:
            df.at[idx, "correct"] = False
            continue
        inputs.append(chat_input); idxs.append(idx)

    if inputs:
        gens = llm.generate(inputs, SamplingParams(max_tokens=1, temperature=0.0, top_k=1))
        for idx, gen in zip(idxs, gens):
            df.at[idx, "correct"] = ( "Yes" in gen.outputs[0].text )

    # ----------------- answer extraction -----------------
    ex_inputs, ex_idxs = [], []
    for idx, row in finished.iterrows():
        chat_input = apply_hf_chat(tokenizer, get_answer_extraction_prompt(row["prompt"], row["response"], args.thinking_token))
        if len(tokenizer(chat_input).input_ids) > args.max_model_len:
            continue
        ex_inputs.append(chat_input); ex_idxs.append(idx)

    if ex_inputs:
        extracts = llm.generate(ex_inputs, SamplingParams(max_tokens=8, temperature=0.0, top_k=1))
        for idx, gen in zip(ex_idxs, extracts):
            df.at[idx, "extracted_answer"] = gen.outputs[0].text.strip().strip('"').strip()

    # Persist
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")
    print(f"✓ wrote prelabels to {out_path}")

if __name__ == "__main__":
    main()