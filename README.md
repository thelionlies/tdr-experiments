# TDR Experiments

## Overview

This project evaluates two 8B-parameter language models — **SEA-LION**
(`aisingapore/Llama-SEA-LION-v2-8B-IT`) and **Llama 3.1**
(`meta-llama/Llama-3.1-8B-Instruct`) — on a Tagalog homograph
diacritization multiple-choice benchmark, across three conditions:
zero-shot, few-shot, and LoRA fine-tuned. The task: given a sentence with
a homograph written without its diacritic marks (tuldik), pick which of
four candidate spellings is correct.

## Setup

```bash
pip install -r requirements.txt
```

**Requires a CUDA-capable GPU** (developed against an NVIDIA L4, 24GB)
for everything except `test_common_local.py`.

**`meta-llama/Llama-3.1-8B-Instruct` is a gated model on Hugging Face** —
you must accept Meta's license on the model's HF page and authenticate
locally via `huggingface-cli login` (or set the `HF_TOKEN` environment
variable) before running anything that loads it, or the download will
fail with an authorization error.

## File structure

- `common.py` — shared config, prompt building, response parsing,
  model/adapter loading. Imported by everything else.
- `model_setup.py` — one-time sanity check: loads both models, confirms
  chat templates work, runs one test generation each, checks GPU memory.
  Run this FIRST.
- `finetune.py` — LoRA fine-tunes one model on the diacritization task.
  Run twice (once per model).
- `run_eval.py` — runs the MCQ benchmark for one (model, mode)
  combination. Run six times (2 models x 3 modes).
- `compute_metrics.py` — aggregates all available results files into
  final accuracy/F1 tables. Safe to re-run anytime, reports on whatever
  results exist so far.
- `test_common_local.py` — CPU-only test suite for `common.py`'s non-GPU
  logic. Already run and passing (57/57) as of this handoff — rerun if
  you touch `common.py`.
- `data/` — all prepared datasets (see [Data files](#data-files-already-prepared-do-not-regenerate) below).
- `evals/{sealion,llama}/adapter/` — where `finetune.py` saves LoRA
  weights.
- `evals/{sealion,llama}/results/{zero_shot,few_shot,finetuned}.jsonl` —
  where `run_eval.py` saves raw outputs.
- `evals/final_results.csv`, `evals/final_results_by_asymmetry.csv` —
  final output of `compute_metrics.py`.

## Data files (already prepared, do not regenerate)

- `data/finetune_train.jsonl` (28,742 examples), `data/finetune_val.jsonl`
  (3,199 examples) — used by `finetune.py`.
- `data/benchmark.jsonl` (7,989 instances) — the evaluation set, used by
  `run_eval.py`. Ambiguous same-homograph/different-variant sentences
  already removed.
- `data/few_shot_examples.jsonl` (164 examples) — used by `run_eval.py`
  in `few_shot` mode.

## RUN ORDER

Run these in this exact order. Each step depends on the previous one
completing successfully.

1. **Sanity check** (a few minutes, no training):

   ```bash
   python model_setup.py
   ```

   Confirms both models load, GPU memory is sufficient, and the
   single-letter answer format works under real generation. **STOP and
   report back if this fails or if either model doesn't produce a valid
   A/B/C/D answer** — do not proceed to fine-tuning until this passes.

2. **Fine-tune SEA-LION** (budget a few hours):

   ```bash
   python finetune.py --model sealion
   ```

   Saves the adapter to `evals/sealion/adapter/`.

3. **Fine-tune Llama** (budget a few hours):

   ```bash
   python finetune.py --model llama
   ```

   Saves the adapter to `evals/llama/adapter/`.

4. **Run all 6 evaluation conditions** (order within this step doesn't
   matter, but `finetuned` mode for a given model requires that model's
   Step 2/3 fine-tune to have completed first):

   ```bash
   python run_eval.py --model sealion --mode zero_shot
   python run_eval.py --model sealion --mode few_shot
   python run_eval.py --model sealion --mode finetuned
   python run_eval.py --model llama --mode zero_shot
   python run_eval.py --model llama --mode few_shot
   python run_eval.py --model llama --mode finetuned
   ```

   Each writes to `evals/{model}/results/{mode}.jsonl`. These can be run
   in any order and interrupted/resumed independently — each is a
   self-contained run over the full benchmark.

5. **Compute final metrics** (fast, no GPU needed, can rerun anytime):

   ```bash
   python compute_metrics.py
   ```

   Produces `evals/final_results.csv` and
   `evals/final_results_by_asymmetry.csv`. Safe to run after only some of
   Step 4's conditions are done — it reports on whatever exists and
   clearly flags what's missing.

## Known constraints / things to watch for

- Both models must use IDENTICAL LoRA/training config (already enforced
  via `common.py` — do not add per-model overrides).
- `zero_shot` and `few_shot` modes must NEVER load a LoRA adapter;
  `finetuned` mode must ALWAYS load one. This is asserted in
  `run_eval.py` — if you see an assertion error here, something is wrong
  with the `--mode`/adapter logic, do not bypass it.
- The system prompt requires a single-letter answer only (A/B/C/D) — no
  reasoning/explanation. Responses that don't match this exact format are
  logged as parse errors, not retried or coerced.
- Report back GPU memory usage from `model_setup.py`'s output before
  starting fine-tuning, in case batch size needs adjusting for the actual
  hardware.

## Questions / issues

If anything fails or produces unexpected output, stop and report back
rather than guessing a fix — especially for `model_setup.py`'s initial
results, since everything downstream depends on that working correctly
first.
