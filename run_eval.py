"""Evaluation CLI: `python run_eval.py --model {sealion|llama} --mode
{zero_shot|few_shot|finetuned}`.

Requires a GPU, meant to run on the GCP L4 instance for all 6
combinations (--model sealion/llama x --mode zero_shot/few_shot/finetuned).
"""

import argparse
import json
import re
import time
from collections import defaultdict

import torch

from common import (
    BENCHMARK_PATH,
    EVALS_DIR,
    FEWSHOT_PATH,
    MAX_NEW_TOKENS,
    build_chat_prompt,
    build_fewshot_block,
    build_mcq_question,
    load_model_and_tokenizer,
    parse_response,
    read_jsonl,
)

BATCH_SIZE = 16  # tune down if OOM on the L4, tune up if there's headroom


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def _extract_sentence_from_prompt(prompt: str) -> str | None:
    """benchmark.jsonl records don't carry a separate sentence field --
    only the fully-built `prompt` string with the sentence embedded
    inside it (verified against a real record; same recovery logic as
    model_setup.py)."""
    match = re.search(r'in the sentence "(?P<sentence>.*)"\?$', prompt, re.DOTALL)
    return match.group("sentence") if match else None


def summarize(records: list[dict]) -> dict:
    total = len(records)
    parse_errors = sum(1 for r in records if r["parse_error"])
    correct = sum(1 for r in records if r["correct"])
    parsed_total = total - parse_errors

    return {
        "total": total,
        "parse_errors": parse_errors,
        "correct": correct,
        "parsed_total": parsed_total,
        # Accuracy among successfully parsed responses only.
        "acc_among_parsed": (correct / parsed_total) if parsed_total else float("nan"),
        # Accuracy treating parse errors as wrong answers.
        "acc_all": (correct / total) if total else float("nan"),
    }


def print_summary_block(label: str, stats: dict):
    print(f"{label}")
    print(f"  Total instances:       {stats['total']}")
    parse_error_pct = (stats["parse_errors"] / stats["total"]) if stats["total"] else 0.0
    print(f"  Parse errors:          {stats['parse_errors']} ({parse_error_pct:.1%})")
    print(f"  Accuracy (parsed only): {stats['acc_among_parsed']:.1%} ({stats['correct']}/{stats['parsed_total']})")
    print(f"  Accuracy (all, errors=wrong): {stats['acc_all']:.1%} ({stats['correct']}/{stats['total']})")


def main():
    # ------------------------------------------------------------------
    # Step 1: CLI args and mode validation
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["sealion", "llama"])
    parser.add_argument("--mode", required=True, choices=["zero_shot", "few_shot", "finetuned"])
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Load model according to mode (STRICT guardrails)
    # ------------------------------------------------------------------
    use_adapter = (args.mode == "finetuned")
    model, tokenizer = load_model_and_tokenizer(args.model, adapter=use_adapter, for_training=False)

    # Required for correct batched generation with decoder-only models:
    # left-padding keeps the "real" content right-aligned against the
    # generation start, so (a) the model isn't attending to padding in
    # the middle of its context and (b) every sequence in a batch has its
    # newly-generated tokens starting at the SAME position, which is what
    # makes the simple `output_ids[i][inputs['input_ids'].shape[1]:]`
    # slicing below valid for the whole batch at once. Must be set before
    # any prompts are tokenized.
    tokenizer.padding_side = "left"

    # Explicit assertion, not just implicit behavior -- this must be
    # impossible to get wrong silently.
    if args.mode == "finetuned":
        assert use_adapter, "finetuned mode must load the adapter"
    else:
        assert not use_adapter, f"{args.mode} mode must NOT load any adapter"

    print(f"Model:   {args.model}")
    print(f"Mode:    {args.mode}")
    print(f"Adapter attached: {use_adapter}")

    # ------------------------------------------------------------------
    # Step 3: Load benchmark and (if needed) few-shot data
    # ------------------------------------------------------------------
    benchmark = read_jsonl(BENCHMARK_PATH)
    fewshot_records = read_jsonl(FEWSHOT_PATH) if args.mode == "few_shot" else None

    print(f"Benchmark instances: {len(benchmark)}")
    if fewshot_records is not None:
        print(f"Few-shot records:    {len(fewshot_records)}")

    # ------------------------------------------------------------------
    # Step 4: Build (record, user_message) pairs -- logic unchanged from
    # the per-instance version, just collected up front instead of
    # generated one at a time.
    # ------------------------------------------------------------------
    pairs = []
    for record in benchmark:
        homograph = record["homograph"]
        # `choices` here is the ALREADY-SHUFFLED per-instance list baked
        # into the benchmark record itself -- NOT re-derived from
        # CHOICE_MAP, which would shuffle differently.
        choices = record["choices"]
        sentence = _extract_sentence_from_prompt(record["prompt"])
        question_text = build_mcq_question(homograph, sentence, choices)

        if args.mode == "few_shot":
            fewshot_block = build_fewshot_block(homograph, fewshot_records)
            user_message = f"{fewshot_block}\nNow answer the following question:\n\n{question_text}"
        else:
            user_message = question_text

        pairs.append((record, user_message))

    # ------------------------------------------------------------------
    # Step 5: Batched generation + parsing, with incremental saving
    # ------------------------------------------------------------------
    results_path = EVALS_DIR / args.model / "results" / f"{args.mode}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    total = len(pairs)
    batches = list(batched(pairs, BATCH_SIZE))
    n_batches = len(batches)
    start_time = time.time()

    # Opened once in "w" mode (fresh run each invocation -- no resume
    # logic requested) and flushed after every batch, so an interruption
    # mid-run only loses the batch in progress, not everything completed
    # so far.
    with open(results_path, "w", encoding="utf-8") as results_file:
        processed = 0
        for batch_idx, batch in enumerate(batches, start=1):
            batch_records = [r for r, _ in batch]
            user_messages = [m for _, m in batch]
            prompts = [build_chat_prompt(tokenizer, m) for m in user_messages]

            try:
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Left-padding means every prompt in this batch ends at
                # the same position, so this single slice index is valid
                # for the whole batch at once.
                input_len = inputs["input_ids"].shape[1]
                decoded_texts = [
                    tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
                    for i in range(len(batch))
                ]

                if batch_idx == 1:
                    # Runtime sanity check on the first batch only: if
                    # the left-padding slicing assumption were wrong,
                    # decoded output would still contain the prompt/
                    # question text rather than just the model's answer.
                    for record, user_message, decoded in zip(batch_records, user_messages, decoded_texts):
                        probe = user_message[:30]
                        if probe and probe in decoded:
                            print(f"WARNING: decoded output for {record['id']!r} still appears to "
                                  f"contain prompt text -- left-padding slicing assumption may be "
                                  f"wrong. Decoded (truncated): {decoded[:120]!r}")

                for record, decoded_text in zip(batch_records, decoded_texts):
                    parsed = parse_response(decoded_text)
                    result = {
                        "id": record["id"],
                        "homograph": record["homograph"],
                        "asymmetry_type": record["asymmetry_type"],
                        "gold_variant": record["gold_variant"],
                        "label": record["label"],
                        "raw_output": decoded_text,
                        "parsed_answer": parsed["answer"] if parsed else None,
                        "parse_error": parsed is None,
                        "correct": (parsed is not None and ord(parsed["answer"]) - ord("a") == record["label"]),
                    }
                    results_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                # Simpler-to-implement-correctly choice: skip and log the
                # whole failed batch rather than retrying with a halved
                # batch size (which risks compounding failures / repeated
                # OOM churn). One bad batch is logged as parse errors for
                # its records and the run continues -- it does NOT crash
                # and does NOT lose any already-written batches.
                bad_ids = [r["id"] for r in batch_records]
                print(f"ERROR in batch {batch_idx}/{n_batches} (record ids: {bad_ids}): {type(e).__name__}: {e}")
                for record in batch_records:
                    result = {
                        "id": record["id"],
                        "homograph": record["homograph"],
                        "asymmetry_type": record["asymmetry_type"],
                        "gold_variant": record["gold_variant"],
                        "label": record["label"],
                        "raw_output": f"BATCH_FAILED: {type(e).__name__}: {e}",
                        "parsed_answer": None,
                        "parse_error": True,
                        "correct": False,
                    }
                    results_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            results_file.flush()
            processed += len(batch)
            print(f"processed {processed}/{total} (batch {batch_idx}/{n_batches})...")

    elapsed = time.time() - start_time
    print(f"\nSaved {total} result(s) to {results_path.resolve()}")
    print(f"Wall-clock time: {elapsed:.1f}s total, {elapsed / total:.2f}s/instance average")

    # ------------------------------------------------------------------
    # Step 7: Print summary -- read back the full written file rather
    # than relying on an in-memory list, so this reflects exactly what's
    # on disk even after an interrupted/resumed run.
    # ------------------------------------------------------------------
    results = read_jsonl(results_path)

    print("\n" + "=" * 70)
    print(f"SUMMARY -- model={args.model}  mode={args.mode}")
    print("=" * 70)

    overall_stats = summarize(results)
    print_summary_block("Overall:", overall_stats)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["asymmetry_type"]].append(r)

    print("\nBreakdown by asymmetry_type:")
    for asymmetry_type in sorted(by_type, key=lambda k: (k is None, k)):
        stats = summarize(by_type[asymmetry_type])
        print()
        print_summary_block(f"{asymmetry_type}:", stats)


if __name__ == "__main__":
    main()
