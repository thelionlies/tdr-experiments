"""Local, CPU-only, no-GPU-required test script for common.py. Run this
BEFORE handing the repo off to a GPU machine, to catch bugs early.

Never calls load_model_and_tokenizer or anything else that would trigger
a model download or GPU usage. Importing common.py itself pulls in
torch/transformers/peft/bitsandbytes at module level (for
load_model_and_tokenizer's use elsewhere) -- that alone needs no GPU and
downloads nothing, so it's expected and fine here.
"""

import re
import tempfile
from collections import defaultdict
from pathlib import Path

from common import (
    BENCHMARK_PATH,
    BNB_CONFIG,
    CHOICE_MAP,
    FEWSHOT_PATH,
    LORA_CONFIG,
    MAX_NEW_TOKENS,
    MODEL_REGISTRY,
    RESPONSE_PATTERN,
    SYSTEM_PROMPT,
    TRAINING_CONFIG,
    adapter_dir,
    build_fewshot_block,
    build_mcq_question,
    parse_response,
    read_jsonl,
    write_jsonl,
)

results = []  # (description, passed, detail)


def check(description: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" -- {detail}" if detail and not condition else ""
    print(f"[{status}] {description}{suffix}")
    results.append((description, condition, detail))
    return condition


def _extract_sentence_from_prompt(prompt: str) -> str | None:
    """Same recovery logic as model_setup.py -- benchmark.jsonl records
    don't carry a separate sentence field, only the fully-built `prompt`
    string with the sentence embedded inside it."""
    match = re.search(r'in the sentence "(?P<sentence>.*)"\?$', prompt, re.DOTALL)
    return match.group("sentence") if match else None


def test_1_choice_map_integrity():
    print("\n" + "=" * 70)
    print("Test 1: CHOICE_MAP integrity")
    print("=" * 70)

    check("len(CHOICE_MAP) == 40", len(CHOICE_MAP) == 40, f"got {len(CHOICE_MAP)}")

    bad_len = [(h, len(c)) for h, c in CHOICE_MAP.items() if len(c) != 4]
    check("every homograph has exactly 4 choices", not bad_len, f"bad: {bad_len}")

    bad_dup = [(h, c) for h, c in CHOICE_MAP.items() if len(set(c)) != len(c)]
    check("no duplicate strings within any homograph's 4 choices", not bad_dup, f"bad: {bad_dup}")


def test_2_jsonl_roundtrip():
    print("\n" + "=" * 70)
    print("Test 2: read_jsonl / write_jsonl round-trip")
    print("=" * 70)

    benchmark_records = read_jsonl(BENCHMARK_PATH)
    fewshot_records = read_jsonl(FEWSHOT_PATH)
    print(f"benchmark.jsonl records:         {len(benchmark_records)}")
    print(f"few_shot_examples.jsonl records: {len(fewshot_records)}")

    sample = benchmark_records[:5]
    tmp_path = Path(tempfile.gettempdir()) / "test_common_local_roundtrip.jsonl"
    try:
        write_jsonl(tmp_path, sample)
        reread = read_jsonl(tmp_path)
        check(
            "read/write round-trip identical (first 5 benchmark records, diacritics included)",
            reread == sample,
            "mismatch between original and round-tripped records" if reread != sample else "",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return benchmark_records, fewshot_records


def test_3_build_mcq_question(benchmark_records):
    print("\n" + "=" * 70)
    print("Test 3: build_mcq_question")
    print("=" * 70)

    for i, rec in enumerate(benchmark_records[:3]):
        sentence = _extract_sentence_from_prompt(rec["prompt"])
        question = build_mcq_question(rec["homograph"], sentence, rec["choices"])
        print(f"\n--- sample {i} ({rec['id']}) ---")
        print(question)

        lines = question.split("\n")
        option_lines = [ln for ln in lines if re.match(r"^[A-D]\.\s", ln)]
        letters_found = [ln[0] for ln in option_lines]
        check(f"sample {i}: exactly 4 lines starting with A./B./C./D. in order", letters_found == ["A", "B", "C", "D"], f"found: {letters_found}")
        check(f"sample {i}: contains homograph name", rec["homograph"] in question)
        check(f"sample {i}: contains sentence text", sentence in question)

    def _raises_assertion(choices):
        try:
            build_mcq_question("kaya", "test sentence", choices)
            return False
        except AssertionError:
            return True

    check("build_mcq_question raises AssertionError for 3 choices", _raises_assertion(["a", "b", "c"]))
    check("build_mcq_question raises AssertionError for 5 choices", _raises_assertion(["a", "b", "c", "d", "e"]))


def test_4_build_fewshot_block(fewshot_records):
    print("\n" + "=" * 70)
    print("Test 4: build_fewshot_block")
    print("=" * 70)

    non_null_counts = defaultdict(int)
    for r in fewshot_records:
        if r.get("sentence"):
            non_null_counts[r["homograph"]] += 1

    candidates = [h for h, c in non_null_counts.items() if c >= 2]
    check("found at least 3 homographs with >=2 non-null-sentence records", len(candidates) >= 3, f"found {len(candidates)}")
    test_homographs = candidates[:3]

    for h in test_homographs:
        expected_count = non_null_counts[h]
        output = build_fewshot_block(h, fewshot_records)
        print(f"\n--- build_fewshot_block('{h}') ---")
        print(output)

        blocks = [b for b in output.split("Example:\n") if b.strip()]
        check(f"{h}: number of 'Example:' blocks matches non-null-sentence record count", len(blocks) == expected_count, f"expected {expected_count}, got {len(blocks)}")

        per_block_answers = [re.findall(r"^ANSWER:\s*([A-Da-d])\s*$", b, re.MULTILINE) for b in blocks]
        one_each = all(len(a) == 1 for a in per_block_answers)
        all_valid = all(a[0] in "ABCDabcd" for a in per_block_answers if a)
        check(f"{h}: every block has exactly one ANSWER: line with a valid letter", one_each and all_valid, f"per-block answer counts: {[len(a) for a in per_block_answers]}")

        output2 = build_fewshot_block(h, fewshot_records)
        check(f"{h}: build_fewshot_block deterministic across repeated calls", output == output2)


def test_5_parse_response_valid():
    print("\n" + "=" * 70)
    print("Test 5: parse_response -- valid inputs")
    print("=" * 70)

    VALID_CASES = [
        ("A", "a"), ("b", "b"), (" C ", "c"), ("D\n", "d"),
        ("a", "a"), ("  d  ", "d"),
    ]
    for inp, expected in VALID_CASES:
        result = parse_response(inp)
        ok = result is not None and result["answer"] == expected
        check(f"parse_response({inp!r}) == {{'answer': {expected!r}}}", ok, f"got {result!r}")


def test_6_parse_response_invalid():
    print("\n" + "=" * 70)
    print("Test 6: parse_response -- invalid inputs (must return None)")
    print("=" * 70)

    INVALID_CASES = [
        "The answer is A", "A.", "AB", "", "   ", "E", "1",
        "REASON: because\nANSWER: A",  # old format should now fail
        "A B", "Answer: A",
    ]
    for inp in INVALID_CASES:
        result = parse_response(inp)
        ok = result is None
        check(f"parse_response({inp!r}) is None", ok, f"incorrectly accepted, got {result!r}")


def test_7_config_sanity():
    print("\n" + "=" * 70)
    print("Test 7: Config sanity")
    print("=" * 70)

    print(f"LORA_CONFIG.r = {LORA_CONFIG.r}")
    print(f"LORA_CONFIG.lora_alpha = {LORA_CONFIG.lora_alpha}")
    print(f"LORA_CONFIG.lora_dropout = {LORA_CONFIG.lora_dropout}")
    print(f"LORA_CONFIG.target_modules = {LORA_CONFIG.target_modules}")
    check("LORA_CONFIG.r == 16", LORA_CONFIG.r == 16)
    check("LORA_CONFIG.lora_alpha == 32", LORA_CONFIG.lora_alpha == 32)
    check("LORA_CONFIG.lora_dropout == 0.05", LORA_CONFIG.lora_dropout == 0.05)
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    check(
        "LORA_CONFIG.target_modules contains all 7 expected modules",
        expected_modules.issubset(set(LORA_CONFIG.target_modules)),
        f"got {LORA_CONFIG.target_modules}",
    )

    print(f"\nBNB_CONFIG.load_in_4bit = {BNB_CONFIG.load_in_4bit}")
    print(f"BNB_CONFIG.bnb_4bit_quant_type = {BNB_CONFIG.bnb_4bit_quant_type}")
    check("BNB_CONFIG.load_in_4bit == True", BNB_CONFIG.load_in_4bit is True)
    check("BNB_CONFIG.bnb_4bit_quant_type == 'nf4'", BNB_CONFIG.bnb_4bit_quant_type == "nf4")

    print(f"\nTRAINING_CONFIG.num_train_epochs = {TRAINING_CONFIG['num_train_epochs']}")
    print(f"TRAINING_CONFIG.learning_rate = {TRAINING_CONFIG['learning_rate']}")
    print(f"TRAINING_CONFIG.per_device_train_batch_size = {TRAINING_CONFIG['per_device_train_batch_size']}")
    check("TRAINING_CONFIG.num_train_epochs == 3", TRAINING_CONFIG["num_train_epochs"] == 3)
    check("TRAINING_CONFIG.learning_rate == 2e-4", TRAINING_CONFIG["learning_rate"] == 2e-4)
    check("TRAINING_CONFIG.per_device_train_batch_size == 4", TRAINING_CONFIG["per_device_train_batch_size"] == 4)

    print(f"\nMODEL_REGISTRY = {MODEL_REGISTRY}")
    check("MODEL_REGISTRY has exactly 2 entries", len(MODEL_REGISTRY) == 2, f"got {len(MODEL_REGISTRY)}")
    check("MODEL_REGISTRY['sealion'] correct", MODEL_REGISTRY.get("sealion") == "aisingapore/Llama-SEA-LION-v2-8B-IT")
    check("MODEL_REGISTRY['llama'] correct", MODEL_REGISTRY.get("llama") == "meta-llama/Llama-3.1-8B-Instruct")

    sealion_adapter = adapter_dir("sealion")
    llama_adapter = adapter_dir("llama")
    print(f"\nadapter_dir('sealion') = {sealion_adapter}")
    print(f"adapter_dir('llama') = {llama_adapter}")
    check("adapter_dir('sealion') resolves to .../evals/sealion/adapter", sealion_adapter.parts[-3:] == ("evals", "sealion", "adapter"), f"got {sealion_adapter}")
    check("adapter_dir('llama') resolves to .../evals/llama/adapter", llama_adapter.parts[-3:] == ("evals", "llama", "adapter"), f"got {llama_adapter}")

    print(f"\nMAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
    check("MAX_NEW_TOKENS == 8", MAX_NEW_TOKENS == 8, f"got {MAX_NEW_TOKENS}")

    print(f"\nSYSTEM_PROMPT = {SYSTEM_PROMPT!r}")
    # Checks for the old "REASON:" format TAG specifically -- not the
    # bare word "reason" (SYSTEM_PROMPT legitimately says "...reasoning,
    # punctuation..." as normal English, which a naive substring check
    # on "reason" alone would incorrectly flag).
    check("SYSTEM_PROMPT does not mention the old 'REASON:' format tag", "REASON:" not in SYSTEM_PROMPT)


def print_final_summary():
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{total} checks passed")

    failed = [(desc, detail) for desc, ok, detail in results if not ok]
    if failed:
        print("\nFAILED checks:")
        for desc, detail in failed:
            suffix = f" ({detail})" if detail else ""
            print(f"  - {desc}{suffix}")
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    test_1_choice_map_integrity()
    benchmark_records, fewshot_records = test_2_jsonl_roundtrip()
    test_3_build_mcq_question(benchmark_records)
    test_4_build_fewshot_block(fewshot_records)
    test_5_parse_response_valid()
    test_6_parse_response_invalid()
    test_7_config_sanity()
    print_final_summary()
