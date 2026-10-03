"""Re-parse CLI: `python reparse_results.py --results-dir <dir> [--benchmark-path <path>]`.

No GPU needed -- re-parses `raw_output` on already-generated results
files (records where `parse_error == True`) using looser recovery rules
than `common.parse_response`'s strict single-letter match, and writes
the results to new `{stem}_reparsed.jsonl` files alongside the originals.
Never overwrites an original results file.

Works against either folder layout without modification:
(a) the standard `evals/{model}/results/{mode}.jsonl` nested structure, or
(b) a flat folder containing `zero_shot.jsonl` / `few_shot.jsonl` /
    `finetuned.jsonl` directly (e.g. a trial run folder like
    `lanz_trial_code_results/`).
"""

import argparse
import re
from pathlib import Path

from common import BENCHMARK_PATH, read_jsonl, write_jsonl

RESULT_FILENAMES = ["zero_shot.jsonl", "few_shot.jsonl", "finetuned.jsonl"]

LETTER_RE = re.compile(r"\b([A-Da-d])\.")


def extract_letters(text: str) -> list[str]:
    return sorted(set(m.group(1).lower() for m in LETTER_RE.finditer(text)))


def extract_whole_word_choices(text: str, choices: list[str]) -> list[int]:
    """Case-insensitive (handles sentence-initial capitalization like
    'Kayâ'), but diacritic-sensitive -- .lower() normalizes case without
    merging distinct diacritic marks (á/à/â stay distinct)."""
    text_lower = text.lower()
    found = []
    for i, choice in enumerate(choices):
        pattern = r"(?<!\w)" + re.escape(choice.lower()) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(i)
    return found


def smart_parse(raw_output: str, choices: list[str]) -> dict:
    """
    Returns {"answer": <letter or None>, "method": <str>, "ambiguous": <bool>}
    Priority:
      1. Strict single-letter-only match (common.py's parse_response) --
         cleanest case, unchanged.
      2. Whole-word matching, checked FIRST:
         - Multiple distinct choices found as whole words -> immediately
           ambiguous, letters are not even consulted.
         - Exactly one choice found as a whole word -> tentatively
           accepted, then cross-validated against letters (step 3).
         - Zero choices found as whole words -> fall through to
           letter-only resolution (step 4).
      3. Cross-validation (only runs when step 2 found exactly one word
         match): if exactly one distinct letter is also present, it must
         agree with the word's position ("letter_and_word_agree") or the
         whole thing is ambiguous ("letter_word_conflict") -- this
         catches cases like "A. kayâ" where position A is actually a
         different choice than kayâ, a genuine inconsistency worth
         flagging rather than silently resolving one way. If zero or
         multiple letters are present, they don't provide a clean
         confirmation OR contradiction, so the word match stands alone
         ("whole_word").
      4. Letter-only fallback (only runs when step 2 found ZERO word
         matches): exactly one distinct letter -> accept
         ("leading_letter"). Multiple distinct letters -> ambiguous
         ("leading_letter"). Zero letters -> unrecoverable ("none").
    """
    from common import parse_response

    strict = parse_response(raw_output)
    if strict is not None:
        return {"answer": strict["answer"], "method": "strict", "ambiguous": False}

    word_matches = extract_whole_word_choices(raw_output, choices)

    # Step 2: whole-word matching checked FIRST.
    if len(word_matches) > 1:
        return {"answer": None, "method": "whole_word", "ambiguous": True}

    if len(word_matches) == 1:
        word_answer = "abcd"[word_matches[0]]
        letters = extract_letters(raw_output)

        # Step 3: cross-validate against letters, only when exactly one
        # letter is present -- otherwise the word match stands alone.
        if len(letters) == 1:
            letter_answer = letters[0]
            if letter_answer == word_answer:
                return {"answer": word_answer, "method": "letter_and_word_agree", "ambiguous": False}
            else:
                return {"answer": None, "method": "letter_word_conflict", "ambiguous": True}

        return {"answer": word_answer, "method": "whole_word", "ambiguous": False}

    # Step 4: no word match at all -- fall back to letters only.
    letters = extract_letters(raw_output)
    if len(letters) == 1:
        return {"answer": letters[0], "method": "leading_letter", "ambiguous": False}
    if len(letters) > 1:
        return {"answer": None, "method": "leading_letter", "ambiguous": True}

    return {"answer": None, "method": "none", "ambiguous": False}


# --------------------------------------------------------------------------
# Step 1: Detect folder layout and find results files
# --------------------------------------------------------------------------


def find_results_files(results_dir: Path) -> list[tuple[Path, str]]:
    """Returns (path, label) pairs. Flat layout is checked first -- if
    ANY of the 3 expected filenames sit directly inside results_dir,
    that's the detected layout and only that layout is used, even if an
    evals/ subfolder also happens to exist. Otherwise falls back to the
    nested evals/{model}/results/ layout."""
    flat_matches = []
    for filename in RESULT_FILENAMES:
        path = results_dir / filename
        if path.exists():
            flat_matches.append((path, filename.removesuffix(".jsonl")))

    if flat_matches:
        print(f"Detected FLAT layout under {results_dir.resolve()}")
        missing = [f for f in RESULT_FILENAMES if not (results_dir / f).exists()]
        for f in missing:
            print(f"  WARNING: {f} not found in {results_dir.resolve()} -- skipping.")
        return flat_matches

    nested_matches = []
    for results_subdir in sorted(results_dir.glob("evals/*/results")):
        model_key = results_subdir.parent.name
        for filename in RESULT_FILENAMES:
            path = results_subdir / filename
            if path.exists():
                nested_matches.append((path, f"{model_key}/{filename.removesuffix('.jsonl')}"))

    if nested_matches:
        print(f"Detected NESTED layout (evals/{{model}}/results/) under {results_dir.resolve()}")
        return nested_matches

    raise FileNotFoundError(
        f"No results files found under {results_dir.resolve()} -- expected either "
        f"{RESULT_FILENAMES} directly inside it, or an evals/{{model}}/results/ "
        f"subtree containing them."
    )


# --------------------------------------------------------------------------
# Step 2: Re-parse a single results file
# --------------------------------------------------------------------------


def reparse_file(path: Path, label: str, benchmark_lookup: dict) -> tuple[list[dict], dict]:
    """Returns (updated_records, stats) where stats collects everything
    Step 3 needs to report, without re-deriving it from mutated records
    afterward."""
    records = read_jsonl(path)

    original_parse_errors = [r for r in records if r["parse_error"]]
    stats = {
        "label": label,
        "total": len(records),
        "original_parse_error_count": len(original_parse_errors),
        "no_benchmark_match": [],
        "recovered": {"leading_letter": [], "whole_word": [], "letter_and_word_agree": []},
        "ambiguous": {"leading_letter": [], "whole_word": [], "letter_word_conflict": []},
        "unrecoverable": [],
    }

    for record in records:
        if not record["parse_error"]:
            record["recovery_method"] = "strict"
            record["ambiguous"] = False
            continue

        choices = benchmark_lookup.get(record["id"])
        if choices is None:
            record["recovery_method"] = "no_benchmark_match"
            record["ambiguous"] = False
            stats["no_benchmark_match"].append(record)
            continue

        result = smart_parse(record["raw_output"], choices)
        resolved = result["answer"] is not None and not result["ambiguous"]

        record["parsed_answer"] = result["answer"]
        record["parse_error"] = not resolved
        record["recovery_method"] = result["method"]
        record["ambiguous"] = result["ambiguous"]

        if resolved:
            record["correct"] = ord(result["answer"]) - ord("a") == record["label"]
            stats["recovered"][result["method"]].append(record)
        elif result["ambiguous"]:
            stats["ambiguous"][result["method"]].append(record)
        else:
            stats["unrecoverable"].append(record)

    return records, stats


# --------------------------------------------------------------------------
# Step 3: Report
# --------------------------------------------------------------------------


def print_report(stats: dict, benchmark_lookup: dict):
    print(f"\n{'=' * 70}")
    print(f"{stats['label']}  ({stats['total']} total record(s))")
    print("=" * 70)
    print(f"Original parse errors: {stats['original_parse_error_count']}")

    recovered_total = sum(len(v) for v in stats["recovered"].values())
    print(f"\nRecovered: {recovered_total}")
    for method, items in stats["recovered"].items():
        print(f"  {method:24s} {len(items)}")

    ambiguous_total = sum(len(v) for v in stats["ambiguous"].values())
    print(f"\nAmbiguous: {ambiguous_total}")
    for method, items in stats["ambiguous"].items():
        print(f"  {method:24s} {len(items)}")

    if stats["no_benchmark_match"]:
        print(f"\nNo benchmark match (id not found in benchmark.jsonl): {len(stats['no_benchmark_match'])}")

    print(f"\nFully unrecoverable (method=none): {len(stats['unrecoverable'])}")

    if ambiguous_total:
        print("\n--- Ambiguous cases (full detail, for manual review) ---")
        for method, items in stats["ambiguous"].items():
            if not items:
                continue
            print(f"\n[{method}]")
            for r in items:
                print(f"  id={r['id']!r}")
                print(f"    raw_output={r['raw_output']!r}")
                print(f"    choices={benchmark_lookup.get(r['id'])!r}")

    print("\n--- Example recovered records (up to 5 per method) ---")
    for method, items in stats["recovered"].items():
        if not items:
            continue
        print(f"\n[{method}]")
        for r in items[:5]:
            print(f"  {r['raw_output']!r} -> {r['parsed_answer']!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", required=True,
        help="Folder containing zero_shot.jsonl / few_shot.jsonl / finetuned.jsonl "
             "directly (flat layout), or an evals/{model}/results/ subtree (nested layout).",
    )
    parser.add_argument(
        "--benchmark-path", default=None,
        help="Path to benchmark.jsonl. Defaults to common.py's BENCHMARK_PATH if not given.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmark_path = Path(args.benchmark_path) if args.benchmark_path else BENCHMARK_PATH

    print(f"Results dir:    {results_dir.resolve()}")
    print(f"Benchmark path: {benchmark_path.resolve()}")

    benchmark_records = read_jsonl(benchmark_path)
    benchmark_lookup = {r["id"]: r["choices"] for r in benchmark_records}
    print(f"Loaded {len(benchmark_records)} benchmark record(s), {len(benchmark_lookup)} unique id(s).")

    results_files = find_results_files(results_dir)
    print(f"\nFound {len(results_files)} results file(s):")
    for path, label in results_files:
        print(f"  {label:24s} {path.resolve()}")

    for path, label in results_files:
        records, stats = reparse_file(path, label, benchmark_lookup)

        out_path = path.parent / f"{path.stem}_reparsed.jsonl"
        write_jsonl(out_path, records)
        print(f"\nWrote {len(records)} record(s) to {out_path.resolve()}")

        print_report(stats, benchmark_lookup)


if __name__ == "__main__":
    main()
