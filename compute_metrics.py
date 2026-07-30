"""Computes final metrics from run_eval.py's results files. Reads only
JSONL files already on disk -- no GPU or model imports needed, kept
independent of common.py's heavier torch/transformers/peft/bitsandbytes
imports on purpose, so this can be re-run cheaply and often as results
come in.

Can be run repeatedly as results files arrive -- doesn't need all 6
conditions to exist, reports on whatever is available and clearly flags
what's missing.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(__file__).parent / "data"
EVALS_DIR = Path(__file__).parent / "evals"
BENCHMARK_PATH = DATA_DIR / "benchmark.jsonl"

LETTERS = ["a", "b", "c", "d"]

CONDITIONS = [
    ("sealion", "zero_shot"), ("sealion", "few_shot"), ("sealion", "finetuned"),
    ("llama", "zero_shot"), ("llama", "few_shot"), ("llama", "finetuned"),
]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_conditions() -> dict:
    """Returns {(model, mode): [result records]} for whichever of the 6
    CONDITIONS have a results file on disk; missing ones are warned about
    and skipped, not treated as an error."""
    loaded = {}
    for model, mode in CONDITIONS:
        path = EVALS_DIR / model / "results" / f"{mode}.jsonl"
        if not path.exists():
            print(f"WARNING: missing results file, skipping ({model}, {mode}): {path}")
            continue
        records = read_jsonl(path)
        loaded[(model, mode)] = records
        print(f"Loaded {len(records)} record(s) for ({model}, {mode}) from {path}")
    return loaded


def build_dataframe(records: list[dict], benchmark_by_id: dict) -> pd.DataFrame:
    """Joins each result record against its benchmark.jsonl record (by
    id) to recover the per-instance `choices` list -- needed to identify
    which letter corresponds to the bare/undiacritized form for the
    bare-selection-rate diagnostic, since results files themselves don't
    carry `choices`."""
    rows = []
    for r in records:
        bench = benchmark_by_id.get(r["id"])
        picked_bare = False
        if bench is not None:
            choices = bench["choices"]
            homograph = r["homograph"]
            if homograph in choices:
                bare_letter = LETTERS[choices.index(homograph)]
                picked_bare = (r["parsed_answer"] == bare_letter)

        row = dict(r)
        row["picked_bare"] = picked_bare
        row["gold_letter"] = LETTERS[r["label"]]
        # Parse errors mapped to an explicit sentinel class so they count
        # against accuracy/F1 rather than being silently dropped.
        row["pred_letter"] = r["parsed_answer"] if not r["parse_error"] else "PARSE_ERROR"
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n": 0,
            "pooled_accuracy": float("nan"),
            "pooled_accuracy_excl_parse_errors": float("nan"),
            "parse_error_rate": float("nan"),
            "macro_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "bare_selection_rate": float("nan"),
        }

    total = len(df)
    parse_errors = int(df["parse_error"].sum())
    parsed_total = total - parse_errors

    # Pooled accuracy: parse errors count as wrong (PARSE_ERROR never
    # equals a real gold letter, so accuracy_score handles this for free).
    pooled_accuracy = accuracy_score(df["gold_letter"], df["pred_letter"])

    if parsed_total:
        parsed = df.loc[~df["parse_error"]]
        pooled_accuracy_excl_parse_errors = accuracy_score(parsed["gold_letter"], parsed["pred_letter"])
    else:
        pooled_accuracy_excl_parse_errors = float("nan")

    parse_error_rate = parse_errors / total

    # Per-homograph accuracy and F1 (F1 = multi-class macro-F1 across
    # that homograph's own A/B/C/D + PARSE_ERROR classes), then averaged
    # (macro) across homographs.
    per_homograph_acc = []
    per_homograph_f1 = []
    for _, group in df.groupby("homograph"):
        per_homograph_acc.append(accuracy_score(group["gold_letter"], group["pred_letter"]))
        per_homograph_f1.append(
            f1_score(group["gold_letter"], group["pred_letter"], average="macro", zero_division=0)
        )

    macro_accuracy = sum(per_homograph_acc) / len(per_homograph_acc) if per_homograph_acc else float("nan")
    macro_f1 = sum(per_homograph_f1) / len(per_homograph_f1) if per_homograph_f1 else float("nan")

    bare_selection_rate = df["picked_bare"].mean()

    return {
        "n": total,
        "pooled_accuracy": pooled_accuracy,
        "pooled_accuracy_excl_parse_errors": pooled_accuracy_excl_parse_errors,
        "parse_error_rate": parse_error_rate,
        "macro_accuracy": macro_accuracy,
        "macro_f1": macro_f1,
        "bare_selection_rate": bare_selection_rate,
    }


def main():
    assert BENCHMARK_PATH.exists(), f"{BENCHMARK_PATH} not found"
    benchmark_records = read_jsonl(BENCHMARK_PATH)
    benchmark_by_id = {r["id"]: r for r in benchmark_records}
    print(f"Loaded {len(benchmark_records)} benchmark record(s) for the choices/bare-form join.\n")

    loaded = load_all_conditions()
    missing = [c for c in CONDITIONS if c not in loaded]
    if missing:
        print(f"\nMissing condition(s), will not appear in the tables below: {missing}")
    if not loaded:
        print("\nNo results files found at all -- nothing to report.")
        return

    final_rows = []
    by_asymmetry_rows = []

    for (model, mode), records in loaded.items():
        df = build_dataframe(records, benchmark_by_id)

        metrics = compute_metrics(df)
        final_rows.append({"model": model, "mode": mode, **metrics})

        for asymmetry_type, group in df.groupby("asymmetry_type"):
            group_metrics = compute_metrics(group)
            by_asymmetry_rows.append({
                "model": model, "mode": mode, "asymmetry_type": asymmetry_type, **group_metrics,
            })

    df_final = pd.DataFrame(final_rows)
    df_by_asymmetry = pd.DataFrame(by_asymmetry_rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    print("\n" + "=" * 70)
    print("Final Results (pooled per model/mode)")
    print("=" * 70)
    print(df_final.to_string(index=False))

    print("\n" + "=" * 70)
    print("Final Results by Asymmetry_Type")
    print("=" * 70)
    print(df_by_asymmetry.to_string(index=False))

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out_final = EVALS_DIR / "final_results.csv"
    out_by_asymmetry = EVALS_DIR / "final_results_by_asymmetry.csv"
    df_final.to_csv(out_final, index=False)
    df_by_asymmetry.to_csv(out_by_asymmetry, index=False)

    print(f"\nSaved {out_final.resolve()}")
    print(f"Saved {out_by_asymmetry.resolve()}")


if __name__ == "__main__":
    main()
