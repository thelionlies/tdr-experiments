"""One-time sanity check for both models (SEA-LION, Llama). Run directly,
not imported elsewhere."""

import re

import torch

from common import (
    BENCHMARK_PATH,
    MAX_NEW_TOKENS,
    MODEL_REGISTRY,
    build_chat_prompt,
    build_mcq_question,
    load_model_and_tokenizer,
    parse_response,
    read_jsonl,
)


def _extract_sentence_from_prompt(prompt: str) -> str | None:
    """benchmark.jsonl's exported records don't carry a separate
    Context_Sentence field -- dataprep.ipynb's Step 12 only exported the
    fully-built `prompt` string (the sentence embedded inside it), not
    the raw sentence on its own. Recovered here instead of reading a
    field that doesn't exist."""
    match = re.search(r'in the sentence "(?P<sentence>.*)"\?$', prompt, re.DOTALL)
    return match.group("sentence") if match else None


def _print_gpu_memory(label: str):
    if not torch.cuda.is_available():
        print(f"{label}: CUDA not available, skipping GPU memory report.")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: allocated={allocated:.2f} GB  peak={peak:.2f} GB")


def main():
    print("=" * 70)
    print("Step 1: Load both models")
    print("=" * 70)

    model_keys = list(MODEL_REGISTRY)
    loaded = {}
    load_ok = {}

    for i, model_key in enumerate(model_keys):
        if i > 0 and torch.cuda.is_available():
            # Reset peak stats before loading the SECOND model so its own
            # peak reading isn't polluted by the first model already
            # resident in memory. Nothing is freed between loads, so both
            # models stay resident afterward -- this doubles as the
            # "load both simultaneously" test.
            torch.cuda.reset_peak_memory_stats()

        print(f"\nLoading {model_key} ({MODEL_REGISTRY[model_key]})...")
        try:
            model, tokenizer = load_model_and_tokenizer(model_key, adapter=False, for_training=False)
        except Exception as e:
            print(f"FAILED to load {model_key}: {type(e).__name__}: {e}")
            load_ok[model_key] = False
            continue

        load_ok[model_key] = True
        loaded[model_key] = (model, tokenizer)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model_key:         {model_key}")
        print(f"  num_parameters:    {n_params:,}")
        print(f"  model dtype:       {next(model.parameters()).dtype}")
        print(f"  chat_template set: {tokenizer.chat_template is not None}")
        _print_gpu_memory(f"  GPU memory after loading {model_key}")

    print("\nCombined memory usage after loading both models:")
    if len(loaded) == len(model_keys):
        _print_gpu_memory("  Combined (both models resident)")
        print("  Both models loaded simultaneously: OK (no OOM).")
    else:
        print(f"  Only {len(loaded)}/{len(model_keys)} model(s) loaded successfully -- see failures above.")

    print("\n" + "=" * 70)
    print("Step 2: Check enable_thinking support")
    print("=" * 70)

    enable_thinking_support = {}
    for model_key, (model, tokenizer) in loaded.items():
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            enable_thinking_support[model_key] = True
        except TypeError:
            enable_thinking_support[model_key] = False
        print(f"  {model_key}: enable_thinking supported = {enable_thinking_support[model_key]}")

    print("\n" + "=" * 70)
    print("Step 3: Run one real test generation per model")
    print("=" * 70)

    records = read_jsonl(BENCHMARK_PATH)
    assert records, f"No records found in {BENCHMARK_PATH}"
    record = records[0]

    sentence = _extract_sentence_from_prompt(record["prompt"])
    assert sentence is not None, f"Could not extract sentence from prompt: {record['prompt']!r}"

    print(f"\nUsing benchmark.jsonl record: {record['id']}")
    print(f"  homograph:    {record['homograph']}")
    print(f"  gold_variant: {record['gold_variant']}")
    print(f"  label:        {record['label']}")

    user_message = build_mcq_question(record["homograph"], sentence, record["choices"])

    generation_outputs = {}
    for model_key, (model, tokenizer) in loaded.items():
        full_prompt = build_chat_prompt(tokenizer, user_message)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generation_outputs[model_key] = generated_text

        print(f"\n--- {model_key} raw output ---")
        print(generated_text)

    print("\n" + "=" * 70)
    print("Step 4: Parse and validate")
    print("=" * 70)

    letters = ["a", "b", "c", "d"]
    gold_letter = letters[record["label"]]

    parse_results = {}
    for model_key, generated_text in generation_outputs.items():
        parsed = parse_response(generated_text)
        parse_results[model_key] = parsed is not None
        print(f"\n{model_key}:")
        if parsed is None:
            print("  PARSE ERROR")
            print(f"  Raw text: {generated_text!r}")
        else:
            print(f"  answer: {parsed['answer']}")
            print(f"  matches gold label ({gold_letter}): {parsed['answer'] == gold_letter}")

    print("\n" + "=" * 70)
    print("Step 5: Summary")
    print("=" * 70)

    print(f"Models loaded successfully: {[k for k, ok in load_ok.items() if ok]}")
    failed = [k for k, ok in load_ok.items() if not ok]
    if failed:
        print(f"Models FAILED to load: {failed}")

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU memory headroom remaining: {free_bytes / 1e9:.2f} GB free / {total_bytes / 1e9:.2f} GB total")
    else:
        print("GPU memory headroom remaining: CUDA not available.")

    print(f"enable_thinking supported per model: {enable_thinking_support}")
    print(f"Step 3 outputs parsed successfully under single-letter format: {parse_results}")


if __name__ == "__main__":
    main()
