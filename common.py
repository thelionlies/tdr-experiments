
"""Shared module imported by every other script in this project
(model_setup.py, finetune.py, run_eval.py, compute_metrics.py)."""

import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
EVALS_DIR = Path(__file__).parent / "evals"

FINETUNE_TRAIN_PATH = DATA_DIR / "finetune_train.jsonl"
FINETUNE_VAL_PATH = DATA_DIR / "finetune_val.jsonl"
BENCHMARK_PATH = DATA_DIR / "benchmark.jsonl"
FEWSHOT_PATH = DATA_DIR / "few_shot_examples.jsonl"

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

MODEL_REGISTRY = {
    "sealion": "aisingapore/Llama-SEA-LION-v2-8B-IT",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def adapter_dir(model_key: str) -> Path:
    """Where finetune.py saves, and run_eval.py loads, the LoRA adapter
    for a given model key."""
    assert model_key in MODEL_REGISTRY, f"Unknown model key: {model_key}"
    return EVALS_DIR / model_key / "adapter"


# --------------------------------------------------------------------------
# Generation / prompting config
# --------------------------------------------------------------------------

MAX_NEW_TOKENS = 8

SYSTEM_PROMPT = (
    "You are a Tagalog diacritic restoration assistant. Answer using "
    "ONLY a single letter: A, B, C, or D. Do not include any explanation, "
    "reasoning, punctuation, or extra text -- output ONLY the letter."
)

# --------------------------------------------------------------------------
# JSONL helpers (defined first -- CHOICE_MAP derivation below depends on
# read_jsonl)
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CHOICE_MAP -- derived from benchmark.jsonl, NOT hardcoded
# --------------------------------------------------------------------------


def _derive_choice_map() -> dict:
    """Reconstructs {homograph: [4 base choice strings]} from
    benchmark.jsonl, using the first record found per homograph. The
    order of the 4 strings here doesn't matter -- callers (e.g.
    build_fewshot_block) re-shuffle deterministically per use."""
    records = read_jsonl(BENCHMARK_PATH)
    choice_map = {}
    for r in records:
        h = r["homograph"]
        if h not in choice_map:
            choice_map[h] = list(r["choices"])
    return choice_map


CHOICE_MAP = _derive_choice_map()
assert len(CHOICE_MAP) == 40, f"Expected 40 homographs, got {len(CHOICE_MAP)}"
assert all(len(v) == 4 for v in CHOICE_MAP.values())

# --------------------------------------------------------------------------
# LoRA / quantization / training config
# --------------------------------------------------------------------------

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# These must be IDENTICAL for both models -- no per-model branching or
# overrides anywhere in this file.
TRAINING_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=512,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    logging_steps=50,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def load_model_and_tokenizer(model_key: str, adapter: bool = False, for_training: bool = False):
    assert model_key in MODEL_REGISTRY, f"Unknown model key: {model_key}"
    # for_training=True means finetune.py is about to attach a FRESH,
    # untrained LoraConfig itself (via LORA_CONFIG), so this function must
    # NOT load any existing/saved adapter in that case, even if one
    # already exists on disk from a prior run.
    assert not (adapter and for_training), "adapter=True and for_training=True are mutually exclusive"

    model_name = MODEL_REGISTRY[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter and not for_training:
        path = adapter_dir(model_key)
        assert path.exists(), f"Adapter directory not found: {path}"
        model = PeftModel.from_pretrained(model, path)
        print(f"Loaded {model_key} base model with adapter from {path}")
    else:
        print(f"Loaded {model_key} base model (no adapter) from {model_name}")

    return model, tokenizer


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------


def build_chat_prompt(tokenizer, user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def build_mcq_question(homograph: str, sentence: str, choices: list[str]) -> str:
    assert len(choices) == 4, f"choices must have exactly 4 entries, got {len(choices)}"
    letters = ["A", "B", "C", "D"]
    lines = [
        f'How should the word {homograph} be written with the correct diacritic '
        f'marks in the sentence "{sentence}"?',
        "",
    ]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_fewshot_block(homograph: str, all_fewshot_records: list[dict]) -> str:
    """
    Pulls ALL few-shot records for the given homograph (across all its
    variants), formats each as a solved MCQ (question + ANSWER only).
    Returns the combined block as a string.
    """
    relevant = [
        r for r in all_fewshot_records
        if r["homograph"] == homograph and r.get("sentence")
    ]
    choices = CHOICE_MAP[homograph]
    letters = ["A", "B", "C", "D"]
    blocks = []

    for i, ex in enumerate(relevant):
        rng = random.Random(f"{homograph}_{i}")
        shuffled = choices[:]
        rng.shuffle(shuffled)
        answer_idx = shuffled.index(ex["variant"])
        answer_letter = letters[answer_idx]

        question_text = build_mcq_question(homograph, ex["sentence"], shuffled)
        blocks.append(f"Example:\n{question_text}\nANSWER: {answer_letter}\n")

    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

RESPONSE_PATTERN = re.compile(r"^\s*([A-Da-d])\s*$")


def parse_response(text: str) -> dict | None:
    """Returns {'answer': 'a'|'b'|'c'|'d'} on success, or None if the
    response is anything other than a single A/B/C/D letter (a parse
    error -- caller must log/count this, NOT retry or coerce it)."""
    match = RESPONSE_PATTERN.match(text.strip())
    if not match:
        return None
    return {"answer": match.group(1).lower()}
