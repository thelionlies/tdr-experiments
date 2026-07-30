"""QLoRA fine-tuning CLI: `python finetune.py --model {sealion|llama}`.

Requires a GPU and will download model weights on first run if not
already cached -- meant to run on the GCP L4 instance, not locally.

Both models use IDENTICAL config: LORA_CONFIG/BNB_CONFIG/TRAINING_CONFIG
are all imported from common.py, never redefined or branched on
model_key here.
"""

import argparse
import inspect
import math

import torch
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from common import (
    FINETUNE_TRAIN_PATH,
    FINETUNE_VAL_PATH,
    LORA_CONFIG,
    TRAINING_CONFIG,
    adapter_dir,
    load_model_and_tokenizer,
    read_jsonl,
)


def format_example(tokenizer, record: dict) -> str:
    messages = [
        {"role": "system", "content": record["instruction"]},
        {"role": "user", "content": record["input"]},
        {"role": "assistant", "content": record["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_sft_config(output_dir) -> SFTConfig:
    """Builds SFTConfig from TRAINING_CONFIG (common.py) plus this run's
    output_dir/dataset_text_field. TRAINING_CONFIG's keys are passed
    through as-is where SFTConfig accepts them; `max_seq_length` is
    remapped to `max_length` if the installed trl version renamed the
    field (this happened across trl releases) -- anything else that
    doesn't match is reported rather than silently dropped, since
    TRAINING_CONFIG must stay untouched in common.py per spec."""
    accepted = inspect.signature(SFTConfig.__init__).parameters
    rename_map = {"max_seq_length": "max_length"}

    config_kwargs = {}
    for key, value in TRAINING_CONFIG.items():
        target_key = key if key in accepted else rename_map.get(key, key)
        if target_key in accepted:
            config_kwargs[target_key] = value
        else:
            print(f"WARNING: TRAINING_CONFIG key '{key}' has no matching SFTConfig field in this trl version; dropped.")

    config_kwargs["output_dir"] = str(output_dir)
    if "dataset_text_field" in accepted:
        config_kwargs["dataset_text_field"] = "text"

    return SFTConfig(**config_kwargs)


# SEA-LION-v2-8B-IT and Llama-3.1-8B-Instruct are both Llama-3-based, so
# they likely share this assistant-turn delimiter -- but that's an
# assumption, not a guarantee, and must be verified per-tokenizer at
# runtime rather than trusted blindly. `verify_response_template` below
# checks it against a real formatted example and fails loudly (not
# silently) if it's ever wrong for a given model/tokenizer.
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def verify_response_template(sample_formatted_text: str) -> str:
    """Prints a real formatted training example and confirms
    RESPONSE_TEMPLATE actually occurs in it, exactly as
    apply_chat_template renders it for THIS tokenizer -- visual
    confirmation plus a hard check, not an assumption."""
    print("\nFormatted example (Step 3) -- visually confirm the assistant "
          "turn's start delimiter below matches RESPONSE_TEMPLATE:")
    print(repr(sample_formatted_text))

    if RESPONSE_TEMPLATE not in sample_formatted_text:
        raise ValueError(
            f"RESPONSE_TEMPLATE {RESPONSE_TEMPLATE!r} was not found in this "
            f"tokenizer's formatted output -- completion-only masking would "
            f"mask EVERYTHING (or crash) rather than just the system/user "
            f"portion. Inspect the printed example above and update "
            f"RESPONSE_TEMPLATE to match this tokenizer's actual chat "
            f"template before proceeding."
        )
    print(f"RESPONSE_TEMPLATE {RESPONSE_TEMPLATE!r} found in formatted output -- OK.")
    return RESPONSE_TEMPLATE


def verify_completion_only_masking(tokenizer, collator, sample_formatted_text: str):
    """Tokenizes one formatted example, runs it through the collator, and
    prints which positions ended up masked (label == -100, i.e. no loss)
    vs unmasked (real label id, i.e. loss computed) -- then decodes just
    the unmasked span so it can be visually confirmed to be ONLY the
    assistant's response, not the system/user portion."""
    tokenized = tokenizer(sample_formatted_text)
    batch = collator([tokenized])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    masked_count = sum(1 for lab in labels if lab == -100)
    unmasked_count = len(labels) - masked_count
    print(f"\nCompletion-only masking check: {len(labels)} total token(s), "
          f"{masked_count} masked (-100), {unmasked_count} unmasked (real labels).")

    unmasked_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded_unmasked = tokenizer.decode(unmasked_ids)
    print("Decoded UNMASKED span (should be ONLY the assistant's response):")
    print(repr(decoded_unmasked))

    if unmasked_count == 0:
        raise ValueError("Every token was masked -- the collator found no response_template match; masking is broken.")


def build_trainer(model, tokenizer, sft_config, train_dataset, eval_dataset, collator) -> SFTTrainer:
    kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    try:
        return SFTTrainer(**kwargs, processing_class=tokenizer)
    except TypeError:
        return SFTTrainer(**kwargs, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["sealion", "llama"])
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------------------
    # Step 2: Load base model for training
    # --------------------------------------------------------------------
    print(f"\nLoading base model for training: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, adapter=False, for_training=True)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    # Gradient checkpointing and KV caching are mutually incompatible
    # during training -- use_cache=True is fine/expected at inference
    # time in run_eval.py, but must be off for this training run.
    model.config.use_cache = False
    model.print_trainable_parameters()

    # --------------------------------------------------------------------
    # Step 3: Load and format training data
    # --------------------------------------------------------------------
    train_records = read_jsonl(FINETUNE_TRAIN_PATH)
    val_records = read_jsonl(FINETUNE_VAL_PATH)

    train_dataset = Dataset.from_list([
        {"text": format_example(tokenizer, r)} for r in train_records
    ])
    eval_dataset = Dataset.from_list([
        {"text": format_example(tokenizer, r)} for r in val_records
    ])

    # --------------------------------------------------------------------
    # Step 3.5: Completion-only loss masking -- verify BEFORE training
    # --------------------------------------------------------------------
    verify_response_template(train_dataset[0]["text"])

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )
    verify_completion_only_masking(tokenizer, collator, train_dataset[0]["text"])

    # --------------------------------------------------------------------
    # Step 4: Configure SFTTrainer
    # --------------------------------------------------------------------
    checkpoints_dir = adapter_dir(args.model).parent / "checkpoints"
    sft_config = build_sft_config(checkpoints_dir)
    print(f"sft_config.gradient_checkpointing = {sft_config.gradient_checkpointing}")
    trainer = build_trainer(model, tokenizer, sft_config, train_dataset, eval_dataset, collator)

    # --------------------------------------------------------------------
    # Step 5: Train
    # --------------------------------------------------------------------
    effective_batch_size = TRAINING_CONFIG["per_device_train_batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    expected_total_steps = steps_per_epoch * TRAINING_CONFIG["num_train_epochs"]

    print(f"\nModel:               {args.model}")
    print(f"Train examples:       {len(train_dataset)}")
    print(f"Val examples:         {len(eval_dataset)}")
    print(f"Effective batch size: {effective_batch_size} "
          f"(per_device={TRAINING_CONFIG['per_device_train_batch_size']} x "
          f"grad_accum={TRAINING_CONFIG['gradient_accumulation_steps']}, single device assumed)")
    print(f"Expected total steps: {expected_total_steps} "
          f"({steps_per_epoch} steps/epoch x {TRAINING_CONFIG['num_train_epochs']} epochs)")
    print(f"Checkpoints dir:      {checkpoints_dir}")
    print("\nStarting training...")

    trainer.train()

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    final_train_loss = train_losses[-1] if train_losses else None
    best_eval_loss = min(eval_losses) if eval_losses else None

    print(f"\nTraining complete.")
    print(f"Final train loss: {final_train_loss}")
    print(f"Best eval loss:   {best_eval_loss}")

    # --------------------------------------------------------------------
    # Step 6: Save the final adapter
    # --------------------------------------------------------------------
    save_path = adapter_dir(args.model)
    save_path.mkdir(parents=True, exist_ok=True)

    # trainer.model is the PeftModel -- save_pretrained here writes only
    # the LoRA adapter weights/config, not the full base model.
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\nSaved adapter to: {save_path.resolve()}")
    for f in sorted(save_path.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.2f} MB)")

    # --------------------------------------------------------------------
    # Step 7: Cleanup note
    # --------------------------------------------------------------------
    print(f"\nNote: {checkpoints_dir} (training checkpoints) can be deleted "
          f"once you're satisfied with the saved adapter above, if disk space "
          f"is needed -- run_eval.py's 'finetuned' condition only reads from "
          f"{save_path}, not the checkpoints directory.")


if __name__ == "__main__":
    main()
