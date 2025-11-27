import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict
from inspect import signature

from datasets import load_from_disk
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed as hf_set_seed,
)

from models.tapex_loader import load_tapex
from utils.seed import set_seed_all
from utils.logger import get_logger, HFLossLoggingCallback


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(project_root: Path, config_path_str: str) -> Dict[str, Any]:
    config_path = Path(config_path_str)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 (Answer generator) with TAPEX on TABMWP"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stage2_tapex_large.json",
        help="Path to JSON config with hyper-parameters (relative to project root).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/stage2",
        help="Path to HF dataset for Stage 2 (created by build_stage2.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/stage2_tapex_large",
        help="Where to save TAPEX checkpoints for Stage 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    log_file = project_root / "logs" / "train_stage2.log"
    logger = get_logger("train_stage2", log_file=log_file)
    logger.info(f"Project root: {project_root}")

    get_logger("transformers.trainer", log_file=log_file)

    # ---------- Load config ----------
    cfg = load_config(project_root, args.config_path)
    logger.info(f"Loaded config from {args.config_path}: {cfg}")

    model_name = cfg.get("model_name", "microsoft/tapex-large")
    max_input_length = int(cfg.get("max_input_length", 768))
    max_output_length = int(cfg.get("max_output_length", 256))
    learning_rate = float(cfg.get("learning_rate", 3e-5))
    batch_size = int(cfg.get("batch_size", 4))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    num_train_epochs = float(cfg.get("num_train_epochs", 20))
    save_steps = int(cfg.get("save_steps", 1000))
    eval_steps = int(cfg.get("eval_steps", 1000))
    logging_steps = int(cfg.get("logging_steps", 200))
    predict_with_generate = bool(cfg.get("predict_with_generate", False))
    fp16 = bool(cfg.get("fp16", True))
    warmup_fraction = float(cfg.get("warmup_fraction", 0.1))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    # ---------- Seed ----------
    set_seed_all(args.seed)
    hf_set_seed(args.seed)
    logger.info(f"Using seed = {args.seed}")

    # ---------- Load dataset ----------
    dataset_path = project_root / args.dataset_path
    logger.info(f"Loading Stage 2 dataset from {dataset_path}")
    raw_datasets = load_from_disk(str(dataset_path))

    # ---------- Load TAPEX ----------
    logger.info(f"Loading TAPEX model '{model_name}' for Stage 2")
    tokenizer, model = load_tapex(model_name)

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        getattr(model.config, "decoder_start_token_id", None) is None
        and getattr(tokenizer, "bos_token_id", None) is not None
    ):
        model.config.decoder_start_token_id = tokenizer.bos_token_id

    # ---------- Tokenization ----------
    column_names = raw_datasets["train"].column_names
    logger.info(f"Columns in raw Stage 2 dataset: {column_names}")

    # Stage 2: input_text = table + question + CoT, target_text = answer
    def preprocess_function(examples):
        # encode encoder input
        model_inputs = tokenizer(
            answer=examples["input_text"],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                answer=examples["target_text"],
                max_length=max_output_length,
                padding="max_length",
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info(
        f"Tokenizing Stage 2 dataset (max_input_length={max_input_length}, "
        f"max_output_length={max_output_length})"
    )

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing Stage 2",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation", None)

    logger.info(f"Stage 2 train size: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"Stage 2 validation size: {len(eval_dataset)}")
    else:
        logger.warning("No validation split found for Stage 2; evaluation will be disabled.")

    # ---------- Training arguments ----------
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_train_batch = batch_size * grad_accum
    logger.info(
        f"Stage 2: per-device batch size = {batch_size}, "
        f"gradient_accumulation_steps = {grad_accum} "
        f"-> effective batch size = {total_train_batch}"
    )

    train_steps_per_epoch = math.ceil(len(train_dataset) / total_train_batch)
    max_train_steps = int(train_steps_per_epoch * num_train_epochs)
    warmup_steps = int(max_train_steps * warmup_fraction)
    logger.info(
        f"Stage 2: train steps/epoch ≈ {train_steps_per_epoch}, "
        f"num_train_epochs = {num_train_epochs}, "
        f"max_train_steps ≈ {max_train_steps}, "
        f"warmup_fraction = {warmup_fraction} -> warmup_steps = {warmup_steps}"
    )

    args_dict: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "linear",
        "max_steps": -1,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "save_strategy": "steps",
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "predict_with_generate": predict_with_generate,
        "fp16": fp16,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": False,
        "load_best_model_at_end": False,
        "save_total_limit": 2,
        "report_to": ["none"],
        "label_smoothing_factor": 0.0,
        "max_grad_norm": 1.0,
    }

    # filter args for current transformers version
    sig = signature(Seq2SeqTrainingArguments.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self", "*args", "**kwargs"}
    filtered_args = {k: v for k, v in args_dict.items() if k in valid_keys}

    missing = sorted(set(args_dict.keys()) - set(filtered_args.keys()))
    if missing:
        logger.info(
            f"Seq2SeqTrainingArguments in your transformers version "
            f"does not support: {missing}. They are skipped."
        )

    training_args = Seq2SeqTrainingArguments(**filtered_args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.add_callback(HFLossLoggingCallback(logger))

    logger.info("Starting training for Stage 2 (answer generation)")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset is not None:
        logger.info("Running final loss-only evaluation on Stage 2 validation set")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        logger.info(f"Final Stage 2 eval (loss) metrics: {eval_metrics}")

    logger.info("Training for Stage 2 finished.")


if __name__ == "__main__":
    main()
