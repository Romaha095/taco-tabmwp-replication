import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

DATA_DIR = "data/stage2"
MODEL_NAME = "google/flan-t5-large"   # можна змінити на t5-base / llama-8b-instruct


def load_dataset():
    print("Loading Stage2 dataset from:", DATA_DIR)
    return load_from_disk(DATA_DIR)


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["input_text"],
        text_target=example["target_text"],
        truncation=True,
        max_length=512
    )


def main():
    # 1) Load data
    ds = load_dataset()

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3) Tokenize
    tokenized = ds.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["input_text", "target_text", "id"]
    )

    # 4) Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir="checkpoints_stage2",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=200,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        learning_rate=3e-5,
        num_train_epochs=3,

        predict_with_generate=True,
        fp16=True,          # якщо GPU підтримує
        bf16=False,
    )

    # 5) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    # 6) Train
    print("Starting Stage2 training...")
    trainer.train()

    # 7) Save final model
    trainer.save_model("stage2_model_final")
    print("Stage2 training complete! Saved to stage2_model_final/")


if __name__ == "__main__":
    main()
