from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk

ds = load_from_disk("data/stage1")

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        text_target=batch["target_text"],
        truncation=True
    )

tokenized = ds.map(tokenize, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir="checkpoints_stage1",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
