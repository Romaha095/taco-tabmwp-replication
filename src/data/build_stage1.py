import json
from datasets import Dataset, DatasetDict

IN_PATH = "data/processed/tabmwp.jsonl"
OUT_DIR = "data/stage1"


def load_processed():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    data = load_processed()

    train, val, test = [], [], []

    for ex in data:
        item = {
            "id": ex["id"],
            "input_text": (
                f"{ex['table_title']}\n"
                f"{ex['table_linearized']}\n\n"
                f"Question: {ex['question']}\n"
                "Let's think step by step."
            ),
            "target_text": ex["cot_solution"]
        }

        if ex["split"] == "train":
            train.append(item)
        elif ex["split"] == "dev":
            val.append(item)
        else:
            test.append(item)

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
    })

    ds.save_to_disk(OUT_DIR)
    print("Saved Stage1:", OUT_DIR)
