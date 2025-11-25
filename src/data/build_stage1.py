import json
from pathlib import Path
from datasets import Dataset, DatasetDict


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


project_root = get_project_root()
IN_PATH = project_root / "data" / "processed" / "tabmwp.jsonl"
OUT_DIR = project_root / "data" / "stage1"


def load_processed():
    return [
        json.loads(line)
        for line in IN_PATH.read_text(encoding="utf-8").splitlines()
    ]


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT_DIR))

    print("Saved Stage1:", OUT_DIR)
