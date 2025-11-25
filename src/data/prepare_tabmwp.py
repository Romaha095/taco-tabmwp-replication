import json
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


project_root = get_project_root()
RAW_DIR = project_root / "data" / "raw" / "tabmwp"
OUT_PATH = project_root / "data" / "processed" / "tabmwp.jsonl"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def linearize_table(table_pd: dict) -> str:
    cols = list(table_pd.keys())
    col1, col2 = cols[0], cols[1]

    rows = [f"HEAD: {col1.strip()} | {col2.strip()}"]

    for i, (a, b) in enumerate(zip(table_pd[col1], table_pd[col2]), start=1):
        rows.append(f"ROW{i}: {a} | {b}")

    return "\n".join(rows)


def read_split(filename, split_name):
    path = RAW_DIR / filename
    print(f"Loading {split_name} from {path}...")

    raw = json.load(open(path, "r", encoding="utf-8"))
    examples = []

    for ex in raw.values():
        ex["split"] = split_name   # <<< ВАЖНО
        examples.append(ex)

    return examples


if __name__ == "__main__":
    ensure_dir(OUT_PATH.parent)

    train = read_split("problems_train.json", "train")
    dev   = read_split("problems_dev.json", "dev")
    test  = read_split("problems_test.json", "test")

    all_examples = train + dev + test

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, ex in enumerate(all_examples):
            item = {
                "id": i,
                "split": ex["split"],
                "table_title": ex["table_title"],
                "table_linearized": linearize_table(ex["table_for_pd"]),
                "question": ex["question"],
                "cot_solution": ex["solution"],
                "answer": ex["answer"],
                "ques_type": ex["ques_type"],
                "ans_type": ex["ans_type"],
                "grade": ex["grade"]
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("DONE! Saved", len(all_examples), "examples →", OUT_PATH)
