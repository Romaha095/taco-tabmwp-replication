import json
import os
import re

RAW_DIR = "data/raw/tabmwp"
OUT_PATH = "data/processed/tabmwp.jsonl"


def linearize_table(table_pd: dict) -> str:
    """Линеаризація using table_for_pd."""
    cols = list(table_pd.keys())
    col1 = cols[0]
    col2 = cols[1]

    rows = []
    rows.append(f"HEAD: {col1.strip()} | {col2.strip()}")

    for i, (a, b) in enumerate(zip(table_pd[col1], table_pd[col2]), start=1):
        rows.append(f"ROW{i}: {a} | {b}")

    return "\n".join(rows)


def load_raw():
    files = [
        "problems_train.json",
        "problems_dev.json",
        "problems_test.json"
    ]

    all_examples = []

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        raw = json.load(open(path, "r", encoding="utf-8"))
        all_examples.extend(list(raw.values()))

    return all_examples


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    raw = load_raw()

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, ex in enumerate(raw):
            table_lin = linearize_table(ex["table_for_pd"])

            item = {
                "id": i,
                "split": ex["split"],
                "table_title": ex["table_title"],
                "table_linearized": table_lin,
                "question": ex["question"],
                "cot_solution": ex["solution"],
                "answer": ex["answer"],
                "ques_type": ex["ques_type"],
                "ans_type": ex["ans_type"],
                "grade": ex["grade"]
            }

            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Created:", OUT_PATH)
