import argparse
import json
import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"[!] Expected dict at top-level in {path}, got {type(data)}", file=sys.stderr)
        sys.exit(1)
    return data


def build_question(prob: dict, option_inds=("A", "B", "C", "D", "E", "F")) -> str:
    q = str(prob.get("question", "")).strip()

    unit = str(prob.get("unit", "")).strip()
    if unit:
        q = f"{q} (Unit: {unit})"

    choices = prob.get("choices")
    if choices:
        for i, c in enumerate(choices):
            if i >= len(option_inds):
                break
            q += f" ({option_inds[i]}) {c}"

    return q.strip()


def normalize_solution(sol) -> str:
    if sol is None:
        return ""
    if isinstance(sol, list):
        parts = [str(s).strip() for s in sol if str(s).strip()]
        return "\n".join(parts)
    return str(sol).strip()


def process_split(input_path: Path, output_path: Path, split: str) -> None:
    print(f"[*] Processing split='{split}' from {input_path}")
    problems = load_split(input_path)

    ensure_dir(output_path.parent)

    pids = list(problems.keys())
    num_total = len(pids)
    num_written = 0
    num_skipped_empty_table = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for pid in pids:
            prob = problems[pid]

            table_text = str(prob.get("table", "")).strip()
            if not table_text:
                num_skipped_empty_table += 1
                continue

            example = {
                "id": pid,
                "split": split,
                "question": build_question(prob),
                "answer": str(prob.get("answer", "")).strip(),
                "solution": normalize_solution(prob.get("solution")),
                "table_linearized": table_text,
            }

            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            num_written += 1

    print(
        f"[+] Split '{split}': total={num_total}, "
        f"written={num_written}, skipped_empty_table={num_skipped_empty_table}"
    )
    print(f"[+] Saved processed split to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TABMWP in a simple, TaCo-style processed format."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw/tabmwp",
        help="Directory with problems_train/dev/test.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/tabmwp",
        help="Where to write processed JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    raw_dir = project_root / args.raw_dir
    out_dir = project_root / args.output_dir
    ensure_dir(out_dir)

    splits = {
        "train": "problems_train.json",
        "dev": "problems_dev.json",
        "test": "problems_test.json",
    }

    for split, filename in splits.items():
        input_path = raw_dir / filename
        output_path = out_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"[!] File not found for split '{split}': {input_path}", file=sys.stderr)
            continue
        process_split(input_path, output_path, split)


if __name__ == "__main__":
    main()
