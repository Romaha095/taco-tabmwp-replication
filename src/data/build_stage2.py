import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset, DatasetDict


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def has_non_empty_solution(example: Dict[str, Any]) -> bool:
    sol = example.get("solution")
    if sol is None:
        return False
    if isinstance(sol, list):
        return any(str(s).strip() for s in sol)
    return bool(str(sol).strip())


def solution_to_text(sol: Any) -> str:
    if sol is None:
        return ""
    if isinstance(sol, list):
        return "\n".join(str(s).strip() for s in sol if str(s).strip())
    return str(sol).strip()


def answer_to_text(ans: Any) -> str:
    if ans is None:
        return ""
    return str(ans).strip()


def build_input_text_stage2(example: Dict[str, Any]) -> str:
    table_part = example.get("table_linearized", "")
    question = example.get("question", "")
    solution = solution_to_text(example.get("solution"))

    parts = []
    if table_part:
        parts.append(table_part)
    if question:
        parts.append(f"Question: {question}")
    parts.append("Let's think step by step.")
    if solution:
        parts.append(solution)

    return "\n".join(parts)


def has_non_empty_solution_and_answer(example: Dict[str, Any]) -> bool:
    if not has_non_empty_solution(example):
        return False
    ans = example.get("answer")
    if ans is None:
        return False
    return bool(str(ans).strip())


def parse_args_stage2() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HF dataset for Stage 2 (answer inference) from processed TABMWP."
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/tabmwp",
        help="Directory with processed TABMWP JSONL files (relative to project root).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/stage2",
        help="Where to save HF dataset (relative to project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args_stage2()
    project_root = get_project_root()

    processed_dir = project_root / args.processed_dir
    output_path = project_root / args.output_path

    data_files = {}
    mapping = {
        "train": "train.jsonl",
        "validation": "dev.jsonl",
        "test": "test.jsonl",
    }
    for split, filename in mapping.items():
        path = processed_dir / filename
        if path.exists():
            data_files[split] = str(path)
        else:
            print(f"[!] Processed file for split '{split}' not found: {path}", file=sys.stderr)

    if "train" not in data_files:
        print("[!] No train split found, cannot build Stage 2 dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading processed data from: {processed_dir}")
    raw_ds = load_dataset("json", data_files=data_files)

    print("[*] Filtering examples without gold solution or gold answer (Stage 2 needs both).")
    filtered_ds = DatasetDict()
    for split, ds in raw_ds.items():
        before = ds.num_rows
        ds_filtered = ds.filter(has_non_empty_solution_and_answer)
        after = ds_filtered.num_rows
        print(f"[+] Split '{split}': before={before}, after={after}")
        filtered_ds[split] = ds_filtered

    def map_example(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input_text": build_input_text_stage2(example),
            "target_text": answer_to_text(example.get("answer")),
        }

    print("[*] Mapping to (input_text, target_text) pairs for Stage 2.")
    stage2_ds = DatasetDict()
    for split, ds in filtered_ds.items():
        mapped = ds.map(map_example)
        stage2_ds[split] = mapped

    ensure_dir(output_path)
    print(f"[*] Saving Stage 2 dataset to {output_path}")
    stage2_ds.save_to_disk(str(output_path))
    print("[✓] Stage 2 dataset saved successfully.")


if __name__ == "__main__":
    main()
