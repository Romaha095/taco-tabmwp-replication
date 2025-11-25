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


def build_input_text(example: Dict[str, Any]) -> str:
    """
    Вход для Stage 1: T* + Q + триггер.
    """
    table_part = example.get("table_linearized", "")
    question = example.get("question", "")
    parts = []
    if table_part:
        parts.append(table_part)
    if question:
        parts.append(f"Question: {question}")
    parts.append("Let's think step by step.")
    return "\n".join(parts)


def parse_args_stage1() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HF dataset for Stage 1 (CoT generation) from processed TABMWP."
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
        default="data/stage1",
        help="Where to save HF dataset (relative to project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args_stage1()
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
        print("[!] No train split found, cannot build Stage 1 dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading processed data from: {processed_dir}")
    raw_ds = load_dataset("json", data_files=data_files)

    print("[*] Filtering examples without gold solutions (Stage 1 needs gold CoT).")
    filtered_ds = DatasetDict()
    for split, ds in raw_ds.items():
        before = ds.num_rows
        ds_filtered = ds.filter(has_non_empty_solution)
        after = ds_filtered.num_rows
        print(f"[+] Split '{split}': before={before}, after={after}")
        filtered_ds[split] = ds_filtered

    def map_example(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input_text": build_input_text(example),
            "target_text": solution_to_text(example.get("solution")),
        }

    print("[*] Mapping to (input_text, target_text) pairs for Stage 1.")
    stage1_ds = DatasetDict()
    for split, ds in filtered_ds.items():
        mapped = ds.map(map_example)
        stage1_ds[split] = mapped

    ensure_dir(output_path)
    print(f"[*] Saving Stage 1 dataset to {output_path}")
    stage1_ds.save_to_disk(str(output_path))
    print("[✓] Stage 1 dataset saved successfully.")


if __name__ == "__main__":
    main()