import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to load JSON from {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_id(example: Dict[str, Any], idx: int, split: str) -> str:
    for key in ("pid", "id", "question_id", "problem_id"):
        if key in example:
            return str(example[key])
    return f"{split}_{idx}"


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_table_matrix(example: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    table_pd = example.get("table_for_pd")
    if isinstance(table_pd, dict) and table_pd:
        headers = list(table_pd.keys())

        col_lists: List[List[str]] = []
        max_len = 0
        for h in headers:
            col = table_pd[h]
            if isinstance(col, list):
                col_norm = [normalize_cell(v) for v in col]
            else:
                col_norm = [normalize_cell(col)]
            col_lists.append(col_norm)
            max_len = max(max_len, len(col_norm))

        for col in col_lists:
            if len(col) < max_len:
                col.extend([""] * (max_len - len(col)))

        rows: List[List[str]] = []
        for row_idx in range(max_len):
            row = [col_lists[col_idx][row_idx] for col_idx in range(len(headers))]
            rows.append(row)

        return headers, rows

    table = None
    for key in ("table", "structured_table", "table_struct"):
        if key in example:
            table = example[key]
            break

    if table is None:
        return None, None

    if isinstance(table, dict):
        if "table" in table:
            table_data = table["table"]
        elif "rows" in table:
            table_data = table["rows"]
        else:
            values = list(table.values())
            if values and all(isinstance(v, list) for v in values):
                table_data = values
            else:
                return None, None
    else:
        table_data = table

    if not isinstance(table_data, list) or not table_data:
        return None, None

    if isinstance(table_data[0], dict):
        headers = list(table_data[0].keys())
        rows: List[List[str]] = []
        for row_dict in table_data:
            row = [normalize_cell(row_dict.get(h, "")) for h in headers]
            rows.append(row)
    else:
        rows_2d: List[List[str]] = []
        for row in table_data:
            if isinstance(row, list):
                rows_2d.append([normalize_cell(cell) for cell in row])
            else:
                rows_2d.append([normalize_cell(row)])

        if not rows_2d:
            return None, None

        headers = rows_2d[0]
        body_rows = rows_2d[1:]

        def is_numeric_str(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            return all(ch.isdigit() or ch in ".,-" for ch in s)

        if (not any(h.strip() for h in headers)) or all(is_numeric_str(h) for h in headers):
            num_cols = len(headers)
            headers = [f"Column header {i+1}" for i in range(num_cols)]
            body_rows = rows_2d

        rows = body_rows

    max_len = max(len(r) for r in rows) if rows else len(headers)
    if len(headers) < max_len:
        headers = headers + [f"Column header {i+1}" for i in range(len(headers), max_len)]
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    return headers, rows


def linearize_table(headers: List[str], rows: List[List[str]]) -> str:
    header_part = "[HEAD] : " + " | ".join(headers)
    row_parts = []
    for i, row in enumerate(rows, start=1):
        row_parts.append(f"[ROW] {i} : " + " | ".join(row))
    return " ".join([header_part] + row_parts)


def extract_question(example: Dict[str, Any]) -> str:
    for key in ("question", "question_text", "q", "body"):
        if key in example:
            return normalize_cell(example[key])
    return ""


def extract_answer(example: Dict[str, Any]) -> str:
    for key in ("answer", "label", "gold_ans", "gold_answer"):
        if key in example:
            return normalize_cell(example[key])
    return ""


def extract_solution(example: Dict[str, Any]) -> str:
    sol = None
    for key in ("solution", "solutions", "rationale", "steps"):
        if key in example:
            sol = example[key]
            break

    if sol is None:
        return ""

    if isinstance(sol, list):
        return "\n".join(normalize_cell(s) for s in sol)
    return normalize_cell(sol)


def convert_example(raw_ex: Dict[str, Any], idx: int, split: str) -> Optional[Dict[str, Any]]:
    ex_id = extract_id(raw_ex, idx, split)
    question = extract_question(raw_ex)
    answer = extract_answer(raw_ex)
    solution = extract_solution(raw_ex)

    headers, rows = extract_table_matrix(raw_ex)
    if headers is None or rows is None:
        return None

    table_linearized = linearize_table(headers, rows)

    processed: Dict[str, Any] = {
        "id": ex_id,
        "split": split,
        "question": question,
        "answer": answer,
        "solution": solution,
        "table_linearized": table_linearized,
        "table_headers": headers,
        "table_rows": rows,
    }

    for key in ("ques_type", "ans_type", "grade", "table_title", "row_num", "column_num"):
        if key in raw_ex:
            processed[key] = raw_ex[key]

    return processed


def process_split(input_path: Path, output_path: Path, split: str) -> None:
    print(f"[*] Processing split='{split}' from {input_path}")
    raw = load_json(input_path)

    ensure_dir(output_path.parent)

    num_total = 0
    num_converted = 0
    num_no_table = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        if isinstance(raw, dict):
            if "problems" in raw and isinstance(raw["problems"], list):
                iterable = ((idx, ex) for idx, ex in enumerate(raw["problems"]))
            elif all(isinstance(v, dict) for v in raw.values()):
                iterable = ((idx, ex) for idx, ex in enumerate(raw.values()))
                keys = list(raw.keys())
            else:
                print(f"[!] Could not interpret dict structure in {input_path}", file=sys.stderr)
                sys.exit(1)

            if not ("problems" in raw and isinstance(raw["problems"], list)):
                for idx, key in enumerate(raw.keys()):
                    ex = raw[key]
                    if not isinstance(ex, dict):
                        continue
                    ex = dict(ex)
                    if not any(k in ex for k in ("pid", "id", "question_id", "problem_id")):
                        ex["id"] = key
                    num_total += 1
                    processed = convert_example(ex, idx, split)
                    if processed is None:
                        num_no_table += 1
                        continue
                    out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    num_converted += 1
            else:
                for idx, ex in enumerate(raw["problems"]):
                    num_total += 1
                    processed = convert_example(ex, idx, split)
                    if processed is None:
                        num_no_table += 1
                        continue
                    out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    num_converted += 1

        elif isinstance(raw, list):
            for idx, ex in enumerate(raw):
                num_total += 1
                if not isinstance(ex, dict):
                    continue
                processed = convert_example(ex, idx, split)
                if processed is None:
                    num_no_table += 1
                    continue
                out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                num_converted += 1
        else:
            print(f"[!] Unexpected JSON structure in {input_path}: type={type(raw)}", file=sys.stderr)
            sys.exit(1)

    print(
        f"[+] Split '{split}': total={num_total}, "
        f"converted={num_converted}, skipped_no_table={num_no_table}"
    )
    print(f"[+] Saved processed split to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw TABMWP JSON into processed tabular format."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw/tabmwp",
        help="Path to directory with raw TABMWP JSON (relative to project root).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/tabmwp",
        help="Where to write processed JSONL files (relative to project root).",
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