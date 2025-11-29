import argparse
import json
import os
import re
from typing import Dict, List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.eval.evaluate_acc import get_scores, print_scores
from src.data.build_stage2_from_stage1 import extract_prediction

def normalize_answer(text: str, unit: str) -> str:
    text = re.sub(r"^[\$]", "", text)
    text = re.sub(r"[\,\.\,\/]$", "", text)

    result = re.match(r"^[-+]?[\d,./]+$", text)

    if result is not None:
        text_clean = text.replace(",", "")
        result_int = re.match(r"[-+]?\d+$", text_clean)

        if result_int is not None:
            number = int(text_clean)
        elif "/" in text_clean:
            nums = text_clean.split("/")
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text_clean), 3)

        number_str = str(number)
        number_str = re.sub(r"\.[0]+$", "", number_str)
        return number_str
    else:
        if unit:
            text = text.replace(unit, "").strip()
        return text


PRETRAINED_MODELS = {
    "tapex-base": "microsoft/tapex-base",
    "tapex-large": "microsoft/tapex-large",
    "tapex-base-finetuned-wtq": "microsoft/tapex-base-finetuned-wtq",
    "tapex-large-finetuned-wtq": "microsoft/tapex-large-finetuned-wtq",
}


def run_qt2a_inference(
    model_name_or_path: str,
    data_file: str,
    batch_size: int,
    max_length: int,
    split: str = "test",
) -> Dict[str, Dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    print(f"[*] Loading problems from: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        problems: Dict[str, Dict] = json.load(f)

    all_pids = sorted(problems.keys())
    pids = [
        pid for pid in all_pids
        if not isinstance(problems[pid], dict) or problems[pid].get("split", split) == split
    ]
    print(f"[*] Total problems in file: {len(all_pids)}, used for split '{split}': {len(pids)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    model.eval()

    option_inds = ["A", "B", "C", "D", "E", "F"]

    results: Dict[str, Dict] = {}

    num_batches = (len(pids) + batch_size - 1) // batch_size
    for start_idx in tqdm(
        range(0, len(pids), batch_size),
        total=num_batches,
        desc=f"Inference QT->A ({model_name_or_path})",
    ):
        end_idx = min(start_idx + batch_size, len(pids))
        batch_pids = pids[start_idx:end_idx]

        tables = []
        questions = []
        gold_answers = []
        units = []
        choices_batch = []

        for pid in batch_pids:
            prob = problems[pid]

            table_for_pd = prob["table_for_pd"]
            pd_table = pd.DataFrame.from_dict(table_for_pd)

            question = prob["question"]
            unit = prob.get("unit", None)
            choices = prob.get("choices", None)

            if unit:
                question = question + f" (Unit: {unit})"
            if choices:
                for i, c in enumerate(choices):
                    if i >= len(option_inds):
                        break
                    question += f" ({option_inds[i]}) {c}"

            tables.append(pd_table)
            questions.append(question.strip())
            gold_answers.append(str(prob["answer"]))
            units.append(unit)
            choices_batch.append(choices)

        enc = tokenizer(
            tables,
            questions,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_length=max_length,
            )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for pid, output, gold, unit, choices in zip(
            batch_pids, decoded, gold_answers, units, choices_batch
        ):
            output_str = str(output).strip()
            gold_str = str(gold).strip()
            unit_str = unit if unit is not None else ""

            pred_answer = extract_prediction(output_str, choices)

            answer_norm = normalize_answer(gold_str, unit_str)
            prediction_norm = normalize_answer(pred_answer, unit_str)

            is_correct = str(answer_norm).lower() == str(prediction_norm).lower()

            results[pid] = {
                "answer": gold_str,
                "output": output_str,
                "prediction": pred_answer,
                "answer_norm": answer_norm,
                "prediction_norm": prediction_norm,
                "true_false": bool(is_correct),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/raw/tabmwp/problems_test.json",
        help="TabMWP JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pretrained_qta",
        help="Directory to save result JSON files.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(PRETRAINED_MODELS.keys()),
        choices=list(PRETRAINED_MODELS.keys()),
        help="Which logical TAPEX models to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to filter by in data_file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Max length for generated answers.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_key in args.models:
        hf_name = PRETRAINED_MODELS[model_key]
        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_key}  (HF: {hf_name})")
        print("=" * 80)

        results = run_qt2a_inference(
            model_name_or_path=hf_name,
            data_file=args.data_file,
            batch_size=args.batch_size,
            max_length=args.max_length,
            split=args.split,
        )

        num = len(results)
        num_correct = sum(1 for r in results.values() if r["true_false"])
        acc = round(num_correct / num * 100, 3) if num > 0 else 0.0

        out_file = os.path.join(
            args.output_dir,
            f"{model_key}_qt2a_{args.split}.json",
        )

        result_data = {
            "acc": acc,
            "results": results,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"[{model_key}] Exact-match accuracy (official normalization): {acc:.3f}")
        print(f"[{model_key}] Saved results to: {out_file}")

        try:
            scores = get_scores([out_file], args.data_file)
            print_scores(scores)
        except Exception as e:
            print(f"[WARN] Failed to compute detailed scores via evaluate_acc: {e}")


if __name__ == "__main__":
    main()
