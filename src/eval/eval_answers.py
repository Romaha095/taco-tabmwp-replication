import argparse
import json
import os
from typing import Dict

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

from evaluate_acc import get_scores, print_scores


def normalize_answer_simple(text: str) -> str:
    return str(text).strip().lower()


def run_inference(
    checkpoint_dir: str,
    dataset_path: str,
    split: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> Dict[str, Dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load Stage2 dataset split
    dataset = load_from_disk(dataset_path)[split]
    print(f"[*] Loaded split '{split}' from {dataset_path}: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    results: Dict[str, Dict] = {}

    num_examples = len(dataset)
    num_batches = (num_examples + batch_size - 1) // batch_size

    for start_idx in tqdm(
        range(0, num_examples, batch_size),
        total=num_batches,
        desc="Stage2 inference",
    ):
        end_idx = min(start_idx + batch_size, num_examples)
        batch = dataset[start_idx:end_idx]

        input_texts = batch["input_text"]
        gold_answers = batch["answer"]
        batch_pids = [str(pid) for pid in batch["pid"]]

        # Tokenize input
        enc = tokenizer(
            answer=input_texts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Generate predictions
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_target_length,
                num_beams=num_beams,
            )

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Compare predictions to gold answers
        for pid, pred, gold in zip(batch_pids, preds, gold_answers):
            pred_str = str(pred).strip()
            gold_str = str(gold).strip()

            is_correct = (
                normalize_answer_simple(pred_str)
                == normalize_answer_simple(gold_str)
            )

            results[pid] = {
                "prediction": pred_str,
                "answer": gold_str,
                "true_false": bool(is_correct),
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage2 model on TabMWP: generate answers + exact-match accuracy."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Folder containing the HF checkpoint of Stage2.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Stage2 HF dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/raw/tabmwp/problems_test.json",
        help="Original TabMWP JSON (problems_test.json) for evaluate_acc.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to save JSON with predictions for evaluate_acc.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    # 1) Stage2 inference
    results = run_inference(
        checkpoint_dir=args.checkpoint_dir,
        dataset_path=args.dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
    )

    result_data = {
        "results": results,
    }

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 3) Official detailed evaluation script
    try:
        scores = get_scores([args.output_file], args.data_file)
        print_scores(scores)
    except Exception as e:
        print(f"Failed to compute detailed scores via evaluate_acc: {e}")


if __name__ == "__main__":
    main()
