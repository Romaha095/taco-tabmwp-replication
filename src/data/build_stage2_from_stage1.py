import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.cot_calculator import fix_cot_with_calculator

def _replace_punctuation(s: str) -> str:
    return s.replace("\"", "").replace("'", "")


def _fix_buggy_characters(s: str) -> str:
    return re.sub(r"[{}^\\`\u2047<]", " ", s)


def _score_string_similarity(str1: str, str2: str) -> float:
    if str1 == str2:
        return 3.0
    str1 = _fix_buggy_characters(_replace_punctuation(str1))
    str2 = _fix_buggy_characters(_replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        s1 = str1.split(" ")
        s2 = str2.split(" ")
        overlap = list(set(s1) & set(s2))
        return len(overlap) / max(len(s1), len(s2))
    else:
        return 1.0 if str1 == str2 else 0.0


def extract_prediction(output: str, options: List[str] = None) -> str:
    if options:
        scores = [_score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))
        return options[max_idx]

    patterns = [
        r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',
        r'([\-\d\$\.\,\/\:]{0,}[\d]+)',
    ]
    for p in patterns:
        pattern = re.compile(p)
        res = pattern.findall(output)
        if res:
            return res[-1].strip()

    return output.strip()


def build_stage1_input_text(problem: Dict) -> str:
    table = problem["table"]
    question = problem["question"]
    return (
        "Table:\n"
        f"{table}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Explain your reasoning step by step."
    )


def build_stage2_input_text(problem: Dict, cot: str, calc_answer_raw: str) -> str:
    table = problem["table"]
    question = problem["question"]
    return (
        "Table:\n"
        f"{table}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Reasoning:\n"
        f"{cot}\n\n"
        "According to the reasoning above, the final answer (number/text only) is "
        f"{calc_answer_raw}.\n"
        "Please output only the final answer."
    )


def run_stage1_inference(
    stage1_ckpt_dir: str,
    problems: Dict[str, Dict],
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_beams: int = 4,
) -> Dict[str, Dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(stage1_ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(stage1_ckpt_dir).to(device)
    model.eval()

    pids: List[str] = sorted(list(problems.keys()))
    inputs: List[str] = [build_stage1_input_text(problems[pid]) for pid in pids]

    results: Dict[str, Dict] = {}

    total_stats = {
        "num_equations": 0,
        "num_eval_success": 0,
        "num_rhs_numeric": 0,
        "num_rhs_correct": 0,
        "num_rhs_fixed": 0,
    }


    for start in tqdm(range(0, len(pids), batch_size), desc="Stage1 inference"):
        end = min(start + batch_size, len(pids))
        batch_inputs = inputs[start:end]
        batch_pids = pids[start:end]

        enc = tokenizer(
            answer=batch_inputs,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_target_length,
                num_beams=num_beams,
            )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for pid, cot in zip(batch_pids, decoded):
            cot = str(cot).strip()

            cot_fixed, stats = fix_cot_with_calculator(cot)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            prob = problems[pid]

            options = prob.get("choices") or []
            calc_answer_raw = extract_prediction(cot_fixed, options)

            results[pid] = {
                "cot": cot,
                "calc_answer_raw": calc_answer_raw,
            }

    print(
        "[Calculator stats] "
        f"equations={total_stats['num_equations']}, "
        f"eval_success={total_stats['num_eval_success']}, "
        f"rhs_numeric={total_stats['num_rhs_numeric']}, "
        f"rhs_correct={total_stats['num_rhs_correct']}, "
        f"rhs_fixed={total_stats['num_rhs_fixed']}"
    )

    return results


def build_stage2_dataset_from_stage1(
    problems: Dict[str, Dict],
    stage1_outputs: Dict[str, Dict],
    split_name: str = "test",
) -> DatasetDict:

    pids_sorted = sorted(list(problems.keys()))

    input_texts: List[str] = []
    answers: List[str] = []
    pids: List[str] = []

    for pid in tqdm(pids_sorted, desc="Building Stage2 dataset"):
        prob = problems[pid]
        cot_info = stage1_outputs[pid]

        cot = cot_info["cot"]
        calc_answer_raw = cot_info["calc_answer_raw"]

        input_text = build_stage2_input_text(prob, cot, calc_answer_raw)
        gold_answer = str(prob["answer"])

        input_texts.append(input_text)
        answers.append(gold_answer)
        pids.append(pid)

    ds = Dataset.from_dict(
        {"input_text": input_texts, "answer": answers, "pid": pids}
    )
    return DatasetDict({split_name: ds})


def main():
    parser = argparse.ArgumentParser(
        description="Stage1 (CoT) -> calculator -> build Stage2 HF dataset for TabMWP."
    )
    parser.add_argument(
        "--stage1_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the Stage1 TAPEX checkpoint.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/tabmwp/problems_test.json",
        help="TabMWP JSON with problems.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/stage2_from_stage1",
        help="Where to save the HF dataset for Stage2.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length_stage1", type=int, default=512)
    parser.add_argument("--max_target_length_stage1", type=int, default=128)
    parser.add_argument("--num_beams_stage1", type=int, default=4)
    args = parser.parse_args()

    print("Loading problems from:", args.data_file)
    with open(args.data_file, "r", encoding="utf-8") as f:
        problems: Dict[str, Dict] = json.load(f)
    print(f"Loaded {len(problems)} problems.")

    print("Running Stage1 inference (CoT + calculator)...")
    stage1_outputs = run_stage1_inference(
        stage1_ckpt_dir=args.stage1_checkpoint_dir,
        problems=problems,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length_stage1,
        max_target_length=args.max_target_length_stage1,
        num_beams=args.num_beams_stage1,
    )

    print("Building Stage2 dataset from Stage1 outputs...")
    stage2_ds = build_stage2_dataset_from_stage1(
        problems, stage1_outputs, split_name="test"
    )

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    stage2_ds.save_to_disk(args.output_dir)
    print(f"Stage2 HF dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
