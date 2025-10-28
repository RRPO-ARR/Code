import argparse
import re
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from math_verify import parse, verify, ExprExtractionConfig


SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>."
)


def build_prompt(tokenizer: AutoTokenizer, question: str) -> Tuple[str, torch.Tensor]:
    tip_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(tip_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return tip_text, input_ids


def extract_numeric_answer(text: str):
    # Match decimals, simple fractions, or integers; take the last occurrence
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, text)
    return nums[-1] if nums else None


def check_correct(pred_text: str, gt_text: str) -> bool:
    pred_num = extract_numeric_answer(pred_text)
    if pred_num is None:
        return False
    try:
        ans = parse(pred_num, extraction_config=[ExprExtractionConfig()])
        gt = parse(gt_text, extraction_config=[ExprExtractionConfig()])
        return bool(verify(ans, gt))
    except Exception:
        return False


def check_format(text: str) -> bool:
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = text.count("<think>") + text.count("</think>")
    answer_count = text.count("<answer>") + text.count("</answer>")
    return bool(re.match(pattern, text, re.DOTALL | re.VERBOSE)) and think_count == 2 and answer_count == 2


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model accuracy on GSM8K")
    parser.add_argument("--model_path", default='./save/00/step_20')
    parser.add_argument("--device", default="cuda", help="Device to run on, e.g., cuda, cuda:0, cpu")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--do_sample", type=bool, default=False, help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling when do_sample is enabled")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=1.0 if args.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load GSM8K
    dataset = load_from_disk("../data2/datasets/gsm8k")[args.split]
    print(f"Loaded GSM8K {args.split} split with {len(dataset)} samples.")
    total = min(args.num_samples, len(dataset))

    correct = 0
    format_ok = 0
    start_time = time.time()

    for i in tqdm(range(total), desc="Evaluating"):
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip()

        tip_text, input_ids = build_prompt(tokenizer, question)
        plen = input_ids.shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids.to(args.device),
                generation_config=generation_config,
            )

        gen_ids = outputs[0, plen:]  # Only the generated part
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Some models may insert spaces/newlines; trim
        pred_text = pred_text.strip()

        # Metrics
        if check_correct(pred_text, gt):
            correct += 1
        if check_format(pred_text):
            format_ok += 1

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    format_rate = format_ok / total if total > 0 else 0.0

    print("\n===== GSM8K Evaluation Result =====")
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"Format compliance: {format_rate*100:.2f}% ({format_ok}/{total})")
    print(f"Elapsed: {elapsed:.2f}s  ({elapsed/total:.2f}s/sample)")

    result = {
        "model": args.model_path,
        "split": args.split,
        "samples": total,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "elapsed_time_sec": elapsed,
        "time_per_sample_sec": elapsed / total if total > 0 else 0.0,
    }
    with open(f"{args.model_path}/eval_result.txt", "a") as f:
        f.write(str(result) + "\n")


if __name__ == "__main__":
    main()