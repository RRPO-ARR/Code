import os
if "VLLM_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["VLLM_CUDA_VISIBLE_DEVICES"]

import argparse
import re
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from math_verify import parse, verify, ExprExtractionConfig

from qwen_with_adaptor import QwenWithAdaptor, QwenWithRolloutHead
from vllm import ModelRegistry, LLM, SamplingParams
# from vllm.logger import init_logger
# init_logger(progress_bar_enabled=False)

ModelRegistry.register_model("QwenWithRolloutHead", QwenWithRolloutHead)


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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path; defaults to --model_path")
    parser.add_argument("--dataset_path", required=True, help="Path to a local GSM8K dataset saved with datasets.save_to_disk")
    parser.add_argument("--device", default="cuda", help="Device to run on, e.g., cuda, cuda:0, cpu")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate, 0 means ALL")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--do_sample", type=bool, default=False, help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling when do_sample is enabled")
    parser.add_argument("--num_per_q", type=int, default=1, help="trajectory num of one question")
    parser.add_argument("--save_rollout", type=int, default=0, help="save first k rollouts")
    parser.add_argument("--save_info", type=str, default=None)
    parser.add_argument("--use_adaptor", type=bool, default=False)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16,
    #     _attn_implementation="sdpa",
    # ).to(args.device)


    # config = AutoConfig.from_pretrained(args.model_path)

    # print(">>> loading model...")
    # model = QwenWithRolloutHead.from_pretrained(
    #     args.model_path, 
    #     config=config,  
    #     torch_dtype=torch.bfloat16,
    #     # _attn_implementation="sdpa", 
    # ).to(args.device)
    # model.use_adaptor = args.use_adaptor
    # model.eval()

    # print("------------------------")
    # print(model.lm_head.weight[:10][:10])
    # print("------------------------")
    # # print(model.adaptor.weight[:10][:10])
    # # print("------------------------")
    # exit()
    # print(torch.equal(model.lm_head.weight, model.adaptor.weight))
    model = LLM(model=args.model_path, gpu_memory_utilization=0.5, trust_remote_code=True)
    model.llm_engine.model_executor.driver_worker.model_runner.model.use_adaptor = args.use_adaptor
    num_per_q = args.num_per_q if args.do_sample==True else 1
    save_rollout = args.save_rollout
    if save_rollout > 0:
        rollouts = {}

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=1.0 if args.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_per_q,
    )

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, n=args.num_per_q)
    sampling_params.update_from_generation_config(generation_config.to_dict())
    # if generation_config.num_return_sequences > 1:
    #     sampling_params.n = generation_config.num_return_sequences

    print(sampling_params)

    # Load GSM8K
    dataset = load_from_disk(args.dataset_path)[args.split]
    print(f"Loaded GSM8K {args.split} split with {len(dataset)} samples.")
    args.num_samples = len(dataset) if args.num_samples <= 0 else args.num_samples
    total = min(args.num_samples, len(dataset))

    correct = 0
    format_ok = 0
    start_time = time.time()
    pbar = tqdm(range(total), desc="Evaluating")

    for i in pbar:
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip()

        tip_text, _ = build_prompt(tokenizer, question)

        prompts = [tip_text] * num_per_q
        outputs = model.generate(prompts, sampling_params)

        for j, request_output in enumerate(outputs):
            for completion in request_output.outputs:
                pred_text = completion.text.strip()
                # print(pred_text)

                if check_correct(pred_text, gt):
                    correct_increment = 1.0 / (num_per_q * len(outputs)) if num_per_q > 1 else 1
                    correct += correct_increment
                if check_format(pred_text):
                    format_increment = 1.0 / (num_per_q * len(outputs)) if num_per_q > 1 else 1
                    format_ok += format_increment

                if save_rollout > 0 and i < save_rollout:
                    if i not in rollouts:
                        rollouts[i] = {
                            "question": question,
                            "gt": gt,
                            "results": []
                        }
                    correct_reward = 1 if check_correct(pred_text, gt) else -1
                    format_reward = 1.25 if check_format(pred_text) else -1
                    rollouts[i]["results"].append({
                        "answer": pred_text,
                        "correct_reward": correct_reward,
                        "format_reward": format_reward
                    })

        pbar.set_postfix({"correct": f'{correct:.4f}', 'format': f'{format_ok:.4f}'})

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
    if args.save_info is not None:
        result["info"] = args.save_info
    with open(f"{args.model_path}/eval_result.txt", "a") as f:
        f.write(str(result) + "\n")
    
    if save_rollout > 0:  
        import json
        import os
        output_file_path = os.path.join(args.model_path, "saved_rollouts.json")
        print(f"Saving {save_rollout} rollouts to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(rollouts, f, ensure_ascii=False, indent=4)
    print(outputs)


if __name__ == "__main__":
    main()
