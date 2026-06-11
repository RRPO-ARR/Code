import argparse
import os
import re
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from math_verify import parse, verify, ExprExtractionConfig

from qwen3_with_adaptor import QwenWithAdaptor, QwenWithRolloutHead

from code_excutor import CodeExecutor



SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question about python code generation, and the Assistant solves it by coding. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the user with the python code. "
    "The python code is enclosed within \'```python\n\' \'\n```\' tags, respectively, "
    "i.e., ```python\n python code here ```."
    "IMPORTANT: The entire Python code must be written as a single, complete code block within the \'```python\n\' \'\n```\' tags. "
    "Do not split the code into multiple parts or write incomplete code."
    "Once you've finished your code, you don't need to explain it."
    "Don't think too long about the solution, write the code directly after a brief thought if thought is necessary."
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


# def extract_numeric_answer(text: str):
#     # Match decimals, simple fractions, or integers; take the last occurrence
#     pattern = r"\d+\.\d+|\d+/\d+|\d+"
#     nums = re.findall(pattern, text)
#     return nums[-1] if nums else None


# def check_correct(pred_text: str, gt_text: str) -> bool:
#     pred_num = extract_numeric_answer(pred_text)
#     if pred_num is None:
#         return False
#     try:
#         ans = parse(pred_num, extraction_config=[ExprExtractionConfig()])
#         gt = parse(gt_text, extraction_config=[ExprExtractionConfig()])
#         return bool(verify(ans, gt))
#     except Exception:
#         return False


# def check_format(text: str) -> bool:
#     pattern = r"^<think>.*?</think>[\n ]*<python>.*?</python>$"
#     think_count = text.count("<think>") + text.count("</think>")
#     python_count = text.count("<python>") + text.count("</python>")
#     return bool(re.match(pattern, text, re.DOTALL | re.VERBOSE)) and think_count == 2 and python_count == 2

def code_from_python_tags(text: str) -> str:
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model accuracy on GSM8K")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path; defaults to --model_path")
    parser.add_argument("--dataset_path", required=True, help="Path to a local MBPP dataset saved with datasets.save_to_disk")
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
    parser.add_argument("--save_detail", type=bool, default=False)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16,
    #     _attn_implementation="sdpa",
    # ).to(args.device)


    config = AutoConfig.from_pretrained(args.model_path)

    print(">>> loading model...")
    model = QwenWithRolloutHead.from_pretrained(
        args.model_path, 
        config=config,  
        torch_dtype=torch.bfloat16,
        # _attn_implementation="sdpa", 
    ).to(args.device)
    model.use_adaptor = args.use_adaptor
    model.eval()

    # print("------------------------")
    # print(model.lm_head.weight[:10][:10])
    # print("------------------------")
    # # print(model.adaptor.weight[:10][:10])
    # # print("------------------------")
    # exit()
    # print(torch.equal(model.lm_head.weight, model.adaptor.weight))
    # model = LLM(model=model_path, gpu_memory_utilization=0.5)
    num_per_q = args.num_per_q if args.do_sample==True else 1
    save_rollout = args.save_rollout
    save_detail = args.save_detail
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

    # sampling_params = SamplingParams()
    # sampling_params.update_from_generation_config(generation_config.to_dict())

    # Load GSM8K
    dataset = load_from_disk(args.dataset_path)[args.split]
    print(f"Loaded MBPP {args.split} split with {len(dataset)} samples.")
    args.num_samples = len(dataset) if args.num_samples <= 0 else args.num_samples
    total = min(args.num_samples, len(dataset))
    if save_detail:
        detail = {"dataset": "MBPP", "num_samples": total, "correct":[], "acc": 0}

    correct = 0
    format_ok = 0
    start_time = time.time()
    pbar = tqdm(range(total), desc="Evaluating")

    executor = CodeExecutor(timeout=10, memory_limit=4096)
    pss = {1:0, 3:0, 5:0}
    cur_pss = {1:False, 3:False, 5:False}
    total_pass_rate = 0
    pss[num_per_q] = 0

    for i in pbar:
        question = dataset[i]["text"]
        test_list = dataset[i]["test_list"]
        set_up_code = dataset[i].get("setup_code", "")
        if set_up_code !=  "":
            question += " Set up code is given: <setup_code>" + set_up_code + "</setup_code>. "
        question += "\n Three tests are given: " + str(test_list)
        if i == 0: print(question)

        tip_text, input_ids = build_prompt(tokenizer, question)
        plen = input_ids.shape[1]
        
        with torch.inference_mode():
            outputs = model.generate(
                    input_ids.to(args.device),
                    generation_config,
                )

        gen_ids = outputs[:, plen:] 
        pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # print(f"Predicted Text: {pred_text[0]}")

        code_text = [code_from_python_tags(pt) for pt in pred_text]
        sub_correct = 0
        if set_up_code.strip() != "":
            code_text = [set_up_code + "\n" + code for code in code_text]
        for idx, code in enumerate(code_text):
            # print(f"Executing Code:\n{code}")
            result = executor.execute_python_code(code, test_list)
            # print(f"Execution Result: {result}")
            if result["passed"]:
                for p in cur_pss.keys():
                    if idx < p :
                        cur_pss[p] = True
            total_pass_rate += result["num_passed"] / result["total_tests"]
        
        for p in cur_pss.keys():
            pss[p] = pss[p] + 1 if cur_pss[p] else pss[p]
        cur_pss = {1:False, 3:False, 5:False}

        # if save_detail and check_correct(pred_text, gt):
        #     detail["correct"].append(i)

        # if i < save_rollout:
        #     rollouts[i] = {"question": question, "gt": gt}
        #     results = []
        #     answers = tokenizer.batch_decode(outputs[:, plen:], skip_special_tokens=True)
        #     for idx, ans in enumerate(answers):
        #         ans = ans.strip()
        #         correct_reward = 1 if check_correct(ans, gt) else -1
        #         format_reward = 1.25 if check_format(ans) else -1
        #         results.append({"answer": ans, "correct_reward": correct_reward, "format_reward": format_reward})
        #     rollouts[i]["results"] = results
                
        pbar.set_postfix({f"PS{p}": f"{pss[p]/(i+1):.4f}" for p in pss.keys()})

    for p in pss.keys():
        pss[p] = pss[p] / total
    total_pass_rate = total_pass_rate / total
    elapsed = time.time() - start_time
    # accuracy = correct / total if total > 0 else 0.0
    # format_rate = format_ok / total if total > 0 else 0.0

    print("\n===== MBPP Evaluation Result =====")
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Samples: {total}")
    for p in pss.keys():
        print(f"PS{p}: {pss[p]:.4f}")
    print(f"Total Pass Rate: {total_pass_rate:.4f}")
    print(f"Elapsed: {elapsed:.2f}s  ({elapsed/total:.2f}s/sample)")

    result = {
        "dataset": "MBPP",
        "model": args.model_path,
        "split": args.split,
        "samples": total,
        "pass": pss,
        "total_pass_rate": total_pass_rate,
        "elapsed_time_sec": elapsed,
        "time_per_sample_sec": elapsed / total if total > 0 else 0.0,
    }
    if args.save_info is not None:
        result["info"] = args.save_info

    with open(f"{args.model_path}/eval_result.txt", "a") as f:
        f.write(str(result) + "\n")
    
    if save_detail:
        import os
        import json
        output_file_path = os.path.join(args.model_path, "detail")
        detail["acc"] = accuracy
        print(f"Saving details to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, indent=4)
    
    if save_rollout > 0:  
        import json
        import os
        output_file_path = os.path.join(args.model_path, "saved_rollouts.json")
        print(f"Saving {save_rollout} rollouts to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(rollouts, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
