import argparse
import re
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import math

from math_verify import parse, verify, ExprExtractionConfig

from qwen_with_adaptor import QwenWithAdaptor, QwenWithRolloutHead

import os

SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    # "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    # "i.e., <think> reasoning process here </think><answer> answer here </answer>."
    "Below is a specific skill requirement for the assistant to solve math problems in GSM8K dataset. Please strictly follow it."
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


def check_format(text: str):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = text.count("<think>") + text.count("</think>")
    answer_count = text.count("<answer>") + text.count("</answer>")
    if bool(re.match(pattern, text, re.DOTALL | re.VERBOSE)):
        if think_count == 2 and answer_count == 2:
            return True, True
        else:
            return True, False
    return False, False

def count_words(text: str) -> int:
    words = [word for word in text.split() if word.strip()]
    return len(words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model accuracy on GSM8K")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path; defaults to --model_path")
    parser.add_argument("--dataset_path", required=True, help="Path to a local GSM8K dataset saved with datasets.save_to_disk")
    parser.add_argument("--skill_prompt_path", default=None, help="Optional prompt file appended to the system prompt")
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

    global SYSTEM_PROMPT
    if args.skill_prompt_path and os.path.exists(args.skill_prompt_path):
        with open(args.skill_prompt_path, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT += "\n\n" + f.read()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(args.device)


    # config = AutoConfig.from_pretrained(args.model_path)

    # print(">>> loading model...")
    # model = QwenWithRolloutHead.from_pretrained(
    #     args.model_path, 
    #     config=config,  
    #     torch_dtype=torch.bfloat16,
    #     # _attn_implementation="sdpa", 
    # ).to(args.device)
    # model.use_adaptor = args.use_adaptor
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
        import os
        import json
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
    print(f"Loaded GSM8K {args.split} split with {len(dataset)} samples.")
    args.num_samples = len(dataset) if args.num_samples <= 0 else args.num_samples
    total = min(args.num_samples, len(dataset))
    if save_detail:
        detail = {"dataset": "GSM8K", "num_samples": total, "correct":[], "acc": 0}

    correct = 0
    format_ok = 0
    format_ok_strict = 0
    start_time = time.time()
    pbar = tqdm(range(total), desc="Evaluating")
    corr_words = []
    err_words = []
    pks = [pow(2, i) for i in range(0, 10) if pow(2, i) < num_per_q]
    pass_k = {pk: 0 for pk in pks}

    for i in pbar:
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip()

        tip_text, input_ids = build_prompt(tokenizer, question)
        plen = input_ids.shape[1]
        
        with torch.inference_mode():
            outputs = model.generate(
                    input_ids.to(args.device),
                    generation_config,
                )

        if len(outputs) == 1:
            gen_ids = outputs[0, plen:]  # Only the generated part
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # print(pred_text)

            # Some models may insert spaces/newlines; trim
            pred_text = pred_text.strip()
            # print(f"answer: {pred_text}")

            # Metrics
            co = check_correct(pred_text, gt)
            if co:
                correct += 1
                corr_words.append(count_words(pred_text))
            else:
                err_words.append(count_words(pred_text))

            fo, fo_strict = check_format(pred_text)
            if fo:
                format_ok += 1
            if fo_strict:
                format_ok_strict += 1

            if save_detail and check_correct(pred_text, gt):
                detail["correct"].append(i)
        else:
            ids = outputs[:, plen:]
            pad_ids_cnt = (ids == tokenizer.pad_token_id).sum(dim=1)
            pred_text = tokenizer.batch_decode(outputs[:, plen:], skip_special_tokens=True)
            acc, fo, fos = [], [], []
            for idx, p in enumerate(pred_text):
                co = check_correct(p, gt)
                acc.append(1 if co else 0)
                if co:
                    corr_words.append(len(ids[0] - pad_ids_cnt[idx]))
                else:
                    err_words.append(len(ids[0] - pad_ids_cnt[idx]))
                f, fs = check_format(p)
                fo.append(1 if f else 0)
                fos.append(1 if fs else 0)

            correct += 1 if any(acc) else 0 # sum(acc) / len(pred_text)
            format_ok += sum(fo) / len(pred_text)
            format_ok_strict += sum(fos) / len(pred_text)

            for pk in pks:
                score = 1 - math.comb(num_per_q - sum(acc), pk) / math.comb(num_per_q, pk)
                pass_k[pk] += score
            pass_at_k = {pk: round(pass_k[pk]/(i+1), 4) for pk in pks}


        if i < save_rollout:
            rollouts[i] = {"question": question, "gt": gt, "pass@k": {pk: pass_k[pk]/(i+1) for pk in pks}}
            results = []
            answers = tokenizer.batch_decode(outputs[:, plen:], skip_special_tokens=True)
            for idx, ans in enumerate(answers):
                ans = ans.strip()
                correct_reward = 1 if check_correct(ans, gt) else -1
                fo, fos = check_format(ans)
                format_reward = 1.25 if fo else -1
                format_reward_strict = 1 if fos else -1
                results.append({"answer": ans, "correct_reward": correct_reward, "format_reward": format_reward, "format_reward_strict": format_reward_strict})
            rollouts[i]["results"] = results
            # if (i + 1) % 10 == 0:
            #     output_file_path = os.path.join(args.model_path, "saved_rollouts.jsonl")
            #     with open(output_file_path, 'a', encoding='utf-8') as f:
            #         for item in rollouts.values():
            #             f.write(json.dumps(item, ensure_ascii=False) + "\n")
            #     rollouts = {}
            #     print(f"Step {i+1}: rollouts saved to {output_file_path}")
                
        pbar.set_postfix({"correct": f'{correct:.4f}'}) #, 'pass@k': f'{pass_at_k}'})

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    format_rate = format_ok / total if total > 0 else 0.0
    format_strict_rate = format_ok_strict / total if total > 0 else 0.0
    avg_word_cnt = (sum(corr_words)+sum(err_words)) / (len(corr_words)+len(err_words))
    avg_corr_word_cnt = sum(corr_words) / len(corr_words)
    avg_err_word_cnt = sum(err_words) / len(err_words)

    print("\n===== GSM8K Evaluation Result =====")
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"Format compliance: {format_rate*100:.2f}% ({format_ok}/{total})")
    print(f"Format compliance strict: {format_strict_rate*100:.2f}% ({format_ok_strict}/{total})")
    print(f"Avg answer word cnt: {avg_word_cnt}")
    print(f"Avg correct answer word cnt: {avg_corr_word_cnt}")
    print(f"Avg error answer word cnt: {avg_err_word_cnt}")
    print(f"Elapsed: {elapsed:.2f}s  ({elapsed/total:.2f}s/sample)")

    result = {
        "model": args.model_path,
        "split": args.split,
        "samples": total,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "format_rate_strict": format_strict_rate,
        "avg_words": avg_word_cnt,
        "avg_corr_ans_words": avg_corr_word_cnt,
        "avg_err_ans_words": avg_err_word_cnt,
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
        output_file_path = os.path.join(args.model_path, "saved_rollouts_skill_8b.json")
        print(f"Saving {save_rollout} rollouts to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(rollouts, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
