from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from typing import Callable, Optional, Union
import copy
from qwen_with_adaptor import QwenWithAdaptor, QwenWithRolloutHead
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ.setdefault('LD_LIBRARY_PATH', '/usr/lib/x86_64-linux-gnu')
os.environ['NCCL_ALGO'] = 'Ring'  # 使用环算法，通常更稳定
os.environ['NCCL_MAX_NCHANNELS'] = '12'  # 增加通信通道
os.environ['NCCL_MIN_NCHANNELS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '8'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_BUFFSIZE'] = '4194304'  # 4MB缓冲区
os.environ['NCCL_TIMEOUT'] = '180'
Q_batch_size = 1
assert Q_batch_size == 1

import argparse

parser = argparse.ArgumentParser(description="Evaluate saved model accuracy on GSM8K")
parser.add_argument("--model_path", required=True)
parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path; defaults to --model_path")
parser.add_argument("--dataset_path", required=True, help="Path to a local GSM8K dataset saved with datasets.save_to_disk")
parser.add_argument("--output_dir", default="./save")
parser.add_argument("--pre_step", type=int, default=3000)
parser.add_argument("--all_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="38")
parser.add_argument("--ref_server", type=str, default="http://localhost:59875")
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument("--reset_adaptor", action="store_true", help="Reset adaptor weights before training")
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

random.seed(args.seed)
model_path = args.model_path
tokenizer_path = args.tokenizer_path or model_path
beta = 0.04
num_pre_Q = 8
all_steps = args.all_steps
max_prompt_length = 360   
all_save_steps = [100, 120, 140, 160, 180, 200]
save_steps = args.save_steps
clip_param = 0.2
ref_server = args.ref_server
save_dir = args.save_dir
mode = 1
pre_step = args.pre_step
reset_ad = args.reset_adaptor
# reset_ad = True
compute_gen_logps = (mode != 0)
run_output_dir = os.path.join(args.output_dir, save_dir)
os.makedirs(run_output_dir, exist_ok=True)

SHARED_FILE = os.path.join(run_output_dir, "shared_num.json")
def read_shared_num(default=0):
    with open(SHARED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return int(data["value"]), int(data["step"])

def write_shared_num(num, step):
    data = {"value": num, "step": step}
    with open(SHARED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        # "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) >= 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    if len(dd) == 6: data['a_gen_logps'] = bytes_to_tensor(dd[5])
    return data

batch = get_batch()
while batch is not None:
    batch = get_batch()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = AutoConfig.from_pretrained(model_path)

THINK_STR ="<think></think>"
THINK_END_STR = "</think>"

print(">>> loading model...")
# model = QwenWithAdaptor.from_pretrained(
#     model_path, 
#     config=config,  
#     torch_dtype=torch.bfloat16,
#     # _attn_implementation="sdpa", 
#     device_map="auto"
# )

model = QwenWithRolloutHead.from_pretrained(
    model_path, 
    config=config,  
    # torch_dtype=torch.bfloat16,
    # # _attn_implementation="sdpa", 
    # device_map="auto"
)
if reset_ad:
    model.reset_adaptor()
model.use_adaptor = True
    
print(">>> model loaded.")

if mode == 0:
    for name, param in model.named_parameters():
        if "adaptor" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
elif mode == 1:
    for name, param in model.named_parameters():
        if "rollout_head" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
else:
    for name, param in model.named_parameters():
        param.requires_grad = True

gen_model = model

gradient_size_bytes = 0
for param in model.parameters():
    if param.requires_grad:
        gradient_size_bytes += param.numel() * param.element_size()

print(f"Total gradient size: {gradient_size_bytes / 1024**3:.2f} GB")

# 动态调整bucket size
if gradient_size_bytes > 5 * 1024**3:  # 超过5GB
    ds_config["zero_optimization"]["allgather_bucket_size"] = 1e9  # 1GB
    ds_config["zero_optimization"]["reduce_bucket_size"] = 1e9

r_count = {"-2.00":0, "0.00":0, "0.25":0, "2.25":0}

from datasets import load_dataset, load_from_disk
# dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = load_from_disk(args.dataset_path)
dataset = dataset['train']
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. </think><answer> should be conected and </answer> is the last word. Your reply should be concise."""
def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: return []
    tip_inputs = {k: v.to(gen_model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '').replace('<|im_end|>', '') for x in completion_ids]
    return answers

from math_verify import parse, verify, ExprExtractionConfig
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1
def reward_format(item, answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


def gen_samples(inputs, add=False):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: return None, None, None, None
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    for r in rewards:
        k = f"{r:.2f}"
        r_count[k] += 1
    if add:
        answers = insert_think(answers, torch.tensor(rewards))
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)["input_ids"]
    return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers

def reward_rarity(rewards):
    unique, counts = rewards.unique(return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    rarity_scores = torch.tensor([1.0 / freq[r.item()] for r in rewards], device=rewards.device)
    rarity_scores = (rarity_scores - rarity_scores.mean()) / (rarity_scores.std() + 1e-6)
    return rarity_scores


def insert_think(
    answers, rewards
):
    mask = torch.isclose(rewards, torch.tensor(2.25, dtype=rewards.dtype))
    indices = torch.where(mask)[0]
    if len(indices) == 0 :
        return answers
    b = random.randrange(len(indices))
    ans = answers[indices[b]]
    position = ans.find(THINK_END_STR)
    if position == -1:
        return answers
    insert_pos = position + len(THINK_END_STR)

    new_ans = ans[:insert_pos] + THINK_STR + ans[insert_pos:]
    print(new_ans)
    answers[b] = new_ans

    return answers

def generate_mode(num=10, rank=0, add=False):
    if rank == 0: print('enter generate mode')
    else: return
    global rand_step
    tic = time.time()
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        rand_step += 1
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs, add)
        if prompt_inputs is None: continue
        rew_mean = rewards.mean().item()
        rew_var = rewards.var().item()
        if rank == 0: 
            print(rand_step, 'rewards:', rewards)
            if ii == 4: print('answers:', answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        a_rewards = reward_rarity(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        rewards = torch.cat([rewards, a_rewards])
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        data = [json.dumps({"plen": prompt_length, "rmean": rew_mean, "rvar": rew_var}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)]       

        if compute_gen_logps:
            # Reduntant calculations? For modifiability!
            with torch.inference_mode():
                mids = merged_ids.to(engine.module.device)
                output = engine.module(mids)
                logits = output.lm_logits
                a_logits = output.logits
                gen_logps = get_per_token_logps(logits[:, :-1, :], mids[:, 1:])
                a_gen_logps = get_per_token_logps(a_logits[:, :-1, :], mids[:, 1:])
            data.append(tensor_to_bytes(gen_logps[:,prompt_length-1:].cpu()))
            data.append(tensor_to_bytes(a_gen_logps[:,prompt_length-1:].cpu()))

        xdata = make_bytes_list(data)
        # requests.post(f"{ref_server}/upload", data=xdata)
        try:
            # print("[DEBUG] sending POST to:", f"{ref_server}/upload")
            resp = requests.post(f"{ref_server}/upload", data=xdata, timeout=10)

            # print("[DEBUG] HTTP status code:", resp.status_code)
            # print("[DEBUG] Response text:", resp.text[:500])  # 只打印前500字符
            resp.raise_for_status()

        except requests.exceptions.RequestException as e:
            print("requests.post failed!")
            print("Exception type:", type(e))
            print("Exception:", e)

    if rank == 0: print('exit generate mode')
    print(r_count)
    print(f'{rank}: {time.time()-tic:.3f}s')

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

import deepspeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine
print(engine.module.rollout_head.layer1.weight[:5, :5])

def hook_fn(module, grad_input, grad_output):
    module_name = module.__class__.__name__
    if grad_output[0] is not None:
        grad_output_norm = grad_output[0].norm().item()
    else:
        grad_output_norm = 0
    print(f"Module {module_name}: grad_output norm: {grad_output_norm:.8f}")

# for name, module in engine.named_modules():
#     if "rollout_head" in name:
#         module.register_full_backward_hook(hook_fn)
#         print(name)

# print(gen_model)

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather


def compute_entropy_from_logits(logits, mask, prompt_length):

    logits = logits[:, :-1, :]
    logits = logits[:, prompt_length - 1:, :]  # [B, L', V]
    mask = mask[:, :logits.size(1)].float()

    z_max = logits.max(dim=-1, keepdim=True).values
    logits_minus_z = logits - z_max
    logZ = torch.log(torch.exp(logits_minus_z).sum(dim=-1)) + z_max.squeeze(-1)  # [B, L']

    log_probs = logits_minus_z - logZ.unsqueeze(-1)  # [B, L', V]
    chunk_size = 2048  # 小块大小，可调
    H = torch.zeros(logits.size(0), logits.size(1), device=logits.device)
    for i in range(0, logits.size(2), chunk_size):
        chunk = log_probs[..., i:i + chunk_size]
        probs_chunk = torch.exp(chunk)
        H -= (probs_chunk * chunk).sum(dim=-1)

    sum_entropy = (H * mask).sum()
    num_tokens = mask.sum()
    avg_entropy = sum_entropy / (num_tokens + 1e-8)
    return avg_entropy



def GRPO_step(batch, mode):
    rank = deepspeed.comm.get_rank()
    prompt_length = batch['plen']
    rew_var = torch.tensor(batch['rvar']).to(engine.device)
    inputs = batch['inputs'].to(engine.device)
    adv = batch['rewards'].to(engine.device)
    advantages = adv[:len(adv)//2].unsqueeze(1)   # normalized in generation
    a_advantages = adv[len(adv)//2:].unsqueeze(1)
    
    a = 0
    b = 1

    output = engine(inputs)
    logits = output.lm_logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

    adaptor_logits = output.logits[:, :-1, :]
    a_per_token_logps = get_per_token_logps(adaptor_logits, input_ids)[:,prompt_length-1:]
    # print(a_per_token_logps[0, 0])
    # average_entropy = compute_entropy_from_logits(adaptor_logits, completion_mask, prompt_length)
    average_entropy = 0

    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if 'gen_logps' in batch:
        # if rank == 0:
        #     print('----------------------------------------------------------------')
        #     print(f"gen_logps, {torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))}")
        #     print('----------------------------------------------------------------')
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        # if rank == 0:
            # print('----------------------------------------------------------------')
            # print(f"no gen_logps, {torch.exp(per_token_logps - per_token_logps.detach())}")
            # print('----------------------------------------------------------------')
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    rollout_loss = 0
    if 'a_gen_logps' in batch and a > 0:
        ratio = torch.exp(a_per_token_logps - batch['a_gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        # rollout_loss = torch.min(rew_var.repeat(num_pre_Q).unsqueeze(1) * ratio, rew_var.repeat(num_pre_Q).unsqueeze(1) * clipped_ratio)
        rollout_loss = torch.min(ratio * a_advantages, clipped_ratio * a_advantages)
        rollout_loss = ((rollout_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print(rollout_loss, rew_var, clipped_ratio[:5,:5])

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # alpha = 5e-5 # 这是一个超参数，需要调节

    # 最终损失 = GRPO_Loss - alpha * Entropy
    loss_total = b * loss - a * rollout_loss
    loss_total = loss_total.float()
    return loss_total


if pre_step == 0:
    rand_step = 0
else:
    rand_step, pre_step = read_shared_num()
    
for i in range(rand_step):
    inputs = random.sample(QAs, Q_batch_size)
    
# # mode = 0
generate_mode(rank=torch.distributed.get_rank(), num=min(10, save_steps), add=False)


from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)

next_save_step = save_steps
saves = 0
for step in progress:

    batch = get_batch()
    while batch is None:
        batch = get_batch()
        if batch is None:
            generate_mode(rank=torch.distributed.get_rank(), num=min(10, 2*(next_save_step - step + 1)))
        

    loss = GRPO_step(batch, mode)
    if torch.isnan(loss).any():
        print("NaN detected in loss, skipping step")
        continue
    engine.backward(loss)
    engine.step()

    # print(engine.module.lm_head.weight[:10][:10])
    # print(engine.module.adaptor.weight[:10][:10])

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0 or step in all_save_steps or step+pre_step in all_save_steps:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print(r_count)
            print('saving model')
            save_name = os.path.join(run_output_dir, f"step_{step+pre_step}") if step + pre_step in all_save_steps else os.path.join(run_output_dir, "step_temp")
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            try:
                engine.module.save_pretrained(save_name, state_dict=state_dict)
            except Exception as e:
                print(f"保存模型时发生错误: {e}")
            # tokenizer.save_pretrained(save_name)
            # for param in engine.module.parameters():
            #     param.requires_grad = not param.requires_grad
            # # engine.reinit_optimizer(model_parameters=model.parameters())
            # mode += 1
            next_save_step += save_steps
            saves += 1
            write_shared_num(rand_step, step + pre_step)
            
        dist.barrier()

batch = get_batch()
while batch is not None:
    batch = get_batch()
