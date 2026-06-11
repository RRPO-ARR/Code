from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from collections import deque
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
parser.add_argument("--pre_step", type=int, default=0)
parser.add_argument("--all_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--save_dir", type=str, default="37")
parser.add_argument("--ref_server", type=str, default="http://localhost:59875")
parser.add_argument("--seed", type=int, default=11451)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--freeze', type=bool, default=False)
args = parser.parse_args()

model_path = args.model_path
tokenizer_path = args.tokenizer_path or model_path
pre_step = args.pre_step
beta = 0.04
num_pre_Q = 8
all_steps = args.all_steps
max_prompt_length = 360   
save_steps = args.save_steps
# if type(save_steps) == int:
#     save_steps = range(0, all_steps+1, save_steps)
# print(save_steps[0])
save_steps = [100, 150, 200, 400, 600, 800, 1000, 1250, 1500]
compute_gen_logps = True
clip_param = 0.2
ref_server = args.ref_server
save_dir = args.save_dir
freeze = args.freeze
run_output_dir = os.path.join(args.output_dir, save_dir)
os.makedirs(run_output_dir, exist_ok=True)

SHARED_FILE = os.path.join(run_output_dir, "shared_num.json")
def read_shared_num(default=0):
    with open(SHARED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return int(data["value"]), int(data["step"])

def write_shared_num(num, num2):
    data = {"value": num, "step": num2}
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
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
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
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

if freeze:
    for n,p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False
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

r_count = {"-2.00":0, "0.00":0, "0.25":0, "2.25":0}

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. </think><answer> should be conected and </answer> is the last word. Your reply should be concise."""

THINK_END_STR = "</think>"
THINK_STR = "<think></think>"

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
    answers = [tokenizer.decode(x).replace('<|im_end|>', '') for x in completion_ids]
    answers = [a.replace('<|endoftext|>', '') for a in answers]
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
    if add:
        answers = insert_think(answers, torch.tensor(rewards))
    for r in rewards:
        k = f"{r:.2f}"
        r_count[k] += 1
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

def generate_mode(num=10, rank=0, add=False):
    if rank == 0: print('enter generate mode')
    else: return None
    global rand_step
    tic = time.time()
    cnt = 0
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs, add)
        if prompt_inputs is None: continue
        rand_step += 1
        raw_mean = rewards.mean().item()
        rew_var = rewards.var().item()
        if rank == 0: 
            print('rewards:', rewards)
            if ii == 5: print('answers:', answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        rarity_r = reward_rarity(rewards)
        rewards = rewards + 0.05 * rarity_r
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        data = [json.dumps({"plen": prompt_length, "rmean": raw_mean, "rvar": rew_var}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)]       

        if compute_gen_logps:
            # Reduntant calculations? For modifiability!
            with torch.inference_mode():
                mids = merged_ids.to(model.device)
                gen_logps = get_per_token_logps(model(mids).logits[:, :-1, :], mids[:, 1:])
                gen_logps = gen_logps[:,prompt_length-1:]
                completion_mask = (mids[:, prompt_length:] != tokenizer.pad_token_id).int()
                # ppl = compute_ppl_from_logps(gen_logps, completion_mask)
                # print(ppl.item())
                # ppls.append(ppl.item())
            data.append(tensor_to_bytes(gen_logps.cpu()))

        xdata = make_bytes_list(data)
        requests.post(f"{ref_server}/upload", data=xdata)
        cnt += 1

    if rank == 0: print('exit generate mode')
    print(f'{rank}: {time.time()-tic:.3f}s')
    print(r_count)
    return cnt

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

import deepspeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def compute_ppl_from_logps(per_token_logps, mask):
    sum_logps = (per_token_logps * mask).sum(dim=1)
    num_tokens = mask.sum(dim=1)
    avg_logp = sum_logps / (num_tokens + 1e-8)
    ppl = torch.exp(-avg_logp)
    return ppl

def pearsonr_torch(x, y):
    x = x.flatten()
    y = y.flatten()
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    cov = ((x - x_mean) * (y - y_mean)).mean()
    x_std = x.std(unbiased=False)  # 使用总体标准差以与NumPy等保持一致
    y_std = y.std(unbiased=False)
    
    corr = cov / (x_std * y_std + 1e-8)  # 加一个小数防止除零
    return corr

def compute_entropy_from_logits(logits, mask, prompt_length):

    logits = logits[:, :-1, :]
    logits = logits[:, prompt_length - 1:, :]  # [B, L', V]
    mask = mask[:, :logits.size(1)].float()

    z_max = logits.max(dim=-1, keepdim=True).values
    logits_minus_z = logits - z_max
    logZ = torch.log(torch.exp(logits_minus_z).sum(dim=-1)) + z_max.squeeze(-1)  # [B, L']

    log_probs = logits_minus_z - logZ.unsqueeze(-1)  # [B, L', V]
    chunk_size = 4096  # 小块大小，可调
    H = torch.zeros(logits.size(0), logits.size(1), device=logits.device)
    for i in range(0, logits.size(2), chunk_size):
        chunk = log_probs[..., i:i + chunk_size]
        probs_chunk = torch.exp(chunk)
        H -= (probs_chunk * chunk).sum(dim=-1)

    sum_entropy = (H * mask).sum()
    num_tokens = mask.sum()
    avg_entropy = sum_entropy / (num_tokens + 1e-8)
    return avg_entropy


def GRPO_step(batch):
    rank = deepspeed.comm.get_rank()
    prompt_length = batch['plen']
    rew_var = batch['rvar']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)   # normalized in generation

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # ppl = compute_ppl_from_logps(per_token_logps, completion_mask)
    # pear = pearsonr_torch(ppl, advantages.squeeze())
    # average_entropy = 0
    # if ppl > sum(ppls) / len(ppls) * 1.1:
    #     average_entropy = compute_entropy_from_logits(logits, completion_mask, prompt_length)


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

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # alpha = 0
    # print(loss.item(), alpha*pear.item())
    # loss = loss - alpha * pear
    return loss

if pre_step == 0:
    rand_step = 0
else:
    rand_step, pre_step = read_shared_num()
    
for i in range(rand_step):
    inputs = random.sample(QAs, Q_batch_size)

# ppls = deque(maxlen=50)
generate_mode(rank=torch.distributed.get_rank(), add=False)

from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)

all_suc = 0
brk_step = 0
saves = 0
for step in progress:
    batch = get_batch()
    while batch is None:
        cnt = generate_mode(rank=torch.distributed.get_rank())
        if cnt == 0:
            all_suc += 1
            if all_suc == 10:
                brk_step = step
                break
        else:
            all_suc = 0
        # dist.barrier()
        batch = get_batch()
    # dist.barrier()

    loss = GRPO_step(batch)
    engine.backward(loss)
    engine.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step in save_steps:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            save_name = os.path.join(run_output_dir, "step_temp")
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            # tokenizer.save_pretrained(save_name)
            saves += 1
            write_shared_num(rand_step, step+pre_step)
        dist.barrier()

