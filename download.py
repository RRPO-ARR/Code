# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Qwen/Qwen2.5-3B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

# print("Model downloaded to:", local_dir)

import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Download and save a Hugging Face dataset")
parser.add_argument("--dataset_name", default="hotpot_qa")
parser.add_argument("--dataset_config", default="distractor")
parser.add_argument("--save_path", required=True)
args = parser.parse_args()

dataset = load_dataset(args.dataset_name, args.dataset_config)

dataset.save_to_disk(args.save_path)

print(dataset[0].keys())
print("Example data:", dataset['train'][0])

print("Dataset saved to:", args.save_path)
