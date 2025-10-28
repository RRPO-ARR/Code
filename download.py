# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Qwen/Qwen2.5-3B"
# local_dir = "./data2/Qwen/Qwen2.5-3B"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

# print("Model downloaded to:", local_dir)

from datasets import load_dataset, DatasetDict

dataset = load_dataset("hotpot_qa", "distractor")

save_path = "./data2/datasets/hotpot_qa_distractor"
dataset.save_to_disk(save_path)

print(dataset[0].keys())
print("Example data:", dataset['train'][0])

print("Dataset saved to:", save_path)