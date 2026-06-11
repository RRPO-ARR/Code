# simple_GRPO Experimental Fork

This repository is an experimental fork of
[lsdefine/simple_GRPO](https://github.com/lsdefine/simple_GRPO). The upstream
project keeps the GRPO training loop intentionally small. This fork keeps that
spirit and adds experiments under `simple_grpo_v1/` for:

- split reference-model serving over HTTP;
- GRPO training on GSM8K and MBPP;
- rollout-head / adaptor-based training variants;
- evaluation scripts for GSM8K, MATH500, MBPP, and vLLM-backed inference.

The original upstream README is preserved in [origin_README.md](origin_README.md).

## Repository Layout

```text
.
|-- grpo_vllm_one.py              # upstream-style vLLM GRPO entry point
|-- ref_server.py                 # upstream reference server
|-- requirements.txt              # Python dependencies used by this fork
|-- simple_grpo_v1/
|   |-- ref_server.py             # reference-model HTTP server for v1 scripts
|   |-- grpo_ref_split.py         # GSM8K GRPO training with split reference model
|   |-- grpo_mbpp.py              # MBPP GRPO training
|   |-- rollout_head_train.py     # train rollout head on GSM8K
|   |-- rollout_head_train_mbpp.py # train rollout head on MBPP
|   |-- adaptor_grpo_split.py     # GSM8K adaptor GRPO stage
|   |-- adaptor_mbpp.py           # MBPP adaptor GRPO stage
|   |-- infer_gsm8k.py            # GSM8K evaluation
|   |-- infer_math500.py          # MATH500 evaluation
|   |-- infer_mbpp.py             # MBPP evaluation
|   |-- infer_vllm.py             # vLLM GSM8K evaluation
|   |-- qwen_with_adaptor.py      # Qwen2 adaptor / rollout-head model class
|   |-- qwen3_with_adaptor.py     # Qwen3 adaptor / rollout-head model class
|   `-- code_excutor.py           # MBPP code execution helper
|-- regroup_ver/                  # upstream regroup experiment
|-- simple-reinforce++/           # upstream Reinforce++ experiment
|-- kernel/                       # optional Triton loss kernel
`-- Auto_Program/                 # upstream auto-program experiment
```

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The training scripts assume CUDA GPUs and DeepSpeed. Most examples use one GPU
for the reference server and one or more GPUs for training. Model and dataset
paths are command-line parameters:

```text
--model_path       Path or Hugging Face id for the checkpoint to train/evaluate.
--tokenizer_path   Optional tokenizer path; defaults to --model_path.
--dataset_path     Local dataset directory saved with datasets.save_to_disk.
--output_dir       Checkpoint root directory; defaults to ./save.
```

## Quick Start

Start the reference server in one terminal:

```bash
cd simple_grpo_v1
CUDA_VISIBLE_DEVICES=0 python ref_server.py \
  --model_path Qwen/Qwen2.5-3B \
  --port 59875
```

Run GSM8K GRPO training in another terminal:

```bash
cd simple_grpo_v1
CUDA_VISIBLE_DEVICES=1 deepspeed --master_port=29500 grpo_ref_split.py \
  --model_path Qwen/Qwen2.5-3B \
  --dataset_path /path/to/gsm8k \
  --ref_server http://localhost:59875 \
  --save_dir 37 \
  --output_dir ./save \
  --all_steps 200 \
  --save_steps 200
```

Run MBPP GRPO training:

```bash
cd simple_grpo_v1
CUDA_VISIBLE_DEVICES=1 deepspeed --master_port=29500 grpo_mbpp.py \
  --model_path Qwen/Qwen3-8B \
  --dataset_path /path/to/mbpp \
  --ref_server http://localhost:59875 \
  --save_dir 39
```

## Two-Stage / Adaptor Experiments

The shell scripts in `simple_grpo_v1/` encode longer experiment schedules:

- `3-8b_train.sh`: repeated GRPO stages for Qwen3-8B.
- `2stage_train.sh`: rollout-head training followed by adaptor GRPO.
- `2stage_train_freezelm.sh`: staged training with alternating freeze settings.

Before running them, set the environment variables used by the scripts, for
example:

```bash
DATASET_PATH=/path/to/gsm8k \
BASE_MODEL_PATH=Qwen/Qwen3-8B \
TRAIN_GPUS=0,1 \
REF_SERVER=http://localhost:59875 \
bash 3-8b_train.sh
```

The scripts are experiment schedules rather than generic launchers.

## Evaluation

Example GSM8K evaluation:

```bash
cd simple_grpo_v1
python infer_gsm8k.py \
  --model_path ./save/37/step_200 \
  --dataset_path /path/to/gsm8k \
  --split test \
  --num_samples 100
```

Example MATH500 and MBPP evaluation:

```bash
python infer_math500.py --model_path ./save/37/step_200 --dataset_path /path/to/math500 --split test
python infer_mbpp.py --model_path ./save/39/step_100 --dataset_path /path/to/mbpp --split test
```

`infer_gsm8k.py` and `infer_math500.py` can append a local prompt file with
`--skill_prompt_path`.

## Notes

- Generated checkpoints, logs, local datasets, and local model weights are
  ignored by `.gitignore`.
- This fork keeps the code close to the original simple_GRPO style: most
  experiment parameters live near the top of the scripts or are exposed as a
  small set of command-line arguments.
- The upstream citation and project background are available in
  [origin_README.md](origin_README.md).
