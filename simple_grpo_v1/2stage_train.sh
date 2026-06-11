#!/bin/bash

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
: "${DATASET_PATH:?Set DATASET_PATH to a local GSM8K dataset saved with datasets.save_to_disk}"
OUTPUT_DIR="${OUTPUT_DIR:-./save}"
REF_SERVER="${REF_SERVER:-http://localhost:59875}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
GRPO_SAVE_DIR="${GRPO_SAVE_DIR:-38}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-39}"
ROLLING_MODEL_PREFIX="${ROLLING_MODEL_PREFIX:-$OUTPUT_DIR/39_}"
ROLLOUT_MODEL_PATH="${ROLLOUT_MODEL_PATH:-$OUTPUT_DIR/$GRPO_SAVE_DIR/100rollout_head}"

steps=(0 40 80 100 120 140 160 180)

for ((i=0; i<${#steps[@]}-1; i++)); do
    current_step=${steps[$i]}
    next_step=${steps[$((i+1))]}

    if [ $current_step -eq 0 ]; then
        model_path="$BASE_MODEL_PATH"
    else
        model_path="$ROLLING_MODEL_PREFIX"
    fi

    pre_step=$current_step
    save_steps=$((next_step - current_step))
    all_steps=$((next_step - current_step))

    echo "====================================="
    echo "轮次 $i："
    echo "current_step: $current_step, next_step: $next_step"
    echo "model_path: $model_path"
    echo "pre_step: $pre_step, save_steps: $save_steps, all_steps: $all_steps"
    echo "====================================="

    echo "开始执行 rollout_head_train.py ..."
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" deepspeed --master_port="$MASTER_PORT" ./rollout_head_train.py \
        --pre_step="$pre_step" \
        --ref_server="$REF_SERVER" \
        --model_path="$model_path" \
        --tokenizer_path="$BASE_MODEL_PATH" \
        --dataset_path="$DATASET_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --save_dir="$ROLLOUT_SAVE_DIR"
    
    # 检查上一条命令是否执行成功，失败则退出
    if [ $? -ne 0 ]; then
        echo "ERROR: rollout_head_train_mbpp.py 执行失败，退出脚本"
        exit 1
    fi

    model_path="$ROLLOUT_MODEL_PATH"

    echo "开始执行 adaptor_grpo_split.py ..."
    # if [ $i -ne 0 ]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" deepspeed --master_port="$MASTER_PORT" ./adaptor_grpo_split.py \
            --model_path="$model_path" \
            --tokenizer_path="$BASE_MODEL_PATH" \
            --dataset_path="$DATASET_PATH" \
            --output_dir="$OUTPUT_DIR" \
            --save_dir="$GRPO_SAVE_DIR" \
            --pre_step="$pre_step" \
            --save_steps="$save_steps" \
            --all_steps="$all_steps" \
            --ref_server="$REF_SERVER"
                
        # 检查上一条命令是否执行成功，失败则退出
    if [ $? -ne 0 ]; then
        echo "ERROR: adaptor_grpo_split.py 执行失败，退出脚本"
        exit 1
    fi
    # fi



    echo "轮次 $i 执行完成！"
    echo "-------------------------------------"
done

echo "所有轮次执行完成！"
