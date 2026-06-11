#!/bin/bash

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
: "${DATASET_PATH:?Set DATASET_PATH to a local GSM8K dataset saved with datasets.save_to_disk}"
OUTPUT_DIR="${OUTPUT_DIR:-./save}"
REF_SERVER="${REF_SERVER:-http://localhost:59875}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_DIR="${SAVE_DIR:-37}"

steps=(0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)

for ((i=0; i<${#steps[@]}-1; i++)); do
    current_step=${steps[$i]}
    next_step=${steps[$((i+1))]}

    if [ $current_step -eq 0 ]; then
        model_path="$BASE_MODEL_PATH"
        pre_step=0
    else
        model_path="$OUTPUT_DIR/$SAVE_DIR/step_temp"
        pre_step=-1
    fi
    
    save_steps=50
    all_steps=1500


    echo "====================================="
    echo "轮次 $i："
    echo "current_step: $current_step, next_step: $next_step"
    echo "model_path: $model_path"
    echo "pre_step: $pre_step, save_steps: $save_steps, all_steps: $all_steps"
    echo "====================================="

    echo "开始执行 adaptor_grpo_split.py ..."
    # if [ $i -ne 0 ]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" deepspeed --master_port="$MASTER_PORT" ./grpo_ref_split.py \
            --model_path="$model_path" \
            --tokenizer_path="$BASE_MODEL_PATH" \
            --dataset_path="$DATASET_PATH" \
            --output_dir="$OUTPUT_DIR" \
            --save_dir="$SAVE_DIR" \
            --pre_step="$pre_step" \
            --save_steps="$save_steps" \
            --all_steps="$all_steps" \
            --ref_server="$REF_SERVER"
                
    #     # 检查上一条命令是否执行成功，失败则退出
    # if [ $? -ne 0 ]; then
    #     echo "ERROR: grpo_ref_split.py 执行失败，退出脚本"
    #     exit 1
    # fi

    
done

echo "所有轮次执行完成！"
