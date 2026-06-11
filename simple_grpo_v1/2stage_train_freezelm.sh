#!/bin/bash

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B}"
: "${DATASET_PATH:?Set DATASET_PATH to a local GSM8K dataset saved with datasets.save_to_disk}"
OUTPUT_DIR="${OUTPUT_DIR:-./save}"
REF_SERVER="${REF_SERVER:-http://localhost:59875}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_DIR="${SAVE_DIR:-36}"

steps=(0 200 400 600 800 1000 1200 1400 1600 1800 2000 2250 2500 2750 3000)

for ((i=0; i<${#steps[@]}-1; i++)); do
    current_step=${steps[$i]}
    next_step=${steps[$((i+1))]}

    if [ $current_step -eq 0 ]; then
        model_path="$BASE_MODEL_PATH"
    else
        model_path="$OUTPUT_DIR/$SAVE_DIR/step_$current_step"
    fi

    pre_step=$current_step
    save_steps=$((next_step - current_step))
    all_steps=$((next_step - current_step))

    if [ $((i % 2)) -eq 0 ]; then
        freeze="false"
    else
        freeze="true"
    fi

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
            --ref_server="$REF_SERVER" \
            --freeze=$freeze 
                
        # 检查上一条命令是否执行成功，失败则退出
    if [ $? -ne 0 ]; then
        echo "ERROR: grpo_ref_split.py 执行失败，退出脚本"
        exit 1
    fi
    # fi

    
done

echo "所有轮次执行完成！"
