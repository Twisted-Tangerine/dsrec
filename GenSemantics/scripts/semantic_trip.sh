#!/usr/bin/env bash
set -e

# Semantic Embedding Generation Script (bf16)
# Usage: execute from Generation/ directory
#   bash scripts/semantic_trip.sh

dataset="trip"
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_path="resources/Llama-2-7b-hf"
data_dir="data/${dataset}/"
checkpoint_dir="saved"
MAX_STEPS=4000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="trip"
MAX_SOURCE_LENGTH=1024

PER_DEVICE_BATCH_SIZE=32
GRAD_ACCUMULATION_STEPS=1

peft_path="" 

# --- Train Command  ---
deepspeed --num_gpus=2 --master_port $MASTER_PORT main_llm.py \
    --dataset_choice $dataset \
    --deepspeed llm/ds.config \
    --do_train \
    --train_file $data_dir/item_str.jsonline \
    --cache_dir $data_dir \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_path $model_path \
    --output_dir $checkpoint_dir/lora-$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --max_steps ${MAX_STEPS} \
    --logging_steps 100 \
    --save_steps $(($MAX_STEPS / 2)) \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --pool_type avg \
    --dropout_ratio 0.4 \
    --bf16 


# --- Generate Command  ---
deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm.py \
    --do_predict \
    --test_file $data_dir/item_str.jsonline \
    --cache_dir $data_dir \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_path $model_path \
    --peft_path $checkpoint_dir/lora-$date/checkpoint-$MAX_STEPS \
    --output_dir results/${dataset}-semantic-bf16 \
    --output_file $date.json \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --pool_type avg