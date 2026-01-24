#!/bin/bash
# DSRec Embedding Generation Script - Yelp Dataset
# This script trains a base model and saves item/user embeddings for collaborative injection

GPU=0
DATASET="yelp"
SEEDS=(42)

# Thresholds for popularity-based metrics
THRESHOLD_USER=12
THRESHOLD_ITEM=13

MODEL="sasrec"

# Step 1: Train the base model
for SEED in ${SEEDS[@]}; do
    echo "Training $MODEL with seed $SEED"
    
    python run.py \
        --model_name $MODEL \
        --dataset $DATASET \
        --hidden_size 128 \
        --train_batch_size 128 \
        --max_len 200 \
        --gpu_id $GPU \
        --num_workers 8 \
        --num_train_epochs 200 \
        --seed $SEED \
        --check_path "" \
        --patience 20 \
        --threshold_user $THRESHOLD_USER \
        --threshold_item $THRESHOLD_ITEM \
        --log
done

# Step 2: Generate embeddings
# Output: ./data/${DATASET}/item_id_embeddings.pkl and ./data/${DATASET}/user_id_embeddings.pkl
for SEED in ${SEEDS[@]}; do
    echo "Generating embeddings for $MODEL with seed $SEED"
    
    python run.py \
        --model_name $MODEL \
        --dataset $DATASET \
        --hidden_size 128 \
        --train_batch_size 128 \
        --max_len 200 \
        --gpu_id $GPU \
        --num_workers 8 \
        --num_train_epochs 200 \
        --seed $SEED \
        --check_path "" \
        --patience 20 \
        --threshold_user $THRESHOLD_USER \
        --threshold_item $THRESHOLD_ITEM \
        --do_emb
done
