#!/bin/bash
# DSRec Training Script - Yelp Dataset

GPU=0
DATASET="yelp"
SEEDS=(42)

# Hyperparameters
HIDDEN=128
LINEAR=64
ALPHA=0.01
BETA=-0.2
TAU=2
THRESHOLD_USER=12
THRESHOLD_ITEM=13

for MODEL in dsrec_sasrec dsrec_bert4rec dsrec_gru4rec; do
    for SEED in ${SEEDS[@]}; do
        echo "Training $MODEL with seed $SEED"
        
        python run.py \
            --model_name $MODEL \
            --dataset $DATASET \
            --hidden_size $HIDDEN \
            --linear_dim $LINEAR \
            --train_batch_size 128 \
            --max_len 200 \
            --gpu_id $GPU \
            --num_workers 8 \
            --num_train_epochs 200 \
            --seed $SEED \
            --check_path DSRec \
            --patience 20 \
            --threshold_user $THRESHOLD_USER \
            --threshold_item $THRESHOLD_ITEM \
            --freeze_emb \
            --sem_emb semantics_embeddings \
            --alpha $ALPHA \
            --beta $BETA \
            --tau $TAU \
            --log
    done
done
