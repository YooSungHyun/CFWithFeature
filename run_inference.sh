#!/bin/bash
GPU_IDS=3

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 ./inference.py \
    --model_path="../models/" \
    --seed=42 \
    --accelerator=gpu \
    --devices=1 \
    --auto_select_gpus=true \
    --n_users= \
    --n_items= \
    --dropout= \
    --emb_dim= \
    --layer_dim= \
    --n_users_features= \
    --n_items_features=
