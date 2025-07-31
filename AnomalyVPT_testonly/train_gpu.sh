#!/bin/bash

# 定义固定的数据集和标志
dataset="mvtec"
suffix="-2_seg_cls_1024_output_dim_gs05"

# 定义配置名称
config_name="vitl14_ep20"  # "vitl14_ep20", "eva02_l14_336"

# 定义输出目录
output_dir="./output/train_vpt_${dataset}_${suffix}_3/"

# 执行训练命令
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config-file "./configs/${config_name}.yaml" \
  --output-dir "$output_dir" \
  --name "$dataset" \
  --seed 1003 \
  --device 0 \
  --pixel

# --pixel