#!/bin/bash
# root dir is Server

testset="mvtec"
config_name="vitl14_ep20"  # "vitl14_ep20", "eva02_l14_336"

model="./weights/train-visa-model-latest.pth.tar"

CUDA_VISIBLE_DEVICES=1 python main.py \
    --config-file ./configs/${config_name}.yaml \
    --output-dir "./output/${testset}_1" \
    --name "$testset" \
    --device 0 \
    --vis \
    --pixel