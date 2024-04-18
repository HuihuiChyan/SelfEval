#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4
export CUDA_LAUNCH_BLOCKING=1
MODEL_NAME=llama2-7b-chat
BENCH_NAME=vicuna_bench
python -u gen_model_answer.py \
    --model-path ../models/$MODEL_NAME \
    --bench-name $BENCH_NAME \
    --model-id $MODEL_NAME \
    --num-choices 2 \
    --num-gpus-total 2

# python gen_judgment.py \
#     --model-list $MODEL_NAME \
#     --bench-name $BENCH_NAME \
#     --parallel 4 \
#     --judge-model gpt-4

# python show_result.py
