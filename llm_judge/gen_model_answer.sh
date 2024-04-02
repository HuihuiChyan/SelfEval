#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
python -u gen_model_answer.py \
    --model-path ../models/llama2-7b-chat \
    --bench-name vicuna_bench \
    --model-id llama2-7b-chat

python -u gen_judgment.py \
    --model-list llama2-7b-chat \
    --bench-name vicuna_bench \
    --parallel 4 \
    --judge-model gpt-4

python show_result.py
