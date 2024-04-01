#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u llm_judge/gen_model_answer.py \
    --model-path ./models/llama2-7b-chat \
    --model-id llama2-7b-chat

python -u llm_judge/gen_judgment.py \
    --model-list llama2-7b-chat \
    --parallel 4 \
    --judge-model gpt-4

python show_result.py