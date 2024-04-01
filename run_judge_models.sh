#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u run_judge_models.py \
    --model-name-or-path "./models/prometheus-7b-v1.0" \
    --model-type "prometheus" \
    --data-path ./llm_judge/data/mt_bench/model_answer/llama2-7b-chat.jsonl