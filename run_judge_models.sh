#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u run_judge_models.py \
    --model-name-or-path "../CrossEval/models/JudgeLM-7B" \
    --model-type "judgelm" \
    --data-path ./llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-entropy.jsonl