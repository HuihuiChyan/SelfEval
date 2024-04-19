#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u run_judge_models.py \
    --model-name-or-path "./models/Auto-J-13B" \
    --model-type "auto-j" \
    --data-path-question ./llm_judge/data/vicuna_bench/question.jsonl \
    --data-path-answer ./llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-entropy.jsonl