#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=llama2-7b-chat
BENCH_NAME=mt_bench
ESTIMATION_MODE=ensemble_logprobs
python -u llm_judge/gen_model_answer_selfeval.py \
    --model-path ./models/$MODEL_NAME \
    --model-id $MODEL_NAME

python -u llm_judge/cal_correlation.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --estimation_mode $ESTIMATION_MODE