#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_LAUNCH_BLOCKING=1
MODEL_NAME=llama2-7b-chat
BENCH_NAME=vicuna_bench
ESTIMATION_MODE=logprobs-entropy
python -u gen_model_answer_selfeval.py \
     --model-path ../models/$MODEL_NAME \
     --model-id $MODEL_NAME \
     --estimation-mode $ESTIMATION_MODE \
     --num-gpus-total 2

python -u cal_correlation.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --estimation_mode $ESTIMATION_MODE
