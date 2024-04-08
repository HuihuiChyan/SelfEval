#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
MODEL_NAME=vicuna-7b
BENCH_NAME=vicuna_bench
ESTIMATION_MODE=ensemble-temper-logprobs-entropy
python -u gen_model_answer_selfeval.py \
    --model-path ../models/$MODEL_NAME \
    --model-id $MODEL_NAME \
    --bench-name $BENCH_NAME \
    --estimation-mode $ESTIMATION_MODE \
    --num-gpus-total 4

python -u cal_correlation.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --estimation_mode $ESTIMATION_MODE
