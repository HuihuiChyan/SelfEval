#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
MODEL_NAME=vicuna-7b
BENCH_NAME=ultra_feedback
ESTIMATION_MODE=ensemble-prompt-logprobs-both
python -u gen_answer_eval_ultra_feedback.py \
    --model-path ../models/$MODEL_NAME \
    --model-id $MODEL_NAME \
    --bench-name $BENCH_NAME \
    --estimation-mode $ESTIMATION_MODE \
    --num-choices 2 \
    --num-gpus-total 1

# python -u cal_correlation.py \
#     --model_name $MODEL_NAME \
#     --bench_name $BENCH_NAME \
#     --estimation_mode $ESTIMATION_MODE