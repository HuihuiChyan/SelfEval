import sys
import json
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau

judge_file = "./llama2-7b-chat-logprobs-entropy-auto-j.jsonl"
eval_file1 = "/Users/dxm/Desktop/SelfEval/llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-variance.jsonl"

with open(judge_file, "r") as fsys:
    judge_lines = [json.loads(line.strip())["prometheus_score"] for line in fsys.readlines()]
    judge_lines = [int(line[0]>line[1]) for line in judge_lines]

with open(eval_file1, "r") as feva:
    eval_lines = [json.loads(line.strip())["evaluations"] for line in feva.readlines()]
    eval_lines = [int(line[0]>line[1]) for line in eval_lines]

acc_cnt = 0
for l in zip(judge_lines, eval_lines):
    if l[0] == l[1]:
        acc_cnt += 1

print("Accuracy is "+str(acc_cnt/len(judge_lines)))
