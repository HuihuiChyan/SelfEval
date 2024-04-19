import sys
import json
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau

judge_file = "./llama2-7b-chat-logprobs-entropy-auto-j.jsonl"
eval_file1 = "/Users/dxm/Desktop/SelfEval/llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-entropy.jsonl"

with open(judge_file, "r") as fsys:
    judge_lines = [json.loads(line.strip())["judge_score"] for line in fsys.readlines()]
    new_judge_lines = []
    for line in judge_lines:
        new_judge_lines.append(line[0])
        new_judge_lines.append(line[1])

with open(eval_file1, "r") as feva:
    eval_lines = [json.loads(line.strip())["evaluations"] for line in feva.readlines()]
    new_eval_lines = []
    for line in eval_lines:
        new_eval_lines.append(line[0])
        new_eval_lines.append(line[1])

pearson = pearsonr(new_judge_lines, new_eval_lines)[0]
kendalltau = kendalltau(new_judge_lines, new_eval_lines)[0]
spearman = spearmanr(new_judge_lines, new_eval_lines)[0]

# add metrics to dict
metrics_dict = {
    'pearson': pearson,
    'kendalltau': kendalltau,
    'spearman': spearman,
}

print(metrics_dict)