import sys
import json
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau

eval_file1 = "/Users/dxm/Desktop/SelfEval/llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-entropy.jsonl"
eval_file2 = "/Users/dxm/Desktop/SelfEval/llm_judge/data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-variance.jsonl"

with open(eval_file1, "r") as feva:
    new_eval_lines1 = [json.loads(line.strip())["evaluations"] for line in feva.readlines()]
    # new_eval_lines1 = [[line[0], line[1]] for line in zip(eval_lines[::2], eval_lines[1::2])]
    # new_eval_lines1 = []
    # for line in eval_lines:
    #     new_eval_lines1.append(line[0])
    #     new_eval_lines1.append(line[1])

with open(eval_file2, "r") as feva:
    new_eval_lines2 = [json.loads(line.strip())["evaluations"] for line in feva.readlines()]
    # new_eval_lines2 = [[line[0], line[1]] for line in zip(eval_lines[::2], eval_lines[1::2])]
    # new_eval_lines2 = []
    # for line in eval_lines:
    #     new_eval_lines2.append(line[0])
    #     new_eval_lines2.append(line[1])

# pearson = pearsonr(new_eval_lines1, new_eval_lines2)[0]
# kendalltau = kendalltau(new_eval_lines1, new_eval_lines2)[0]
# spearman = spearmanr(new_eval_lines1, new_eval_lines2)[0]

# # add metrics to dict
# metrics_dict = {
#     'pearson': pearson,
#     'kendalltau': kendalltau,
#     'spearman': spearman,
# }

# print(metrics_dict)

eval_lines1 = []
for line in new_eval_lines1:
    if line[0] > line[1]:
        eval_lines1.append([1, 0])
    else:
        eval_lines1.append([0, 1])

eval_lines2 = []
for line in new_eval_lines2:
    if line[0] > line[1]:
        eval_lines2.append([1, 0])
    else:
        eval_lines2.append([0, 1])

# acc_cnt = 0
# for line in zip(eval_lines2, eval_lines1):
#     if line[0] == line[1]:
#         acc_cnt += 1
    
# acc = acc_cnt / len(eval_lines1)
# print("The accuracy is "+str(acc))

judge_file = "./llama2-7b-chat-logprobs-entropy-auto-j-gpt4.jsonl"
with open(judge_file, "r") as fjudge:
    judge_lines = [json.loads(line.strip())["judge_score"] for line in fjudge.readlines()]


acc_cnt = 0
for line in zip(eval_lines1, judge_lines):
    if line[0] == line[1]:
        acc_cnt += 1
    
acc = acc_cnt / len(eval_lines1)
print("The accuracy is "+str(acc))