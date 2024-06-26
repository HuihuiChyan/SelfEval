import sys
import json
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama2-7b-chat")
parser.add_argument("--bench_name", type=str, default="mt_bench")
parser.add_argument("--estimation_mode", type=str, default="logprobs")

args = parser.parse_args()

with open("./data/" + args.bench_name + "/model_judgment/gpt-4_single.jsonl", "r") as fsys:
    lines = [json.loads(line.strip()) for line in fsys.readlines()]
    syslines = []
    for line in lines:
        if line["model"] == args.model_name:
            syslines.append(line)
            
with open("./data/" + args.bench_name + "/model_answer/" + args.model_name + "-" + args.estimation_mode + ".jsonl", "r") as feva:
    evalines = [json.loads(line.strip()) for line in feva.readlines()]

syslines = sorted(syslines, key=lambda d: d['turn'])
syslines = sorted(syslines, key=lambda d: d['question_id'])

syslines = [line["score"] for line in syslines]
evalines = sorted(evalines, key=lambda d: d['question_id'])

if args.bench_name == "mt_bench":
    lines = []
    for line in evalines:
        lines.append(line["evaluations"][0])
        lines.append(line["evaluations"][1])

    assert len(syslines) == len(lines) == 160

else:
    lines = []
    for line in evalines:
        lines.append(line["evaluations"][0])

    assert len(syslines) == len(lines) == 80

# # exclude the influcence of NaN value
# import math
# avearge = []
# for l in lines:
#     if not math.isnan(l):
#         avearge.append(l)
# avg = sum(avearge) / len(avearge)
# lines = [l if not math.isnan(l) else avg for l in lines]

pearson = pearsonr(syslines, lines)[0]
kendalltau = kendalltau(syslines, lines)[0]
spearman = spearmanr(syslines, lines)[0]

# add metrics to dict
metrics_dict = {
    'pearson': pearson,
    'kendalltau': kendalltau,
    'spearman': spearman,
}

print(metrics_dict)
