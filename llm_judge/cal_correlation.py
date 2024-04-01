import sys
import json
from scipy.stats import pearsonr, spearmanr, kendalltau

model_name = "llama2-7b-chat"
bench_name = "mt_bench"
method_name = "ensemble-logprobs-var-means"
with open("./data/" + bench_name + "/model_judgment/gpt-4_single.jsonl", "r") as fsys:
    lines = [json.loads(line.strip()) for line in fsys.readlines()]
    syslines = []
    for line in lines:
        if line["model"] == model_name:
            syslines.append(line)
            
with open("./data/" + bench_name + "/model_answer/" + model_name + "-" + method_name + ".jsonl", "r") as feva:
    evalines = [json.loads(line.strip()) for line in feva.readlines()]

syslines = sorted(syslines, key=lambda d: d['turn'])
syslines = sorted(syslines, key=lambda d: d['question_id'])

syslines = [line["score"] for line in syslines]
evalines = sorted(evalines, key=lambda d: d['question_id'])

lines = []
for line in evalines:
    lines.append(line["evaluations"][0])
    lines.append(line["evaluations"][1])

assert len(syslines) == len(lines) == 160

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