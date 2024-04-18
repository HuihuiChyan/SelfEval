import pandas as pd
import numpy as np
import json

excel_data = pd.read_excel("./data/human_eval/vicuna_bench_ref.xlsx")
human_eval = excel_data[["judge1", "judge2", "judge3", "judge4", "judge5"]].to_numpy()
human_eval = [np.bincount(line).argmax() for line in human_eval]

model_name = excel_data["first_model"].tolist()

new_human_eval = []
for m, h in zip(model_name, human_eval):
    if m == "vicuna":
        if h == 1:
            new_human_eval.append(2)
        elif h == 2:
            new_human_eval.append(1)
    else:
        new_human_eval.append(h)

answer_file1 = "data/vicuna_bench/model_answer/llama2-7b-chat-logprobs-variance.jsonl"
answer_file2 = "data/vicuna_bench/model_answer/vicuna-7b-logprobs-variance.jsonl"

def normalize(lines, bias=0.0):
    max_line = max(lines)
    min_line = min(lines)

    lines = [(max_line-line)/(max_line-min_line) + bias for line in lines]

    return lines

with open(answer_file1, "r") as fans1, open(answer_file2, "r") as fans2:

    self_eval1 = [json.loads(l)["evaluations"][0] for l in fans1]
    self_eval2 = [json.loads(l)["evaluations"][0] for l in fans2]

    self_eval1 = normalize(self_eval1)
    self_eval2 = normalize(self_eval2)

# self_eval1 = excel_data["score1"]
# self_eval2 = excel_data["score2"]

# new_self_eval1 = []
# new_self_eval2 = []
# for m, s1, s2 in zip(model_name, self_eval1, self_eval2):
#     if m == "vicuna":
#         new_self_eval1.append(s2)
#         new_self_eval2.append(s1)
#     else:
#         new_self_eval1.append(s1)
#         new_self_eval2.append(s2)

# self_eval1 = new_self_eval1
# self_eval2 = new_self_eval2

new_self_eval = []
for line in zip(self_eval1, self_eval2):
    if line[0] > line[1]:
        new_self_eval.append(1)
    else:
        new_self_eval.append(2)

acc_cnt = 0
for l in zip(new_human_eval, new_self_eval):
    if l[0] == l[1]:
        acc_cnt += 1

print("Accuracy is "+str(acc_cnt/len(new_human_eval)))