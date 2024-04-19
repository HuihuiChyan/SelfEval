import json
import csv
import random

model_name = "llama2-7b-chat"

quest_file = "data/vicuna_bench/question.jsonl"
refere_file = "data/vicuna_bench/reference_answer/gpt-4.jsonl"
answer_file1 = "data/vicuna_bench/model_answer/"+model_name+"-logprobs-variance.jsonl"
answer_file2 = "data/vicuna_bench/model_answer/"+model_name+"-logprobs-entropy.jsonl"

answers = {}
with open(quest_file, "r") as fqes, open(refere_file, "r") as fref, \
open(answer_file1, "r") as fans1, open(answer_file2, "r") as fans2:

    qes_lines = [json.loads(l) for l in fqes]
    ref_lines = [json.loads(l) for l in fref]
    ans_lines1 = [json.loads(l) for l in fans1]
    ans_lines2 = [json.loads(l) for l in fans2]

    final_dict = {}
    for line in qes_lines:
        qid = str(line["question_id"])
        final_dict[qid] = {}
        final_dict[qid]["qes"] = line["turns"][0]

    for line in ref_lines:
        qid = str(line["question_id"])
        final_dict[qid]["ref"] = line["choices"][0]["turns"][0]    

    for line in ans_lines1:
        qid = str(line["question_id"])
        final_dict[qid]["ans1"] = line["choices"][0]["turns"][0]
        final_dict[qid]["ans2"] = line["choices"][1]["turns"][0]
        final_dict[qid]["var1"] = line["evaluations"][0]
        final_dict[qid]["var2"] = line["evaluations"][1]

    for line in ans_lines2:
        qid = str(line["question_id"])
        final_dict[qid]["ent1"] = line["evaluations"][0]
        final_dict[qid]["ent2"] = line["evaluations"][1]

with open(model_name+'.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')

    random.seed(42)
    
    final_lines = []
    for qid in range(80):
        qid = str(qid+1)
        qes = final_dict[qid]["qes"]
        if random.random() >= 0.5:
            var1 = final_dict[qid]["var1"]
            var2 = final_dict[qid]["var2"]
            ent1 = final_dict[qid]["ent1"]
            ent2 = final_dict[qid]["ent2"]
            ans1 = final_dict[qid]["ans1"]
            ans2 = final_dict[qid]["ans2"]
            switch = "False"
        else:
            var1 = final_dict[qid]["var2"]
            var2 = final_dict[qid]["var1"]
            ent1 = final_dict[qid]["ent2"]
            ent2 = final_dict[qid]["ent1"]
            ans1 = final_dict[qid]["ans2"]
            ans2 = final_dict[qid]["ans1"]
            switch = "True"

        if "ref" in final_dict[qid].keys():
            final_line = [qid, qes, final_dict[qid]["ref"], ans1, ans2, ent1, ent2, var1, var2, switch]
        else:
            final_line = [qid, qes, "", ans1, ans2, ent1, ent2, var1, var2, switch]
        
        final_lines.append(final_line)

    random.shuffle(final_lines)

    for line in final_lines:
        writer.writerow(line)