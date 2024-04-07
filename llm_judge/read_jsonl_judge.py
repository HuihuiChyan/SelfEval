import json
import csv

judge_file = "data/vicuna_bench/model_judgment/gpt-4_single.jsonl"
quest_file = "data/vicuna_bench/question.jsonl"
refere_file = "data/vicuna_bench/reference_answer/gpt-4.jsonl"
answer_file1 = "data/vicuna_bench/model_answer/llama2-7b-chat.jsonl"
answer_file2 = "data/vicuna_bench/model_answer/vicuna-7b.jsonl"

answers = {}
with open(judge_file, "r") as fjud, open(quest_file, "r") as fqes, open(refere_file, "r") as fref, \
open(answer_file1, "r") as fans1, open(answer_file2, "r") as fans2:

    qes_lines = [json.loads(l) for l in fqes]
    jud_lines = [json.loads(l) for l in fjud]
    ref_lines = [json.loads(l) for l in fref]
    ans_lines1 = [json.loads(l) for l in fans1]
    ans_lines2 = [json.loads(l) for l in fans2]

    final_dict = {}
    for line in qes_lines:
        qid = str(line["question_id"])
        final_dict[qid] = {}
        final_dict[qid]["qes"] = line["turns"][0]

    for line in jud_lines:
        qid = str(line["question_id"])
        if line["model"] == "llama2-7b-chat":
            final_dict[qid]["scr1"] = line["score"]
        else:
            final_dict[qid]["scr2"] = line["score"]

    for line in ref_lines:
        qid = str(line["question_id"])
        final_dict[qid]["ref"] = line["choices"][0]["turns"][0]    

    for line in ans_lines1:
        qid = str(line["question_id"])
        final_dict[qid]["ans1"] = line["choices"][0]["turns"][0]

    for line in ans_lines2:
        qid = str(line["question_id"])
        final_dict[qid]["ans2"] = line["choices"][0]["turns"][0]

with open('gpt-4.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    
    for qid in range(80):
        qid = str(qid+1)
        qes = final_dict[qid]["qes"]
        scr1 = final_dict[qid]["scr1"]
        scr2 = final_dict[qid]["scr2"]
        ans1 = final_dict[qid]["ans1"]
        ans2 = final_dict[qid]["ans2"]

        if "ref" in final_dict[qid].keys():
            final_line = [qid, qes, final_dict[qid]["ref"], ans1, ans2, scr1, scr2]
        else:
            final_line = [qid, qes, "", ans1, ans2, scr1, scr2]

        writer.writerow(final_line)