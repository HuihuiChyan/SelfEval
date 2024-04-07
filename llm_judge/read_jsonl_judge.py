import json
import csv

answer_file = "data/vicuna_bench/model_judgment/gpt-4_single.jsonl"
refere_file = "data/vicuna_bench/reference_answer/gpt-4.jsonl" 

answers = {}
with open(answer_file, "r") as fin, open(refere_file, "r") as fref, \
open('gpt-4.tsv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter='\t')

    lines = [json.loads(l) for l in fin]
    ref_lines = [json.loads(l) for l in fref]
    ref_dict = {}
    for line in ref_lines:
        ref_dict[line["question_id"]] = line["choices"][0]["turns"][0]
    final_lines = []
    for line in lines:
        qid = line["question_id"]
        scr = line["score"]
        line = line["user_prompt"].split("[Question]")[1].strip()
        qes = line.split("[The Start of Assistant's Answer]")[0].strip()
        line = line.split("[The Start of Assistant's Answer]")[1].strip()
        ans = line.split("[The End of Assistant's Answer]")[0].strip()

        final_line = [qid, qes, ans, scr]

        if qid in ref_dict.keys():
            final_line = [qid, qes, ref_dict[qid], ans]
        else:
            final_line = [qid, qes, "", ans]

        writer.writerow(final_line)