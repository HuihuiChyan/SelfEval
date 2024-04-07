import json
answer_file = "gpt-4_single.jsonl"

answers = {}
with open(answer_file, "r") as fin:
    answers = [json.loads(l) for l in fin]
    answers = sorted(answers, key=lambda x:x['judge'][1])
    answers = sorted(answers, key=lambda x:x['question_id'])
    answers = sorted(answers, key=lambda x:x['model'])
    
with open(answer_file, "w") as fout:
    for ans in answers:
        fout.write(json.dumps(ans)+"\n")