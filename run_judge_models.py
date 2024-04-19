import os
import json
import argparse
import random
import torch
import datasets
import re
import ray
import copy
import vllm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from build_prompt import create_prompt_predefined

def parse_score_autoj_single(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0

def parse_score_prometheus_single(review):
    try:
        score = review.split('[RESULT]')[1].strip()
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        else:
            return 1
    except:
        return 1
    # try:
    #     score = re.search(r"Assistant 1: [0-9]+\nAssistant 2: [0-9]+", review).group()
    #     score = score.lstrip("Assistant 1:")
    #     sp = score.split("\nAssistant 2: ")
    #     return [float(sp[0]), float(sp[1])]
    # except:
    #     print(review)
    #     return [1.0, 1.0]

@torch.inference_mode()
def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("start load model")
    model = vllm.LLM(model=model_path, tensor_parallel_size=1)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("model loaded")

    pred_list = model.generate(prompts, sampling_params)
    pred_list = [it.outputs[0].text for it in pred_list]

    return pred_list


def main(args):
    random.seed(42)
    with open(args.data_path_question, "r") as fin:
        lines = [line.strip() for line in fin.readlines()]
        dataset_qes = [json.loads(line) for line in lines]
        dataset_qes = [line['turns'][0] for line in dataset_qes]

    with open(args.data_path_answer, "r") as fin:
        lines = [line.strip() for line in fin.readlines()]
        dataset_ans = [json.loads(line) for line in lines]
        dataset_ans = [[line['choices'][0]['turns'][0], line['choices'][1]['turns'][0]] for line in dataset_ans]

    instruction = create_prompt_predefined(args.model_type)
    prompts = []

    import pdb;pdb.set_trace()

    for qes, ans in zip(dataset_qes, dataset_ans):
        example = {"rubric": "Please rate the helpfulness, relevance, accuracy, level of details of their responses."}
        example{"question_body"} = qes[0]
        example{"answer1_body"} = ans[0]
        example{"answer2_body"} = ans[1]
        prompt = instruction["single"].format(question=example["question_body"],
                                              rubric=example["rubric"],
                                              answer1=example["answer1_body"],
                                              answer2=example["answer2_body"])     
        prompts.append(prompt)
    
    import pdb;pdb.set_trace()

    predictions = batched_generation(args.model_name_or_path, prompts, 
                                     max_new_token=args.max_new_token, 
                                     temperature=args.temperature,
                                     top_p=args.top_p)

    if args.model_type == "judgelm":
        pred_scores = [parse_score_judgelm_single(pred) for pred in predictions]
    elif args.model_type == "auto-j":
        pred_scores = [parse_score_autoj_single(pred) for pred in predictions]
    elif args.model_type == "pandalm":
        pred_scores = [parse_score_pandalm_single(pred) for pred in predictions]
    elif args.model_type == "prometheus":
        pred_scores = [parse_score_prometheus_single(pred) for pred in predictions]

    with open(args.data_path.rstrip(".jsonl")+args.model_type+".jsonl", "w") as fout:
        for i,line in enumerate(pred_scores):
            fout.write(json.dumps({"question_id": i, "evaluations": line}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus"),
        default=None,
    )
    parser.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--data-path-question",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-path-answer",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=2048,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The temperature for sampling.",
    )
    args = parser.parse_args()

    main(args)