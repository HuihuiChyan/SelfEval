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
        return 5.0

def parse_score_prometheus_single(review):
    try:
        score = review.split('[RESULT]')[1].strip()
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        else:
            return 3
    except:
        return 3

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
        lines_qes = [line.strip() for line in fin.readlines()]
        lines_qes = [json.loads(line) for line in lines_qes]
        dataset_qes = [line['turns'][0] for line in lines_qes]

    with open(args.data_path_answer, "r") as fin:
        lines_ans = [line.strip() for line in fin.readlines()]
        lines_ans = [json.loads(line) for line in lines_ans]
        dataset_ans = [[line['choices'][0]['turns'][0], line['choices'][1]['turns'][0]] for line in lines_ans]

    instruction = create_prompt_predefined(args.model_type)
    prompts = []

    for qes, ans in zip(dataset_qes, dataset_ans):
        example = {"rubric": "Please rate the helpfulness, relevance, accuracy, level of details of their responses."}
        example["question_body"] = qes[0]
        example["answer1_body"] = ans[0]
        example["answer2_body"] = ans[1]
        prompt = instruction.format(question=example["question_body"],
                                    rubric=example["rubric"],
                                    answer=example["answer1_body"])
        prompts.append(prompt)
        prompt = instruction.format(question=example["question_body"],
                                    rubric=example["rubric"],
                                    answer=example["answer2_body"])   
        prompts.append(prompt)

    predictions = batched_generation(args.model_name_or_path, prompts, 
                                     max_new_token=args.max_new_token, 
                                     temperature=args.temperature,
                                     top_p=args.top_p)

    if args.model_type == "auto-j":
        pred_scores = [parse_score_autoj_pair(pred) for pred in predictions]
    elif args.model_type == "prometheus":
        pred_scores = [parse_score_prometheus_pair(pred) for pred in predictions]

    with open(args.data_path_answer.rstrip(".jsonl")+"-"+args.model_type+".jsonl", "w") as fout:
        for line, score in zip(lines_ans, pred_scores):
            line["judge_score"] = score
            fout.write(json.dumps(line)+"\n")


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
        default=1024,
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