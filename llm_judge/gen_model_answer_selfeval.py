"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import copy
import numpy as np

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

from modeling_llama_dropout import LlamaDropoutForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from gen_single_answer_selfeval import get_single_answer

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    estimation_mode,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                estimation_mode=estimation_mode,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    estimation_mode,
):
    # model, tokenizer = load_model(
    #     model_path,
    #     revision=revision,
    #     device="cuda",
    #     num_gpus=num_gpus_per_model,
    #     max_gpu_memory=max_gpu_memory,
    #     dtype=dtype,
    #     load_8bit=False,
    #     cpu_offloading=False,
    #     debug=False,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        revision=revision,
        trust_remote_code=True,
    )
    # Adjust the dropout rate in modeling_llama_dropout.py, default is zero
    model = LlamaDropoutForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda()

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        evaluations = []
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                output_tokens, evaluation = get_single_answer(
                    tokenizer,
                    model,
                    prompt,
                    conv_stop_token_ids=conv.stop_token_ids,
                    conv_stop_str=conv.stop_str,
                    temperature=temperature,
                    max_new_token=max_new_token,
                    estimation_mode=estimation_mode,
                )

                conv.update_last_message(output_tokens)
                turns.append(output_tokens)
                evaluations.append(evaluation.tolist()[0])
            
            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file).rstrip(".jsonl")+"-"+estimation_mode+".jsonl", "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "evaluations": evaluations,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@torch.inference_mode()
def get_model_answers_ensemble(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    estimation_mode,
):
    # model, tokenizer = load_model(
    #     model_path,
    #     revision=revision,
    #     device="cuda",
    #     num_gpus=num_gpus_per_model,
    #     max_gpu_memory=max_gpu_memory,
    #     dtype=dtype,
    #     load_8bit=False,
    #     cpu_offloading=False,
    #     debug=False,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        revision=revision,
        trust_remote_code=True,
    )
    # Adjust the dropout rate in modeling_llama_dropout.py, default is zero
    model = LlamaDropoutForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda()

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        evaluations = []
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                ensemble_num = 10
                for k in range(ensemble_num):
                    # some models may error out when generating long outputs
                    torch.manual_seed(k)

                    output_tokens, evaluation = get_single_answer(
                        tokenizer,
                        model,
                        prompt,
                        conv_stop_token_ids=conv.stop_token_ids,
                        conv_stop_str=conv.stop_str,
                        temperature=temperature,
                        max_new_token=max_new_token,
                        estimation_mode=estimation_mode,
                    )

                conv.update_last_message(output_tokens)
                turns.append(output_tokens)
                evaluations.append(evaluation.tolist())
            
            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file).rstrip(".jsonl")+"-"+estimation_mode+".jsonl", "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "evaluations": evaluations,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--estimation-mode",
        type=str,
        default="logprobs",
        help="The model revision to load.",
    )


    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        estimation_mode=args.estimation_mode,
    )

    reorg_answer_file(answer_file)
