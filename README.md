# SelfEval

This is the official repository for paper **Self-Evaluation of Large Language Model based on Glass-box Features**.

If you have any quesions, you can contact me with Wechat huanghui20200708.

## ⚡️ Usage
### Preparation
Please first install [Python 3.10](https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/#:~:text=Python%203.10-,Miniconda3%20Linux%2064%2Dbit,-127.9%20MiB) and [ray](https://docs.ray.io/en/latest/ray-overview/installation.html#:~:text=Linux%20Python%203.10%20(x86_64)) from the respective links.

Then refer to the following command to prepare your environment.
```shell
pip install -r requirements.txt
```
Please download pre-trained LLMs and put them under ``llm_judge/models``

Specifically, our validation is based on two popular open-source LLMs: [vicuna-7b-v1.3](https://www.modelscope.cn/models/Xorbits/vicuna-7b-v1.3/summary) and [Llama-2-7b-chat-hf](https://www.modelscope.cn/models/shakechen/Llama-2-7b-chat-hf/summary)

## Answer Generation and GPT4-based Evaluation
We have already prepared the answers and GPT4-based evaluations of Llama2 and Vicuna in ``data`` folder.

Alternatively, you can generate the answers of LLMs to MT-bench and Vicuna-bench, and then evaluate them with GPT-4 as the judge.

Notice this implementation is based on [llm-as-a-judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

```shell
MODEL_NAME=llama2-7b-chat
BENCH_NAME=mt_bench
python -u gen_model_answer.py \
    --model-path ../models/$MODEL_NAME \
    --bench-name $BENCH_NAME \
    --model-id $MODEL_NAME

python gen_judgment.py \
    --model-list $MODEL_NAME \
    --bench-name $BENCH_NAME \
    --parallel 4 \
    --judge-model gpt-4
```


## Self Evaluation based on Glass-box Features
With the previous created answers, you can induce the glass-box features of LLMs as the evaluation result.

Adjust the glass-box feature with ``ESTIMATION_MODE``. We recommend you use ``scores`` and ``logprobs-variance``. 

```shell
MODEL_NAME=llama2-7b-chat
BENCH_NAME=mt_bench
ESTIMATION_MODE=logprobs-variance
python -u gen_model_answer_selfeval.py \
    --model-path ../models/$MODEL_NAME \
    --model-id $MODEL_NAME \
    --bench-name $BENCH_NAME \
    --estimation-mode $ESTIMATION_MODE \
    --num-gpus-total 1

python -u cal_correlation.py \
    --model_name $MODEL_NAME \
    --bench_name $BENCH_NAME \
    --estimation_mode $ESTIMATION_MODE
```
# 💬 Citation
If you find our work is helpful, please cite as:

```
@misc{huang2024selfevaluation,
      title={Self-Evaluation of Large Language Model based on Glass-box Features}, 
      author={Hui Huang and Yingqi Qu and Jing Liu and Muyun Yang and Tiejun Zhao},
      year={2024},
      eprint={2403.04222},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```