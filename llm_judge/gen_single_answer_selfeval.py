import copy
import torch

def get_single_answer_evaluation(
    tokenizer,
    model,
    prompt,
    conv_stop_token_ids=None,
    conv_stop_str=None,
    temperature=0.1,
    max_new_token=2048,
):
    output_tokens, prefix_len, target_len, output_ids = get_single_answer(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        conv_stop_token_ids=conv_stop_token_ids,
        conv_stop_str=conv_stop_str,
        temperature=temperature,
        max_new_token=max_new_token,
    )
    evaluation = get_single_evaluation(
        model,
        output_ids,
        prefix_len,
        target_len,
        estimation_mode,
    )
    return output_tokens, evaluation

@torch.inference_mode()
def get_single_answer(
    tokenizer,
    model,
    prompt,
    conv_stop_token_ids=None,
    conv_stop_str=None,
    temperature=0.1,
    max_new_token=2048,
):
    input_ids = tokenizer([prompt]).input_ids

    if temperature < 1e-4:
        do_sample = False
    else:
        do_sample = True

    # some models may error out when generating long outputs
    outputs = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_token,
        output_attentions=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # sequences, attentions, scores
    prefix_len = len(input_ids[0])
    target_len = len(outputs["sequences"][0]) - prefix_len
    output_ids = outputs["sequences"][0][prefix_len:]

    # be consistent with the template's stop_token_ids
    if conv_stop_token_ids:
        stop_token_ids_index = [
            i
            for i, id in enumerate(output_ids)
            if id in conv_stop_token_ids
        ]
        if len(stop_token_ids_index) > 0:
            output_ids = output_ids[: stop_token_ids_index[0]]

    output_tokens = tokenizer.decode(
        output_ids,
        spaces_between_special_tokens=False,
    )
    if conv_stop_str and isinstance(conv_stop_str, list):
        stop_str_indices = sorted(
            [
                output_tokens.find(stop_str)
                for stop_str in conv_stop_str
                if output_tokens.find(stop_str) > 0
            ]
        )
        if len(stop_str_indices) > 0:
            output_tokens = output_tokens[: stop_str_indices[0]]
    elif conv_stop_str and output_tokens.find(conv_stop_str) > 0:
        output_tokens = output_tokens[: output_tokens.find(conv_stop_str)]

    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output_tokens = output_tokens.replace(special_tok, "")
        else:
            output_tokens = output_tokens.replace(special_token, "")
    
    output_tokens = output_tokens.strip()

    return output_tokens, prefix_len, target_len, outputs["sequences"]


def get_single_evaluation(
    model,
    output_ids,
    prefix_len,
    target_len,
    estimation_mode,
):
    # output_ids: The predicted ids consist of both instruction and response, shape is [1, sequence_len]
    # prefix_len: The length of the instruction part
    # target_len: The length of the response part
    # estimation_mode: Choose from logprobs, logprobs-entropy and logprobs-variance

    if estimation_mode == "logprobs":
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1) # the predict ids should be shifted left
        logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        logprobs[output_ids==-100] = 0 # instruction masking
        evaluation = torch.gather(logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        evaluation = evaluation.sum(-1)[0] / target_len # averaged on target length

    elif estimation_mode == "logprobs-entropy":
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1) # the predict ids should be shifted left
        logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        logprobs[output_ids==-100] = 0 # instruction masking
        # The original entropy has a minus sign, but we remove it to keep the positive correlation
        logprobs_entropy = torch.mean(logprobs * outputs["logits"])
        evaluation = logprobs_entropy.sum(-1)[0] / target_len # averaged on target length

    elif estimation_mode == "logprobs-variance":
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1) # the predict ids should be shifted left
        logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        logprobs_variance = torch.var(logprobs, dim=-1)
        logprobs_variance[output_ids==-100] = 0 # instruction masking
        evaluation = logprobs_variance.sum(-1)[0] / target_len # averaged on target length
    
    elif estimation_mode == "logprobs-both":
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1) # the predict ids should be shifted left
        logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

        logprobs_variance = torch.var(logprobs, dim=-1)
        logprobs_variance[output_ids==-100] = 0 # instruction masking
        evaluation_var = logprobs_variance.sum(-1)[0] / target_len # averaged on target length

        logprobs[output_ids==-100] = 0 # instruction masking
        # The original entropy has a minus sign, but we remove it to keep the positive correlation
        import pdb;pdb.set_trace()
        logprobs_entropy = torch.mean(logprobs * outputs["logits"])
        evaluation_ent = logprobs_entropy.sum(-1)[0] / target_len # averaged on target length

        return {"entropy": evaluation_ent, "variance": evaluation_var}
    
    else:
        raise Exception("Please check your estimation mode!")

    return evaluation.tolist()