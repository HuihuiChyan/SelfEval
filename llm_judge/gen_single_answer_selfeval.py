import copy
import torch

@torch.inference_mode()
def get_single_answer(
    tokenizer,
    model,
    prompt,
    conv_stop_token_ids=None,
    conv_stop_str=None,
    temperature=0.1,
    max_new_token=2048,
    estimation_mode="logprobs-entropy",
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
    total_len = len(outputs["sequences"][0])
    target_len = total_len - prefix_len
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

    if estimation_mode == "logprobs":
        output_ids = copy.deepcopy(outputs["sequences"])
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        second_outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1)
        log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
        log_probs[output_ids==-100] = 0 # instruction masking
        evaluation = torch.gather(log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1).sum(-1) / target_len

    elif estimation_mode == "logprobs-entropy":
        output_ids = copy.deepcopy(outputs["sequences"])
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        second_outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1)
        log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
        log_probs[output_ids==-100] = 0 # instruction masking
        log_probs = log_probs * second_outputs["logits"]
        evaluation = (log_probs.sum(-1) / target_len).sum(-1) / 32000

    elif estimation_mode == "logprobs-variance":
        output_ids = copy.deepcopy(outputs["sequences"])
        input_ids = copy.deepcopy(output_ids)
        output_ids[0][:prefix_len] = -100 # instruction masking
        second_outputs = model(
            input_ids=torch.as_tensor(input_ids).cuda(),
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        shifted_input_ids = torch.roll(input_ids, shifts=-1)
        log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
        log_probs = torch.var(log_probs, dim=-1)
        log_probs[output_ids==-100] = 0 # instruction masking
        evaluation = log_probs.sum(-1) / target_len
    
    # elif estimation_mode == "attention-variance":
    #     evaluation = 0.0
    #     for attn in outputs['attentions'][1:]: # (layer_num * [batch_size, head_num, 1, curr_len])
    #         attn = torch.cat(attn, dim=0).squeeze(dim=2) # [layer_num, head_num, curr_len]
    #         evaluation += torch.var(attn, dim=-1).sum(dim=0).sum(dim=0) / 32 #/ 32 少除了一个防止下溢出
    #     evaluation = evaluation / (len(outputs['attentions'])-1)

    elif estimation_mode == "attention-average":
        evaluation = 0.0
        for attn in outputs['attentions'][1:]: # (layer_num * [batch_size, head_num, 1, curr_len])
            attn = torch.cat(attn, dim=0).squeeze(dim=2) # [layer_num, head_num, curr_len]
            attn = torch.nn.functional.log_softmax(attn, dim=-1) * attn # [layer_num, head_num, curr_len]
            evaluation += attn.sum(dim=0).sum(dim=0).sum(dim=0) / 32 / attn.size(-1) #/ 32 少除了一个防止下溢出
        evaluation = evaluation / (len(outputs['attentions'])-1)

    elif estimation_mode == "attention-minimal":
        evaluation = 0.0
        for attn in outputs['attentions'][1:]: # (layer_num * [batch_size, head_num, 1, curr_len])
            attn = torch.cat(attn, dim=0).squeeze(dim=2) # [layer_num, head_num, curr_len]
            attn = torch.nn.functional.log_softmax(attn, dim=-1) * attn # [layer_num, head_num, curr_len]
            evaluation += (attn.sum(dim=-1) / attn.size(-1)).max() # logprob为负数，所以取maximal为minimal
        evaluation = evaluation / (len(outputs['attentions'])-1)

    elif estimation_mode == "scores":
        evaluation = torch.gather(torch.vstack(outputs["scores"]), dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1).sum(-1) / target_len
        evaluation = evaluation.unsqueeze(0) # 统一形状为 [batch_size]
    
    else:
        raise Exception("Please check your estimation mode!")

    return output_tokens, evaluation