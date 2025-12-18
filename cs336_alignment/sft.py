import torch
import einops
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_tokens_list = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_strs]
    output_tokens_list = [tokenizer(o, add_special_tokens=False)["input_ids"] for o in output_strs]

    input_ids_list = []
    labels_list = []
    response_mask_list = []

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    for p_ids, o_ids in zip(prompt_tokens_list, output_tokens_list):
        # Full sequence
        full_seq = p_ids + o_ids
        
        # input_ids: full sequence (will be sliced after padding)
        curr_input_ids = full_seq
        
        # labels: shifted sequence + EOS (will be sliced after padding)
        # But we need EOS to be at the right position after slicing
        curr_labels = full_seq[1:] + [eos_id]
        
        # Mask
        mask_zeros = [0] * (len(p_ids) - 1)
        mask_ones = [1] * (len(o_ids) )
        mask_zero_eos = [0]
        curr_mask = mask_zeros + mask_ones + mask_zero_eos
        #curr_mask = mask_zeros + mask_ones
        
        input_ids_list.append(torch.tensor(curr_input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(curr_labels, dtype=torch.long))
        response_mask_list.append(torch.tensor(curr_mask, dtype=torch.long))

    # Pad sequences
    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_val)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=eos_id)
    response_mask = torch.nn.utils.rnn.pad_sequence(response_mask_list, batch_first=True, padding_value=0)
    
    # Slice off the final column from all three
    input_ids = input_ids[:, :-1]
    labels = labels[:, :-1]
    response_mask = response_mask[:, :-1]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
   # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute log probabilities (more numerically stable)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy = -sum(p * log(p)) over vocab dimension
    # Using einops to sum over the vocabulary dimension
    entropy = -einops.reduce(probs * log_probs, 'batch seq vocab -> batch seq', 'sum')
    
    return entropy

def get_response_log_probs(
model: torch.nn.Module,
input_ids: torch.Tensor,
labels: torch.Tensor,
return_token_entropy: bool = False,
):
    output = model(input_ids)
    logits = output.logits
    log_probs_all = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)
    log_probs = torch.gather(
        log_probs_all, 
        dim=-1, 
        index=labels.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
    ).squeeze(-1)
    result = {"log_probs": log_probs}

    if return_token_entropy :
       entropy=  compute_entropy(logits)
       result["token_entropy"] =entropy
    
    return result
    