from grpo_utils import compute_group_normalized_rewards,grpo_microbatch_train_step
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch    

# 1.Model and VLLM init
def init_policy_model(model_id: str, device: str) -> tuple:
    print(f"Loading policy model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)
    model.train()
    return model, tokenizer