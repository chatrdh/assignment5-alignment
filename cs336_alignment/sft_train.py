import os
import json
import torch
import typer
import wandb
import gc
from typing import List, Optional, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd

# Import your adapter assignments
from cs336_alignment.sftutils import (
    tokenize_prompt_and_output, 
    sft_microbatch_train_step, 
    get_response_log_probs
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


app = typer.Typer()

# --- 1. Model & vLLM Initialization ---

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

def init_vllm_engine(model_id: str, device: str, seed: int = 42) -> LLM:
    print(f"Initializing vLLM engine on {device}...")
    vllm_set_random_seed(seed)
    
    # Patches for single-process/single-device vLLM
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            # CRITICAL: Reserve only 20-30% of GPU for vLLM so training fits
            gpu_memory_utilization=0.3, 
            tensor_parallel_size=1, 
        )

def sync_policy_to_vllm(policy: torch.nn.Module, vllm_engine: LLM):
    """
    Copies weights from the trained policy to the vLLM instance.
    """
    state_dict = policy.state_dict()
    llm_model = vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# --- 2. Data Loading (Same as before) ---

def get_r1_prompt(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant "
        "solves it. The Assistant first thinks about the reasoning process in the mind and then "
        "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
        "and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> "
        "reasoning process here </think> <answer> answer here </answer>.\n"
        f"User: {question}\nAssistant: <think>"
    )

def load_validation_data(limit: int = 128) -> List[Dict]:
    print("Fetching validation data...")
    subsets = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    # Base URL for Parquet files
    base_url = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/refs%2Fconvert%2Fparquet"

    # Load and combine all test examples
    test_dfs = []
    for subset in subsets:
        print(f"Loading {subset} test split...")
        test_url = f"{base_url}/{subset}/test/0000.parquet"
        df = pd.read_parquet(test_url)
        df['subset'] = subset  # Add a column to track which subset each row came from
        test_dfs.append(df)

    test_data = pd.concat(test_dfs, ignore_index=True)
    dataset = Dataset.from_pandas(test_data)
    
    #dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    val_data = []
    for i, example in enumerate(dataset):
        if i >= limit: break
        val_data.append({
            "prompt": get_r1_prompt(example["problem"]),
            "answer": example["solution"]
        })
    return val_data

class SFTDataset(Dataset):
    def __init__(self, path: str, limit: Optional[int] = None):
        self.data = []
        print(f"Loading SFT data from {path}...")
        with open(path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        if limit:
            self.data = self.data[:limit]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_train_dataloader(data_path, tokenizer, batch_size, device, limit=None):
    dataset = SFTDataset(data_path, limit=limit)
    
    def collate_fn(batch):
        prompts = [x["prompt"] for x in batch]
        responses = [x["response"] for x in batch]
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        return {
            "input_ids": tokenized["input_ids"].to(device),
            "labels": tokenized["labels"].to(device),
            "response_mask": tokenized["response_mask"].to(device),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --- 3. Evaluation Logic ---

def run_evaluation(model, vllm_engine, val_data, step):
    print(f"Running evaluation at step {step}...")
    
    # 1. Clear Cache to make room for inference
    torch.cuda.empty_cache()
    
    # 2. Sync weights
    sync_policy_to_vllm(model, vllm_engine)
    
    # 3. Generate
    sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=1024, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    prompts = [x["prompt"] for x in val_data]
    outputs = vllm_engine.generate(prompts, sampling_params)
    
    # 4. Score
    total_score = 0
    for i, out in enumerate(outputs):
        generated_text = out.outputs[0].text
        ground_truth = val_data[i]["answer"]
        metrics = r1_zero_reward_fn(generated_text, ground_truth)
        if metrics.get("reward", 0) == 1.0:
            total_score += 1
            
    accuracy = total_score / len(val_data)
    wandb.log({"eval/accuracy": accuracy}, step=step)
    print(f"Step {step} | Eval Accuracy: {accuracy:.2%}")
    
    # 5. Switch back to train and clear cache again
    model.train()
    torch.cuda.empty_cache()

# --- 4. Main Training Loop ---

@app.command()
def main(
    exp_name: str = "sft_single_device",
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    sft_data_path: str = "sft.jsonl", 
    output_dir: str = "./checkpoints",
    lr: float = 1e-5,
    num_epochs: int = 5,
    train_batch_size: int = 16,     
    micro_batch_size: int = 2,      
    eval_every_steps: int = 50,
    device: str = "cuda:0", # Single device for everything
    dataset_limit: int = 256
):
    wandb.init(project="cs336-assignment5-sft", name=exp_name)
    wandb.define_metric("train_step")
    os.makedirs(output_dir, exist_ok=True)
    
    grad_accum_steps = train_batch_size // micro_batch_size

    # --- Initialize BOTH on the same device ---
    # Note: Initialize vLLM first to reserve its block manager memory, 
    # then the training model. 
    vllm_engine = init_vllm_engine(model_id, device)
    model, tokenizer = init_policy_model(model_id, device)
    
    train_loader = get_train_dataloader(sft_data_path, tokenizer, micro_batch_size, device,limit=dataset_limit)
    val_data = load_validation_data(limit=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    global_step = 0
    print(f"Starting Single-Device Training on {device}...")
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            
            # Forward & Backward
            log_probs_out = get_response_log_probs(
                model=model, 
                input_ids=batch["input_ids"], 
                labels=batch["labels"]
            )
            
            loss, metrics = sft_microbatch_train_step(
                policy_log_probs=log_probs_out["log_probs"],
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=grad_accum_steps
            )
            
            # Optimizer Step
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                wandb.log({
                    "train_step": global_step,
                    "train/loss": metrics["loss"].item(),
                    "train/tokens": metrics["num_response_tokens"].item()
                })

                # Evaluation
                if global_step % eval_every_steps == 0:
                    run_evaluation(model, vllm_engine, val_data, global_step)

    print(f"Saving artifacts to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    app()