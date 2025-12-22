import os
import torch
import typer
import wandb
import pandas as pd
from typing import List, Dict, Optional
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.sftutils import get_response_log_probs
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

# --- 2. Prompt Formatting & Data Loading ---

def get_r1_prompt(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant "
        "solves it. The Assistant first thinks about the reasoning process in the mind and then "
        "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
        "and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> "
        "reasoning process here </think> <answer> answer here </answer>.\n"
        f"User: {question}\nAssistant: <think>"
    )

def load_train_prompts(limit: Optional[int] = None) -> List[Dict]:
    """
    Load training prompts (questions + ground truth answers) from MATH dataset.
    Returns list of {"prompt": str, "answer": str}
    """
    print("Loading training prompts from MATH dataset...")
    subsets = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]
    
    base_url = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/refs%2Fconvert%2Fparquet"
    
    train_dfs = []
    for subset in subsets:
        print(f"Loading {subset} train split...")
        train_url = f"{base_url}/{subset}/train/0000.parquet"
        df = pd.read_parquet(train_url)
        df['subset'] = subset
        train_dfs.append(df)
    
    train_data = pd.concat(train_dfs, ignore_index=True)
    dataset = Dataset.from_pandas(train_data)
    
    prompts = []
    for i, example in enumerate(dataset):
        if limit and i >= limit:
            break
        prompts.append({
            "prompt": get_r1_prompt(example["problem"]),
            "answer": example["solution"]
        })
    
    print(f"Loaded {len(prompts)} training prompts")
    return prompts

def sample_prompt_batch(
    prompts: List[Dict], 
    batch_size: int, 
    group_size: int
) -> tuple[List[str], List[str]]:
    """
    Sample a batch of prompts and repeat each prompt group_size times for rollouts.
    Returns: (repeated_prompts, repeated_ground_truths)
    """
    import random
    
    # Sample batch_size // group_size unique prompts
    n_unique = batch_size // group_size
    sampled = random.sample(prompts, min(n_unique, len(prompts)))
    
    # Repeat each prompt group_size times
    repeated_prompts = []
    repeated_ground_truths = []
    for item in sampled:
        for _ in range(group_size):
            repeated_prompts.append(item["prompt"])
            repeated_ground_truths.append(item["answer"])
    
    return repeated_prompts, repeated_ground_truths

def generate_rollouts(
    vllm_engine: LLM,
    prompts: List[str],
    temperature: float = 1.0,
    min_tokens: int = 4,
    max_tokens: int = 1024,
) -> List[str]:
    """
    Generate rollouts (completions) for a batch of prompts using vLLM.
    Returns list of generated response strings.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    outputs = vllm_engine.generate(prompts, sampling_params)
    
    # Extract the generated text from each output
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)
    
    return responses

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
    
    val_data = []
    for i, example in enumerate(dataset):
        if i >= limit: break
        val_data.append({
            "prompt": get_r1_prompt(example["problem"]),
            "answer": example["solution"]
        })
    return val_data

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
    exp_name: str = "grpo_single_device",
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    output_dir: str = "./checkpoints",
    device: str = "cuda:0",
    # GRPO hyperparameters
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",  # "no_baseline", "reinforce_with_baseline", "grpo_clip"
    use_std_normalization: bool = True,
    eval_every_steps: int = 50,
    dataset_limit: int = 256,
):
    """GRPO Training Loop"""
    wandb.init(project="cs336-assignment5-grpo", name=exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Microbatch size = train_batch_size / gradient_accumulation_steps
    micro_batch_size = train_batch_size // gradient_accumulation_steps
    
    # --- Initialize vLLM and Policy Model ---
    vllm_engine = init_vllm_engine(model_id, device)
    policy, tokenizer = init_policy_model(model_id, device)
    
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    val_data = load_validation_data(limit=128)
    
    # Load training prompts
    train_prompts = load_train_prompts(limit=dataset_limit)
    
    print(f"Starting GRPO Training on {device}...")
    print(f"  n_grpo_steps: {n_grpo_steps}")
    print(f"  rollout_batch_size: {rollout_batch_size}")
    print(f"  group_size: {group_size}")
    print(f"  train_batch_size: {train_batch_size}")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  micro_batch_size: {micro_batch_size}")
    
    global_step = 0
    
    for step in range(n_grpo_steps):
        print(f"\n=== GRPO Step {step + 1}/{n_grpo_steps} ===")
        
        # 1. Sync policy weights to vLLM for generation
        sync_policy_to_vllm(policy, vllm_engine)
        torch.cuda.empty_cache()
        
        # 2. Sample prompts
        prompts, ground_truths = sample_prompt_batch(
            train_prompts, 
            rollout_batch_size, 
            group_size
        )
        
        # 3. Generate rollouts with vLLM
        rollout_responses = generate_rollouts(
            vllm_engine,
            prompts,
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
        )
        
        # 4. Compute rewards and group-normalized advantages
        advantages, raw_rewards, reward_metrics = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        
        # Convert to tensors
        advantages = torch.stack(advantages).unsqueeze(1).to(device)  # (B, 1)
        raw_rewards = raw_rewards.unsqueeze(1).to(device)  # (B, 1)
        
        print(f"  Mean reward: {reward_metrics['mean']:.4f}, Std: {reward_metrics['std']:.4f}")
        
        # 5. Tokenize prompts + responses for training
        full_texts = [p + r for p, r in zip(prompts, rollout_responses)]
        tokenized = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        
        # Create response mask (tokens after prompt)
        prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        response_mask = torch.zeros_like(tokenized["input_ids"], dtype=torch.bool)
        for i, plen in enumerate(prompt_lengths):
            response_mask[i, plen:] = True
        
        # 6. Get old log probs (for GRPO clip) before training
        policy.eval()
        with torch.no_grad():
            old_log_probs_out = get_response_log_probs(
                model=policy,
                input_ids=tokenized["input_ids"],
                labels=tokenized["input_ids"],
            )
            old_log_probs = old_log_probs_out["log_probs"]
        policy.train()
        
        # 7. Training epochs over this rollout batch
        for epoch in range(epochs_per_rollout_batch):
            # Shuffle indices for microbatches
            indices = torch.randperm(len(prompts))
            
            for mb_start in range(0, len(prompts), micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, len(prompts))
                mb_indices = indices[mb_start:mb_end]
                
                # Get microbatch data
                mb_input_ids = tokenized["input_ids"][mb_indices]
                mb_response_mask = response_mask[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_raw_rewards = raw_rewards[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Forward pass to get current log probs
                log_probs_out = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_input_ids,
                )
                policy_log_probs = log_probs_out["log_probs"]
                
                # GRPO microbatch train step
                loss, metrics = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=0.2,
                )
                
                # Optimizer step after gradient accumulation
                if (mb_start // micro_batch_size + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/mean_reward": reward_metrics["mean"].item(),
                        "train/std_reward": reward_metrics["std"].item(),
                        "train/clipped_ratio": metrics.get("clipped_or_not", torch.tensor(0.0)).item(),
                    }, step=global_step)
        
        # 8. Evaluation
        if (step + 1) % eval_every_steps == 0:
            run_evaluation(policy, vllm_engine, val_data, global_step)
    
    print(f"Saving artifacts to {output_dir}")
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    app()
