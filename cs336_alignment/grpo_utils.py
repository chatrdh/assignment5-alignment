import numpy as np
from typing import Callable,Literal
import torch
import einops


def compute_group_normalized_rewards(
reward_fn :Callable[[str, str], dict[str, float]],rollout_responses : list[str],
repeated_ground_truths : list[str],group_size : int,advantage_eps : float,
normalize_by_std : bool,):
    raw_rewards = []
    for rollout_response,ground_truth in zip(rollout_responses,repeated_ground_truths):
        rewards = reward_fn(rollout_response,ground_truth)
        raw_rewards.append(rewards["reward"])
    raw_rewards = torch.tensor(raw_rewards)
    
    advantages = []
    for i in range(len(raw_rewards)):
        if i % group_size == 0:  # Start of a new group
            mean_group = torch.mean(raw_rewards[i:i+group_size])
            if normalize_by_std:
                std_group = torch.std(raw_rewards[i:i+group_size])
            for j in range(i, i+group_size):
                if normalize_by_std:
                    advantages.append((raw_rewards[j] - mean_group) / (std_group + advantage_eps))
                else:
                    advantages.append(raw_rewards[j] - mean_group)
    
    
    return (advantages,raw_rewards,{"mean":torch.mean(raw_rewards[group_size-1::group_size]),"std":torch.std(raw_rewards[group_size-1::group_size])})



def compute_naive_policy_gradient_loss(
raw_rewards_or_advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    b,seq = policy_log_probs.shape
    raw_rewards_or_advantages = einops.repeat(raw_rewards_or_advantages,'b 1 -> b s',s=seq)
    return -(raw_rewards_or_advantages * policy_log_probs) 

def compute_grpo_clip_loss(
advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
old_log_probs: torch.Tensor,
cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = einops.repeat(advantages,'b 1 -> b s',s=policy_log_probs.shape[1])
    # Probability ratio: π_θ / π_old = exp(log_π_θ - log_π_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    standard_loss = ratio * advantages
    clip_loss = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    
    grpo_loss = -torch.min(standard_loss, clip_loss)
    return grpo_loss, {"clipped_or_not": ratio.gt(1 + cliprange).float().mean()}
    
    
def compute_policy_gradient_loss(
policy_log_probs: torch.Tensor,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None = None,
advantages: torch.Tensor | None = None,
old_log_probs: torch.Tensor | None = None,
cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs),{"clipped_or_not":torch.tensor(0.0)}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages,policy_log_probs),{"clipped_or_not":torch.tensor(0.0)}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)




def masked_mean(
tensor: torch.Tensor,
mask: torch.Tensor,
dim: int | None = None,
) -> torch.Tensor:
    if dim is None:
        return torch.sum(tensor * mask) / torch.sum(mask)
    else:
        return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None = None,
advantages: torch.Tensor | None = None,
old_log_probs: torch.Tensor | None = None,
cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns:
tuple[torch.Tensor, dict[str, torch.Tensor]].
loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
this so we can log it.
metadata Dict with metadata from the underlying loss call, and any other statistics you
might want to log."""
    per_token_loss,metadata = compute_policy_gradient_loss(policy_log_probs,loss_type,raw_rewards,advantages,old_log_probs,cliprange)
    loss = masked_mean(per_token_loss,response_mask)
    
    loss = loss/gradient_accumulation_steps
    loss.backward()
    return loss,metadata
    
    