from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .qwen_policy import QwenPolicy


@dataclass
class QwenGRPOConfig:
    group_size: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.05
    lr: float = 2e-4
    temperature: float = 1.0
    advantage_clip: float = 4.0
    grad_clip: float = 1.0


class QwenGRPOTrainer:
    """GRPO-style trainer for QwenPolicy with LoRA adapters.

    For each prompt, samples a group of actions from the current policy, computes
    rewards via the provided rollout function, normalizes to advantages with a
    group baseline, and applies PPO-style clipped policy gradient with a KL
    penalty to the frozen base (reference) model.
    """

    def __init__(self, policy: QwenPolicy, cfg: QwenGRPOConfig, rollout_fn: Callable[[str, int], float]) -> None:
        self.policy = policy
        self.cfg = cfg
        self.rollout_fn = rollout_fn
        self.opt = torch.optim.Adam(self.policy.model.parameters(), lr=cfg.lr)

    def step(self, prompts: Sequence[str]) -> dict[str, float]:
        g = self.cfg.group_size
        clip_eps = self.cfg.clip_eps
        kl_beta = self.cfg.kl_beta

        device = self.policy.device

        losses: list[torch.Tensor] = []
        pre_kls: list[torch.Tensor] = []
        reward_stds: list[float] = []
        mean_abs_adv: list[float] = []

        saved_actions_per_prompt: list[list[int]] = []
        saved_prompts: list[str] = []
        saved_logp_old_per_prompt: list[torch.Tensor] = []

        for prompt in prompts:
            actions: list[int] = []
            logp_old: list[float] = []
            for _ in range(g):
                a, lp = self.policy.act(prompt, temperature=self.cfg.temperature)
                actions.append(a)
                logp_old.append(lp)
            saved_actions_per_prompt.append(actions)
            saved_prompts.append(prompt)
            logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=device)
            saved_logp_old_per_prompt.append(logp_old_t)

            # CPU-side rollouts to compute rewards
            rewards = np.array([self.rollout_fn(prompt, a) for a in actions], dtype=float)
            r_mean = rewards.mean()
            r_std = rewards.std() if rewards.std() > 1e-8 else 1.0
            adv = (rewards - r_mean) / r_std
            adv = np.clip(adv, -self.cfg.advantage_clip, self.cfg.advantage_clip)
            reward_stds.append(float(rewards.std()))
            mean_abs_adv.append(float(np.mean(np.abs(adv))))

            adv_t = torch.from_numpy(adv).float().to(device)

            # Recompute current/ref logprobs for sampled actions
            logp_curr = torch.stack([self.policy.log_prob(prompt, a) for a in actions]).to(device)
            with torch.no_grad():
                logp_ref = torch.stack([self.policy.ref_log_prob(prompt, a) for a in actions]).to(device)

            ratio = torch.exp(logp_curr - logp_old_t)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = -torch.mean(torch.minimum(ratio * adv_t, clipped * adv_t))

            pre_kl = torch.mean(logp_curr - logp_ref)
            loss = pg_loss + kl_beta * pre_kl

            losses.append(loss)
            pre_kls.append(pre_kl.detach())

        self.opt.zero_grad(set_to_none=True)
        total = torch.stack(losses).mean()
        total.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=self.cfg.grad_clip)
        self.opt.step()

        # Post-update KL on same prompts/actions
        with torch.no_grad():
            post_kls: list[torch.Tensor] = []
            for prompt, actions in zip(saved_prompts, saved_actions_per_prompt, strict=False):
                logp_curr = torch.stack([self.policy.log_prob(prompt, a) for a in actions]).to(device)
                logp_ref = torch.stack([self.policy.ref_log_prob(prompt, a) for a in actions]).to(device)
                post_kls.append(torch.mean(logp_curr - logp_ref))
            post_kl = torch.stack(post_kls).mean()

        stats: dict[str, float] = {}
        stats["loss"] = float(total.item())
        stats["kl_pre"] = float(torch.stack(pre_kls).mean().item())
        stats["kl_post"] = float(post_kl.item())
        stats["reward_std"] = float(np.mean(reward_stds))
        stats["mean_abs_adv"] = float(np.mean(mean_abs_adv))
        stats["batch"] = float(len(prompts))
        return stats
