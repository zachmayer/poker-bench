from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from policy_net import MLPPolicy
from policy_net import PolicyConfig
from policy_net import copy_policy


@dataclass
class GRPOConfig:
    input_dim: int
    num_actions: int
    hidden_dim: int = 64
    group_size: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.05
    lr: float = 3e-4
    temperature: float = 1.0
    advantage_clip: float = 4.0


class GRPOTrainer:
    """Minimal GRPO trainer (no critic): group baseline + PPO-style clip + KL.

    rollouts_fn should, given (features, action_idx), return a scalar normalized
    terminal reward (e.g., chips delta in big blinds) for that action sampled at
    the given state, by simulating the rest of the hand.
    """

    def __init__(
        self,
        cfg: GRPOConfig,
        rollouts_fn: Callable[[np.ndarray, int], float],
    ) -> None:
        self.cfg = cfg
        self.policy = MLPPolicy(PolicyConfig(cfg.input_dim, cfg.num_actions, cfg.hidden_dim))
        self.ref_policy = copy_policy(self.policy)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.rollouts_fn = rollouts_fn

    def step(self, batch_states: Sequence[np.ndarray]) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        g = self.cfg.group_size
        clip_eps = self.cfg.clip_eps
        kl_beta = self.cfg.kl_beta

        losses: list[torch.Tensor] = []
        pre_kls: list[torch.Tensor] = []
        reward_stds: list[float] = []
        mean_abs_adv: list[float] = []
        saved_actions_per_state: list[list[int]] = []

        for s_np in batch_states:
            # Sample G actions from current policy
            s = torch.from_numpy(s_np).float().to(device)
            actions: list[int] = []
            logp_new: list[float] = []
            for _ in range(g):
                a, lp = self.policy.act(s, temperature=self.cfg.temperature)
                actions.append(a)
                logp_new.append(lp)
            saved_actions_per_state.append(actions)

            # Compute rewards via environment rollouts (CPU side)
            rewards = np.array([self.rollouts_fn(s_np, a) for a in actions], dtype=float)
            r_mean = rewards.mean()
            r_std = rewards.std() if rewards.std() > 1e-8 else 1.0
            adv = (rewards - r_mean) / r_std
            adv = np.clip(adv, -self.cfg.advantage_clip, self.cfg.advantage_clip)
            reward_stds.append(float(rewards.std()))
            mean_abs_adv.append(float(np.mean(np.abs(adv))))

            # Build tensors
            adv_t = torch.from_numpy(adv).float().to(device)
            acts_t = torch.tensor(actions, dtype=torch.long, device=device)
            logp_old_t = torch.tensor(logp_new, dtype=torch.float32, device=device)

            # Recompute logprobs and ref logprobs under current/ref policies
            logp_curr: list[torch.Tensor] = []
            logp_ref: list[torch.Tensor] = []
            for a_idx in actions:
                logp_curr.append(self.policy.log_prob(s, a_idx))
                with torch.no_grad():
                    logp_ref.append(self.ref_policy.log_prob(s, a_idx))  # type: ignore[attr-defined]
            logp_curr_t = torch.stack(logp_curr)
            logp_ref_t = torch.stack(logp_ref)

            # Importance ratio vs. fixed snapshot (use old logp from sampling)
            ratio = torch.exp(logp_curr_t - logp_old_t)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = -torch.mean(torch.minimum(ratio * adv_t, clipped * adv_t))

            # KL penalty to reference policy
            # Here use reverse KL estimator: KL(pi||ref) ≈ E_a[ log pi(a) - log ref(a) ]
            # for sampled actions in the group.
            pre_kl = torch.mean(logp_curr_t - logp_ref_t)
            loss = pg_loss + kl_beta * pre_kl

            losses.append(loss)
            pre_kls.append(pre_kl.detach())

        self.opt.zero_grad(set_to_none=True)
        total = torch.stack(losses).mean()
        total.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        # Compute post-update KL on same states/actions
        with torch.no_grad():
            post_kls: list[torch.Tensor] = []
            for s_np, actions in zip(batch_states, saved_actions_per_state, strict=False):
                s = torch.from_numpy(s_np).float().to(device)
                logp_curr = torch.stack([self.policy.log_prob(s, a) for a in actions])
                logp_ref = torch.stack([self.ref_policy.log_prob(s, a) for a in actions])
                post_kls.append(torch.mean(logp_curr - logp_ref))
            post_kl = torch.stack(post_kls).mean()

        # Slow update of reference policy (EMA-like) AFTER measuring KL
        with torch.no_grad():
            for p_ref, p in zip(self.ref_policy.parameters(), self.policy.parameters(), strict=False):
                p_ref.mul_(0.99).add_(0.01 * p)

        stats: dict[str, float] = {}
        stats["loss"] = float(total.item())
        stats["kl_pre"] = float(torch.stack(pre_kls).mean().item())
        stats["kl_post"] = float(post_kl.item())
        stats["reward_std"] = float(np.mean(reward_stds))
        stats["mean_abs_adv"] = float(np.mean(mean_abs_adv))
        stats["batch"] = float(len(batch_states))
        return stats
