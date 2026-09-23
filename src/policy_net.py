from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    input_dim: int
    num_actions: int
    hidden_dim: int = 128


class MLPPolicy(nn.Module):
    """A tiny, readable policy network for discrete actions.

    - Input: float feature vector of length ``input_dim``
    - Output: categorical logits over ``num_actions`` actions
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    @torch.no_grad()
    def act(self, features: torch.Tensor, temperature: float = 1.0) -> tuple[int, float]:
        """Sample an action and return (action_index, logprob).

        Features: shape (input_dim,)
        """
        logits = self.forward(features)
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())

    def log_prob(self, features: torch.Tensor, action_idx: int) -> torch.Tensor:
        logits = self.forward(features)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[action_idx]


def copy_policy(src: nn.Module) -> nn.Module:
    dst = type(src)(src.config)  # type: ignore[arg-type]
    dst.load_state_dict(src.state_dict())
    for p in dst.parameters():
        p.requires_grad_(False)
    return dst
