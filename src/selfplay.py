from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np

from simple_features import ACTION_SPACE
from simple_features import encode_state


class SelfPlayEnv:
    """Very small adapter around the external poker engine interfaces.

    This class expects a `Game`-like runner and exposes two helpers:
    - encode_current_state(game_state, actor_idx) -> np.ndarray features
    - simulate_from_action(game_state, actor_idx, action_idx) -> float reward
    """

    def __init__(
        self,
        step_fn: Callable[[Any, int, str, int | None], Any],
        finish_and_reward_fn: Callable[[Any], float],
        clone_fn: Callable[[Any], Any],
    ) -> None:
        self.step_fn = step_fn
        self.finish_and_reward_fn = finish_and_reward_fn
        self.clone_fn = clone_fn

    def features(self, state: Any, actor_idx: int) -> np.ndarray:
        return encode_state(state, actor_idx)

    def rollout_once(self, state: Any, actor_idx: int, action_idx: int) -> float:
        """Clone state, apply the chosen action, finish the hand, return normalized reward."""
        s = self.clone_fn(state)

        # Map our discrete action to engine action name/amount.
        valid = getattr(s, "valid_actions", {})
        amount_to_call = int(valid.get("check_or_call", 0))
        pots = getattr(s, "pots", [])
        pot_total = int(sum(getattr(p, "unraked_amount", 0) for p in pots))
        stacks = getattr(s, "stacks", [])
        my_stack = int(stacks[actor_idx]) if stacks else 0

        name = ACTION_SPACE[action_idx]
        amount: int | None = None
        if name == "fold":
            pass
        elif name == "check_or_call":
            amount = amount_to_call
        elif name in ("half_pot", "pot", "all_in"):
            if name == "half_pot":
                amount = max(amount_to_call, pot_total // 2)
            elif name == "pot":
                amount = max(amount_to_call, pot_total)
            else:
                amount = my_stack

        s = self.step_fn(s, actor_idx, name, amount)
        reward_chips = self.finish_and_reward_fn(s)
        # Normalize by big blind if available, else by max(1, pot)
        bb = int(getattr(s, "big_blind", 1)) or 1
        denom = max(bb, 1)
        return float(reward_chips / denom)
