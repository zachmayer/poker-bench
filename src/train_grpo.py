from __future__ import annotations

import argparse
import random
from typing import Any
from typing import Iterable

import numpy as np
from PokerBots import Game
from PokerBots import RandomPlayer

# PokerBots engine imports
from PokerBots.players.BasePlayer import BasePlayer

from .grpo_trainer import GRPOConfig
from .grpo_trainer import GRPOTrainer
from .policy_net import MLPPolicy
from .policy_net import PolicyConfig
from .simple_features import ACTION_SPACE
from .simple_features import encode_state


def _choose_raise_to(allowed: Iterable[int], target: int) -> int | None:
    vals = list(allowed)
    if not vals:
        return None
    # pick the smallest allowed >= target, else the max allowed
    ge = [v for v in vals if v >= target]
    return min(ge) if ge else max(vals)


class PolicyPlayer(BasePlayer):
    """Hero controlled by our policy network.

    If forced_action_idx is set, it will be used for the first decision only.
    Afterwards, the policy is used for all subsequent decisions.
    """

    def __init__(self, name: str, policy: MLPPolicy, forced_action_idx: int | None = None) -> None:
        super().__init__(name)
        self.policy = policy
        self.forced_action_idx = forced_action_idx
        self._used_forced = False

    def play(self, valid_actions: dict[str, Any], state: Any) -> tuple[str, int | None]:  # type: ignore[override]
        # Decide discrete action index using full feature vector
        if (self.forced_action_idx is not None) and (not self._used_forced):
            action_idx = self.forced_action_idx
            self._used_forced = True
        else:
            # Build feature vector from current state
            actor_idx = int(getattr(state, "actor_index", 0))
            feats_np = encode_state(state, actor_idx, valid_actions).astype(np.float32)
            import torch

            action_idx, _ = self.policy.act(torch.from_numpy(feats_np))

        name = ACTION_SPACE[action_idx]

        # Map to engine action
        can_call = "check_or_call" in valid_actions
        can_raise = (
            "complete_bet_or_raise_to" in valid_actions
            and len(list(valid_actions.get("complete_bet_or_raise_to", []))) > 0
        )

        amount_to_call = int(valid_actions.get("check_or_call", 0)) if can_call else 0

        # Pot / stacks to size raises approximately
        pots = getattr(state, "pots", [])
        pot_total = int(sum(getattr(p, "unraked_amount", 0) for p in pots))
        stacks = getattr(state, "stacks", [])
        actor_idx = int(getattr(state, "actor_index", 0))
        my_stack = int(stacks[actor_idx]) if stacks else 0

        if name == "fold":
            # If folding when check is free, convert to check
            if can_call and amount_to_call == 0:
                return "check_or_call", 0
            return "fold", None
        if name == "check_or_call":
            return "check_or_call", amount_to_call if can_call else 0

        # Raise buckets
        if not can_raise:
            # Fallback to call if raise illegal
            if can_call:
                return "check_or_call", amount_to_call
            return "fold", None

        targets = {
            "half_pot": max(amount_to_call, pot_total // 2),
            "pot": max(amount_to_call, pot_total),
            "all_in": my_stack if my_stack > 0 else amount_to_call,
        }
        target = targets[name]
        raise_to_values = list(valid_actions.get("complete_bet_or_raise_to", []))
        chosen = _choose_raise_to(raise_to_values, target)
        if chosen is None:
            # No legal raise_to values; fallback
            return ("check_or_call", amount_to_call) if can_call else ("fold", None)
        return "complete_bet_or_raise_to", int(chosen)


def rollout_episode(
    forced_action_idx: int | None, policy: MLPPolicy, n_opponents: int = 3, initial_stack: int = 30_000
) -> float:
    """Play a full round; hero uses a forced first action (optional), then policy.

    Returns hero chips delta (end_stack - initial_stack).
    """
    hero = PolicyPlayer("Hero", policy, forced_action_idx=forced_action_idx)
    opponents = [RandomPlayer(name=f"R{i}") for i in range(n_opponents)]
    players = [hero] + opponents
    game = Game(players=players, initial_stack=initial_stack)
    game.play_round(verbose=False)
    # Hero is seat 0
    return float(game.stacks[0] - initial_stack)


def evaluate_policy(policy: MLPPolicy, episodes: int = 40, initial_stack: int = 30_000) -> float:
    """Average chips/BB over several hands vs. RandomPlayers."""
    total = 0.0
    for _ in range(episodes):
        total += rollout_episode(None, policy, initial_stack=initial_stack)
    bb = 100.0
    return total / (episodes * bb)


class ProbePlayer(BasePlayer):
    """A player that records a feature vector at its first decision and plays a safe action."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.captured: np.ndarray | None = None
        self._captured_once = False

    def play(self, valid_actions: dict[str, Any], state: Any) -> tuple[str, int | None]:  # type: ignore[override]
        if not self._captured_once:
            actor_idx = int(getattr(state, "actor_index", 0))
            self.captured = encode_state(state, actor_idx, valid_actions).astype(np.float32)
            self._captured_once = True
        # Play safe: check/call if possible, otherwise fold
        if "check_or_call" in valid_actions:
            return "check_or_call", int(valid_actions.get("check_or_call", 0))
        return "fold", None


def infer_input_dim() -> int:
    # Spin a tiny game with a probe player to capture a live feature vector
    probe = ProbePlayer("Probe")
    opponents = [RandomPlayer(name="R0"), RandomPlayer(name="R1"), RandomPlayer(name="R2")]
    game = Game(players=[probe] + opponents, initial_stack=10_000)
    game.play_round(verbose=False)
    feats = probe.captured
    if feats is None:
        raise RuntimeError("Failed to capture features from ProbePlayer.")
    return int(feats.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Configure policy dims based on live engine state
    input_dim = infer_input_dim()
    num_actions = len(ACTION_SPACE)

    # Build policy and trainer
    policy = MLPPolicy(PolicyConfig(input_dim=input_dim, num_actions=num_actions, hidden_dim=args.hidden))

    cfg = GRPOConfig(
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden,
        group_size=args.group,
        clip_eps=0.2,
        kl_beta=args.kl_beta,
        lr=args.lr,
        temperature=args.temperature,
        advantage_clip=args.adv_clip,
    )

    def rollout_fn(_feats: np.ndarray, action_idx: int) -> float:
        # Use the real engine to play a hand with a forced first action
        chips = rollout_episode(action_idx, policy)
        # Normalize by big blind ~ assume 100 for simplicity if BB not exposed; use 100 to keep scale sane
        bb = 100.0
        return chips / bb

    trainer = GRPOTrainer(cfg, rollout_fn)

    # Training loop
    steps = args.steps
    batch_size = args.batch
    eval_every = args.eval_every
    eval_episodes = args.eval_episodes
    print("starting eval:", round(evaluate_policy(policy, episodes=eval_episodes), 4))
    for step in range(1, steps + 1):
        # Build a fresh batch of probe states
        batch_states: list[np.ndarray] = []
        for _ in range(batch_size):
            probe = ProbePlayer("ProbeBatch")
            opponents = [RandomPlayer(name="RB0"), RandomPlayer(name="RB1"), RandomPlayer(name="RB2")]
            game = Game(players=[probe] + opponents, initial_stack=10_000)
            game.play_round(verbose=False)
            feats = probe.captured
            if feats is not None:
                batch_states.append(feats)
        if not batch_states:
            continue
        stats = trainer.step(batch_states)
        if step % eval_every == 0:
            score = evaluate_policy(policy, episodes=eval_episodes)
            log = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}
            print(f"step {step:03d} score {score:.4f} stats {log}")


if __name__ == "__main__":
    main()
