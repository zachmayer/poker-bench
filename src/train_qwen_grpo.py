from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer

from .qwen_grpo_trainer import QwenGRPOConfig
from .qwen_grpo_trainer import QwenGRPOTrainer
from .qwen_policy import QwenConfig
from .qwen_policy import QwenPolicy
from .simple_features import ACTION_SPACE
from .simple_features import encode_state


class PolicyPlayer(BasePlayer):
    def __init__(self, name: str, policy: QwenPolicy, forced_action_idx: int | None = None) -> None:
        super().__init__(name)
        self.policy = policy
        self.forced_action_idx = forced_action_idx
        self._used_forced = False

    def play(self, valid_actions: dict[str, Any], state: Any) -> tuple[str, int | None]:  # type: ignore[override]
        if (self.forced_action_idx is not None) and (not self._used_forced):
            action_idx = self.forced_action_idx
            self._used_forced = True
        else:
            actor_idx = int(getattr(state, "actor_index", 0))
            feats = encode_state(state, actor_idx, valid_actions)
            # Build a tiny prompt from numeric features
            prompt = (
                "state:"
                + ",".join(map(lambda x: f"{x:.3f}", feats.tolist()))
                + "\nchoose one of: "
                + ",".join(ACTION_SPACE)
            )
            action_idx, _ = self.policy.act(prompt)

        name = ACTION_SPACE[action_idx]

        can_call = "check_or_call" in valid_actions
        can_raise = (
            "complete_bet_or_raise_to" in valid_actions
            and len(list(valid_actions.get("complete_bet_or_raise_to", []))) > 0
        )
        amount_to_call = int(valid_actions.get("check_or_call", 0)) if can_call else 0

        pots = getattr(state, "pots", [])
        pot_total = int(sum(getattr(p, "unraked_amount", 0) for p in pots))
        stacks = getattr(state, "stacks", [])
        actor_idx = int(getattr(state, "actor_index", 0))
        my_stack = int(stacks[actor_idx]) if stacks else 0

        if name == "fold":
            if can_call and amount_to_call == 0:
                return "check_or_call", 0
            return "fold", None
        if name == "check_or_call":
            return "check_or_call", amount_to_call if can_call else 0

        if not can_raise:
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
        chosen = min(
            [v for v in raise_to_values if v >= target], default=(max(raise_to_values) if raise_to_values else None)
        )
        if chosen is None:
            return ("check_or_call", amount_to_call) if can_call else ("fold", None)
        return "complete_bet_or_raise_to", int(chosen)


def rollout_episode(
    forced_action_idx: int | None, policy: QwenPolicy, n_opponents: int = 3, initial_stack: int = 30_000
) -> float:
    hero = PolicyPlayer("Hero", policy, forced_action_idx=forced_action_idx)
    opponents = [RandomPlayer(name=f"R{i}") for i in range(n_opponents)]
    game = Game(players=[hero] + opponents, initial_stack=initial_stack)
    game.play_round(verbose=False)
    return float(game.stacks[0] - initial_stack)


def evaluate_policy(policy: QwenPolicy, episodes: int = 40, initial_stack: int = 30_000) -> float:
    total = 0.0
    for _ in range(episodes):
        total += rollout_episode(None, policy, initial_stack=initial_stack)
    bb = 100.0
    return total / (episodes * bb)


class ProbePromptPlayer(BasePlayer):
    """Capture a realistic prompt built from live features at the first hero decision."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt: str | None = None
        self._captured = False

    def play(self, valid_actions: dict[str, Any], state: Any) -> tuple[str, int | None]:  # type: ignore[override]
        if not self._captured:
            actor_idx = int(getattr(state, "actor_index", 0))
            feats = encode_state(state, actor_idx, valid_actions)
            prompt = (
                "state:"
                + ",".join(map(lambda x: f"{x:.3f}", feats.tolist()))
                + "\nchoose one of: "
                + ",".join(ACTION_SPACE)
            )
            self.prompt = prompt
            self._captured = True
        # Safe action: check/call if possible else fold
        if "check_or_call" in valid_actions:
            return "check_or_call", int(valid_actions.get("check_or_call", 0))
        return "fold", None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = QwenPolicy(QwenConfig(model_name=args.model))

    # Build trainer with GRPO objective using the Qwen policy
    def rollout_fn(_prompt: str, action_idx: int) -> float:
        chips = rollout_episode(action_idx, policy)
        return chips / 100.0

    trainer = QwenGRPOTrainer(policy, QwenGRPOConfig(group_size=args.group), rollout_fn)

    print("starting eval:", round(evaluate_policy(policy, episodes=args.eval_episodes), 4))

    # Training loop: sample prompts from fresh engine states, run GRPO step
    for step in range(1, args.steps + 1):
        prompts: list[str] = []
        for _ in range(args.batch):
            # Build a single prompt from a probe state
            hero = ProbePromptPlayer("ProbeHero")
            opponents = [RandomPlayer(name="RB0"), RandomPlayer(name="RB1"), RandomPlayer(name="RB2")]
            game = Game(players=[hero] + opponents, initial_stack=10_000)
            game.play_round(verbose=False)
            prompt = hero.prompt or ("state:probe\nchoose one of: " + ",".join(ACTION_SPACE))
            prompts.append(prompt)

        stats = trainer.step(prompts)
        if step % args.eval_every == 0:
            score = evaluate_policy(policy, episodes=args.eval_episodes)
            log = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}
            print(f"step {step:03d} score {score:.4f} stats {log}")


if __name__ == "__main__":
    main()
