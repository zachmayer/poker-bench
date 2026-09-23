#!/usr/bin/env python3
"""Optimized mini LLM poker training - faster execution for 2000 iterations."""

import random
import time
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# PokerBots imports
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer


@dataclass
class Config:
    """Optimized configuration."""

    # Model - smaller for speed
    vocab_size: int = 50
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    max_seq_len: int = 16

    # GRPO settings
    learning_rate: float = 2e-3
    batch_size: int = 50  # Smaller batches for speed
    kl_beta: float = 0.01
    clip_ratio: float = 0.3

    # Training
    num_iterations: int = 2000
    eval_hands: int = 20  # Fewer eval hands for speed

    # Poker
    initial_stack: int = 5000  # Smaller stacks for faster games
    num_opponents: int = 2  # Fewer opponents


class FastTransformer(nn.Module):
    """Ultra-fast tiny transformer."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Simple embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Single transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=0.0,  # No dropout for speed
            batch_first=True,
        )

        # Output
        self.out = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.out(x)

    @torch.no_grad()
    def get_action(self, x, temp=1.0):
        logits = self(x)[:, -1, :]
        if temp != 1.0:
            logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[0, action]).item()


class SimpleTokenizer:
    """Minimal tokenizer."""

    def __init__(self):
        self.actions = ["fold", "call", "raise"]
        self.action_to_id = {a: i + 10 for i, a in enumerate(self.actions)}

    def encode_state(self, state, valid_actions):
        # Super simple: just encode pot size and position
        tokens = [1]  # Start token

        # Pot category (small/medium/large)
        pots = getattr(state, "pots", [])
        pot = sum(getattr(p, "unraked_amount", 0) for p in pots)
        if pot < 500:
            tokens.append(2)
        elif pot < 2000:
            tokens.append(3)
        else:
            tokens.append(4)

        # Position
        actor_idx = getattr(state, "actor_index", 0)
        tokens.append(5 + min(actor_idx, 2))

        return tokens

    def decode_action(self, action_id):
        if 10 <= action_id <= 12:
            return self.actions[action_id - 10]
        return "call"


class FastPlayer(BasePlayer):
    """Fast poker player."""

    def __init__(self, name, model, tokenizer):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def play(self, valid_actions, state):
        tokens = self.tokenizer.encode_state(state, valid_actions)
        x = torch.tensor([tokens], dtype=torch.long)

        action_id, log_prob = self.model.get_action(x)
        action = self.tokenizer.decode_action(action_id)

        self.history.append({"tokens": tokens, "action_id": action_id, "log_prob": log_prob})

        # Quick mapping
        to_call = valid_actions.get("check_or_call", 0)

        if action == "fold" and to_call > 0:
            return "fold", None
        elif action == "raise" and "complete_bet_or_raise_to" in valid_actions:
            raises = list(valid_actions["complete_bet_or_raise_to"])
            if raises:
                return "complete_bet_or_raise_to", raises[len(raises) // 2]  # Middle value

        return "check_or_call", to_call


class FastGRPOTrainer:
    """Optimized GRPO trainer."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = SimpleTokenizer()

        self.model = FastTransformer(config)
        self.ref_model = FastTransformer(config)
        self.ref_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Fast model: {params:,} parameters")

    def rollout_batch(self):
        """Fast batch rollout."""
        trajectories = []
        rewards = []

        for _ in range(self.config.batch_size):
            hero = FastPlayer("H", self.model, self.tokenizer)
            opps = [RandomPlayer(f"R{i}") for i in range(self.config.num_opponents)]

            game = Game(players=[hero] + opps, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            reward = (game.stacks[0] - self.config.initial_stack) / 100.0

            if hero.history:
                trajectories.append(hero.history)
                rewards.append(reward)

        return trajectories, rewards

    def update_batch(self, trajectories, rewards):
        """Fast batch update."""
        if not trajectories:
            return {}

        # Normalize advantages
        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = 0
        updates = 0

        # Process in mini-batches for speed
        for i, traj in enumerate(trajectories):
            adv = advantages[i]

            for step in traj[:2]:  # Only update on first 2 actions for speed
                tokens = torch.tensor([step["tokens"]], dtype=torch.long)
                action_id = step["action_id"]
                old_log_prob = step["log_prob"]

                # Forward
                logits = self.model(tokens)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = log_probs[0, action_id]

                # PPO loss
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                loss = -torch.min(ratio * adv, clipped * adv)

                # Quick update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                updates += 1

        return {"loss": total_loss / max(updates, 1)}

    def evaluate(self):
        """Quick evaluation."""
        total = 0
        for _ in range(self.config.eval_hands):
            hero = FastPlayer("H", self.model, self.tokenizer)
            opps = [RandomPlayer(f"R{i}") for i in range(2)]

            game = Game(players=[hero] + opps, initial_stack=5000)
            game.play_round(verbose=False)

            total += (game.stacks[0] - 5000) / 100.0

        return total / self.config.eval_hands

    def train(self):
        """Fast training loop."""
        print("\n" + "=" * 60)
        print(f"FAST Training: {self.config.num_iterations} iterations")
        print("=" * 60)

        # Tracking
        iterations = []
        rewards = []
        eval_iters = []
        eval_scores = []

        # Initial eval
        score = self.evaluate()
        print(f"Initial: {score:.2f} BB/hand")
        eval_iters.append(0)
        eval_scores.append(score)

        start_time = time.time()

        for iter in range(1, self.config.num_iterations + 1):
            # Rollout and update
            trajs, rews = self.rollout_batch()
            stats = self.update_batch(trajs, rews)

            avg_reward = np.mean(rews) if rews else 0
            iterations.append(iter)
            rewards.append(avg_reward)

            # Progress
            if iter % 50 == 0:
                elapsed = time.time() - start_time
                iters_per_sec = iter / elapsed
                eta = (self.config.num_iterations - iter) / iters_per_sec
                print(
                    f"Iter {iter:4d}: {avg_reward:+.2f} BB, Loss={stats['loss']:.3f}, Speed={iters_per_sec:.1f} it/s, ETA={eta / 60:.1f}m"
                )

            # Eval
            if iter % 200 == 0:
                score = self.evaluate()
                print(f"  → Eval: {score:.2f} BB/hand")
                eval_iters.append(iter)
                eval_scores.append(score)

        # Final
        final = self.evaluate()
        print(f"\nFinal: {final:.2f} BB/hand")
        eval_iters.append(self.config.num_iterations)
        eval_scores.append(final)

        # Plot
        self.plot(iterations, rewards, eval_iters, eval_scores)

        # Save
        torch.save(
            {"model": self.model.state_dict(), "history": {"rewards": rewards, "evals": eval_scores}},
            "fast_poker_2000.pt",
        )

        return final

    def plot(self, iters, rewards, eval_iters, eval_scores):
        """Quick plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Training rewards
        ax1.plot(iters, rewards, alpha=0.3, color="blue")

        # Moving average
        if len(rewards) > 50:
            window = 50
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(iters[window - 1 :], ma, "b-", linewidth=2)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("BB/hand")
        ax1.set_title("Training Rewards (2000 iterations)")
        ax1.grid(True, alpha=0.3)

        # Evaluation
        ax2.plot(eval_iters, eval_scores, "ro-", linewidth=2, markersize=8)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("BB/hand")
        ax2.set_title("Evaluation Performance")
        ax2.grid(True, alpha=0.3)

        # Improvement
        improvement = eval_scores[-1] - eval_scores[0]
        ax2.text(
            0.05,
            0.95,
            f"Total: {improvement:+.1f} BB/hand",
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        plt.tight_layout()
        plt.savefig("poker_2000_iterations.png", dpi=150)
        print("\nPlot saved: poker_2000_iterations.png")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config = Config()
    trainer = FastGRPOTrainer(config)
    score = trainer.train()

    print(f"\n✓ Complete: {score:.2f} BB/hand after 2000 iterations")


if __name__ == "__main__":
    main()
