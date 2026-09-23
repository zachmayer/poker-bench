#!/usr/bin/env python3
"""Ultra-lightweight LLM poker training with GRPO - uses a tiny custom transformer."""

import random
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# PokerBots imports
from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer


@dataclass
class Config:
    """Configuration for mini LLM poker training."""

    # Model
    vocab_size: int = 100  # Small vocabulary for poker actions
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 32

    # GRPO settings
    learning_rate: float = 1e-3
    batch_size: int = 100  # 100 hands per update as requested
    kl_beta: float = 0.05
    clip_ratio: float = 0.2

    # Training
    num_iterations: int = 2000  # Extended training
    eval_hands: int = 50

    # Poker
    initial_stack: int = 10000
    num_opponents: int = 3


class MiniTransformer(nn.Module):
    """Tiny transformer for poker decisions."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output head
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln(x)
        logits = self.head(x)

        return logits

    def generate_action(self, context: torch.Tensor, temperature: float = 1.0) -> Tuple[int, float]:
        """Generate next action token given context."""
        logits = self(context)[:, -1, :]  # Get last position

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action]).item()

        return action, log_prob


class PokerTokenizer:
    """Simple tokenizer for poker states and actions."""

    def __init__(self):
        # Basic poker vocabulary
        self.vocab = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "street": 3,
            "pre": 4,
            "flop": 5,
            "turn": 6,
            "river": 7,
            "stack": 8,
            "small": 9,
            "medium": 10,
            "large": 11,
            "pot": 12,
            "empty": 13,
            "low": 14,
            "high": 15,
            "position": 16,
            "early": 17,
            "middle": 18,
            "late": 19,
            "action": 20,
            "fold": 21,
            "call": 22,
            "raise": 23,
            "check": 24,
            "bet": 25,
            "allin": 26,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode_state(self, state: Any, valid_actions: dict) -> List[int]:
        """Encode poker state as token sequence."""
        tokens = [self.vocab["<start>"]]

        # Street info
        street_idx = getattr(state, "street_index", 0)
        street_names = ["pre", "flop", "turn", "river"]
        tokens.extend([self.vocab["street"], self.vocab[street_names[min(street_idx, 3)]]])

        # Stack size category
        stacks = getattr(state, "stacks", [])
        actor_idx = getattr(state, "actor_index", 0)
        my_stack = stacks[actor_idx] if stacks else 0

        tokens.append(self.vocab["stack"])
        if my_stack < 2000:
            tokens.append(self.vocab["small"])
        elif my_stack < 8000:
            tokens.append(self.vocab["medium"])
        else:
            tokens.append(self.vocab["large"])

        # Pot size
        pots = getattr(state, "pots", [])
        pot_total = sum(getattr(p, "unraked_amount", 0) for p in pots)

        tokens.append(self.vocab["pot"])
        if pot_total == 0:
            tokens.append(self.vocab["empty"])
        elif pot_total < 1000:
            tokens.append(self.vocab["low"])
        else:
            tokens.append(self.vocab["high"])

        # Position
        num_players = len(stacks)
        tokens.append(self.vocab["position"])
        if actor_idx < num_players // 3:
            tokens.append(self.vocab["early"])
        elif actor_idx < 2 * num_players // 3:
            tokens.append(self.vocab["middle"])
        else:
            tokens.append(self.vocab["late"])

        tokens.append(self.vocab["action"])  # Prompt for action

        return tokens

    def decode_action(self, token_id: int) -> str:
        """Decode action token to string."""
        token = self.inv_vocab.get(token_id, "call")
        if token in ["fold", "call", "raise", "check", "bet", "allin"]:
            return token
        return "call"  # Default


class MiniLLMPlayer(BasePlayer):
    """Player using mini transformer model."""

    def __init__(self, name: str, model: MiniTransformer, tokenizer: PokerTokenizer, temperature: float = 1.0):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.history = []

    def play(self, valid_actions: dict, state: Any) -> Tuple[str, Any]:
        # Encode state
        tokens = self.tokenizer.encode_state(state, valid_actions)
        context = torch.tensor([tokens], dtype=torch.long)

        # Generate action
        action_id, log_prob = self.model.generate_action(context, self.temperature)
        action_str = self.tokenizer.decode_action(action_id)

        # Store for learning
        self.history.append({"tokens": tokens, "action_id": action_id, "log_prob": log_prob, "action_str": action_str})

        # Map to game action
        return self._map_to_game_action(action_str, valid_actions, state)

    def _map_to_game_action(self, action: str, valid_actions: dict, state: Any) -> Tuple[str, Any]:
        """Map model action to game action."""
        to_call = valid_actions.get("check_or_call", 0)

        if action == "fold":
            if to_call == 0:  # Don't fold when check is free
                return "check_or_call", 0
            return "fold", None

        elif action in ["call", "check"]:
            return "check_or_call", to_call

        elif action in ["raise", "bet", "allin"]:
            if "complete_bet_or_raise_to" in valid_actions:
                raises = list(valid_actions["complete_bet_or_raise_to"])
                if raises:
                    if action == "allin":
                        return "complete_bet_or_raise_to", max(raises)
                    else:
                        # Pot-sized raise
                        pots = getattr(state, "pots", [])
                        pot_total = sum(getattr(p, "unraked_amount", 0) for p in pots)
                        target = to_call + pot_total
                        best_raise = min(raises, key=lambda x: abs(x - target))
                        return "complete_bet_or_raise_to", best_raise

        return "check_or_call", to_call


class MiniGRPOTrainer:
    """GRPO trainer for mini transformer."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = PokerTokenizer()

        # Create model and reference model
        self.model = MiniTransformer(config)
        self.ref_model = MiniTransformer(config)
        self.ref_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        print(f"Mini transformer ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def rollout_batch(self) -> Tuple[List[Dict], List[float]]:
        """Rollout batch of hands."""
        all_trajectories = []
        all_rewards = []

        for _ in range(self.config.batch_size):
            # Play one hand
            hero = MiniLLMPlayer("Hero", self.model, self.tokenizer, temperature=1.2)

            # Mix opponents
            opponent_types = [RandomPlayer, CallingPlayer, GamblingPlayer]
            opponents = []
            for i in range(self.config.num_opponents):
                OpponentClass = random.choice(opponent_types)
                opponents.append(OpponentClass(f"Opp{i}"))

            game = Game(players=[hero] + opponents, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            # Calculate reward
            reward = (game.stacks[0] - self.config.initial_stack) / 100.0

            if hero.history:
                all_trajectories.append(hero.history)
                all_rewards.append(reward)

        return all_trajectories, all_rewards

    def update_model(self, trajectories: List[List[Dict]], rewards: List[float]):
        """Update model using GRPO."""
        if not trajectories:
            return {}

        # Normalize rewards to advantages
        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = 0
        total_kl = 0
        num_updates = 0

        for traj_idx, trajectory in enumerate(trajectories):
            advantage = advantages[traj_idx]

            for step in trajectory:
                tokens = torch.tensor([step["tokens"]], dtype=torch.long)
                action_id = step["action_id"]
                old_log_prob = step["log_prob"]

                # Get current model log prob
                logits = self.model(tokens)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = log_probs[0, action_id]

                # Get reference model log prob
                with torch.no_grad():
                    ref_logits = self.ref_model(tokens)[:, -1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_log_prob = ref_log_probs[0, action_id]

                # PPO loss with clipping
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage)

                # KL penalty
                kl = new_log_prob - ref_log_prob

                # Total loss
                loss = policy_loss + self.config.kl_beta * kl

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl.item()
                num_updates += 1

        # Update reference model slowly
        with torch.no_grad():
            for p_ref, p in zip(self.ref_model.parameters(), self.model.parameters(), strict=False):
                p_ref.data.mul_(0.99).add_(p.data, alpha=0.01)

        return {"loss": total_loss / max(num_updates, 1), "kl": total_kl / max(num_updates, 1), "updates": num_updates}

    def evaluate(self) -> float:
        """Evaluate model performance."""
        total_reward = 0

        for _ in range(self.config.eval_hands):
            hero = MiniLLMPlayer("Hero", self.model, self.tokenizer, temperature=1.0)
            opponents = [RandomPlayer(f"R{i}") for i in range(self.config.num_opponents)]

            game = Game(players=[hero] + opponents, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            reward = (game.stacks[0] - self.config.initial_stack) / 100.0
            total_reward += reward

        return total_reward / self.config.eval_hands

    def train(self):
        """Main training loop with GRPO."""
        print("\n" + "=" * 60)
        print("Mini LLM Poker Training with GRPO")
        print(f"Rollout: {self.config.batch_size} hands per update")
        print("=" * 60)

        # Tracking for plotting
        iterations_list = []
        rewards_list = []
        eval_iterations = []
        eval_scores = []

        # Initial evaluation
        initial_score = self.evaluate()
        print(f"\nInitial: {initial_score:.3f} BB/hand")
        eval_iterations.append(0)
        eval_scores.append(initial_score)

        for iteration in range(1, self.config.num_iterations + 1):
            # Rollout batch
            trajectories, rewards = self.rollout_batch()

            # Update model
            stats = self.update_model(trajectories, rewards)

            # Log progress
            avg_reward = np.mean(rewards) if rewards else 0
            iterations_list.append(iteration)
            rewards_list.append(avg_reward)

            # Less frequent logging for long runs
            if iteration % 10 == 0 or iteration <= 10:
                print(
                    f"Iter {iteration:4d}: Reward={avg_reward:+.3f} BB, Loss={stats.get('loss', 0):.4f}, KL={stats.get('kl', 0):.4f}"
                )

            # Evaluate periodically
            if iteration % 100 == 0:
                score = self.evaluate()
                print(f"  → Eval: {score:.3f} BB/hand")
                eval_iterations.append(iteration)
                eval_scores.append(score)

        # Final evaluation
        final_score = self.evaluate()
        print(f"\nFinal: {final_score:.3f} BB/hand")
        print(f"Improvement: {final_score - initial_score:+.3f} BB/hand")
        eval_iterations.append(self.config.num_iterations)
        eval_scores.append(final_score)

        # Create plot
        self.plot_training_progress(iterations_list, rewards_list, eval_iterations, eval_scores)

        # Save model
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "training_history": {
                    "iterations": iterations_list,
                    "rewards": rewards_list,
                    "eval_iterations": eval_iterations,
                    "eval_scores": eval_scores,
                },
            },
            "mini_llm_poker.pt",
        )
        print("\nModel saved to mini_llm_poker.pt")

        return final_score

    def plot_training_progress(self, iterations, rewards, eval_iterations, eval_scores):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot training rewards (smoothed)
        ax1.plot(iterations, rewards, alpha=0.3, color="blue", label="Raw rewards")

        # Add moving average
        if len(rewards) > 20:
            window_size = 20
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            ax1.plot(
                iterations[window_size - 1 :],
                moving_avg,
                color="blue",
                linewidth=2,
                label=f"Moving avg ({window_size})",
            )

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("BB/hand (training)")
        ax1.set_title("Training Rewards per Iteration")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot evaluation scores
        ax2.plot(eval_iterations, eval_scores, "ro-", linewidth=2, markersize=8)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("BB/hand (evaluation)")
        ax2.set_title("Evaluation Performance")
        ax2.grid(True, alpha=0.3)

        # Add improvement annotation
        if len(eval_scores) > 1:
            improvement = eval_scores[-1] - eval_scores[0]
            ax2.text(
                0.05,
                0.95,
                f"Improvement: {improvement:+.1f} BB/hand",
                transform=ax2.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        plt.savefig("poker_training_progress.png", dpi=150)
        print("\nPlot saved to poker_training_progress.png")
        plt.show()


def main():
    """Main entry point."""
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create config and trainer
    config = Config()
    trainer = MiniGRPOTrainer(config)

    # Train
    final_score = trainer.train()

    print(f"\n✓ Training complete! Final performance: {final_score:.3f} BB/hand")


if __name__ == "__main__":
    main()
