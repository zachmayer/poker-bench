#!/usr/bin/env python3
"""Fine-tune Qwen model for poker using GRPO reinforcement learning."""

import random
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# PokerBots imports
from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer


@dataclass
class Config:
    """Training configuration."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Small but capable
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # GRPO settings
    learning_rate: float = 1e-4
    batch_size: int = 100  # Rollout 100 hands per update
    kl_beta: float = 0.05
    clip_ratio: float = 0.2
    temperature: float = 0.8

    # Training
    num_iterations: int = 10  # Reduced for faster testing
    eval_hands: int = 50  # Reduced for faster testing

    # Poker
    initial_stack: int = 10000
    num_opponents: int = 3


class QwenPokerPlayer(BasePlayer):
    """Poker player powered by Qwen model."""

    def __init__(self, name: str, model, tokenizer, temperature: float = 1.0):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.history = []  # Store (state, action, reward) for learning

    def play(self, valid_actions: dict, state: Any) -> Tuple[str, Any]:
        # Build concise state description
        prompt = self._build_prompt(state, valid_actions)

        # Get model's action
        action_str, log_prob = self._get_model_action(prompt)

        # Map to game action
        game_action, amount = self._map_to_game_action(action_str, valid_actions, state)

        # Store for learning
        self.history.append(
            {"prompt": prompt, "action": action_str, "log_prob": log_prob, "game_action": game_action, "amount": amount}
        )

        return game_action, amount

    def _build_prompt(self, state: Any, valid_actions: dict) -> str:
        """Build concise prompt for the model."""
        # Get key information
        stacks = getattr(state, "stacks", [])
        bets = getattr(state, "bets", [])
        pots = getattr(state, "pots", [])
        pot_total = sum(getattr(p, "unraked_amount", 0) for p in pots)
        actor_idx = getattr(state, "actor_index", 0)
        my_stack = stacks[actor_idx] if stacks else 0
        to_call = valid_actions.get("check_or_call", 0)

        # Street info
        street_map = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        street = street_map.get(getattr(state, "street_index", 0), "preflop")

        # Build prompt
        prompt = (
            f"Poker decision:\n"
            f"Street: {street}\n"
            f"Stack: {my_stack}\n"
            f"Pot: {pot_total}\n"
            f"To call: {to_call}\n"
            f"Position: {actor_idx}/{len(stacks)}\n"
            f"Action:"
        )
        return prompt

    def _get_model_action(self, prompt: str) -> Tuple[str, float]:
        """Get action from model with log probability."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Get probabilities for action tokens
            action_tokens = {
                "fold": self.tokenizer.encode("fold", add_special_tokens=False)[0],
                "call": self.tokenizer.encode("call", add_special_tokens=False)[0],
                "raise": self.tokenizer.encode("raise", add_special_tokens=False)[0],
            }

            # Extract action logits
            action_logits = torch.tensor([logits[token_id] for token_id in action_tokens.values()])
            action_probs = F.softmax(action_logits, dim=0)

            # Sample action
            action_idx = torch.multinomial(action_probs, 1).item()
            action_names = list(action_tokens.keys())
            action = action_names[action_idx]

            # Get log probability
            log_prob = torch.log(action_probs[action_idx]).item()

        return action, log_prob

    def _map_to_game_action(self, action: str, valid_actions: dict, state: Any) -> Tuple[str, Any]:
        """Map model action to game action."""
        to_call = valid_actions.get("check_or_call", 0)

        if action == "fold":
            # Don't fold when we can check
            if to_call == 0:
                return "check_or_call", 0
            return "fold", None

        elif action == "call":
            return "check_or_call", to_call

        elif action == "raise":
            if "complete_bet_or_raise_to" in valid_actions:
                raise_values = list(valid_actions["complete_bet_or_raise_to"])
                if raise_values:
                    # Choose 2x pot size or middle value
                    pots = getattr(state, "pots", [])
                    pot_total = sum(getattr(p, "unraked_amount", 0) for p in pots)
                    target = to_call + pot_total

                    # Find closest legal raise
                    best_raise = min(raise_values, key=lambda x: abs(x - target))
                    return "complete_bet_or_raise_to", best_raise

            # Can't raise, just call
            return "check_or_call", to_call

        # Default to check/call
        return "check_or_call", to_call


class QwenGRPOTrainer:
    """GRPO trainer for Qwen poker model."""

    def __init__(self, config: Config):
        self.config = config

        # Load model and tokenizer
        print(f"Loading {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Use CPU to avoid MPS issues
            trust_remote_code=True,
        )

        # Add LoRA adapters for efficient fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.train()

        # Reference model for KL divergence
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Use CPU to avoid MPS issues
            trust_remote_code=True,
        )
        self.ref_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        print(f"Model ready with {self.model.num_parameters(only_trainable=True):,} trainable parameters")

    def rollout_hands(self, num_hands: int) -> List[Dict]:
        """Play multiple hands and collect trajectories."""
        all_trajectories = []

        for hand_idx in range(num_hands):
            # Create player with current model
            hero = QwenPokerPlayer("Hero", self.model, self.tokenizer, self.config.temperature)

            # Mix opponent types for diversity
            opponent_types = [RandomPlayer, CallingPlayer, GamblingPlayer]
            opponents = []
            for i in range(self.config.num_opponents):
                OpponentClass = random.choice(opponent_types)
                opponents.append(OpponentClass(f"Opp{i}"))

            # Play hand
            players = [hero] + opponents
            game = Game(players=players, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            # Calculate reward (normalized by big blind)
            initial = self.config.initial_stack
            final = game.stacks[0]
            reward = (final - initial) / 100.0  # BB normalization

            # Add reward to trajectory
            if hero.history:
                for step in hero.history:
                    step["reward"] = reward  # Simple: same reward for all actions in hand
                all_trajectories.append(hero.history)

        return all_trajectories

    def compute_advantages(self, trajectories: List[List[Dict]]) -> np.ndarray:
        """Compute advantages using group normalization."""
        # Collect all rewards
        all_rewards = []
        for traj in trajectories:
            if traj:
                all_rewards.append(traj[0]["reward"])  # Use hand outcome

        if not all_rewards:
            return np.array([])

        # Normalize advantages
        rewards = np.array(all_rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std() if rewards.std() > 1e-8 else 1.0
        advantages = (rewards - mean_reward) / std_reward

        return advantages

    def update_model(self, trajectories: List[List[Dict]], advantages: np.ndarray):
        """Update model using GRPO."""
        if len(trajectories) == 0 or len(advantages) == 0:
            return {}

        total_loss = 0
        total_kl = 0
        num_updates = 0

        self.model.train()

        for traj_idx, trajectory in enumerate(trajectories):
            if not trajectory:
                continue

            advantage = advantages[min(traj_idx, len(advantages) - 1)]

            for step in trajectory:
                prompt = step["prompt"]
                action = step["action"]
                old_log_prob = step["log_prob"]

                # Prepare input
                full_text = prompt + " " + action
                inputs = self.tokenizer(full_text, return_tensors="pt", padding=False, truncation=True, max_length=256)

                # Get current model log prob
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]

                # Get action token
                action_token = self.tokenizer.encode(action, add_special_tokens=False)[0]
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = log_probs[action_token]

                # Get reference model log prob for KL
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs)
                    ref_logits = ref_outputs.logits[0, -1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_log_prob = ref_log_probs[action_token]

                # PPO-style clipped objective
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                # KL divergence penalty
                kl_div = new_log_prob - ref_log_prob

                # Total loss
                loss = policy_loss + self.config.kl_beta * kl_div

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl_div.item()
                num_updates += 1

        stats = {}
        if num_updates > 0:
            stats["loss"] = total_loss / num_updates
            stats["kl"] = total_kl / num_updates
            stats["num_updates"] = num_updates

        return stats

    def evaluate(self, num_hands: int = 100) -> float:
        """Evaluate current model performance."""
        total_reward = 0

        self.model.eval()

        for _ in range(num_hands):
            hero = QwenPokerPlayer("Hero", self.model, self.tokenizer, temperature=1.0)

            # Test against random players
            opponents = [RandomPlayer(f"R{i}") for i in range(self.config.num_opponents)]
            players = [hero] + opponents
            game = Game(players=players, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            # Calculate reward
            initial = self.config.initial_stack
            final = game.stacks[0]
            reward = (final - initial) / 100.0
            total_reward += reward

        self.model.train()

        return total_reward / num_hands

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 50)
        print("Starting Qwen Poker Training with GRPO")
        print("=" * 50)

        # Initial evaluation
        print("\nInitial evaluation...")
        initial_score = self.evaluate(50)
        print(f"Initial performance: {initial_score:.3f} BB/hand")

        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\nIteration {iteration}/{self.config.num_iterations}")

            # Rollout hands
            print(f"  Rolling out {self.config.batch_size} hands...")
            trajectories = self.rollout_hands(self.config.batch_size)

            # Compute advantages
            advantages = self.compute_advantages(trajectories)

            # Update model
            print("  Updating model...")
            stats = self.update_model(trajectories, advantages)

            # Log stats
            if stats:
                print(f"  Loss: {stats.get('loss', 0):.4f}, KL: {stats.get('kl', 0):.4f}")

            # Evaluate periodically
            if iteration % 5 == 0:
                score = self.evaluate(self.config.eval_hands)
                print(f"  Performance: {score:.3f} BB/hand")

        # Final evaluation
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        final_score = self.evaluate(200)
        print(f"Final performance: {final_score:.3f} BB/hand")
        print(f"Improvement: {final_score - initial_score:.3f} BB/hand")

        # Save model
        self.model.save_pretrained("qwen_poker_model")
        self.tokenizer.save_pretrained("qwen_poker_model")
        print("\nModel saved to qwen_poker_model/")

        return final_score


def main():
    """Main entry point."""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create config
    config = Config()

    # Create trainer and train
    trainer = QwenGRPOTrainer(config)
    final_score = trainer.train()

    print(f"\nTraining finished with {final_score:.3f} BB/hand average performance")


if __name__ == "__main__":
    main()
