#!/usr/bin/env python3
"""Fast, lightweight Qwen poker training with GRPO - uses smallest model and optimizations."""

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
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer


@dataclass
class Config:
    """Optimized configuration for fast training."""

    model_name: str = "Qwen/Qwen2.5-0.5B"  # Smallest Qwen model

    # Minimal LoRA for speed
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05

    # GRPO settings
    learning_rate: float = 5e-4
    batch_size: int = 20  # Smaller batches for speed
    kl_beta: float = 0.01

    # Minimal training
    num_iterations: int = 5
    eval_hands: int = 10

    # Poker
    initial_stack: int = 5000
    num_opponents: int = 2  # Fewer opponents for speed


class SimplePokerPlayer(BasePlayer):
    """Simplified poker player for fast inference."""

    def __init__(self, name: str, model, tokenizer):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def play(self, valid_actions: dict, state: Any) -> Tuple[str, Any]:
        # Very simple state encoding
        pot = sum(getattr(p, "unraked_amount", 0) for p in getattr(state, "pots", []))
        to_call = valid_actions.get("check_or_call", 0)

        # Simple decision: fold, call, or raise based on pot odds
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if pot > 0 else 1.0

            # Use model to get action probabilities
            prompt = f"Pot: {pot}, Call: {to_call}, Odds: {pot_odds:.2f}. Action:"
            action = self._get_action(prompt, pot_odds)

            self.history.append({"prompt": prompt, "action": action})

            if action == "fold" and pot_odds > 0.5:
                return "fold", None
            elif action == "raise" and "complete_bet_or_raise_to" in valid_actions:
                raises = list(valid_actions["complete_bet_or_raise_to"])
                if raises:
                    return "complete_bet_or_raise_to", min(raises)

        return "check_or_call", to_call

    def _get_action(self, prompt: str, pot_odds: float) -> str:
        """Simple action selection."""
        # For speed, use simple heuristics with small model influence
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Get top action tokens
            fold_id = self.tokenizer.encode("fold", add_special_tokens=False)[0]
            call_id = self.tokenizer.encode("call", add_special_tokens=False)[0]
            raise_id = self.tokenizer.encode("raise", add_special_tokens=False)[0]

            probs = F.softmax(logits[[fold_id, call_id, raise_id]], dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            return ["fold", "call", "raise"][action_idx]


class FastGRPOTrainer:
    """Lightweight GRPO trainer."""

    def __init__(self, config: Config):
        self.config = config

        print(f"Loading {config.model_name} (fast mode)...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with minimal config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,  # Use half precision for speed
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Add minimal LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],  # Only essential modules
        )

        self.model = get_peft_model(self.model, peft_config)
        print(f"Model ready with {self.model.num_parameters(only_trainable=True):,} trainable parameters")

        # Simple optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def rollout_batch(self) -> Tuple[List[float], List[Dict]]:
        """Fast batch rollout."""
        rewards = []
        experiences = []

        for _ in range(self.config.batch_size):
            # Quick game
            hero = SimplePokerPlayer("Hero", self.model, self.tokenizer)
            opponents = [RandomPlayer(f"R{i}") for i in range(self.config.num_opponents)]

            game = Game(players=[hero] + opponents, initial_stack=self.config.initial_stack)
            game.play_round(verbose=False)

            # Simple reward
            reward = (game.stacks[0] - self.config.initial_stack) / 100.0
            rewards.append(reward)

            if hero.history:
                experiences.extend(hero.history)

        return rewards, experiences

    def update_step(self, experiences: List[Dict], advantage: float):
        """Simple GRPO update."""
        if not experiences:
            return

        total_loss = 0

        for exp in experiences[:10]:  # Limit updates per batch for speed
            prompt = exp["prompt"]
            action = exp["action"]

            # Prepare input
            text = prompt + " " + action
            inputs = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True)

            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Scale by advantage
            scaled_loss = loss * -advantage  # Negative for gradient ascent when advantage is positive

            # Backward
            self.optimizer.zero_grad()
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / min(len(experiences), 10)

    def train(self):
        """Fast training loop."""
        print("\n" + "=" * 50)
        print("Fast Qwen Poker Training")
        print("=" * 50)

        baseline_reward = 0

        for iteration in range(1, self.config.num_iterations + 1):
            # Rollout
            rewards, experiences = self.rollout_batch()
            avg_reward = np.mean(rewards)

            # Compute advantage
            advantage = avg_reward - baseline_reward
            baseline_reward = 0.9 * baseline_reward + 0.1 * avg_reward  # Moving average

            # Update
            if experiences and abs(advantage) > 0.01:
                loss = self.update_step(experiences, advantage)
                print(f"Iter {iteration}: Reward={avg_reward:.3f} BB, Advantage={advantage:.3f}, Loss={loss:.4f}")
            else:
                print(f"Iter {iteration}: Reward={avg_reward:.3f} BB, Advantage={advantage:.3f}")

        # Quick evaluation
        print("\nFinal evaluation...")
        eval_rewards = []
        for _ in range(self.config.eval_hands):
            hero = SimplePokerPlayer("Hero", self.model, self.tokenizer)
            opponents = [RandomPlayer(f"R{i}") for i in range(2)]
            game = Game(players=[hero] + opponents, initial_stack=5000)
            game.play_round(verbose=False)
            eval_rewards.append((game.stacks[0] - 5000) / 100.0)

        final_score = np.mean(eval_rewards)
        print(f"Final performance: {final_score:.3f} BB/hand")

        # Save
        self.model.save_pretrained("qwen_poker_fast")
        self.tokenizer.save_pretrained("qwen_poker_fast")
        print("Model saved to qwen_poker_fast/")

        return final_score


def main():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config = Config()
    trainer = FastGRPOTrainer(config)
    score = trainer.train()

    print(f"\n✓ Training complete: {score:.3f} BB/hand")


if __name__ == "__main__":
    main()
