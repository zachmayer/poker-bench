#!/usr/bin/env python3
"""Simplified and optimized poker training script with enhanced features and rewards."""

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer

# PokerBots imports
from PokerBots import Game
from PokerBots import RandomPlayer
from PokerBots.players.BasePlayer import BasePlayer


@dataclass
class Config:
    """Unified configuration for training."""

    # Model
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Sampling
    num_episodes: int = 1000
    eval_every: int = 50
    eval_episodes: int = 100
    buffer_size: int = 10000

    # Poker specific
    initial_stack: int = 30000
    num_opponents: int = 3
    exploration_temp: float = 1.2

    # Seed
    seed: int = 42


class PokerFeatures:
    """Enhanced feature extraction for poker states."""

    @staticmethod
    def extract(state: Any, actor_idx: int, valid_actions: dict) -> np.ndarray:
        """Extract comprehensive poker features."""
        features = []

        # Basic info
        stacks = np.array(getattr(state, "stacks", []), dtype=float)
        bets = np.array(getattr(state, "bets", []), dtype=float)
        pots = getattr(state, "pots", [])
        pot_total = float(sum(getattr(p, "unraked_amount", 0) for p in pots))

        # Normalize stacks and bets
        initial_stack = 30000.0  # Assumed initial stack
        my_stack = stacks[actor_idx] if len(stacks) > actor_idx else 0.0
        my_bet = bets[actor_idx] if len(bets) > actor_idx else 0.0

        # Stack sizes (normalized) - 2 features
        features.append(my_stack / initial_stack)
        features.append(my_bet / max(1.0, pot_total))

        # Pot odds - 1 feature
        to_call = float(valid_actions.get("check_or_call", 0))
        pot_odds = to_call / (pot_total + to_call) if pot_total > 0 else 0.0
        features.append(pot_odds)

        # Position (early, middle, late) - 1 feature
        num_players = len(stacks)
        position_ratio = actor_idx / max(1, num_players - 1) if num_players > 1 else 0.5
        features.append(position_ratio)

        # Street information - 4 features
        street_index = getattr(state, "street_index", 0)
        features.extend(
            [
                float(street_index == 0),  # Preflop
                float(street_index == 1),  # Flop
                float(street_index == 2),  # Turn
                float(street_index == 3),  # River
            ]
        )

        # Number of active players (approximation) - 1 feature
        active_players = sum(1 for s in stacks if s > 0)
        features.append(active_players / max(1, num_players))

        # Pot size relative to stack - 1 feature
        pot_to_stack = pot_total / max(1.0, my_stack) if my_stack > 0 else 0.0
        features.append(min(5.0, pot_to_stack))  # Cap at 5x

        # Aggression level (bets relative to pot) - 1 feature
        max_bet = max(bets) if len(bets) > 0 else 0.0
        aggression = max_bet / max(1.0, pot_total) if pot_total > 0 else 0.0
        features.append(min(3.0, aggression))

        # Action possibilities - 2 features
        can_check = to_call == 0
        can_raise = "complete_bet_or_raise_to" in valid_actions
        features.extend([float(can_check), float(can_raise)])

        # Stack depth categories - 4 features
        bb = 100.0  # Big blind
        depth = my_stack / bb
        features.extend(
            [
                float(depth < 20),  # Short stack
                float(20 <= depth < 50),  # Medium stack
                float(50 <= depth < 100),  # Deep stack
                float(depth >= 100),  # Very deep
            ]
        )

        # Pot size normalized - 1 feature (to make 18 total)
        features.append(pot_total / initial_stack)

        return np.array(features, dtype=np.float32)


class ActorCriticNetwork(nn.Module):
    """Simplified but effective actor-critic network."""

    def __init__(self, input_dim: int, num_actions: int, config: Config):
        super().__init__()

        # Shared layers
        layers = []
        prev_dim = input_dim
        for _ in range(config.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(prev_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = config.hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, num_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.ReLU(), nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def get_action(self, x: torch.Tensor, temperature: float = 1.0):
        """Sample action with temperature-scaled exploration."""
        logits, value = self.forward(x)

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class OptimizedPolicyPlayer(BasePlayer):
    """Policy player with enhanced decision making."""

    def __init__(self, name: str, model: ActorCriticNetwork, temperature: float = 1.0):
        super().__init__(name)
        self.model = model
        self.temperature = temperature
        self.trajectory = []

    def play(self, valid_actions: dict, state: Any) -> tuple[str, Optional[int]]:
        # Extract features
        actor_idx = getattr(state, "actor_index", 0)
        features = PokerFeatures.extract(state, actor_idx, valid_actions)

        # Get action from model
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            action_idx, log_prob, entropy, value = self.model.get_action(x, self.temperature)
            action_idx = action_idx.item()

        # Store trajectory for learning
        self.trajectory.append(
            {"features": features, "action": action_idx, "log_prob": log_prob.item(), "value": value.item()}
        )

        # Map to poker action
        return self._map_to_poker_action(action_idx, valid_actions, state)

    def _map_to_poker_action(self, action_idx: int, valid_actions: dict, state: Any):
        """Map model action to poker engine action."""
        actions = ["fold", "check_call", "raise_small", "raise_medium", "raise_large"]
        action_name = actions[min(action_idx, len(actions) - 1)]

        # Get basic info
        to_call = int(valid_actions.get("check_or_call", 0))
        pots = getattr(state, "pots", [])
        pot_total = sum(getattr(p, "unraked_amount", 0) for p in pots)
        stacks = getattr(state, "stacks", [])
        actor_idx = getattr(state, "actor_index", 0)
        my_stack = stacks[actor_idx] if stacks else 0

        # Handle fold
        if action_name == "fold":
            if to_call == 0:  # Don't fold when we can check
                return "check_or_call", 0
            return "fold", None

        # Handle check/call
        if action_name == "check_call":
            return "check_or_call", to_call

        # Handle raises
        if "complete_bet_or_raise_to" not in valid_actions:
            return "check_or_call", to_call

        raise_values = list(valid_actions["complete_bet_or_raise_to"])
        if not raise_values:
            return "check_or_call", to_call

        # Calculate raise sizes
        if action_name == "raise_small":
            target = to_call + pot_total * 0.33
        elif action_name == "raise_medium":
            target = to_call + pot_total * 0.75
        else:  # raise_large
            target = to_call + pot_total * 1.5

        # Find closest legal raise
        best_raise = min(raise_values, key=lambda x: abs(x - target))
        return "complete_bet_or_raise_to", best_raise


class ExperienceBuffer:
    """Simple experience buffer for storing trajectories."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory: list):
        self.buffer.append(trajectory)

    def sample(self, batch_size: int):
        """Sample batch of trajectories."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()


class PokerTrainer:
    """Main trainer with PPO and poker-specific optimizations."""

    def __init__(self, config: Config):
        self.config = config

        # Fixed dimensions for poker
        self.input_dim = 18  # We know this from our feature design
        self.num_actions = 5  # fold, check/call, small/medium/large raise

        # Initialize model
        self.model = ActorCriticNetwork(self.input_dim, self.num_actions, config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Experience buffer
        self.buffer = ExperienceBuffer(config.buffer_size)

    def _get_sample_features(self) -> np.ndarray:
        """Get sample features to determine input dimension."""
        probe = OptimizedPolicyPlayer("Probe", self.model)
        opponents = [RandomPlayer(f"R{i}") for i in range(3)]
        game = Game([probe] + opponents, initial_stack=10000)

        # Play one round to get features
        game.play_round(verbose=False)
        if probe.trajectory:
            return probe.trajectory[0]["features"]

        # Fallback dimension
        return np.zeros(18, dtype=np.float32)

    def collect_episode(self, temperature: float = 1.0) -> dict:
        """Collect one episode of experience."""
        # Create players
        hero = OptimizedPolicyPlayer("Hero", self.model, temperature)

        # Mix of opponent types for diversity
        opponent_types = [RandomPlayer, CallingPlayer, GamblingPlayer]
        opponents = []
        for i in range(self.config.num_opponents):
            OpponentClass = random.choice(opponent_types)
            opponents.append(OpponentClass(f"Opp{i}"))

        # Play game
        players = [hero] + opponents
        game = Game(players=players, initial_stack=self.config.initial_stack)
        game.play_round(verbose=False)

        # Calculate reward (chips won/lost in big blinds)
        initial = self.config.initial_stack
        final = game.stacks[0]
        reward = (final - initial) / 100.0  # Normalize by big blind

        # Add terminal reward to trajectory
        if hero.trajectory:
            hero.trajectory[-1]["reward"] = reward
            # Propagate discounted rewards backwards
            for i in range(len(hero.trajectory) - 2, -1, -1):
                hero.trajectory[i]["reward"] = reward * (self.config.gamma ** (len(hero.trajectory) - 1 - i))

        return {"trajectory": hero.trajectory, "reward": reward, "hands_played": len(hero.trajectory)}

    def train_step(self, trajectories: list):
        """Perform one training step with PPO."""
        all_features = []
        all_actions = []
        all_old_log_probs = []
        all_rewards = []
        all_advantages = []

        for traj in trajectories:
            if not traj:
                continue

            features = [step["features"] for step in traj]
            actions = [step["action"] for step in traj]
            old_log_probs = [step["log_prob"] for step in traj]
            rewards = [step.get("reward", 0.0) for step in traj]
            values = [step["value"] for step in traj]

            # Calculate advantages using GAE
            advantages = self._calculate_gae(rewards, values)

            all_features.extend(features)
            all_actions.extend(actions)
            all_old_log_probs.extend(old_log_probs)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)

        if not all_features:
            return {}

        # Convert to tensors
        features_t = torch.FloatTensor(all_features)
        actions_t = torch.LongTensor(all_actions)
        old_log_probs_t = torch.FloatTensor(all_old_log_probs)
        rewards_t = torch.FloatTensor(all_rewards)
        advantages_t = torch.FloatTensor(all_advantages)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update
        logits, values = self.model(features_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)

        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards_t)

        # Entropy bonus
        entropy_loss = -dist.entropy().mean()

        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
        }

    def _calculate_gae(self, rewards: list, values: list) -> list:
        """Calculate Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.config.gamma * next_value - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def evaluate(self, num_episodes: int = 100) -> dict:
        """Evaluate current policy."""
        total_reward = 0
        total_hands = 0

        for _ in range(num_episodes):
            episode = self.collect_episode(temperature=1.0)  # No exploration during eval
            total_reward += episode["reward"]
            total_hands += episode["hands_played"]

        return {"avg_reward_bb": total_reward / num_episodes, "avg_hands": total_hands / num_episodes}

    def train(self):
        """Main training loop."""
        print(f"Starting training with {self.input_dim} input features, {self.num_actions} actions")

        for episode in range(1, self.config.num_episodes + 1):
            # Collect experience with exploration
            ep_data = self.collect_episode(temperature=self.config.exploration_temp)
            self.buffer.add(ep_data["trajectory"])

            # Train when we have enough data
            if len(self.buffer.buffer) >= self.config.batch_size:
                batch = self.buffer.sample(self.config.batch_size)
                train_stats = self.train_step(batch)

                # Evaluate periodically
                if episode % self.config.eval_every == 0:
                    eval_stats = self.evaluate(self.config.eval_episodes)
                    print(
                        f"Episode {episode:4d} | "
                        f"Reward: {eval_stats['avg_reward_bb']:+.3f} BB | "
                        f"Loss: {train_stats.get('loss', 0):.4f} | "
                        f"Entropy: {train_stats.get('entropy', 0):.4f}"
                    )

                    # Clear old experiences periodically
                    if episode % (self.config.eval_every * 5) == 0:
                        self.buffer.clear()

        # Final evaluation
        final_stats = self.evaluate(200)
        print(f"\nFinal performance: {final_stats['avg_reward_bb']:+.3f} BB per hand")

        # Save model
        torch.save(self.model.state_dict(), "poker_model.pt")
        print("Model saved to poker_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train a poker AI")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create config
    config = Config(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Train
    trainer = PokerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
