#!/usr/bin/env python3
"""Comprehensive evaluation of the 2000-iteration trained model."""

import random

import numpy as np
import torch
from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer
from PokerBots import Game
from PokerBots import RandomPlayer

from train_mini_llm_poker_optimized import Config
from train_mini_llm_poker_optimized import FastPlayer
from train_mini_llm_poker_optimized import FastTransformer
from train_mini_llm_poker_optimized import SimpleTokenizer


def evaluate_comprehensive(model_path="fast_poker_2000.pt"):
    """Run comprehensive evaluation."""

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = Config()
    model = FastTransformer(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = SimpleTokenizer()

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION (2000 iterations)")
    print("=" * 60)

    # Test configurations
    test_configs = [
        ("vs 2 Random", lambda: [RandomPlayer(f"R{i}") for i in range(2)], 100),
        ("vs 2 Calling", lambda: [CallingPlayer(f"C{i}") for i in range(2)], 100),
        ("vs 2 Gambling", lambda: [GamblingPlayer(f"G{i}") for i in range(2)], 100),
        ("vs Mixed (R+C)", lambda: [RandomPlayer("R"), CallingPlayer("C")], 100),
        ("vs Mixed (R+G)", lambda: [RandomPlayer("R"), GamblingPlayer("G")], 100),
        ("vs Mixed (C+G)", lambda: [CallingPlayer("C"), GamblingPlayer("G")], 100),
        ("Heads-up Random", lambda: [RandomPlayer("R")], 50),
        ("Heads-up Calling", lambda: [CallingPlayer("C")], 50),
        ("Heads-up Gambling", lambda: [GamblingPlayer("G")], 50),
        ("4-player table", lambda: [RandomPlayer("R1"), CallingPlayer("C"), GamblingPlayer("G")], 50),
    ]

    all_results = []

    for name, opponents_fn, num_games in test_configs:
        profits = []
        wins = 0

        for _ in range(num_games):
            hero = FastPlayer("Hero", model, tokenizer)
            opponents = opponents_fn()

            initial_stack = 5000
            game = Game(players=[hero] + opponents, initial_stack=initial_stack)
            game.play_round(verbose=False)

            profit = (game.stacks[0] - initial_stack) / 100.0
            profits.append(profit)
            if profit > 0:
                wins += 1

        avg_profit = np.mean(profits)
        std_profit = np.std(profits)
        win_rate = wins / num_games

        all_results.append(avg_profit)

        print(f"\n{name:20s} | Games: {num_games}")
        print(f"  Avg: {avg_profit:+6.2f} BB  |  Std: {std_profit:5.2f}  |  Win%: {win_rate:5.1%}")
        print(f"  Max: {max(profits):+6.1f} BB  |  Min: {min(profits):+6.1f} BB")

    # Overall stats
    print("\n" + "-" * 60)
    print("OVERALL STATISTICS:")
    print(f"  Average across all scenarios: {np.mean(all_results):+.2f} BB/hand")
    print(f"  Best scenario average: {max(all_results):+.2f} BB/hand")
    print(f"  Worst scenario average: {min(all_results):+.2f} BB/hand")

    # Training history
    if "history" in checkpoint:
        history = checkpoint["history"]
        print("\n" + "-" * 60)
        print("TRAINING HISTORY:")
        print(f"  Final training reward: {history['rewards'][-1]:+.2f} BB/hand")
        print(f"  Avg last 100 iterations: {np.mean(history['rewards'][-100:]):+.2f} BB/hand")
        if "evals" in history:
            print(f"  Final eval score: {history['evals'][-1]:+.2f} BB/hand")
            improvement = history["evals"][-1] - history["evals"][0]
            print(f"  Total improvement: {improvement:+.2f} BB/hand")

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    evaluate_comprehensive()
