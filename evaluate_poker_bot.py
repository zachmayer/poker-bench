#!/usr/bin/env python3
"""Evaluate the trained poker bot against various opponents."""

import random

import numpy as np
import torch
from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer
from PokerBots import Game
from PokerBots import RandomPlayer

from train_mini_llm_poker import MiniLLMPlayer
from train_mini_llm_poker import MiniTransformer
from train_mini_llm_poker import PokerTokenizer


def load_trained_model(path: str = "mini_llm_poker.pt"):
    """Load the trained model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = MiniTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = PokerTokenizer()

    return model, tokenizer, config


def evaluate_against_opponents(model, tokenizer, num_games: int = 100, verbose: bool = False):
    """Evaluate model against different opponent types."""
    results = {"vs_random": [], "vs_calling": [], "vs_gambling": [], "vs_mixed": []}

    initial_stack = 10000

    # Test against each opponent type
    for game_type, opponents_setup in [
        ("vs_random", lambda: [RandomPlayer(f"R{i}") for i in range(3)]),
        ("vs_calling", lambda: [CallingPlayer(f"C{i}") for i in range(3)]),
        ("vs_gambling", lambda: [GamblingPlayer(f"G{i}") for i in range(3)]),
        ("vs_mixed", lambda: [RandomPlayer("R"), CallingPlayer("C"), GamblingPlayer("G")]),
    ]:
        for _ in range(num_games):
            hero = MiniLLMPlayer("Hero", model, tokenizer, temperature=1.0)
            opponents = opponents_setup()

            game = Game(players=[hero] + opponents, initial_stack=initial_stack)
            game.play_round(verbose=verbose)

            profit = game.stacks[0] - initial_stack
            results[game_type].append(profit / 100.0)  # Convert to BB

    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("POKER BOT EVALUATION RESULTS")
    print("=" * 60)

    for opponent_type, profits in results.items():
        avg_profit = np.mean(profits)
        std_profit = np.std(profits)
        win_rate = sum(1 for p in profits if p > 0) / len(profits)

        print(f"\n{opponent_type.upper().replace('_', ' ')}:")
        print(f"  Average: {avg_profit:+.3f} BB/hand")
        print(f"  Std Dev: {std_profit:.3f} BB")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Best: {max(profits):+.1f} BB")
        print(f"  Worst: {min(profits):+.1f} BB")

    # Overall performance
    all_profits = []
    for profits in results.values():
        all_profits.extend(profits)

    print("\n" + "-" * 60)
    print("OVERALL PERFORMANCE:")
    print(f"  Average: {np.mean(all_profits):+.3f} BB/hand")
    print(f"  Win Rate: {sum(1 for p in all_profits if p > 0) / len(all_profits):.1%}")
    print("=" * 60)


def test_specific_scenarios(model, tokenizer):
    """Test model in specific poker scenarios."""
    print("\n" + "=" * 60)
    print("SCENARIO TESTING")
    print("=" * 60)

    scenarios = [
        # Heads-up play
        ("Heads-up vs Random", lambda: [MiniLLMPlayer("Hero", model, tokenizer), RandomPlayer("Villain")]),
        # Short-handed
        (
            "3-handed vs Calling",
            lambda: [MiniLLMPlayer("Hero", model, tokenizer), CallingPlayer("C1"), CallingPlayer("C2")],
        ),
        # Full table simulation
        (
            "6-handed mixed",
            lambda: [MiniLLMPlayer("Hero", model, tokenizer)]
            + [random.choice([RandomPlayer, CallingPlayer, GamblingPlayer])(f"P{i}") for i in range(5)],
        ),
    ]

    for scenario_name, players_fn in scenarios:
        profits = []
        for _ in range(20):
            players = players_fn()
            game = Game(players=players, initial_stack=10000)
            game.play_round(verbose=False)

            hero_idx = next(i for i, p in enumerate(players) if p.name == "Hero")
            profit = (game.stacks[hero_idx] - 10000) / 100.0
            profits.append(profit)

        avg_profit = np.mean(profits)
        print(f"\n{scenario_name}: {avg_profit:+.3f} BB/hand")


def main():
    """Main evaluation."""
    print("Loading trained model...")
    model, tokenizer, config = load_trained_model()

    print("Running evaluation...")
    results = evaluate_against_opponents(model, tokenizer, num_games=50)

    print_results(results)
    test_specific_scenarios(model, tokenizer)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
