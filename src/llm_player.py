# players/LLMPlayer.py
"""LLM poker bot"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Tuple
from typing import cast

from PokerBots.players.BasePlayer import BasePlayer  # Third-party; no type stubs
from pokerkit import State  # Third-party; no type stubs
from pydantic import Field
from pydantic import create_model
from pydantic_ai import Agent
from pydantic_ai import NativeOutput
from pydantic_ai import ToolOutput
from pydantic_ai import models

STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
}


def history_extractor(state: State) -> list[str]:
    """Return a human-readable summary of selected operations in *chronological* order.

    The summary keeps only the following actions:
    1. BetCollection - shows the list of bets collected in that round.
    2. BlindOrStraddlePosting - shows which player posted and for how much.
    3. Folding - shows which player folded.
    4. BoardDealing - shows the cards dealt to the board.
    """

    # Mapping from operation class name to a tiny formatter.
    formatters: dict[str, Callable[[Any], str]] = {
        "BetCollection": lambda op: f"Bets collected: {list(op.bets)}",
        "BlindOrStraddlePosting": lambda op: f"Player {op.player_index} posts blind/straddle {op.amount}",
        "Folding": lambda op: f"Player {op.player_index} folds",
        "BoardDealing": lambda op: (f"Board dealt: {', '.join(str(card) for card in op.cards)}"),
    }

    lines: list[str] = []
    for op in getattr(state, "operations", []):  # fall back to [] if missing
        formatter = formatters.get(op.__class__.__name__)
        if formatter:
            try:
                lines.append(formatter(op))
            except AttributeError:
                # Skip if expected attributes are missing – stay robust.
                continue

    return lines


class LLMPlayer(BasePlayer):
    """LLM poker bot"""

    def __init__(
        self,
        name: str = "LLM",
        model_name: str = "gpt-4.1-mini",
    ) -> None:
        super().__init__(name)
        self.model_name = model_name
        if models.infer_model(model_name).profile.supports_json_schema_output:
            self.output_wrapper = NativeOutput
        else:
            self.output_wrapper = ToolOutput

    def play(self, valid_actions: Mapping[str, Any], state: State) -> Tuple[str, int]:
        # print("=== POKER STATE DEBUG ===")
        # print(f"State object: {state}")
        # print(f"State attributes: {vars(state)}")
        # print(f"Available methods: {[attr for attr in dir(state) if not attr.startswith('_')]}")
        # print("========================")

        action_tuple = tuple(valid_actions.keys())

        # Treat actor_index as int for type-checking purposes.
        actor_idx = cast(int, state.actor_index)
        can_call = "check_or_call" in valid_actions
        can_raise = "complete_bet_or_raise_to" in valid_actions

        min_call = 0
        min_raise = 0
        max_raise = 0
        if can_call:
            min_call = cast(int, valid_actions["check_or_call"])
        if can_raise:
            raise_to_values = cast(Iterable[int], valid_actions["complete_bet_or_raise_to"])
            # `default=0` handles empty iterables gracefully.
            min_raise = min(raise_to_values, default=0)
            max_raise = max(raise_to_values, default=0)

        prompt = dedent(
            f"""
            <game-info>
            You are playing {state.betting_structure.value} Texas Hold'em with {state.player_count} players.
            Position 0 is small blind, 1 is big blind, 2 is UTG, {state.player_count - 1} is the button.
            You are in position {state.actor_index}.
            </game-info>

            <game-history>
            {history_extractor(state)}
            </game-history>

            <current-street>
            It is currently {STREET_NAMES[state.street_index]} with {sum(state.statuses)} active players.
            </current-street>

            <all-players-info>
            * Pot: {[pot.unraked_amount for pot in state.pots]}
            * Board cards: {[x[0] for x in state.board_cards]}.
            * Stacks by position: {state.stacks}
            * Bets by position: {state.bets}
            </all-players-info>

            <your-info>
            * Your stack: {state.stacks[actor_idx]}
            * Your hole cards: {state.hole_cards[actor_idx]}
            * Bets you've made so far this street: {state.bets[actor_idx]}
            </your-info>

            Choose your next move: {action_tuple}
            {f"{min_call} chips to call." if can_call else ""}
            {f"You may raise to between {min_raise} and {max_raise} chips." if can_raise else ""}
            """
        )
        print(prompt)

        output_schema = create_model(
            "Decision",
            action=Literal[*action_tuple],
            amount=(
                int | None,
                Field(
                    default=min_call,
                    description="Amount of chips to bet or raise to. Omit if folding.",
                ),
            ),
        )
        # Show the schema for debugging purposes.
        print(json.dumps(output_schema.model_json_schema(), indent=2))

        agent = Agent(
            self.model_name,
            output_type=self.output_wrapper(
                output_schema,
                name="poker_hand",
                description="Allowed Poker Actions",
                strict=True,
            ),
            system_prompt="You are playing no-limit Texas Hold'em in an automated tournament.",
            retries=1,
            output_retries=3,
        )

        decision: Any = agent.run_sync(prompt).output

        # If the model says to fold when checking is free, convert to a check.
        if decision.action == "fold" and valid_actions.get("check_or_call") == 0:
            decision.action = "check_or_call"

        return decision.action, decision.amount
