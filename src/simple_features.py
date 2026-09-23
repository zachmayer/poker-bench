from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

ACTION_SPACE = (
    "fold",
    "check_or_call",
    "half_pot",
    "pot",
    "all_in",
)


def canonicalize_action(valid: Mapping[str, Any], amount_to_call: int, pot_size: int, stack: int) -> list[bool]:
    """Build a mask for ACTION_SPACE indicating legality.

    We map raises into three buckets if raises are allowed; otherwise mark as illegal.
    """
    can_call = "check_or_call" in valid
    can_raise = "complete_bet_or_raise_to" in valid and len(list(valid.get("complete_bet_or_raise_to", []))) > 0

    mask = [False] * len(ACTION_SPACE)

    # fold is always legal unless check is free; we still allow fold but trainer can learn avoiding it.
    mask[0] = True
    mask[1] = can_call
    mask[2] = can_raise
    mask[3] = can_raise
    mask[4] = can_raise or (stack > 0)
    return mask


def encode_state(state: Any, actor_idx: int, valid_actions: Mapping[str, Any]) -> np.ndarray:
    """Minimal numeric encoding from `pokerkit.State`-like object.

    We keep this tiny and robust: stacks, bets, pot, board count, street index, to-call.
    """
    # Defensive fetches with fallbacks
    stacks = np.array(getattr(state, "stacks", []), dtype=float)
    bets = np.array(getattr(state, "bets", []), dtype=float)
    pots = getattr(state, "pots", [])
    pot_total = float(sum(getattr(p, "unraked_amount", 0) for p in pots))
    board_cards = getattr(state, "board_cards", [])
    street = float(getattr(state, "street_index", 0))

    # Derived
    to_call = float(valid_actions.get("check_or_call", 0))

    # Actor view
    my_stack = float(stacks[actor_idx]) if stacks.size else 0.0
    my_bet = float(bets[actor_idx]) if bets.size else 0.0

    # Compact vector (keep < 20 dims)
    vec = np.array(
        [
            my_stack,
            my_bet,
            pot_total,
            to_call,
            street,
            float(len(board_cards)),
        ],
        dtype=float,
    )
    # Normalize roughly to avoid huge scales
    denom = max(1.0, pot_total + my_stack)
    vec[:4] = vec[:4] / denom
    return vec
