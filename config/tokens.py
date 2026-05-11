"""
Special token registry for Halo-VLA.

Loads token definitions from config/special_tokens.json so both the
dataloader and model share the same token IDs.

Usage:
    from config.tokens import SPECIAL_TOKENS, get_token_id

    img_id = get_token_id("image_token_id")
    act_id = SPECIAL_TOKENS["token_ids"]["action_token_id"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_TOKENS_FILE = Path(__file__).parent / "special_tokens.json"

def load_special_tokens() -> Dict[str, Any]:
    """Load and return the special tokens config dict."""
    with open(_TOKENS_FILE, "r") as f:
        return json.load(f)


# Module-level singleton — loaded once on import
SPECIAL_TOKENS: Dict[str, Any] = load_special_tokens()


def get_token(name: str) -> str:
    """Get token string by key, e.g. 'image_token' -> '<image>'."""
    return SPECIAL_TOKENS[name]


def get_token_id(name: str) -> int:
    """Get token ID by key, e.g. 'image_token_id' -> 151665."""
    return SPECIAL_TOKENS["token_ids"][name]


def get_all_custom_tokens() -> list[str]:
    """Return the list of custom tokens that need to be added to the tokenizer."""
    return [
        SPECIAL_TOKENS["image_token"],
        SPECIAL_TOKENS["action_token"],
        SPECIAL_TOKENS["state_token"],
    ]


def get_vocab_size() -> int:
    """Return vocab size after adding custom tokens."""
    return SPECIAL_TOKENS["vocab_size_with_custom_tokens"]
