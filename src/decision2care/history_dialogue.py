from __future__ import annotations
from typing import List

def render(turns: List[dict]) -> str:
    """The dialogue as the text that replaces the history of present illness in Jev's state."""
    return "\n".join((f"Doctor: {t['doctor']}\n" if t.get("doctor") else "") + f"Patient: {t['patient']}" for t in turns)
