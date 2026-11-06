from __future__ import annotations
from typing import Any, Dict, List
def chief_complaint(rec: Dict[str, Any]) -> str:
    """The triage chief complaint; one that is only the cohort builder's mask ("____") or empty counts as none."""
    cc = str(rec.get("ED Chief Complaint") or "").strip()
    return "" if re.fullmatch(r"[\s_.,;:/-]*", cc) else cc

def starter(rec: Dict[str, Any]) -> str:
    """The patient's opening line (src/conv.py)."""
    cc = chief_complaint(rec)
    return STARTER.format(primary_symptom=f"primary symptom: {cc}") if cc else STARTER_NO_COMPLAINT

def render(turns: List[dict]) -> str:
    """The dialogue as the text that replaces the history of present illness in Jev's state."""
    return "\n".join((f"Doctor: {t['doctor']}\n" if t.get("doctor") else "") + f"Patient: {t['patient']}" for t in turns)
