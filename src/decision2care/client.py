"""Thin, cached, thread-safe wrapper around the official TypeSafe SDK (Jev).

* Every request is keyed by sha256(model, state, questions) and stored in SQLite, so re-runs
  (or a crash halfway through the cohort) never pay twice and are fully reproducible.
* Usage (input tokens, calls, latency) is accounted per tag (e.g. "agent.route", "dx.L3").
* Questions are passed as plain dicts in the wire format ({"type": "choice", ...}); answers come
  back as plain dicts (JSON of the SDK answer models), so the rest of the code has no SDK types.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from decision2care.paths import CACHE_DIR, load_typesafe_key

JSON = Dict[str, Any]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


USD_PER_MTOK = 0.042  # jev-1.13 list price per million input tokens; output is free


def approx_tokens(obj: Any) -> int:
    """Cheap token estimate (~4 chars/token) used only to stay under Jev's 32k/64k budgets."""
    return len(_canonical(obj)) // 4 + 1


class UsageMeter:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = defaultdict(int)
        self.cached = defaultdict(int)
        self.input_tokens = defaultdict(int)
        self.output_tokens = defaultdict(int)
        self.latency_s = defaultdict(float)
        self.questions = defaultdict(int)

    def add(self, tag: str, tokens_in: int, tokens_out: int, latency: float, n_q: int, cached: bool):
        with self._lock:
            self.calls[tag] += 1
            self.questions[tag] += n_q
            if cached:
                self.cached[tag] += 1
            else:
                self.input_tokens[tag] += tokens_in
                self.output_tokens[tag] += tokens_out
                self.latency_s[tag] += latency

    def snapshot(self) -> dict:
        with self._lock:
            tags = sorted(set(self.calls) | set(self.input_tokens))
            per_tag = {
                t: {
                    "calls": self.calls[t],
                    "cached": self.cached[t],
                    "questions": self.questions[t],
                    "input_tokens": self.input_tokens[t],
                    "latency_s": round(self.latency_s[t], 2),
                }
                for t in tags
            }
            tot_in = sum(self.input_tokens.values())
            return {
                "per_tag": per_tag,
                "total_calls": sum(self.calls.values()),
                "total_cached": sum(self.cached.values()),
                "total_input_tokens": tot_in,
                "est_cost_usd": round(tot_in * USD_PER_MTOK / 1e6, 4),
            }


class DecisionClient:
    def __init__(
        self,
        model: str = "jev-latest",
        cache_path: Optional[Path] = CACHE_DIR / "jev_cache.sqlite",
        timeout: float = 120.0,
        max_retries: int = 6,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        request_model: Optional[str] = None,
        key_salt: Optional[str] = None,
        refresh: bool = False,
    ):
        """`model` names the model in the cache key (the alias every cached answer was stored under); `request_model`, when
        given, is the exact version requested from the API, and every uncached answer must come back from it (version pin).
        `key_salt` is added to every cache key of this client: variants whose requests differ from earlier runs only in dict
        order (the key sorts keys) must be salted, or they would silently reuse the earlier answers."""
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self.model = model
        self.request_model = request_model or model
        self.key_salt = key_salt
        self.refresh = refresh  # fresh calls: never read a stored answer, but store the new ones (the run stays replayable)
        self.usage = UsageMeter()
        self._client = TypeSafeClient(
            api_key=api_key or load_typesafe_key(),
            model=self.request_model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=max_retries, backoff_initial=1.0, backoff_max=30.0, timeout=900.0),
        )
        self._cache = None
        self._lock = threading.Lock()
        if use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=60)
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, response TEXT, created REAL)")
            self._cache.commit()
        self.resolved_model: Optional[str] = None

    # ------------------------------------------------------------------ cache
    def _get(self, key: str) -> Optional[JSON]:
        if self._cache is None:
            return None
        with self._lock:
            row = self._cache.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, resp: JSON) -> None:
        if self._cache is None:
            return
        with self._lock:
            self._cache.execute(
                "INSERT OR REPLACE INTO responses (key, model, response, created) VALUES (?, ?, ?, ?)",
                (key, resp.get("model", self.model), json.dumps(resp, ensure_ascii=False), time.time()),
            )
            self._cache.commit()

    # ------------------------------------------------------------------ main entry point
    def key(self, state: Any, questions: Dict[str, JSON], key_extra: Optional[str] = None) -> str:
        """A request's key in the response store."""
        body = {"model": self.model, "state": state, "questions": questions}
        salt = "|".join(x for x in (self.key_salt, key_extra) if x)
        return hashlib.sha256(_canonical({**body, "key_salt": salt} if salt else body).encode()).hexdigest()

    def ask(self, state: Any, questions: Dict[str, JSON], tag: str = "misc", key_extra: Optional[str] = None) -> JSON:
        """Evaluate `questions` (wire-format dicts) against `state`; returns {qid: answer_dict}. `key_extra` salts the cache key
        of this request only (used by the scorer for option-order variants)."""
        if not questions:
            return {}
        key = self.key(state, questions, key_extra)
        cached = None if self.refresh else self._get(key)
        if cached is not None:
            self.usage.add(tag, 0, 0, 0.0, len(questions), cached=True)
            return cached["answers"]
        t0 = time.time()
        resp = self._client.system_one(state=state, questions=questions)
        dt = time.time() - t0
        if resp.model != self.request_model:
            raise RuntimeError(f"model version drift: requested {self.request_model}, answered by {resp.model}")
        answers = {qid: a.model_dump(mode="json") for qid, a in resp.answers.items()}
        out = {
            "model": resp.model,
            "answers": answers,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            "latency_s": round(dt, 3),
        }
        self.resolved_model = resp.model
        self.usage.add(tag, resp.usage.input_tokens or 0, resp.usage.output_tokens or 0, dt, len(questions), cached=False)
        self._put(key, out)
        return answers

    def close(self):
        self._client.close()
        if self._cache is not None:
            self._cache.close()


# ------------------------------------------------------------------------- question builders
def choice(instructions: Any, criteria: Dict[str, Any]) -> JSON:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Any, criteria: Optional[Dict[str, Any]] = None) -> JSON:
    q: JSON = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


def score(instructions: Any, levels: list) -> JSON:
    return {"type": "score", "instructions": instructions, "criteria": levels}


class OfflineClient:
    """Offline stand-in for tests: uniform choices (first option wins), nouls from a fixed table."""

    def __init__(self, noul_default: float = 0.2, scripted: Optional[Dict[str, Any]] = None):
        self.usage = UsageMeter()
        self.model = "fake"
        self.resolved_model = "fake"
        self.noul_default = noul_default
        self.scripted = scripted or {}
        self.log = []

    def ask(self, state, questions, tag="misc", key_extra=None):
        self.log.append((tag, state, questions))
        out = {}
        for qid, q in questions.items():
            if qid in self.scripted:
                out[qid] = self.scripted[qid]
                continue
            if q["type"] == "choice":
                keys = list(q["criteria"])
                p = {k: 0.0 for k in keys}
                p[keys[0]] = 1.0
                out[qid] = {"type": "choice", "choice": keys[0], "probabilities": p, "confidence": 1.0}
            elif q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": self.noul_default}
            else:
                n = len(q["criteria"])
                out[qid] = {"type": "score", "score": 0.0, "legend": {str(i): c for i, c in enumerate(q["criteria"])}, "probabilities": {str(i): (1.0 if i == 0 else 0.0) for i in range(n)}, "confidence": 1.0}
        self.usage.add(tag, approx_tokens(state) + approx_tokens(questions), 0, 0.0, len(questions), cached=False)
        return out

    def close(self):
        pass


def client_from_config(cfg: Dict[str, Any], dry_run: bool = False, no_cache: bool = False, refresh: bool = False):
    """The client a run uses, built from its config: version pin (`client.request_model`) and cache salts. A state field order
    other than the default is salted automatically, because the cache key sorts the state's keys."""
    if dry_run:
        return OfflineClient()
    c = cfg["client"]
    salts = [c["key_salt"]] if c.get("key_salt") else []
    order = (cfg.get("tools") or {}).get("state_field_order", "default")
    if order != "default":
        salts.append(f"state_field_order={order}")
    return DecisionClient(model=c["model"], timeout=c.get("timeout", 120), max_retries=c.get("max_retries", 6),
                     use_cache=c.get("use_cache", True) and not no_cache, request_model=c.get("request_model"),
                     key_salt="|".join(salts) or None, refresh=refresh)
