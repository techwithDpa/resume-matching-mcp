"""LLM access for the matching agent (Anthropic Claude API, official `anthropic` SDK).

Config (env or .env):
  ANTHROPIC_API_KEY       required for live LLM calls
  ANTHROPIC_MODEL         default claude-sonnet-5-5
  ANTHROPIC_EFFORT        default effort for agentic calls (low|medium|high), default medium
  MATCHING_AGENT_OFFLINE  "1" forces offline mode: every LLM step falls back
                          to its deterministic heuristic (used by tests and
                          when no key is configured)

Every request opts into server-side refusal fallbacks (`fallbacks="default"`):
if the model declines, the API re-runs the request on a fallback model inside
the same call. A request that still ends in a refusal raises, and the caller
falls back to its heuristic.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    """Raised when no live LLM is configured; callers fall back to heuristics."""


class LLMRefusal(RuntimeError):
    """The model (and its fallback) declined the request."""


def _load_env_file(path: str = os.path.join(BASE_DIR, ".env")) -> None:
    """Minimal .env loader (same as Milestone 1): KEY=VALUE lines, no overrides."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


def is_offline() -> bool:
    return (os.environ.get("MATCHING_AGENT_OFFLINE") == "1"
            or not os.environ.get("ANTHROPIC_API_KEY"))


def model_name() -> str:
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def default_effort() -> str:
    return os.environ.get("ANTHROPIC_EFFORT", DEFAULT_EFFORT)


@lru_cache(maxsize=1)
def _client():
    import anthropic

    return anthropic.Anthropic(timeout=75.0, max_retries=3)


def get_client():
    """Return the Anthropic client, or raise LLMUnavailable."""
    if is_offline():
        raise LLMUnavailable("offline mode (no ANTHROPIC_API_KEY or MATCHING_AGENT_OFFLINE=1)")
    return _client()


def _split(messages) -> tuple[str, list[dict]]:
    """LangChain System/Human messages -> (system prompt, Messages API turns)."""
    system, turns = [], []
    for m in messages:
        if m.type == "system":
            system.append(m.content)
        else:
            turns.append({"role": "assistant" if m.type == "ai" else "user", "content": m.content})
    return "\n\n".join(system), turns


def structured(schema, messages, effort: str | None = None):
    """One Claude call whose reply is parsed into the pydantic `schema`."""
    system, turns = _split(messages)
    resp = get_client().beta.messages.parse(
        model=model_name(),
        max_tokens=MAX_TOKENS,
        system=system,
        messages=turns,
        output_format=schema,
        output_config={"effort": effort or default_effort()},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise LLMRefusal(f"no structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def structured_batch(schema, batches: list, max_concurrency: int = 4,
                     effort: str | None = None) -> list:
    """Run several structured calls concurrently. Failed items come back as the
    exception instance so callers can fall back per item."""
    get_client()   # raise LLMUnavailable up front

    def one(msgs):
        try:
            return structured(schema, msgs, effort)
        except Exception as e:   # per-item fallback in the caller
            return e

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(one, batches))


def chat(system: str, messages: list[dict], tools: list[dict], effort: str | None = None):
    """One turn of the tool-calling loop. Thinking summaries are requested so the
    UI can show the model's reasoning in the trace."""
    return get_client().beta.messages.create(
        model=model_name(),
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        tools=tools,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort or default_effort()},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
