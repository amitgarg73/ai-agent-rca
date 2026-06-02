"""
Tool-level cost and latency tracer — opt-in, fail-silent.

Wraps individual tool functions to capture:
  - Wall-clock latency (including nested LLM calls the tool makes internally)
  - Nested LLM cost via an in-span cost accumulator
  - Outcome (success / error) and error message

Design constraints:
  - NEVER raises. Exceptions in tracing code do not propagate.
  - Zero impact on tool return value or exceptions.
  - Writes to c_tool_spans (separate from c_traces — does not touch trading schema).
  - Opt-in: tools that don't use this tracer are unaffected.

Usage (decorator):
    from sdk.tool_tracer import trace_tool

    @trace_tool(agent="research", session_id_fn=lambda: current_session_id())
    def get_earnings(ticker: str) -> dict:
        ...

Usage (context manager for nested LLM cost tracking):
    from sdk.tool_tracer import ToolSpan

    @trace_tool(agent="research", session_id_fn=lambda: sid)
    def search_with_llm(query: str) -> str:
        with ToolSpan.current() as span:
            resp = anthropic_client.messages.create(...)
            span.add_llm_cost(
                model=resp.model,
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
            )
        return resp.content[0].text
"""
from __future__ import annotations

import functools
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any

# Cost per 1M tokens by model (input, output) — approximate, update as pricing changes
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6":          (3.00, 15.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "gpt-4o":                     (5.00, 15.00),
    "gpt-4o-mini":                (0.15,  0.60),
    "text-embedding-3-small":     (0.02,  0.00),
    "text-embedding-3-large":     (0.13,  0.00),
}

_DEFAULT_COST_PER_1M = (1.00, 5.00)  # fallback for unknown models

_local = threading.local()   # thread-local span stack


@dataclass
class ToolSpan:
    """Accumulates cost and outcome for a single tool invocation."""
    span_id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name:       str = ""
    agent:           str = ""
    session_id:      str = ""
    started_at:      str = ""
    latency_ms:      int = 0
    outcome:         str = "success"
    error:           str | None = None
    nested_cost_usd: float = 0.0
    nested_tokens_in: int = 0
    nested_tokens_out: int = 0
    nested_llm_calls:  int = 0

    def add_llm_cost(self, model: str, tokens_in: int, tokens_out: int):
        """Call this inside the tool to record any nested LLM call costs."""
        try:
            in_rate, out_rate = _MODEL_COSTS.get(model, _DEFAULT_COST_PER_1M)
            self.nested_cost_usd   += (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000
            self.nested_tokens_in  += tokens_in
            self.nested_tokens_out += tokens_out
            self.nested_llm_calls  += 1
        except Exception:
            pass  # never let cost tracking break the tool

    @staticmethod
    def current() -> "ToolSpan | _NullSpan":
        """Return the active span for this thread, or a no-op if none."""
        stack = getattr(_local, "stack", [])
        return stack[-1] if stack else _NullSpan()

    def to_db_row(self) -> dict:
        return {
            "id":                str(uuid.uuid4()),
            "span_id":           self.span_id,
            "session_id":        self.session_id,
            "agent":             self.agent,
            "tool_name":         self.tool_name,
            "started_at":        self.started_at,
            "latency_ms":        self.latency_ms,
            "outcome":           self.outcome,
            "error":             self.error,
            "nested_cost_usd":   round(self.nested_cost_usd, 6),
            "nested_tokens_in":  self.nested_tokens_in,
            "nested_tokens_out": self.nested_tokens_out,
            "nested_llm_calls":  self.nested_llm_calls,
        }


class _NullSpan:
    """No-op span returned when no tracer is active."""
    def add_llm_cost(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


def trace_tool(
    agent: str,
    session_id_fn: Callable[[], str],
    db_fn: Callable[[], Any] | None = None,
):
    """
    Decorator that wraps a tool function with a ToolSpan.

    Args:
        agent:         Agent name (e.g. "research").
        session_id_fn: Zero-arg callable that returns the current session ID.
                       Called at decoration time so it can be a closure.
        db_fn:         Zero-arg callable that returns a Supabase client.
                       If None, spans are not persisted (useful for testing).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span = ToolSpan(
                tool_name  = fn.__name__,
                agent      = agent,
                session_id = "",
                started_at = datetime.now(timezone.utc).isoformat(),
            )
            # Push span onto thread-local stack so nested code can call ToolSpan.current()
            if not hasattr(_local, "stack"):
                _local.stack = []
            _local.stack.append(span)

            t0 = time.perf_counter()
            try:
                span.session_id = _safe_call(session_id_fn, "")
                result = fn(*args, **kwargs)
                span.outcome = "success"
                return result
            except Exception as exc:
                span.outcome = "error"
                span.error   = str(exc)[:500]
                raise
            finally:
                span.latency_ms = int((time.perf_counter() - t0) * 1000)
                _local.stack.pop()
                _persist_span(span, db_fn)

        return wrapper
    return decorator


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _persist_span(span: ToolSpan, db_fn):
    """Write the span row to c_tool_spans. Fails silently."""
    try:
        if db_fn is None:
            return
        db = _safe_call(db_fn, None)
        if db is None:
            return
        db.table("c_tool_spans").insert(span.to_db_row()).execute()
    except Exception:
        pass  # tracing must never break the tool
