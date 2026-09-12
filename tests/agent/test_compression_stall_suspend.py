"""Stall-suspend: an over-threshold turn whose compression passes already proved
ineffective must soft-DEFER — never send the oversized request and never report
``compression_exhausted`` (the gateway wipes the session on that contract,
#9893/#35809).

On main before this change, a session whose compressor kept returning no-progress
results (sanitized list, in-place no-op) had only two outcomes: keep sending the
over-threshold request on every iteration, or die at the provider limit — at
which point the exhausted contract let the gateway reset the session. The opt-in
``compression.suspend_on_stall`` flag adds the third outcome: when the
insufficient-progress blocker (``_preflight_compression_blocked``, armed only
after a real pass cuts <5% while still over threshold) is set, the pre-API gate
ends the turn as ``compression_deferred`` so the next inbound message retries
compression instead of burning doomed requests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _compression_deferred_result
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    from agent import retry_utils as _retry_utils
    monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        return a


_PREFILL = [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"},
]


def _install_stall_compressor(a):
    """Compressor stub: pressure only on the fully-assembled request (pre-API
    site); turn-context preflight stands down via the cheap-gate estimate."""
    a.context_compressor = SimpleNamespace(
        protect_first_n=3,
        protect_last_n=20,
        threshold_tokens=100_000,
        context_length=1_000_000,
        last_prompt_tokens=0,
        should_compress=lambda t: t >= 100_000,
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )


def _noop_compress(a):
    """A compress double that returns its input unchanged — a real no-progress
    pass (no lock-skip flag, no transient-block guard)."""

    def _compress(messages, _system_message, **_kwargs):
        a._compression_skipped_due_to_lock = None
        return messages, "You are helpful."

    return _compress


def _oversized_patches(messages_tokens=500_000):
    return (
        patch(
            "agent.turn_context.estimate_request_tokens_rough",
            return_value=10,
        ),
        patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=messages_tokens,
        ),
        patch(
            "agent.model_metadata.estimate_messages_tokens_rough",
            return_value=messages_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Stall suspend: over threshold + proved no-progress → soft defer, no send
# ---------------------------------------------------------------------------


class TestStallSuspend:
    def test_stalled_over_threshold_turn_defers_without_provider_call(self, agent):
        """The agent0 shape: every compress pass returns the same list (sanitized
        no-op). With the flag on, the turn ends as ``compression_deferred``
        before a single provider call goes out — session persisted, not reset."""
        agent.compression_suspend_on_stall = True
        _install_stall_compressor(agent)
        p1, p2, p3 = _oversized_patches()

        with (
            p1, p2, p3,
            patch.object(agent, "_compress_context", side_effect=_noop_compress(agent)),
            patch.object(agent, "_persist_session") as mock_persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_PREFILL))

        agent.client.chat.completions.create.assert_not_called()
        assert result.get("compression_deferred") is True
        assert not result.get("compression_exhausted")
        assert result.get("failed") is False
        assert result.get("completed") is False
        assert result.get("partial") is True
        assert mock_persist.called

    def test_flag_off_sends_oversized_request_as_before(self, agent):
        """Default off: the same stall falls through and the request is sent —
        upstream behavior unchanged without the opt-in."""
        agent.compression_suspend_on_stall = False
        _install_stall_compressor(agent)
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="delivered anyway"),
        ]
        p1, p2, p3 = _oversized_patches()

        with (
            p1, p2, p3,
            patch.object(agent, "_compress_context", side_effect=_noop_compress(agent)),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_PREFILL))

        assert agent.client.chat.completions.create.call_count == 1
        assert result.get("completed") is True
        assert result["final_response"] == "delivered anyway"
        assert not result.get("compression_deferred")
        assert not result.get("compression_exhausted")

    def test_suspended_turn_retries_compression_on_next_turn(self, agent):
        """Suspend is per-turn, not a session lockout: once compression can make
        progress again the next inbound proceeds normally."""
        agent.compression_suspend_on_stall = True
        _install_stall_compressor(agent)
        p1, p2, p3 = _oversized_patches()

        with (
            p1, p2, p3,
            patch.object(agent, "_compress_context", side_effect=_noop_compress(agent)),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            stalled = agent.run_conversation("hello", conversation_history=list(_PREFILL))
        assert stalled.get("compression_deferred") is True

        # Recovery: the compressor now compacts; the len-keyed estimate drops the
        # rebuilt request below threshold so the turn can proceed.
        def _compacting(messages, _system_message, **_kwargs):
            agent._compression_skipped_due_to_lock = None
            return [{"role": "user", "content": "hello"}], "You are helpful."

        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="resumed"),
        ]
        with (
            patch(
                "agent.turn_context.estimate_request_tokens_rough",
                return_value=10,
            ),
            patch(
                "agent.model_metadata.estimate_messages_tokens_rough",
                side_effect=lambda msgs, **_kw: len(msgs) * 50_000,
            ),
            patch.object(agent, "_compress_context", side_effect=_compacting),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            resumed = agent.run_conversation(
                "again", conversation_history=list(stalled["messages"])
            )

        assert resumed.get("completed") is True
        assert resumed["final_response"] == "resumed"
        assert not resumed.get("compression_exhausted")


# ---------------------------------------------------------------------------
# Result contract: the stall reason stays on the soft-defer contract
# ---------------------------------------------------------------------------


class TestStallResultContract:
    def test_stall_reason_is_deferred_never_exhausted(self, agent):
        result = _compression_deferred_result(
            agent,
            [{"role": "user", "content": "hello"}],
            api_call_count=0,
            reason="stall",
        )
        assert result.get("compression_deferred") is True
        assert not result.get("compression_exhausted")
        assert result.get("failed") is False
        assert result.get("partial") is True
        assert "/compress" in result["final_response"]

    def test_unknown_reason_still_lock_copy(self, agent):
        """The default branch keeps the pre-existing lock message."""
        result = _compression_deferred_result(
            agent, [{"role": "user", "content": "x"}], api_call_count=0
        )
        assert "already running" in result["final_response"]
