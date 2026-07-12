"""Unit tests for the native Hermes↔Zenos Runtime turn bridge helpers."""

from gateway.zenos_runtime import (
    agent_usage_snapshot,
    compact_history,
    format_execution_receipt,
    infer_turn_context,
    middleware_settings,
    new_turn_id,
    runtime_session_id,
    usage_delta,
)


def test_turn_usage_delta_separates_current_turn_from_session_totals():
    class Agent:
        session_input_tokens = 1000
        session_output_tokens = 200
        session_cache_read_tokens = 5000
        session_cache_write_tokens = 50
        session_reasoning_tokens = 75

    agent = Agent()
    before = agent_usage_snapshot(agent)
    agent.session_input_tokens += 117
    agent.session_output_tokens += 23
    agent.session_cache_read_tokens += 900
    agent.session_cache_write_tokens += 0
    agent.session_reasoning_tokens += 9

    delta = usage_delta(before, agent_usage_snapshot(agent))

    assert delta == {
        "inputTokens": 117,
        "outputTokens": 23,
        "cacheReadTokens": 900,
        "cacheWriteTokens": 0,
        "reasoningTokens": 9,
        "totalTokens": 1040,
    }


def test_middleware_settings_are_fail_open_and_bounded():
    settings = middleware_settings({
        "zenos_runtime": {
            "enabled": True,
            "timeout_seconds": 9999,
            "max_history_chars": 999999,
            "receipt": "full",
        }
    })

    assert settings["enabled"] is True
    assert settings["fail_open"] is True
    assert settings["timeout_seconds"] == 600.0
    assert settings["max_history_chars"] == 120000
    assert settings["receipt"] == "full"


def test_compact_history_keeps_recent_user_and_assistant_text_without_tool_payloads():
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "content": "secret raw tool output"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": [{"type": "text", "text": "latest answer"}]},
    ]

    compact = compact_history(history, 80)

    assert "latest question" in compact
    assert "latest answer" in compact
    assert "secret raw tool output" not in compact


def test_infer_turn_context_marks_code_mutation_and_live_verification_hints():
    context = infer_turn_context(
        "Fix bug API ini, test sampai bener, lalu cek status live sekarang",
        workspace_root="/tmp/repo",
    )

    assert context["hasFiles"] is True
    assert context["hasCodeChangeIntent"] is True
    assert context["userRequestedVerification"] is True
    assert context["requiresFreshData"] is True
    assert context["intent"] in {"execute", "mutate"}


def test_infer_turn_context_marks_explicit_boss_requests_without_matching_casual_boss_mentions():
    requested = infer_turn_context(
        "coba tanya agent boss ada ga caranya supaya private RPC lebih dekat dengan chain"
    )
    casual = infer_turn_context("boss gue nanya soal jadwal meeting")

    assert requested["userRequestedBoss"] is True
    assert casual["userRequestedBoss"] is False


def test_execution_receipt_exposes_real_role_invocation_and_skips():
    receipt = format_execution_receipt({
        "pipeline": "verified_path",
        "host": {"invoked": True, "model": "grok", "provider": "etla-router"},
        "worker": {"invoked": True, "model": "build", "ok": True},
        "verifier": {"invoked": True, "model": "grok", "verdict": "pass", "ok": True},
        "boss": {"invoked": False},
        "transformed": False,
    })

    assert "Host grok" in receipt
    assert "Worker build" in receipt
    assert "Verifier grok/pass" in receipt
    assert "Boss skipped" in receipt


def test_runtime_and_turn_ids_are_stable_or_unique_as_required():
    assert runtime_session_id("same-session") == runtime_session_id("same-session")
    assert runtime_session_id("same-session") != runtime_session_id("other-session")
    first = new_turn_id("session-1")
    second = new_turn_id("session-1")
    assert first != second
    assert first.startswith("session-1_")
