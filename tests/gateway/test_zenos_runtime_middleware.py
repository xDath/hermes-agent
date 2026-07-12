"""Unit tests for the native Hermes↔Zenos Runtime turn bridge helpers."""

from gateway.zenos_runtime import (
    agent_usage_snapshot,
    apply_host_working_set_limit,
    compact_history,
    format_execution_receipt,
    handoff_messages,
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
    assert settings["context_soft_limit_tokens"] == 160000
    assert settings["handoff_history_chars"] == 240000
    assert settings["handoff_max_messages"] == 300
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


def test_handoff_messages_keep_small_head_recent_tail_and_bound_tool_payloads():
    history = [
        {"role": "user", "content": "project identity"},
        {"role": "assistant", "content": "initial decision"},
        {"role": "tool", "name": "terminal", "content": "x" * 9000},
        *[
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}-" + ("y" * 1000)}
            for index in range(30)
        ],
    ]

    packet = handoff_messages(history, max_chars=12000, max_messages=12)

    assert len(packet) <= 12
    assert packet[0]["content"] == "project identity"
    assert any("message-29" in item["content"] for item in packet)
    tool = next(item for item in packet if item["role"] == "tool")
    assert len(tool["content"]) <= 2500
    assert sum(len(item["content"]) for item in packet) <= 12000


def test_host_working_set_limit_only_lowers_existing_compressor_threshold():
    class Compressor:
        context_length = 1_000_000
        threshold_tokens = 500_000
        threshold_percent = 0.5

    class Agent:
        context_compressor = Compressor()

    agent = Agent()
    applied = apply_host_working_set_limit(agent, 160_000)

    assert applied == {"previous": 500_000, "applied": 160_000}
    assert agent.context_compressor.threshold_tokens == 160_000
    assert agent.context_compressor.threshold_percent == 0.16


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
        "host": {"invoked": True, "model": "grok", "provider": "etla-router", "plannerInvoked": True},
        "worker": {"invoked": True, "model": "build", "ok": True},
        "verifier": {"invoked": True, "model": "grok", "verdict": "pass", "ok": True},
        "boss": {"invoked": False},
        "transformed": False,
    })

    assert "Host grok" in receipt
    assert "plan+final" in receipt
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
