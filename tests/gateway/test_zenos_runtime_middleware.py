"""Unit tests for the native Hermes↔Zenos Runtime turn bridge helpers."""

import inspect
from pathlib import Path
from unittest.mock import patch

from gateway.zenos_runtime import (
    agent_usage_snapshot,
    apply_host_working_set_limit,
    authoritative_host_override,
    claim_gateway_continuation,
    compact_history,
    format_execution_receipt,
    handoff_messages,
    infer_turn_context,
    internal_continuation_prompt,
    middleware_settings,
    new_turn_id,
    omit_none_values,
    resolve_workspace_root,
    restore_host_working_set_limit,
    runtime_session_id,
    structured_execution_receipts,
    usage_delta,
    workspace_root_from_text,
)


def test_postflight_payload_omits_unavailable_optional_workspace_state():
    payload = omit_none_values({
        "sessionId": "session-1",
        "workspaceState": None,
        "failed": False,
    })

    assert payload == {"sessionId": "session-1", "failed": False}


def test_structured_execution_receipts_use_real_tool_results_not_definitions():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-test",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": '{"command":"npm run typecheck"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-test",
            "name": "terminal",
            "content": '{"output":"TypeScript passed","exit_code":0}',
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-patch",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": '{"path":"app/lib/runtime.ts"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-patch",
            "name": "apply_patch",
            "content": '{"status":"completed","changed_files":["app/lib/runtime.ts"]}',
        },
    ]

    receipts = structured_execution_receipts(
        messages,
        workspace_before={"dirtyDiffSha256": "a" * 64, "changedFiles": []},
        workspace_after={
            "dirtyDiffSha256": "b" * 64,
            "changedFiles": [{"path": "app/lib/runtime.ts", "exists": True}],
        },
    )

    validation = next(item for item in receipts if item.get("validationKind") == "typecheck")
    assert validation["kind"] == "validation"
    assert validation["status"] == "passed"
    assert validation["exitCode"] == 0
    mutation = next(item for item in receipts if item.get("tool") == "apply_patch")
    assert mutation["kind"] == "workspace"
    assert mutation["metadata"]["mutating"] is True
    assert mutation["changedFiles"] == ["app/lib/runtime.ts"]
    assert any(item["receiptId"].startswith("hermes-workspace-") for item in receipts)


def test_structured_execution_receipts_fail_nonzero_validation():
    receipts = structured_execution_receipts([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-build",
                "function": {"name": "terminal", "arguments": '{"command":"npm run build"}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-build",
            "content": '{"output":"build failed","exit_code":2}',
        },
    ])

    assert receipts[0]["kind"] == "validation"
    assert receipts[0]["validationKind"] == "build"
    assert receipts[0]["status"] == "failed"
    assert receipts[0]["exitCode"] == 2


def test_internal_continuation_prompt_requires_explicit_bounded_contract():
    assert internal_continuation_prompt(None) == ""
    assert internal_continuation_prompt({"continuation": {"required": False, "prompt": "ignore"}}) == ""
    prompt = internal_continuation_prompt({
        "continuation": {
            "required": True,
            "prompt": "  continue the same durable task  ",
        }
    })
    assert prompt == "continue the same durable task"
    assert len(internal_continuation_prompt({
        "continuation": {"required": True, "prompt": "x" * 30_000}
    })) == 24_000


def test_restart_recovery_claim_passes_a_process_start_cutoff_to_runtime():
    cutoff = "2026-07-19T14:30:51.168000+00:00"
    with (
        patch("gateway.zenos_runtime._runtime_key", return_value="test-runtime-key"),
        patch("gateway.zenos_runtime._json_request", return_value={"ok": True}) as request,
    ):
        result = claim_gateway_continuation(
            "hermes_session/with topic",
            recover_leased_before=cutoff,
            base_url="http://runtime.test",
        )

    assert result == {"ok": True}
    url = request.call_args.args[0]
    assert "sessionId=hermes_session%2Fwith+topic" in url
    assert "recoverLeasedBefore=2026-07-19T14%3A30%3A51.168000%2B00%3A00" in url


def test_gateway_restart_watcher_recovers_only_pre_start_leases():
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._zenos_continuation_watcher)
    cutoff_marker = "restart_recovery_cutoff = datetime.now().astimezone().isoformat()"
    claim_marker = "recover_leased_before=restart_recovery_cutoff"
    assert cutoff_marker in source
    assert claim_marker in source
    assert source.index(cutoff_marker) < source.index("await asyncio.sleep(4)") < source.index(claim_marker)


def test_runtime_continuation_is_queued_before_intermediate_delivery_and_kept_out_of_transcript():
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._run_agent_inner)
    queue_marker = 'result.get("zenos_continuation_prompt")'
    delivery_marker = "if not was_interrupted and not _is_runtime_continuation:"
    recursion_marker = "followup_result = await self._run_agent("
    persistence_marker = 'persist_user_message="" if _is_runtime_continuation else None'

    assert queue_marker in source
    assert delivery_marker in source
    assert recursion_marker in source
    assert persistence_marker in source
    assert source.index(queue_marker) < source.index(delivery_marker) < source.index(recursion_marker)


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
        "source": "hermes-session-delta",
        "valid": True,
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
    assert settings["context_soft_limit_tokens"] == 64000
    assert settings["handoff_history_chars"] == 120000
    assert settings["handoff_max_messages"] == 160
    assert settings["receipt"] == "full"
    assert settings["authoritative_host"] is True
    assert settings["enforce_host_token_budget"] is False
    assert settings["enforce_host_working_set_limit"] is False


def test_runtime_host_override_is_applied_only_when_authority_is_enabled():
    preflight = {
        "hostOverride": {"model": "deepseek", "provider": "etla-router"},
    }
    assert authoritative_host_override(preflight, {"authoritative_host": True}) == {
        "model": "deepseek",
        "provider": "etla-router",
    }
    assert authoritative_host_override(preflight, {"authoritative_host": False}) is None
    assert authoritative_host_override({"hostOverride": {"model": "", "provider": "etla-router"}}, {"authoritative_host": True}) is None


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


def test_host_working_set_limit_can_be_disabled_for_native_hermes_context_ownership():
    class Compressor:
        context_length = 1_000_000
        threshold_tokens = 500_000
        threshold_percent = 0.5
        tail_token_budget = 100_000

    class Agent:
        context_compressor = Compressor()

    agent = Agent()
    state = apply_host_working_set_limit(agent, 48_000, enforce=False)

    assert state["applied"] is False
    assert state["disabled"] is True
    assert agent.context_compressor.threshold_tokens == 500_000
    assert agent.context_compressor.tail_token_budget == 100_000


def test_host_working_set_limit_only_lowers_existing_compressor_threshold():
    class Compressor:
        context_length = 1_000_000
        threshold_tokens = 500_000
        threshold_percent = 0.5
        summary_target_ratio = 0.25
        tail_token_budget = 100_000

    class Agent:
        context_compressor = Compressor()

    agent = Agent()
    applied = apply_host_working_set_limit(agent, 160_000)

    assert applied["applied"] is True
    assert applied["previous"] == 500_000
    assert applied["current"] == 160_000
    assert agent.context_compressor.threshold_tokens == 160_000
    assert agent.context_compressor.threshold_percent == 0.16
    assert agent.context_compressor.tail_token_budget == 40_000

    restore_host_working_set_limit(agent, applied)
    assert agent.context_compressor.threshold_tokens == 500_000
    assert agent.context_compressor.threshold_percent == 0.5
    assert agent.context_compressor.tail_token_budget == 100_000


def test_workspace_resolution_prefers_explicit_repo_then_reuses_session_repo(tmp_path):
    root = Path(tmp_path)
    runtime = root / "zenos-runtime"
    memory = root / "zenos-memory"
    for repo in (runtime, memory):
        repo.mkdir()
        (repo / "package.json").write_text("{}", encoding="utf-8")

    selected = resolve_workspace_root(
        "audit zenos memory secara detail",
        candidates=[str(root)],
        previous=str(runtime),
    )
    assert selected == str(memory.resolve())

    reused = resolve_workspace_root(
        "lanjut benerin yang tadi",
        candidates=[str(root)],
        previous=selected,
    )
    assert reused == selected


def test_workspace_evidence_normalizes_host_aliases_to_the_sandbox_root(tmp_path):
    root = Path(tmp_path)
    repo = root / "zenos-runtime"
    repo.mkdir()
    (repo / "package.json").write_text("{}", encoding="utf-8")

    assert workspace_root_from_text(
        f"used {repo}",
        allowed_parent=str(root),
    ) == str(repo.resolve())
    assert workspace_root_from_text(
        "used /root/openclaw-projects/zenos-runtime",
        allowed_parent=str(root),
    ) == str(repo.resolve())
    assert workspace_root_from_text(
        "used /workspace/zenos-runtime",
        allowed_parent=str(root),
    ) == str(repo.resolve())


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


def test_infer_turn_context_understands_indonesian_live_verification_phrasing():
    context = infer_turn_context("yang live cobain diverifikasi dulu")

    assert context["userRequestedVerification"] is True
    assert context["requiresFreshData"] is True
    assert context["intent"] == "analyze"


def test_infer_turn_context_inherits_unfinished_coding_intent_for_short_continuation():
    history = [
        {"role": "user", "content": "Fix bot.py lalu test dan restart servicenya"},
        {"role": "assistant", "content": "Code sudah dibaca, patch belum ke-apply dan smoke masih pending."},
    ]

    context = infer_turn_context("Gas", history=history, workspace_root="/tmp/repo")

    assert context["hasFiles"] is True
    assert context["hasCodeChangeIntent"] is True
    assert context["intent"] == "mutate"


def test_infer_turn_context_does_not_promote_casual_short_acknowledgement():
    history = [
        {"role": "user", "content": "Jelasin cuaca hari ini"},
        {"role": "assistant", "content": "Cuaca cerah dan tidak ada pekerjaan tertunda."},
    ]

    context = infer_turn_context("Gas", history=history)

    assert context["hasCodeChangeIntent"] is False
    assert context["intent"] == "analyze"


def test_infer_turn_context_estimate_includes_tool_calls_not_only_visible_text():
    context = infer_turn_context(
        "lanjut",
        history=[{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "x" * 4_000}}],
        }],
    )

    assert context["estimatedContextTokens"] >= 1_000


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
