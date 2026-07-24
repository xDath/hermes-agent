"""Local Hermes client for Zenos Cognitive Runtime.

Model authority is session-scoped and single-model: `/model` selects the Host,
and native Hermes workers plus any explicitly requested review cycle inherit
that same model. Legacy multi-role payloads are accepted only for migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_RUNTIME_URL = "http://127.0.0.1:3090"
DEFAULT_ROUTER_URL = "http://127.0.0.1:20128"
RUNTIME_ROLES = ("host",)
DEFAULT_MIDDLEWARE_TIMEOUT = 180.0
_USAGE_COUNTERS = {
    "inputTokens": "session_input_tokens",
    "outputTokens": "session_output_tokens",
    "cacheReadTokens": "session_cache_read_tokens",
    "cacheWriteTokens": "session_cache_write_tokens",
    "reasoningTokens": "session_reasoning_tokens",
}


def agent_usage_snapshot(agent: Any) -> Dict[str, int]:
    """Capture monotonic agent session counters at one turn boundary."""
    return {
        key: max(0, int(getattr(agent, attribute, 0) or 0))
        for key, attribute in _USAGE_COUNTERS.items()
    }


def usage_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    max_calls: int = 1,
    context_length: int = 0,
    max_output_tokens: int = 0,
) -> Dict[str, Any]:
    """Return one-turn usage with monotonic and aggregate plausibility checks."""
    raw_before = {key: int(before.get(key, 0) or 0) for key in _USAGE_COUNTERS}
    raw_after = {key: int(after.get(key, 0) or 0) for key in _USAGE_COUNTERS}
    counter_regressions = [key for key in _USAGE_COUNTERS if raw_after[key] < raw_before[key]]
    result: Dict[str, Any] = {
        key: max(0, raw_after[key] - raw_before[key])
        for key in _USAGE_COUNTERS
    }
    result["totalTokens"] = (
        result["inputTokens"]
        + result["cacheReadTokens"]
        + result["cacheWriteTokens"]
        + result["outputTokens"]
    )
    calls = max(1, min(int(max_calls or 1), 64))
    effective_context = max(24_000, int(context_length or 0))
    plausible_input = effective_context * calls * 2
    plausible_output = max(4_096, int(max_output_tokens or 0) * calls * 2)
    invalid_reason = ""
    if counter_regressions:
        invalid_reason = f"session usage counters regressed: {', '.join(counter_regressions)}"
    elif result["inputTokens"] + result["cacheReadTokens"] + result["cacheWriteTokens"] > plausible_input:
        invalid_reason = "aggregate input/cache usage exceeds context-by-call plausibility bound"
    elif result["outputTokens"] > plausible_output:
        invalid_reason = "aggregate output usage exceeds output-by-call plausibility bound"
    result["source"] = "hermes-session-delta"
    result["valid"] = not invalid_reason
    if invalid_reason:
        result["invalidReason"] = invalid_reason
    return result


def runtime_session_id(session_key: str) -> str:
    """Return a stable, non-sensitive Runtime id for one Hermes conversation."""
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
    return f"hermes_{digest}"


def _json_request(
    url: str,
    *,
    api_key: str = "",
    method: str = "GET",
    body: Any = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout))) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload)
            detail = str(parsed.get("error") or payload)
            issues = parsed.get("issues")
            if isinstance(issues, list) and issues:
                issue_text = "; ".join(
                    f"{str(item.get('path') or '<root>')}: {str(item.get('message') or 'invalid')}"
                    for item in issues[:8]
                    if isinstance(item, Mapping)
                )
                if issue_text:
                    detail = f"{detail} ({issue_text})"
            request_id = str(parsed.get("requestId") or "").strip()
            if request_id:
                detail = f"{detail} [requestId={request_id}]"
        except Exception:
            detail = payload
        raise RuntimeError(f"Zenos request failed ({exc.code}): {detail[:1200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Zenos service unavailable: {exc}") from exc


def _runtime_url(override: str = "") -> str:
    return (override or os.getenv("ZENOS_RUNTIME_URL", DEFAULT_RUNTIME_URL)).rstrip("/")


def _runtime_key() -> str:
    key = os.getenv("ZENOS_RUNTIME_API_KEY", "").strip()
    if key:
        return key
    credential_directory = os.getenv("CREDENTIALS_DIRECTORY", "").strip()
    candidates = [
        os.getenv("ZENOS_RUNTIME_ENV_FILE", ""),
        str(Path(credential_directory) / "zenos-runtime.env") if credential_directory else "",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            for raw_line in Path(candidate).read_text(encoding="utf-8").splitlines():
                if not raw_line.startswith("ZENOS_RUNTIME_API_KEY="):
                    continue
                value = raw_line.split("=", 1)[1].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if value:
                    return value
        except OSError:
            continue
    raise RuntimeError("ZENOS_RUNTIME_API_KEY is not configured")


def apply_host_token_budget(
    agent: Any,
    budget: Mapping[str, Any] | None,
    *,
    enforce: bool = True,
) -> Dict[str, Any]:
    """Optionally apply one Runtime-issued Host cap for the current turn.

    Hermes needs multiple model iterations to discover tools, call them, read
    their results, and continue. Gateway integrations should keep ``enforce``
    disabled unless a deliberately strict execution cap is required.
    """
    if not enforce:
        return {"applied": False, "disabled": True}
    if not isinstance(budget, Mapping):
        return {"applied": False}
    max_calls = max(1, min(int(budget.get("maxCalls") or 1), 32))
    max_output = max(128, min(int(budget.get("maxOutputTokens") or 2048), 32_000))
    previous_iterations = max(1, int(getattr(agent, "max_iterations", max_calls) or max_calls))
    previous_tokens = getattr(agent, "max_tokens", None)
    applied_iterations = min(previous_iterations, max_calls)
    applied_tokens = min(int(previous_tokens), max_output) if previous_tokens else max_output
    agent.max_iterations = applied_iterations
    agent.max_tokens = applied_tokens
    return {
        "applied": True,
        "budgetId": str(budget.get("budgetId") or ""),
        "reservationId": str(budget.get("reservationId") or ""),
        "previousMaxIterations": previous_iterations,
        "previousMaxTokens": previous_tokens,
        "maxIterations": applied_iterations,
        "maxTokens": applied_tokens,
    }


def restore_host_token_budget(agent: Any, state: Mapping[str, Any] | None) -> None:
    if not isinstance(state, Mapping) or not state.get("applied"):
        return
    agent.max_iterations = int(state.get("previousMaxIterations") or agent.max_iterations)
    agent.max_tokens = state.get("previousMaxTokens")


def get_runtime_models(session_id: str) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url()}/api/runtime/models?sessionId={urllib.parse.quote(session_id)}",
        api_key=_runtime_key(),
    )


def save_runtime_models(session_id: str, roles: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    # Compatibility adapter for stale callers: select the first usable model
    # and write it as the single Host slot. Runtime normalizes old role payloads
    # the same way, so no session loses its prior selection during migration.
    entry = roles.get("host") or roles.get("worker") or roles.get("verifier") or roles.get("boss") or {}
    model = str(entry.get("model") or "").strip()
    provider = str(entry.get("provider") or "").strip()
    if not model or not provider:
        raise RuntimeError("model and provider are required")
    payload: Dict[str, str] = {
        "hostModel": model,
        "hostProvider": provider,
    }
    return _json_request(
        f"{_runtime_url()}/api/runtime/models?sessionId={urllib.parse.quote(session_id)}",
        api_key=_runtime_key(),
        method="POST",
        body=payload,
    )


def save_runtime_host(session_id: str, model: str, provider: str) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url()}/api/runtime/models?sessionId={urllib.parse.quote(session_id)}",
        api_key=_runtime_key(),
        method="POST",
        body={"hostModel": model, "hostProvider": provider},
    )


def save_runtime_default_host(model: str, provider: str) -> Dict[str, Any]:
    """Persist the one global Hermes/Runtime model selection."""
    return _json_request(
        f"{_runtime_url()}/api/runtime/models",
        api_key=_runtime_key(),
        method="POST",
        body={"hostModel": model, "hostProvider": provider},
    )


def list_runtime_combos() -> List[Dict[str, Any]]:
    base = os.getenv("NINE_ROUTER_URL", DEFAULT_ROUTER_URL).rstrip("/")
    data = _json_request(f"{base}/api/zenos-runtime/combos")
    combos = data.get("combos")
    return combos if isinstance(combos, list) else []


def role_config_from_response(data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    config = data.get("config") if isinstance(data.get("config"), dict) else data
    configured_roles = config.get("roles") if isinstance(config, dict) else {}
    host = configured_roles.get("host") if isinstance(configured_roles, dict) else {}
    identity = {
        "model": str((host or {}).get("model") or "unknown"),
        "provider": str((host or {}).get("provider") or "default"),
    }
    # Return legacy aliases for stale Telegram picker state, all pointing to the
    # same authoritative Host identity. New user-facing flows use /model only.
    return {role: dict(identity) for role in ("host", "worker", "verifier", "boss")}


def middleware_settings(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Resolve the native Hermes↔Zenos turn middleware settings.

    Behavioral settings live in config.yaml. The Runtime credential remains a
    secret and is resolved by :func:`_runtime_key`.
    """
    raw = config.get("zenos_runtime") if isinstance(config, Mapping) else {}
    section = raw if isinstance(raw, Mapping) else {}
    return {
        "enabled": bool(section.get("enabled", False)),
        "url": str(section.get("url") or DEFAULT_RUNTIME_URL).rstrip("/"),
        "fail_open": bool(section.get("fail_open", True)),
        "fail_closed_mutations": bool(section.get("fail_closed_mutations", True)),
        "continuity_packet_v2": bool(section.get("continuity_packet_v2", True)),
        # Execution receipts remain available in the internal postflight payload
        # and logs. They are not appended to normal user-facing messages unless
        # an operator explicitly enables a debug receipt mode.
        "receipt": str(section.get("receipt") or "off").strip().lower(),
        "timeout_seconds": min(
            max(float(section.get("timeout_seconds") or DEFAULT_MIDDLEWARE_TIMEOUT), 10.0),
            600.0,
        ),
        "max_history_chars": min(
            max(int(section.get("max_history_chars") or 16_000), 0),
            120_000,
        ),
        "context_soft_limit_tokens": min(
            max(int(section.get("context_soft_limit_tokens") or 64_000), 24_000),
            500_000,
        ),
        "handoff_history_chars": min(
            max(int(section.get("handoff_history_chars") or 120_000), 20_000),
            500_000,
        ),
        "handoff_max_messages": min(
            max(int(section.get("handoff_max_messages") or 160), 20),
            400,
        ),
        "disable_streaming_when_verified": bool(
            section.get("disable_streaming_when_verified", True)
        ),
        "report_failures": bool(section.get("report_failures", False)),
        "authoritative_host": bool(section.get("authoritative_host", True)),
        # Hermes is a tool-using agent loop, not a single inference call. Runtime
        # budgets remain useful for accounting and postflight decisions, but
        # shrinking max_iterations or the compressor threshold makes the Host
        # forget tools/context before it can finish the turn. These execution
        # caps are therefore opt-in.
        "enforce_host_token_budget": bool(section.get("enforce_host_token_budget", False)),
        "enforce_host_working_set_limit": bool(section.get("enforce_host_working_set_limit", False)),
    }


def runtime_failure_may_fail_open(
    settings: Mapping[str, Any] | None,
    *,
    message: str = "",
    routing_hints: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    execution_receipts: Sequence[Mapping[str, Any]] | None = None,
    workspace_before: Mapping[str, Any] | None = None,
    workspace_after: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether Runtime failure may release an unverified Host response.

    Read-only low-risk chat may remain available during a sidecar outage. Code
    mutation, deploy/destructive work, and security/secret boundaries pause or
    fail closed by default so a missing Runtime cannot silently bypass policy,
    deterministic validation, or approval checks.
    """
    if not isinstance(settings, Mapping) or not bool(settings.get("fail_open", True)):
        return False
    if not bool(settings.get("fail_closed_mutations", True)):
        return True

    # Postflight has concrete execution evidence. A lexical routing false
    # positive must not replace a valid Host answer with a middleware warning.
    # Keep fail-closed only when a real workspace/tool mutation or an explicit
    # approval boundary is present.
    if execution_receipts is not None or workspace_after is not None:
        decision = preflight.get("decision") if isinstance(preflight, Mapping) else {}
        requires_approval = bool((decision or {}).get("requiresApproval"))
        before_revision = _workspace_revision(workspace_before)
        after_revision = _workspace_revision(workspace_after)
        workspace_mutated = bool(after_revision and before_revision != after_revision)
        receipt_mutated = any(
            bool((receipt.get("metadata") or {}).get("mutating"))
            or bool(receipt.get("changedFiles"))
            for receipt in (execution_receipts or [])
            if isinstance(receipt, Mapping)
        )
        return not (requires_approval or workspace_mutated or receipt_mutated)

    hints = routing_hints if isinstance(routing_hints, Mapping) else {}
    decision = preflight.get("decision") if isinstance(preflight, Mapping) else {}
    task_type = str((decision or {}).get("taskType") or "").strip()
    if task_type in {
        "coding_change",
        "security_or_secret",
        "deploy_or_destructive_action",
    }:
        return False
    if bool(hints.get("hasCodeChangeIntent")) or str(hints.get("intent") or "") == "mutate":
        return False
    text = str(message or "")
    if re.search(
        r"\b(?:deploy|restart|push|commit|install|delete|hapus|wipe|destroy|rotate\s+key|"
        r"secret|credential|private\s+key|api\s*key|token)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return True


def authoritative_host_override(
    preflight: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None,
) -> Dict[str, str] | None:
    """Return the Runtime-selected Host when Runtime owns Host authority."""
    if not isinstance(preflight, Mapping) or not isinstance(settings, Mapping):
        return None
    if not settings.get("authoritative_host", True):
        return None
    raw = preflight.get("hostOverride")
    if not isinstance(raw, Mapping):
        return None
    model = str(raw.get("model") or "").strip()
    provider = str(raw.get("provider") or "").strip()
    if not model or not provider:
        return None
    return {"model": model, "provider": provider}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            elif item.get("type") in {"image_url", "input_image"}:
                parts.append("[image attachment]")
        return "\n".join(parts)
    return str(content or "")


def compact_history(history: Sequence[Mapping[str, Any]] | None, max_chars: int) -> str:
    """Return a bounded recent transcript without raw tool payloads."""
    if not history or max_chars <= 0:
        return ""
    lines: List[str] = []
    for message in reversed(list(history)):
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        text = _content_text(message.get("content")).strip()
        if not text:
            continue
        line = f"{role}: {text}"
        lines.append(line[:8_000])
        if sum(len(item) + 1 for item in lines) >= max_chars:
            break
    rendered = "\n".join(reversed(lines))
    return rendered[-max_chars:]


def handoff_messages(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    max_chars: int = 240_000,
    max_messages: int = 300,
) -> List[Dict[str, str]]:
    """Build a bounded, evidence-preserving transcript for durable compaction.

    The active Hermes transcript remains canonical. This packet keeps a small
    conversation head plus the recent tail, aggressively bounds raw tool
    results, and is only sent to the local Runtime sidecar under context
    pressure.
    """
    if not history or max_chars <= 0 or max_messages <= 0:
        return []

    rendered: List[Dict[str, str]] = []
    for message in history:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        text = _content_text(message.get("content")).strip()
        if role == "assistant" and message.get("tool_calls"):
            try:
                calls = message.get("tool_calls")
                names = []
                if isinstance(calls, list):
                    for call in calls[:12]:
                        if not isinstance(call, Mapping):
                            continue
                        function = call.get("function")
                        if isinstance(function, Mapping):
                            name = str(function.get("name") or "").strip()
                            if name:
                                names.append(name)
                if names:
                    text = f"{text}\n[tool calls: {', '.join(names)}]".strip()
            except Exception:
                pass
        if role == "tool":
            name = str(message.get("name") or message.get("tool_name") or "tool").strip()
            text = f"{name}: {text}" if text else f"{name}: completed"
        if not text:
            continue
        per_message_limit = 2_500 if role == "tool" else 8_000
        item: Dict[str, str] = {
            "role": role,
            "content": text[:per_message_limit],
        }
        name = str(message.get("name") or "").strip()
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        if name:
            item["name"] = name[:200]
        if tool_call_id:
            item["tool_call_id"] = tool_call_id[:500]
        rendered.append(item)

    if not rendered:
        return []
    head_count = min(3, len(rendered))
    head = rendered[:head_count]
    tail_candidates = rendered[head_count:]
    head_chars = sum(len(item["content"]) for item in head)
    remaining_chars = max(0, max_chars - head_chars)
    remaining_messages = max(0, max_messages - len(head))
    tail: List[Dict[str, str]] = []
    used = 0
    for item in reversed(tail_candidates):
        size = len(item["content"])
        if len(tail) >= remaining_messages or used + size > remaining_chars:
            continue
        tail.append(item)
        used += size
    tail.reverse()
    return (head + tail)[-max_messages:]


_CONTINUITY_GOAL_RE = re.compile(
    r"\b(?:buat|bikin|fix|perbaiki|implement|upgrade|audit|deploy|ubah|tambahkan|hapus|selesaikan|kerjakan|build|debug|goal|tujuan|pengen|mau)\b",
    re.IGNORECASE,
)
_CONTINUITY_DECISION_RE = re.compile(
    r"\b(?:decision|decided|final|approved|confirmed|pilih|pakai|gunakan|diputuskan|keputusan)\b",
    re.IGNORECASE,
)
_CONTINUITY_CONSTRAINT_RE = re.compile(
    r"\b(?:must|must not|do not|never|always|jangan|harus|wajib|tanpa|only|hanya|acceptance criteria)\b",
    re.IGNORECASE,
)
_CONTINUITY_PATCH_RE = re.compile(
    r"\b(?:patch|patched|edit|edited|changed|modified|write|replace|mutation|diff|commit)\b",
    re.IGNORECASE,
)
_CONTINUITY_VALIDATION_RE = re.compile(
    r"\b(?:test|tested|typecheck|lint|build|compile|validation|validate|verified|pass(?:ed)?|fail(?:ed)?)\b",
    re.IGNORECASE,
)
_CONTINUITY_BLOCKER_RE = re.compile(
    r"\b(?:blocker|blocked|error|failed|failure|timeout|denied|broken|invalid|regression|crash|ngadat|gagal|pending approval)\b",
    re.IGNORECASE,
)
_CONTINUITY_PENDING_RE = re.compile(
    r"\b(?:todo|pending|next|remaining|lanjut|belum|unfinished|retry|approval|required|needs? to)\b",
    re.IGNORECASE,
)
_CONTINUITY_PATH_RE = re.compile(
    r"(?:/srv/etla/workspaces/|/usr/local/lib/hermes-agent/|app/|gateway/|tests/|scripts/)[A-Za-z0-9_./-]+"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_continuity_packet_hash(packet: Mapping[str, Any]) -> str:
    hashable = {
        key: value
        for key, value in packet.items()
        if key != "contentHash" and value is not None
    }
    return hashlib.sha256(_canonical_json(hashable).encode("utf-8")).hexdigest()


def _stable_occurred_at(message: Mapping[str, Any]) -> str:
    for key in ("occurred_at", "created_at", "timestamp", "time"):
        value = str(message.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    # A stable fallback is required so an identical retry produces the same
    # packet hash and therefore the same Memory checkpoint.
    return "1970-01-01T00:00:00Z"


def _continuity_entry(message: Mapping[str, Any], index: int) -> Dict[str, Any] | None:
    role = str(message.get("role") or "").strip().lower()
    if role not in {"user", "assistant", "tool", "system"}:
        return None
    text = _content_text(message.get("content")).strip()
    if role == "assistant" and message.get("tool_calls"):
        names: List[str] = []
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls[:16]:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if isinstance(function, Mapping):
                    name = str(function.get("name") or "").strip()
                    if name:
                        names.append(name)
        if names:
            text = f"{text}\n[tool calls: {', '.join(names)}]".strip()
    name = str(message.get("name") or message.get("tool_name") or "").strip()
    if role == "tool":
        text = f"{name or 'tool'}: {text}" if text else f"{name or 'tool'}: completed"
    if not text:
        return None
    source_hash = hashlib.sha256(f"{role}\n{text}".encode("utf-8")).hexdigest()
    provided_id = str(message.get("message_id") or message.get("id") or "").strip()
    message_id = provided_id[:500] if provided_id else f"m{index}:{role}:{source_hash[:16]}"
    return {
        "index": index,
        "role": role,
        "content": text,
        "name": name[:200],
        "tool_call_id": str(message.get("tool_call_id") or "").strip()[:500],
        "message_id": message_id,
        "source_hash": source_hash,
        "occurred_at": _stable_occurred_at(message),
    }


def _bounded_packet_messages(
    entries: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
    direction: str,
    limit: int,
) -> List[Dict[str, str]]:
    source = list(entries) if direction == "head" else list(reversed(entries))
    kept: List[Dict[str, str]] = []
    used = 0
    for entry in source:
        if len(kept) >= limit:
            break
        remaining = max_chars - used
        if remaining <= 64:
            break
        content = str(entry.get("content") or "")[: min(24_000, max(64, remaining - 64))]
        item: Dict[str, str] = {
            "role": str(entry.get("role") or "system"),
            "content": content,
            "message_id": str(entry.get("message_id") or "")[:500],
        }
        name = str(entry.get("name") or "").strip()
        tool_call_id = str(entry.get("tool_call_id") or "").strip()
        if name:
            item["name"] = name[:200]
        if tool_call_id:
            item["tool_call_id"] = tool_call_id[:500]
        size = len(_canonical_json(item))
        if size > remaining and kept:
            continue
        kept.append(item)
        used += min(size, remaining)
    return kept if direction == "head" else list(reversed(kept))


def _milestone_kind(entry: Mapping[str, Any]) -> str | None:
    text = str(entry.get("content") or "")
    role = str(entry.get("role") or "")
    if _CONTINUITY_BLOCKER_RE.search(text):
        return "blocker"
    if _CONTINUITY_VALIDATION_RE.search(text):
        return "validation"
    if _CONTINUITY_PATCH_RE.search(text):
        return "patch"
    if role == "tool":
        return "tool_result"
    if _CONTINUITY_DECISION_RE.search(text):
        return "decision"
    if _CONTINUITY_CONSTRAINT_RE.search(text):
        return "constraint"
    if role == "user" and _CONTINUITY_GOAL_RE.search(text):
        return "goal"
    return None


def build_continuity_packet(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    session_id: str,
    turn_id: str,
    estimated_tokens: int,
    max_chars: int = 240_000,
    max_messages: int = 300,
    previous_checkpoint_id: str = "",
) -> Dict[str, Any] | None:
    """Compile an evidence-addressed packet for Runtime-owned continuity.

    The packet is deterministic for identical input. It keeps meaningful head
    instructions, ranked milestones from the full transcript, active tool
    evidence, unfinished work, and a recent tail under independent budgets.
    """
    if not history:
        return None
    entries = [
        entry
        for index, message in enumerate(history)
        if isinstance(message, Mapping)
        for entry in [_continuity_entry(message, index)]
        if entry is not None
    ]
    if not entries:
        return None

    total_chars = min(max(int(max_chars or 240_000), 20_000), 500_000)
    head_budget = int(total_chars * 0.12)
    milestone_budget = int(total_chars * 0.33)
    tool_budget = int(total_chars * 0.20)
    tail_budget = total_chars - head_budget - milestone_budget - tool_budget

    head_candidates = [
        entry for entry in entries[:40]
        if entry["role"] in {"system", "user"}
        and (
            _CONTINUITY_GOAL_RE.search(entry["content"])
            or _CONTINUITY_DECISION_RE.search(entry["content"])
            or _CONTINUITY_CONSTRAINT_RE.search(entry["content"])
        )
    ]
    if not head_candidates:
        head_candidates = [entry for entry in entries[:12] if entry["role"] in {"system", "user"}]
    head = _bounded_packet_messages(
        head_candidates[:8],
        max_chars=head_budget,
        direction="head",
        limit=min(8, max_messages),
    )

    ranked_milestones: List[Dict[str, Any]] = []
    for entry in entries:
        kind = _milestone_kind(entry)
        if not kind:
            continue
        priority = {
            "blocker": 7,
            "validation": 6,
            "patch": 5,
            "tool_result": 4,
            "decision": 3,
            "constraint": 2,
            "goal": 1,
        }[kind]
        ranked_milestones.append({
            "kind": kind,
            "text": str(entry["content"])[:8_000],
            "sourceMessageIds": [str(entry["message_id"])],
            "sourceHash": str(entry["source_hash"]),
            "occurredAt": str(entry["occurred_at"]),
            "_index": int(entry["index"]),
            "_priority": priority,
        })
    ranked_milestones.sort(key=lambda item: (item["_priority"], item["_index"]), reverse=True)
    selected_milestones: List[Dict[str, Any]] = []
    milestone_used = 0
    for item in ranked_milestones:
        public_item = {key: value for key, value in item.items() if not key.startswith("_")}
        size = len(_canonical_json(public_item))
        if len(selected_milestones) >= 100 or milestone_used + size > milestone_budget:
            continue
        selected_milestones.append(public_item)
        milestone_used += size
    selected_milestones.sort(
        key=lambda item: next(
            (entry["index"] for entry in entries if entry["source_hash"] == item["sourceHash"]),
            0,
        )
    )

    active_tool_state: List[Dict[str, Any]] = []
    tool_used = 0
    for entry in reversed(entries):
        if entry["role"] != "tool":
            continue
        text = str(entry["content"])
        status = (
            "blocked" if re.search(r"\b(?:blocked|denied)\b", text, re.IGNORECASE)
            else "failed" if _CONTINUITY_BLOCKER_RE.search(text)
            else "running" if re.search(r"\b(?:running|started|in progress)\b", text, re.IGNORECASE)
            else "passed"
        )
        item = {
            "id": str(entry["message_id"]),
            "tool": str(entry.get("name") or "tool")[:200],
            "status": status,
            "summary": text[:8_000],
            "changedFiles": list(dict.fromkeys(_CONTINUITY_PATH_RE.findall(text)))[:200],
            "artifactIds": [],
            "sourceMessageIds": [str(entry["message_id"])],
            "sourceHash": str(entry["source_hash"]),
            "occurredAt": str(entry["occurred_at"]),
        }
        size = len(_canonical_json(item))
        if len(active_tool_state) >= 80 or tool_used + size > tool_budget:
            continue
        active_tool_state.append(item)
        tool_used += size
    active_tool_state.reverse()

    open_work: List[Dict[str, Any]] = []
    for entry in reversed(entries):
        text = str(entry["content"])
        if not (_CONTINUITY_PENDING_RE.search(text) or _CONTINUITY_BLOCKER_RE.search(text)):
            continue
        lowered = text.lower()
        kind = (
            "approval" if "approval" in lowered
            else "validate" if _CONTINUITY_VALIDATION_RE.search(text)
            else "patch" if _CONTINUITY_PATCH_RE.search(text)
            else "other"
        )
        item = {
            "id": f"work:{entry['message_id']}",
            "kind": kind,
            "text": text[:8_000],
            "status": "blocked" if _CONTINUITY_BLOCKER_RE.search(text) else "retry_pending" if "retry" in lowered else "queued",
            "acceptanceCriteria": [
                line.strip(" -")[:2_000]
                for line in text.splitlines()
                if re.search(r"acceptance|criteria|must|harus|wajib", line, re.IGNORECASE)
            ][:20],
            "blockers": [text[:2_000]] if _CONTINUITY_BLOCKER_RE.search(text) else [],
            "sourceMessageIds": [str(entry["message_id"])],
            "sourceHash": str(entry["source_hash"]),
        }
        if item not in open_work:
            open_work.append(item)
        if len(open_work) >= 80:
            break
    open_work.reverse()

    tail_limit = min(160, max(20, int(max_messages or 300) - len(head)))
    recent_tail = _bounded_packet_messages(
        entries,
        max_chars=tail_budget,
        direction="tail",
        limit=tail_limit,
    )
    source_cursor = f"msg:{len(entries)}:{entries[-1]['source_hash'][:24]}"
    packet: Dict[str, Any] = {
        "version": "continuity-v2",
        "sessionId": str(session_id)[:220],
        "turnId": str(turn_id)[:220],
        "sourceCursor": source_cursor,
        "estimatedTokens": max(0, min(int(estimated_tokens or 0), 10_000_000)),
        "head": head,
        "milestones": selected_milestones,
        "recentTail": recent_tail,
        "activeToolState": active_tool_state,
        "openWork": open_work,
    }
    if previous_checkpoint_id:
        packet["previousCheckpointId"] = str(previous_checkpoint_id)[:500]
    packet["contentHash"] = compute_continuity_packet_hash(packet)
    return packet


def apply_host_working_set_limit(
    agent: Any,
    soft_limit_tokens: int,
    *,
    enforce: bool = True,
) -> Dict[str, Any]:
    """Optionally apply a reversible Host working-set limit for one turn.

    Native Hermes compression remains authoritative by default. Lowering its
    threshold on routine turns can discard conversational and tool context far
    earlier than the configured model window.
    """
    if not enforce:
        return {"applied": False, "disabled": True, "previous": 0, "current": 0}
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return {"applied": False, "previous": 0, "current": 0}
    try:
        context_length = max(1, int(getattr(compressor, "context_length", 0) or 0))
        previous_threshold = max(
            1,
            int(getattr(compressor, "threshold_tokens", context_length) or context_length),
        )
        requested = max(24_000, min(int(soft_limit_tokens), context_length - 1))
        current = min(previous_threshold, requested)
        state: Dict[str, Any] = {
            "applied": True,
            "previous": previous_threshold,
            "current": current,
            "previousThresholdPercent": getattr(compressor, "threshold_percent", None),
            "previousTailTokenBudget": getattr(compressor, "tail_token_budget", None),
            "previousWorkingSet": getattr(agent, "_zenos_host_working_set_tokens", None),
        }
        if current != previous_threshold:
            compressor.threshold_tokens = current
            compressor.threshold_percent = current / context_length
            ratio = float(getattr(compressor, "summary_target_ratio", 0.25) or 0.25)
            ratio = max(0.10, min(ratio, 0.40))
            compressor.tail_token_budget = max(4_000, min(int(current * ratio), current - 1))
        setattr(agent, "_zenos_host_working_set_tokens", current)
        return state
    except Exception:
        return {"applied": False, "previous": 0, "current": 0}


def restore_host_working_set_limit(agent: Any, state: Mapping[str, Any] | None) -> None:
    """Restore compressor state captured by :func:`apply_host_working_set_limit`."""
    if not isinstance(state, Mapping) or not state.get("applied"):
        return
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return
    compressor.threshold_tokens = int(state.get("previous") or compressor.threshold_tokens)
    if state.get("previousThresholdPercent") is not None:
        compressor.threshold_percent = state.get("previousThresholdPercent")
    if state.get("previousTailTokenBudget") is not None:
        compressor.tail_token_budget = state.get("previousTailTokenBudget")
    previous_working_set = state.get("previousWorkingSet")
    if previous_working_set is None:
        try:
            delattr(agent, "_zenos_host_working_set_tokens")
        except AttributeError:
            pass
    else:
        setattr(agent, "_zenos_host_working_set_tokens", previous_working_set)


def _contains_term(text: str, term: str) -> bool:
    """Match routing vocabulary as tokens/phrases, never raw substrings.

    Raw substring checks made ordinary words such as ``profile`` match
    ``file`` and Indonesian chat such as ``buat Gmail`` look like a source-code
    mutation. Phrase-aware boundaries keep routing conservative without losing
    multi-word intents such as ``stack trace`` or ``update service``.
    """
    normalized = str(term or "").strip().lower()
    if not normalized:
        return False
    phrase = r"\s+".join(re.escape(part) for part in normalized.split())
    return bool(re.search(rf"(?<![\w]){phrase}(?![\w])", str(text or "").lower(), re.UNICODE))


def _contains_any_term(text: str, terms: Sequence[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def infer_turn_context(
    message: str,
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
    workspace_root: str = "",
) -> Dict[str, Any]:
    """Derive conservative deterministic routing hints from one Hermes turn."""
    text = str(message or "")
    lower = text.lower()
    history_items = [item for item in (history or []) if isinstance(item, Mapping)]
    history_chars = sum(
        len(_content_text(item.get("content")))
        + len(json.dumps(item.get("tool_calls") or "", ensure_ascii=False, default=str))
        + len(json.dumps(item.get("reasoning") or "", ensure_ascii=False, default=str))
        + len(str(item.get("tool_name") or ""))
        for item in history_items
    )
    estimated_tokens = min(2_000_000, max(1, (len(text) + history_chars) // 4))
    code_terms = (
        "code", "coding", "repo", "repository", "file", "function", "class",
        "bug", "error", "stack trace", "typescript", "javascript", "python",
        "commit", "branch", "test", "lint", "build", "compile", "api", "endpoint",
        "route", "handler", "controller", "service", "schema", "database", "migration",
        "migrasi", "table", "kolom", "column", "query", "sql", "source", "golang",
        "go code", "rust", "java", "frontend", "backend", "component", "hook",
        "kode", "coding", "ngoding", "perbaiki", "benerin", "implement", "refactor",
    )
    mutation_terms = (
        "fix", "ubah", "edit", "buat", "bikin", "tambah", "tambahkan", "implement", "refactor",
        "benerin", "betulin", "ganti", "replace", "migrasi", "migration", "hapus", "delete",
        "deploy", "restart", "push", "commit", "install",
    )
    log_terms = ("log", "journalctl", "traceback", "stack trace", "stdout", "stderr")
    verification_terms = (
        "verify", "verification", "verifikasi", "diverifikasi", "validasi", "validate",
        "pastikan", "cek bener", "cek dulu", "cek live", "live check", "are you sure",
        "yakin", "test", "uji", "buktikan",
    )
    boss_request_terms = (
        "tanya agent boss", "tanya boss", "panggil agent boss", "panggil boss",
        "minta agent boss", "minta boss", "suruh agent boss", "suruh boss",
        "boss review", "review sama boss", "ask agent boss", "ask the boss",
        "ask boss", "consult the boss", "escalate to boss",
    )
    fresh_terms = (
        "latest", "terbaru", "hari ini", "sekarang", "current", "news", "harga",
        "weather", "jadwal", "score", "status live", "cek live", "live check",
        "yang live", "secara live", "real-time", "real time",
    )
    execute_terms = (
        "jalankan", "run ", "deploy", "restart", "push", "commit", "hapus",
        "delete", "kirim", "send", "install", "update service",
    )
    has_code = _contains_any_term(lower, code_terms)
    has_mutation = _contains_any_term(lower, mutation_terms)
    # Short acknowledgements such as "Gas" are common continuation commands.
    # Preserve the active coding intent only when recent history contains both
    # code evidence and an explicitly unfinished mutation, so casual chat does
    # not get promoted to an expensive coding pipeline.
    recent_history = "\n".join(
        _content_text(item.get("content"))
        for item in history_items[-16:]
    ).lower()
    is_short_continuation = bool(re.fullmatch(
        r"\s*(?:ok(?:e|ay)?\s+)?(?:gas+|lanjut(?:kan)?|continue|proceed|jalan(?:kan)?|kerjain)\s*[.!]*\s*",
        lower,
    ))
    unfinished_terms = (
        "belum", "pending", "todo", "in_progress", "in progress", "lanjut",
        "next turn", "belum ke-apply", "belum di-apply", "not applied",
        "belum selesai", "unfinished", "remaining work",
    )
    continuation_code_change = (
        is_short_continuation
        and _contains_any_term(recent_history, code_terms)
        and _contains_any_term(recent_history, mutation_terms)
        and _contains_any_term(recent_history, unfinished_terms)
    )
    if continuation_code_change:
        has_code = True
        has_mutation = True
    intent = "analyze"
    if continuation_code_change:
        intent = "mutate"
    elif _contains_any_term(lower, execute_terms):
        intent = "execute"
    elif has_code and has_mutation:
        intent = "mutate"
    elif _contains_any_term(lower, ("rencana", "plan", "arsitektur", "design")):
        intent = "plan"
    elif _contains_any_term(lower, ("jelasin", "jelaskan", "apa itu", "explain")):
        intent = "explain"
    return {
        "hasFiles": bool(workspace_root and has_code),
        "hasLogs": _contains_any_term(lower, log_terms),
        "hasCodeChangeIntent": bool(has_code and has_mutation),
        "userRequestedVerification": _contains_any_term(lower, verification_terms),
        "userRequestedBoss": _contains_any_term(lower, boss_request_terms),
        "estimatedContextTokens": estimated_tokens,
        "confidence": 0.75,
        "intent": intent,
        "containsUntrustedInput": False,
        "requiresFreshData": _contains_any_term(lower, fresh_terms),
    }


def _is_repository_root(path: Path) -> bool:
    return path.is_dir() and any(
        (path / marker).exists()
        for marker in (".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod")
    )


def resolve_workspace_root(
    message: str,
    *,
    candidates: Sequence[str] | None = None,
    previous: str = "",
) -> str:
    """Resolve a session workspace without treating a multi-repo parent as a repo.

    An explicit repository name/path in the current message wins. Otherwise a
    previously observed repository remains active for follow-up instructions.
    """
    text = str(message or "").lower()
    roots: List[Path] = []
    for raw in candidates or []:
        try:
            path = Path(str(raw)).expanduser().resolve()
        except Exception:
            continue
        if path.is_dir() and path not in roots:
            roots.append(path)

    repositories: List[Path] = []
    for root in roots:
        if _is_repository_root(root):
            repositories.append(root)
            continue
        try:
            for child in list(root.iterdir())[:200]:
                if _is_repository_root(child):
                    repositories.append(child.resolve())
        except OSError:
            continue

    for repository in repositories:
        name = repository.name.lower()
        aliases = {name, name.replace("-", " "), name.replace("_", " ")}
        if str(repository).lower() in text or any(alias and alias in text for alias in aliases):
            return str(repository)

    if previous:
        try:
            previous_path = Path(previous).expanduser().resolve()
            if _is_repository_root(previous_path):
                return str(previous_path)
        except Exception:
            pass

    return str(repositories[0]) if len(repositories) == 1 else ""


def workspace_root_from_text(text: str, *, allowed_parent: str = "/srv/etla/workspaces") -> str:
    """Extract and normalize a repository path from bounded tool evidence."""
    raw = str(text or "")
    parent = Path(allowed_parent).expanduser().resolve()
    aliases = tuple(dict.fromkeys((
        f"{str(parent).rstrip('/')}/",
        "/srv/etla/workspaces/",
        "/root/openclaw-projects/",
        "/workspace/",
    )))
    prefix_pattern = "|".join(re.escape(prefix.rstrip("/")) for prefix in aliases)
    pattern = re.compile(rf"(?:{prefix_pattern})/[A-Za-z0-9._-]+")
    for match in pattern.finditer(raw):
        matched = match.group(0)
        relative = next((matched[len(prefix):] for prefix in aliases if matched.startswith(prefix)), "")
        if not relative:
            continue
        try:
            candidate = (parent / relative).resolve()
            candidate.relative_to(parent)
        except Exception:
            continue
        if _is_repository_root(candidate):
            return str(candidate)
    return ""


def workspace_state_snapshot(workspace_root: str) -> Dict[str, Any] | None:
    """Capture a bounded, non-secret Git/file state for transactional recovery."""
    if not workspace_root:
        return None
    try:
        root = Path(workspace_root).expanduser().resolve()
        if not _is_repository_root(root):
            return None

        def git(*args: str) -> bytes:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            return completed.stdout if completed.returncode == 0 else b""

        git_head = git("rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
        status = git("status", "--porcelain=v1", "-z")
        paths: List[str] = []
        for raw_entry in status.split(b"\0"):
            if not raw_entry:
                continue
            entry = raw_entry.decode("utf-8", errors="replace")
            candidate = entry[3:] if len(entry) > 3 else entry
            if " -> " in candidate:
                candidate = candidate.split(" -> ", 1)[1]
            candidate = candidate.strip()
            if candidate and candidate not in paths:
                paths.append(candidate)
        diff_material = b"\n".join([
            git("diff", "--binary", "--no-ext-diff", "HEAD", "--"),
            status,
        ])
        changed_files: List[Dict[str, Any]] = []
        for relative in paths[:200]:
            try:
                absolute = (root / relative).resolve()
                absolute.relative_to(root)
            except Exception:
                continue
            exists = absolute.is_file()
            item: Dict[str, Any] = {"path": relative, "exists": exists}
            if exists and absolute.stat().st_size <= 20_000_000:
                digest = hashlib.sha256()
                with absolute.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                item["sha256"] = digest.hexdigest()
            changed_files.append(item)
        return {
            "workspaceRoot": str(root),
            "gitHead": git_head,
            "dirtyDiffSha256": hashlib.sha256(diff_material).hexdigest(),
            "changedFiles": changed_files,
            "clean": not bool(paths),
            "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    except Exception:
        return None


def new_turn_id(session_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
    return f"{safe[:80]}_{uuid.uuid4().hex[:20]}"


def gateway_preflight(
    payload: Mapping[str, Any],
    *,
    base_url: str = "",
    timeout: float = DEFAULT_MIDDLEWARE_TIMEOUT,
) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/preflight",
        api_key=_runtime_key(),
        method="POST",
        body=dict(payload),
        timeout=timeout,
    )


def gateway_heartbeat(
    payload: Mapping[str, Any],
    *,
    base_url: str = "",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/heartbeat",
        api_key=_runtime_key(),
        method="POST",
        body=dict(payload),
        timeout=timeout,
    )


def gateway_abort(
    payload: Mapping[str, Any],
    *,
    base_url: str = "",
    timeout: float = DEFAULT_MIDDLEWARE_TIMEOUT,
) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/abort",
        api_key=_runtime_key(),
        method="POST",
        body=dict(payload),
        timeout=timeout,
    )


def omit_none_values(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an API payload without explicit null optional fields.

    Runtime schemas use omission to represent unavailable optional evidence.
    Sending JSON null for an optional object is a different contract and can
    trigger strict validation failures.
    """
    return {str(key): value for key, value in payload.items() if value is not None}


def internal_continuation_prompt(postflight: Mapping[str, Any] | None) -> str:
    """Extract a bounded, explicitly required internal Runtime continuation."""
    if not isinstance(postflight, Mapping):
        return ""
    continuation = postflight.get("continuation")
    if not isinstance(continuation, Mapping) or continuation.get("required") is not True:
        return ""
    prompt = str(continuation.get("prompt") or "").strip()
    return prompt[:24_000]


def internal_continuation_id(postflight: Mapping[str, Any] | None) -> str:
    if not isinstance(postflight, Mapping):
        return ""
    continuation = postflight.get("continuation")
    if not isinstance(continuation, Mapping) or continuation.get("required") is not True:
        return ""
    return str(continuation.get("continuationId") or "").strip()[:220]


def internal_continuation_token(postflight: Mapping[str, Any] | None) -> str:
    if not isinstance(postflight, Mapping):
        return ""
    continuation = postflight.get("continuation")
    if not isinstance(continuation, Mapping) or continuation.get("required") is not True:
        return ""
    return str(continuation.get("leaseToken") or "").strip()[:500]


def claim_gateway_continuation(
    session_id: str,
    *,
    recover_leased_before: str = "",
    lease_owner: str = "hermes-gateway",
    base_url: str = "",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    query = {
        "sessionId": session_id,
        "leaseOwner": str(lease_owner or "hermes-gateway")[:220],
    }
    if recover_leased_before:
        query["recoverLeasedBefore"] = recover_leased_before
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/continuation?{urllib.parse.urlencode(query)}",
        api_key=_runtime_key(),
        timeout=timeout,
    )


def complete_gateway_continuation(
    continuation_id: str,
    *,
    lease_token: str,
    cancelled: bool = False,
    base_url: str = "",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if not continuation_id or not lease_token:
        return {"ok": False, "skipped": True}
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/continuation",
        api_key=_runtime_key(),
        method="POST",
        body={
            "continuationId": continuation_id,
            "leaseToken": lease_token,
            "action": "cancel" if cancelled else "complete",
        },
        timeout=timeout,
    )


def heartbeat_gateway_continuation(
    continuation_id: str,
    *,
    lease_token: str,
    base_url: str = "",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if not continuation_id or not lease_token:
        return {"ok": False, "skipped": True}
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/continuation",
        api_key=_runtime_key(),
        method="POST",
        body={
            "continuationId": continuation_id,
            "leaseToken": lease_token,
            "action": "heartbeat",
        },
        timeout=timeout,
    )


def gateway_postflight(
    payload: Mapping[str, Any],
    *,
    base_url: str = "",
    timeout: float = DEFAULT_MIDDLEWARE_TIMEOUT,
) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url(base_url)}/api/runtime/gateway/postflight",
        api_key=_runtime_key(),
        method="POST",
        body=dict(payload),
        timeout=timeout,
    )


_VALIDATION_PATTERN = re.compile(
    r"\b(?:test|tests|pytest|vitest|jest|lint|eslint|typecheck|tsc|build|compile|py_compile|syntax|smoke)\b",
    re.I,
)
_MUTATING_TOOL_PATTERN = re.compile(
    r"\b(?:patch|apply_patch|edit|write|create_file|delete_file|replace|str_replace)\b",
    re.I,
)


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _content_text(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _integer_field(record: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
    return None


def _string_list(value: Any, limit: int = 200) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:4096])
        elif isinstance(item, Mapping):
            path = str(item.get("path") or item.get("file") or "").strip()
            if path:
                result.append(path[:4096])
        if len(result) >= limit:
            break
    return list(dict.fromkeys(result))


def _workspace_revision(state: Mapping[str, Any] | None) -> str:
    if not isinstance(state, Mapping):
        return ""
    return str(state.get("dirtyDiffSha256") or state.get("dirty_diff_sha256") or "").strip()


def structured_execution_receipts(
    messages: Any,
    *,
    history_offset: int = 0,
    workspace_before: Mapping[str, Any] | None = None,
    workspace_after: Mapping[str, Any] | None = None,
    max_receipts: int = 200,
) -> List[Dict[str, Any]]:
    """Compile actual tool-call/result messages into deterministic receipts.

    Tool definitions are deliberately ignored. Runtime completion evidence must
    come from assistant ``tool_calls`` paired with ``role=tool`` results.
    """
    if not isinstance(messages, list):
        messages = []
    bounded_offset = max(0, min(int(history_offset or 0), len(messages)))
    turn_messages = messages[bounded_offset:] if bounded_offset else messages[-240:]
    calls: Dict[str, Dict[str, Any]] = {}
    receipts: List[Dict[str, Any]] = []

    for message in turn_messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
            call_id = str(call.get("id") or "").strip()
            if not call_id:
                continue
            arguments = _json_object(function.get("arguments"))
            calls[call_id] = {
                "name": str(function.get("name") or call.get("name") or "tool").strip() or "tool",
                "arguments": arguments,
            }

    for message in turn_messages:
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        call = calls.get(call_id, {})
        name = str(message.get("name") or call.get("name") or "tool").strip() or "tool"
        arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        content = _content_text(message.get("content"))
        result = _json_object(message.get("content"))
        nested = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        merged = {**result, **nested}
        exit_code = _integer_field(merged, "exit_code", "exitCode", "returncode", "return_code", "code")
        explicit_error = merged.get("error") or merged.get("errors")
        raw_status = str(merged.get("status") or merged.get("outcome") or "").strip().lower()
        failed = bool(explicit_error) or (exit_code is not None and exit_code != 0) or raw_status in {
            "failed", "failure", "error", "blocked", "cancelled", "timeout", "timed_out",
        }
        passed = (exit_code == 0) or raw_status in {"passed", "success", "succeeded", "completed", "done", "ok"}
        status = "failed" if failed else "passed" if passed else "unknown"
        command = str(
            arguments.get("command") or arguments.get("cmd")
            or merged.get("command") or merged.get("cmd") or ""
        ).strip()[:4000]
        validation_text = f"{name}\n{command}\n{content[:1200]}"
        validation_kind = ""
        if _VALIDATION_PATTERN.search(validation_text):
            lowered = validation_text.lower()
            validation_kind = next((kind for kind in (
                "typecheck", "lint", "build", "compile", "syntax", "smoke", "test"
            ) if kind in lowered), "other")
        mutating = bool(_MUTATING_TOOL_PATTERN.search(name))
        changed_files = _string_list(
            merged.get("changed_files") or merged.get("changedFiles") or merged.get("files_changed")
        )
        artifact_ids = _string_list(merged.get("artifact_ids") or merged.get("artifactIds"), 100)
        kind = "validation" if validation_kind else "workspace" if mutating or changed_files else "artifact" if artifact_ids else "tool"
        summary = re.sub(r"\s+", " ", content).strip()[:4000]
        receipt_basis = json.dumps({
            "call_id": call_id,
            "name": name,
            "command": command,
            "exit_code": exit_code,
            "status": status,
            "summary": summary,
        }, sort_keys=True, ensure_ascii=False)
        receipt = {
            "receiptId": f"hermes-tool-{hashlib.sha256(receipt_basis.encode('utf-8')).hexdigest()[:32]}",
            "kind": kind,
            "tool": name[:200],
            "status": status,
            "summary": summary,
            "changedFiles": changed_files,
            "artifactIds": artifact_ids,
            "metadata": {"mutating": mutating, "toolCallId": call_id[:500]},
        }
        if command:
            receipt["command"] = command
        if exit_code is not None:
            receipt["exitCode"] = exit_code
        if validation_kind:
            receipt["validationKind"] = validation_kind
        receipts.append(receipt)
        if len(receipts) >= max_receipts:
            break

    before_revision = _workspace_revision(workspace_before)
    after_revision = _workspace_revision(workspace_after)
    if after_revision and before_revision != after_revision and len(receipts) < max_receipts:
        after_files = _string_list((workspace_after or {}).get("changedFiles"))
        basis = f"{before_revision}\n{after_revision}\n" + "\n".join(after_files)
        receipts.append({
            "receiptId": f"hermes-workspace-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:32]}",
            "kind": "workspace",
            "status": "passed",
            "summary": "Hermes workspace snapshot changed during this Host cycle.",
            "changedFiles": after_files,
            "artifactIds": [],
            **({"workspaceRevisionBefore": before_revision} if before_revision else {}),
            "workspaceRevisionAfter": after_revision,
            "metadata": {"mutating": True},
        })
    return receipts


def bounded_tool_summary(messages: Any, max_chars: int = 20_000, *, history_offset: int = 0) -> str:
    """Summarize actual structured execution receipts, never tool schemas."""
    receipts = structured_execution_receipts(messages, history_offset=history_offset)
    lines = []
    for receipt in receipts[-40:]:
        exit_suffix = (
            f" exit={receipt.get('exitCode')}"
            if receipt.get("exitCode") is not None
            else ""
        )
        summary_suffix = f" — {receipt.get('summary')}" if receipt.get("summary") else ""
        line = (
            f"{receipt.get('tool') or receipt.get('kind')}: {receipt.get('status')}"
            f"{exit_suffix}{summary_suffix}"
        )
        lines.append(line[:700])
    return "\n".join(lines)[-max_chars:]


def format_execution_receipt(receipt: Mapping[str, Any] | None, mode: str = "concise") -> str:
    if not isinstance(receipt, Mapping) or mode in {"off", "false", "none"}:
        return ""

    def role_text(role: str, label: str) -> str:
        value = receipt.get(role)
        entry = value if isinstance(value, Mapping) else {}
        if not entry.get("invoked"):
            return f"{label} skipped"
        model = str(entry.get("model") or "configured")
        verdict = str(entry.get("verdict") or "").strip()
        ok = entry.get("ok")
        suffix = f"/{verdict}" if verdict else ""
        if role == "host" and entry.get("plannerInvoked"):
            suffix += " · plan+final"
        if ok is False:
            suffix += " ✕"
        elif role != "host":
            suffix += " ✓"
        return f"{label} {model}{suffix}"

    pipeline = str(receipt.get("pipeline") or "unknown")
    transformed = bool(receipt.get("transformed"))
    if mode == "full":
        return "\n".join([
            "Runtime execution receipt",
            f"Pipeline: {pipeline}",
            role_text("host", "Host"),
            role_text("worker", "Worker"),
            role_text("verifier", "Verifier"),
            role_text("boss", "Boss"),
            f"Final draft transformed: {'yes' if transformed else 'no'}",
        ])
    return (
        f"Runtime · {pipeline} · "
        f"{role_text('host', 'Host')} · "
        f"{role_text('worker', 'Worker')} · "
        f"{role_text('verifier', 'Verifier')} · "
        f"{role_text('boss', 'Boss')}"
        + (" · revised" if transformed else "")
    )
