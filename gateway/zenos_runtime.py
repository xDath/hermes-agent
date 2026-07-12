"""Small local client for Zenos Runtime model/session control.

This module intentionally uses the existing gateway edge rather than adding a
model tool.  `/wmodel` calls it from a worker thread so local HTTP never blocks
the Telegram event loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_RUNTIME_URL = "http://127.0.0.1:3090"
DEFAULT_ROUTER_URL = "http://127.0.0.1:20128"
RUNTIME_ROLES = ("host", "worker", "boss")
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


def usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, int]:
    """Return usage attributable to one turn, never cumulative session totals."""
    result = {
        key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))
        for key in _USAGE_COUNTERS
    }
    result["totalTokens"] = (
        result["inputTokens"]
        + result["cacheReadTokens"]
        + result["cacheWriteTokens"]
        + result["outputTokens"]
    )
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
            detail = json.loads(payload).get("error") or payload
        except Exception:
            detail = payload
        raise RuntimeError(f"Zenos request failed ({exc.code}): {detail[:500]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Zenos service unavailable: {exc}") from exc


def _runtime_url(override: str = "") -> str:
    return (override or os.getenv("ZENOS_RUNTIME_URL", DEFAULT_RUNTIME_URL)).rstrip("/")


def _runtime_key() -> str:
    key = os.getenv("ZENOS_RUNTIME_API_KEY", "").strip()
    if key:
        return key
    candidates = [
        os.getenv("ZENOS_RUNTIME_ENV_FILE", ""),
        "/root/openclaw-projects/zenos-runtime/.env.local",
        "/root/openclaw-projects/zenos-runtime/.env",
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


def get_runtime_models(session_id: str) -> Dict[str, Any]:
    return _json_request(
        f"{_runtime_url()}/api/runtime/models?sessionId={urllib.parse.quote(session_id)}",
        api_key=_runtime_key(),
    )


def save_runtime_models(session_id: str, roles: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    payload: Dict[str, str] = {}
    for role in RUNTIME_ROLES:
        entry = roles.get(role) or {}
        model = str(entry.get("model") or "").strip()
        provider = str(entry.get("provider") or "").strip()
        if not model or not provider:
            raise RuntimeError(f"{role} model and provider are required")
        payload[f"{role}Model"] = model
        payload[f"{role}Provider"] = provider
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


def list_runtime_combos() -> List[Dict[str, Any]]:
    base = os.getenv("NINE_ROUTER_URL", DEFAULT_ROUTER_URL).rstrip("/")
    data = _json_request(f"{base}/api/zenos-runtime/combos")
    combos = data.get("combos")
    return combos if isinstance(combos, list) else []


def role_config_from_response(data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    config = data.get("config") if isinstance(data.get("config"), dict) else data
    configured_roles = config.get("roles") if isinstance(config, dict) else {}
    result: Dict[str, Dict[str, str]] = {}
    for role in RUNTIME_ROLES:
        entry = configured_roles.get(role) if isinstance(configured_roles, dict) else {}
        result[role] = {
            "model": str((entry or {}).get("model") or "unknown"),
            "provider": str((entry or {}).get("provider") or "default"),
        }
    return result


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
        "receipt": str(section.get("receipt") or "concise").strip().lower(),
        "timeout_seconds": min(
            max(float(section.get("timeout_seconds") or DEFAULT_MIDDLEWARE_TIMEOUT), 10.0),
            600.0,
        ),
        "max_history_chars": min(
            max(int(section.get("max_history_chars") or 16_000), 0),
            120_000,
        ),
        "disable_streaming_when_verified": bool(
            section.get("disable_streaming_when_verified", True)
        ),
        "report_failures": bool(section.get("report_failures", True)),
    }


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


def infer_turn_context(
    message: str,
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
    workspace_root: str = "",
) -> Dict[str, Any]:
    """Derive conservative deterministic routing hints from one Hermes turn."""
    text = str(message or "")
    lower = text.lower()
    history_chars = sum(
        len(_content_text(item.get("content")))
        for item in (history or [])
        if isinstance(item, Mapping)
    )
    estimated_tokens = max(1, (len(text) + history_chars) // 4)
    code_terms = (
        "code", "coding", "repo", "repository", "file", "function", "class",
        "bug", "error", "stack trace", "typescript", "javascript", "python",
        "commit", "branch", "test", "lint", "build", "api", "endpoint",
        "kode", "ngoding", "perbaiki", "implement", "refactor",
    )
    mutation_terms = (
        "fix", "ubah", "edit", "buat", "bikin", "implement", "refactor",
        "hapus", "delete", "deploy", "restart", "push", "commit", "install",
    )
    log_terms = ("log", "journalctl", "traceback", "stack trace", "stdout", "stderr")
    verification_terms = (
        "verify", "pastikan", "cek bener", "are you sure", "yakin", "test", "uji",
    )
    boss_request_terms = (
        "tanya agent boss", "tanya boss", "panggil agent boss", "panggil boss",
        "minta agent boss", "minta boss", "suruh agent boss", "suruh boss",
        "boss review", "review sama boss", "ask agent boss", "ask the boss",
        "ask boss", "consult the boss", "escalate to boss",
    )
    fresh_terms = (
        "latest", "terbaru", "hari ini", "sekarang", "current", "news", "harga",
        "weather", "jadwal", "score", "status live",
    )
    execute_terms = (
        "jalankan", "run ", "deploy", "restart", "push", "commit", "hapus",
        "delete", "kirim", "send", "install", "update service",
    )
    has_code = any(term in lower for term in code_terms)
    has_mutation = any(term in lower for term in mutation_terms)
    intent = "analyze"
    if any(term in lower for term in execute_terms):
        intent = "execute"
    elif has_code and has_mutation:
        intent = "mutate"
    elif any(term in lower for term in ("rencana", "plan", "arsitektur", "design")):
        intent = "plan"
    elif any(term in lower for term in ("jelasin", "jelaskan", "apa itu", "explain")):
        intent = "explain"
    return {
        "hasFiles": bool(workspace_root and has_code),
        "hasLogs": any(term in lower for term in log_terms),
        "hasCodeChangeIntent": bool(has_code and has_mutation),
        "userRequestedVerification": any(term in lower for term in verification_terms),
        "userRequestedBoss": any(term in lower for term in boss_request_terms),
        "estimatedContextTokens": estimated_tokens,
        "confidence": 0.75,
        "intent": intent,
        "containsUntrustedInput": False,
        "requiresFreshData": any(term in lower for term in fresh_terms),
    }


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


def bounded_tool_summary(tools: Any, max_chars: int = 20_000) -> str:
    """Summarize this turn's tool evidence without forwarding raw long output."""
    if not isinstance(tools, list):
        return ""
    lines: List[str] = []
    for item in tools[-40:]:
        if isinstance(item, Mapping):
            name = str(item.get("name") or item.get("tool") or "tool")
            status = str(item.get("status") or "completed")
            result = item.get("result")
            preview = _content_text(result).replace("\n", " ").strip()[:400]
            lines.append(f"{name}: {status}{f' — {preview}' if preview else ''}")
        else:
            lines.append(str(item)[:500])
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
