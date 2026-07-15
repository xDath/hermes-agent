"""Small local client for Zenos Runtime model/session control.

This module intentionally uses the existing gateway edge rather than adding a
model tool.  `/wmodel` calls it from a worker thread so local HTTP never blocks
the Telegram event loop.
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


def apply_host_token_budget(agent: Any, budget: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Apply one Runtime-issued Host cap without mutating the cached prompt.

    The returned state must be passed to :func:`restore_host_token_budget` at
    the end of the turn so a cached agent never inherits another turn's cap.
    """
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


def apply_host_working_set_limit(agent: Any, soft_limit_tokens: int) -> Dict[str, Any]:
    """Apply one reversible Host working-set limit for the current turn only.

    Cached agents are reused across gateway turns. Every compressor field changed
    here is therefore captured and restored in ``finally``; otherwise one cheap
    chat turn can permanently trap a later coding turn behind a tiny threshold.
    """
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
    estimated_tokens = max(1, (len(text) + history_chars) // 4)
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
        and any(term in recent_history for term in code_terms)
        and any(term in recent_history for term in mutation_terms)
        and any(term in recent_history for term in unfinished_terms)
    )
    if continuation_code_change:
        has_code = True
        has_mutation = True
    intent = "analyze"
    if continuation_code_change:
        intent = "mutate"
    elif any(term in lower for term in execute_terms):
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


def workspace_root_from_text(text: str, *, allowed_parent: str = "/root/openclaw-projects") -> str:
    """Extract a repository path from bounded tool evidence."""
    raw = str(text or "")
    parent = Path(allowed_parent).expanduser().resolve()
    pattern = re.compile(r"(?:/root/openclaw-projects|/workspace)/[A-Za-z0-9._-]+")
    for match in pattern.finditer(raw):
        try:
            candidate = Path(match.group(0)).expanduser().resolve()
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
