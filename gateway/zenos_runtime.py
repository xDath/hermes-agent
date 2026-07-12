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
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RUNTIME_URL = "http://127.0.0.1:3090"
DEFAULT_ROUTER_URL = "http://127.0.0.1:20128"
RUNTIME_ROLES = ("host", "worker", "boss")


def runtime_session_id(session_key: str) -> str:
    """Return a stable, non-sensitive Runtime id for one Hermes conversation."""
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
    return f"hermes_{digest}"


def _json_request(url: str, *, api_key: str = "", method: str = "GET", body: Any = None) -> Dict[str, Any]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
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


def _runtime_url() -> str:
    return os.getenv("ZENOS_RUNTIME_URL", DEFAULT_RUNTIME_URL).rstrip("/")


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
