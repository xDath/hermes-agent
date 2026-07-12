"""Focused tests for the transactional Zenos Runtime `/wmodel` command."""

import types

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


CURRENT = {
    "host": {"model": "grok", "provider": "etla-router"},
    "worker": {"model": "build", "provider": "etla-router"},
    "boss": {"model": "codex", "provider": "etla-router"},
}


class _FakeWmodelAdapter:
    def __init__(self):
        self.kwargs = None

    async def send_wmodel_picker(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(success=True)


def _runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: (
        f"agent:main:telegram:dm:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda _event: None
    return runner


def _event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="5072082650",
            chat_type="dm",
        ),
    )


def _model_response():
    return {
        "config": {
            "roles": {
                role: {"model": value["model"], "provider": value["provider"]}
                for role, value in CURRENT.items()
            }
        }
    }


@pytest.mark.asyncio
async def test_bare_wmodel_opens_transactional_picker(monkeypatch):
    adapter = _FakeWmodelAdapter()
    saved = []
    monkeypatch.setattr("gateway.zenos_runtime.get_runtime_models", lambda _sid: _model_response())
    monkeypatch.setattr(
        "gateway.zenos_runtime.save_runtime_models",
        lambda sid, roles: saved.append((sid, roles)) or {"ok": True},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **_kwargs: [{
            "slug": "etla-router",
            "name": "Etla Router",
            "models": ["grok", "build", "codex"],
            "total_models": 3,
        }],
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"providers": {}})

    result = await _runner(adapter)._handle_wmodel_command(_event("/wmodel"))

    assert result is None
    assert adapter.kwargs["current_roles"] == CURRENT
    assert adapter.kwargs["runtime_session_id"].startswith("hermes_")
    assert saved == []

    draft = {role: dict(value) for role, value in CURRENT.items()}
    draft["worker"] = {"model": "build-fast", "provider": "etla-router"}
    confirmation = await adapter.kwargs["on_save"](
        "5072082650", adapter.kwargs["runtime_session_id"], draft
    )

    assert saved == [(adapter.kwargs["runtime_session_id"], draft)]
    assert "Wmodel saved" in confirmation
    assert "build-fast" in confirmation


@pytest.mark.asyncio
async def test_combo_command_applies_three_roles(monkeypatch):
    saved = []
    monkeypatch.setattr("gateway.zenos_runtime.get_runtime_models", lambda _sid: _model_response())
    monkeypatch.setattr(
        "gateway.zenos_runtime.list_runtime_combos",
        lambda: [{"name": "coding", "roles": CURRENT}],
    )
    monkeypatch.setattr(
        "gateway.zenos_runtime.save_runtime_models",
        lambda sid, roles: saved.append((sid, roles)) or {"ok": True},
    )

    result = await _runner()._handle_wmodel_command(_event("/wmodel combo coding"))

    assert "Runtime combo coding saved" in result
    assert saved and saved[0][1] == CURRENT
