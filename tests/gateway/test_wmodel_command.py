"""Focused tests for the transactional Zenos Runtime `/wmodel` command."""

import types
from unittest.mock import AsyncMock, MagicMock

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
    runner._apply_wmodel_host_model = AsyncMock()
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

    runner = _runner(adapter)
    result = await runner._handle_wmodel_command(_event("/wmodel"))

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
    runner._apply_wmodel_host_model.assert_awaited_once()
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

    runner = _runner()
    result = await runner._handle_wmodel_command(_event("/wmodel combo coding"))

    assert "Runtime combo coding saved" in result
    assert saved and saved[0][1] == CURRENT
    runner._apply_wmodel_host_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_wmodel_host_write_through_updates_the_actual_hermes_session(monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "old-host",
        {
            "provider": "etla-router",
            "base_url": "http://router.test/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"providers": {"etla-router": {}}},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: ModelSwitchResult(
            success=True,
            new_model="grok-next",
            target_provider="etla-router",
            provider_changed=False,
            api_key="test-key",
            base_url="http://router.test/v1",
            api_mode="chat_completions",
            provider_label="Etla Router",
            is_global=False,
        ),
    )

    source = _event("/wmodel").source
    await runner._apply_wmodel_host_model(
        source=source,
        session_key="agent:main:telegram:dm:5072082650",
        model="grok-next",
        provider="etla-router",
    )

    override = runner._session_model_overrides[
        "agent:main:telegram:dm:5072082650"
    ]
    assert override["model"] == "grok-next"
    assert override["provider"] == "etla-router"
    runner.session_store.set_model_override.assert_called_once()
    runner._evict_cached_agent.assert_called_once_with(
        "agent:main:telegram:dm:5072082650"
    )
