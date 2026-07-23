"""Focused tests for the unified Hermes/Runtime `/wmodel` compatibility alias."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._handle_model_command = AsyncMock(return_value="model-updated")
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


@pytest.mark.asyncio
async def test_bare_wmodel_opens_the_single_model_selector():
    runner = _runner()

    result = await runner._handle_wmodel_command(_event("/wmodel"))

    assert result == "model-updated"
    forwarded = runner._handle_model_command.await_args.args[0]
    assert forwarded.text == "/model"
    assert forwarded.source.chat_id == "5072082650"


@pytest.mark.asyncio
async def test_wmodel_combo_maps_router_combo_to_the_single_model_choice():
    runner = _runner()

    result = await runner._handle_wmodel_command(_event("/wmodel combo grok"))

    assert result == "model-updated"
    forwarded = runner._handle_model_command.await_args.args[0]
    assert forwarded.text == "/model grok --provider etla-router"


@pytest.mark.asyncio
async def test_wmodel_direct_model_selection_forwards_without_creating_role_slots():
    runner = _runner()

    result = await runner._handle_wmodel_command(
        _event("/wmodel grok --provider etla-router")
    )

    assert result == "model-updated"
    forwarded = runner._handle_model_command.await_args.args[0]
    assert forwarded.text == "/model grok --provider etla-router"
