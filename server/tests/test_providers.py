"""
Provider abstraction tests.

why these tests use fakes instead of hitting real Bedrock/Ollama/OpenAI
APIs: the shared `chat_with_tools()` hop-loop in providers/base.py is
pure orchestration logic (streaming, tool dispatch, looping) - it should
be correct independent of which backend is plugged in. A FakeProvider
that yields scripted (delta, tool_call) pairs lets us test that loop
deterministically, in milliseconds, with no network/credentials required
- which also means these tests run in CI without AWS/OpenAI secrets.

Provider-specific wire-format translation (Bedrock's toolUse JSON
accumulation, OpenAI's tool_call delta indexing) is exercised separately
by test_bedrock_provider.py / test_openai_provider.py using recorded
event fixtures, not live calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.providers import VALID_PROVIDERS, create_provider
from app.providers.base import LLMProvider, ToolCall


class FakeProvider(LLMProvider):
    """A scripted provider: each call to _stream_once pops the next scripted hop."""

    name = "fake"

    def __init__(self, scripted_hops: list[list[tuple[str, ToolCall | None]]]):
        self._hops = scripted_hops
        self._hop_index = 0
        self.seen_tool_results: list[str] = []

    def _to_wire_tools(self, generic_tools: list[dict]) -> list[Any]:
        return generic_tools

    def _to_wire_history(self, history: list[dict]) -> list[Any]:
        return list(history)

    async def _stream_once(self, system_prompt, wire_messages, wire_tools) -> AsyncIterator[tuple[str, ToolCall | None]]:
        hop = self._hops[self._hop_index]
        self._hop_index += 1
        for delta, tool_call in hop:
            yield delta, tool_call

    def _append_tool_exchange(self, wire_messages, assistant_text, tool_call, tool_result):
        self.seen_tool_results.append(tool_result)
        return [*wire_messages, {"role": "assistant", "content": assistant_text}]


@pytest.mark.asyncio
async def test_chat_with_tools_returns_plain_text_when_no_tool_call():
    provider = FakeProvider(scripted_hops=[[("Hello", None), (" sir.", None)]])
    result = await provider.chat_with_tools(system_prompt="sys", history=[], tools=[])
    assert result == "Hello sir."


@pytest.mark.asyncio
async def test_chat_with_tools_executes_tool_and_continues(monkeypatch):
    import app.providers.base as base_module

    monkeypatch.setattr(base_module, "execute_tool", lambda name, args: "42 degrees", raising=False)

    # patch the deferred import target used inside chat_with_tools
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "execute_tool", lambda name, args: "42 degrees")

    call = ToolCall(id="1", name="get_weather", arguments={"city": "Paris"})
    provider = FakeProvider(
        scripted_hops=[
            [("Let me check...", None), ("", call)],
            [("It's 42 degrees.", None)],
        ]
    )
    result = await provider.chat_with_tools(system_prompt="sys", history=[], tools=[])
    assert result == "It's 42 degrees."
    assert provider.seen_tool_results == ["42 degrees"]


@pytest.mark.asyncio
async def test_chat_with_tools_respects_max_hops(monkeypatch):
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "execute_tool", lambda name, args: "result")

    call = ToolCall(id="1", name="loop_tool", arguments={})
    # every hop calls a tool again - should stop at max_hops, not loop forever
    provider = FakeProvider(scripted_hops=[[("", call)] for _ in range(10)])
    result = await provider.chat_with_tools(system_prompt="sys", history=[], tools=[], max_hops=3)
    assert result == ""  # never got a final plain-text hop
    assert provider._hop_index == 3


def test_create_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_provider(provider="not_a_real_provider")


def test_create_provider_bedrock_is_default():
    from app.config import Settings

    assert Settings.model_fields["llm_provider"].default == "bedrock"


def test_valid_providers_contains_all_three():
    assert set(VALID_PROVIDERS) == {"bedrock", "ollama", "openai"}
