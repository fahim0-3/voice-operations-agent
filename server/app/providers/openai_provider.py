"""
OpenAI provider, via the standard Chat Completions streaming + tool-calling API.

why Chat Completions instead of the newer Responses API: Chat Completions
is still the most widely supported, best-documented surface and is what
most OpenAI-compatible third-party endpoints (Groq, Together, local
llama.cpp servers with an OpenAI-compatible shim, etc) implement - using
it here means this provider also works unmodified against those, not just
api.openai.com (see `base_url` override in config.py).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import LLMProvider, ToolCall


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI

        from app.config import settings

        self.model = model or settings.openai_model
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url or None,
        )

    def _to_wire_tools(self, generic_tools: list[dict]) -> list[Any]:
        # our canonical schema IS OpenAI's function-calling shape already.
        return generic_tools

    def _to_wire_history(self, history: list[dict]) -> list[Any]:
        return list(history)

    async def _stream_once(
        self, system_prompt: str, wire_messages: list[Any], wire_tools: list[Any]
    ) -> AsyncIterator[tuple[str, ToolCall | None]]:
        full_messages = [{"role": "system", "content": system_prompt}, *wire_messages]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=wire_tools or None,
            stream=True,
        )

        # OpenAI streams tool-call arguments as incremental string
        # fragments, indexed by tool_call position - same "accumulate,
        # then parse once complete" pattern as Bedrock (see bedrock.py),
        # just with a slightly different event shape.
        tool_id: str | None = None
        tool_name: str | None = None
        tool_args_json = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content, None
            if delta.tool_calls:
                call_delta = delta.tool_calls[0]
                if call_delta.id:
                    tool_id = call_delta.id
                if call_delta.function and call_delta.function.name:
                    tool_name = call_delta.function.name
                if call_delta.function and call_delta.function.arguments:
                    tool_args_json += call_delta.function.arguments

            if chunk.choices[0].finish_reason == "tool_calls" and tool_name:
                try:
                    args = json.loads(tool_args_json) if tool_args_json else {}
                except json.JSONDecodeError:
                    args = {}
                yield "", ToolCall(id=tool_id or "", name=tool_name, arguments=args)

    def _append_tool_exchange(
        self, wire_messages: list[Any], assistant_text: str, tool_call: ToolCall, tool_result: str
    ) -> list[Any]:
        return [
            *wire_messages,
            {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result},
        ]
