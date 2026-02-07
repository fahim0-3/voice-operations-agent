"""
Ollama provider - fully local inference, no network dependency once the
model is pulled. See docs/06-llm-inference.md for what TTFT/decode-speed
mean in this context, and the root README for why this is the "airplane
mode" fallback path when a user has no AWS/OpenAI credentials.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import LLMProvider, ToolCall


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        from app.config import settings

        self.model = model or settings.ollama_model
        self._client = ollama.AsyncClient(host=host or settings.ollama_host)

    def _to_wire_tools(self, generic_tools: list[dict]) -> list[Any]:
        # Ollama's tool-calling API accepts our canonical OpenAI-function
        # shape unmodified - no translation needed.
        return generic_tools

    def _to_wire_history(self, history: list[dict]) -> list[Any]:
        return list(history)

    async def _stream_once(
        self, system_prompt: str, wire_messages: list[Any], wire_tools: list[Any]
    ) -> AsyncIterator[tuple[str, ToolCall | None]]:
        full_messages = [{"role": "system", "content": system_prompt}, *wire_messages]
        stream = await self._client.chat(
            model=self.model,
            messages=full_messages,
            tools=wire_tools,
            stream=True,
        )
        async for chunk in stream:
            msg = chunk.get("message", {})
            delta = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")
            tool_call = None
            if tool_calls:
                fn = tool_calls[0]["function"]
                tool_call = ToolCall(id="ollama-0", name=fn["name"], arguments=fn.get("arguments", {}))
            yield delta, tool_call

    def _append_tool_exchange(
        self, wire_messages: list[Any], assistant_text: str, tool_call: ToolCall, tool_result: str
    ) -> list[Any]:
        return [
            *wire_messages,
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {"function": {"name": tool_call.name, "arguments": tool_call.arguments}}
                ],
            },
            {"role": "tool", "content": tool_result},
        ]
