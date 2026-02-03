"""
Common LLM provider abstraction.

why an abstraction at all: Voice Operations Agent supports three very different LLM
backends - a fully local Ollama server, AWS Bedrock (managed, multi-model,
what we use by default - see docs/16-decisions-adr.md for why), and
OpenAI's API. Each speaks a different wire format for messages, streaming
events, and tool-calling. Without this abstraction, pipeline.py would need
if/elif branches sprinkled through the whole turn loop every time we
support a new backend. Instead, every provider implements the same four
small methods below, and the shared `chat_with_tools()` hop-loop (call
model -> maybe run a tool -> feed the result back -> repeat) lives in
exactly one place, shared by all three.

why not just use LangChain/LiteLLM for this: for a portfolio project,
hand-rolling ~90 lines per provider demonstrates the underlying protocols
(Bedrock's Converse event stream, OpenAI's tool-call delta accumulation)
far better than delegating to a framework that hides them - and it keeps
the dependency footprint small. See docs/16-decisions-adr.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

TokenCallback = Callable[[str], Awaitable[None]]
ToolCallCallback = Callable[[str, dict], Awaitable[None]]
ToolResultCallback = Callable[[str, str], Awaitable[None]]


@dataclass
class ToolCall:
    """Generic tool-call representation, decoupled from any one provider's wire format."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Subclasses implement the provider-specific wire-format plumbing;
    `chat_with_tools` (the actual agent loop) is shared here so all three
    backends behave identically from pipeline.py's point of view - it
    never needs to know which provider it's talking to.
    """

    name: str = "base"

    @abstractmethod
    def _stream_once(
        self,
        system_prompt: str,
        wire_messages: list[Any],
        wire_tools: list[Any],
    ) -> AsyncIterator[tuple[str, ToolCall | None]]:
        """
        Run exactly one streamed generation call. Yields (text_delta,
        tool_call) pairs; tool_call is None on every chunk except the one
        (at most one per hop, by design) where a complete tool call has
        just finished streaming in.
        """
        raise NotImplementedError

    @abstractmethod
    def _to_wire_tools(self, generic_tools: list[dict]) -> list[Any]:
        """Convert our canonical OpenAI-function-style TOOL_SCHEMAS into this provider's tool format."""
        raise NotImplementedError

    @abstractmethod
    def _to_wire_history(self, history: list[dict]) -> list[Any]:
        """Convert generic [{'role', 'content'}, ...] history into this provider's message format."""
        raise NotImplementedError

    @abstractmethod
    def _append_tool_exchange(
        self,
        wire_messages: list[Any],
        assistant_text: str,
        tool_call: ToolCall,
        tool_result: str,
    ) -> list[Any]:
        """
        Append the assistant's tool-call message + our tool's result onto
        `wire_messages`, in whatever shape this provider expects to see
        for the *next* generation call in the hop loop.
        """
        raise NotImplementedError

    async def chat_with_tools(
        self,
        system_prompt: str,
        history: list[dict],
        tools: list[dict],
        on_token: TokenCallback | None = None,
        on_tool_call: ToolCallCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
        max_hops: int = 3,
    ) -> str:
        """
        The agent loop: stream a reply; if the model asks for a tool, run
        it locally (via app.tools.execute_tool) and feed the result back
        for a follow-up generation - looping until the model produces a
        plain-text answer or we hit `max_hops` (a safety net against a
        model that decides to call tools forever - see
        docs/07-tool-calling.md "why cap the loop").
        """
        from app.tools import execute_tool  # deferred to avoid an import cycle at module load

        wire_messages = self._to_wire_history(history)
        wire_tools = self._to_wire_tools(tools)
        final_text = ""

        for _hop in range(max_hops):
            hop_text = ""
            pending_tool_call: ToolCall | None = None

            async for delta, tool_call in self._stream_once(system_prompt, wire_messages, wire_tools):
                if delta:
                    hop_text += delta
                    if on_token:
                        await on_token(delta)
                if tool_call:
                    pending_tool_call = tool_call

            if not pending_tool_call:
                final_text = hop_text
                break

            if on_tool_call:
                await on_tool_call(pending_tool_call.name, pending_tool_call.arguments)

            result = execute_tool(pending_tool_call.name, pending_tool_call.arguments)

            if on_tool_result:
                await on_tool_result(pending_tool_call.name, result)

            wire_messages = self._append_tool_exchange(wire_messages, hop_text, pending_tool_call, result)

        return final_text
