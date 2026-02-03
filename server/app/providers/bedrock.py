"""
AWS Bedrock provider, via the Converse/ConverseStream API.

why the Converse API instead of each model's native InvokeModel body
format: Bedrock's older InvokeModel API requires a different JSON request
shape *per model family* (Anthropic's shape != Amazon Titan/Nova's shape
!= Meta Llama's shape). Converse is Bedrock's newer unified interface -
one request/response shape works across Anthropic, Amazon Nova, Meta,
Mistral, etc, including a standardized tool-calling contract. That's what
lets a user swap `bedrock_model_id` between Claude and Nova and have
everything (streaming, tool calls) keep working unchanged.

why boto3 instead of an async AWS SDK: there isn't a mature first-party
async boto3 (aioboto3 is third-party and adds a dependency for one
provider). boto3's `converse_stream` returns a blocking generator, so we
run it on a background thread via `_sync_bridge.bridge_sync_iterator` -
see that module's docstring for why that matters for the event loop.

Verified at project setup time (see docs/16-decisions-adr.md) against the
user's own AWS account: streaming + tool-calling both work end-to-end on
anthropic.claude-3-haiku-20240307-v1:0 and amazon.nova-lite-v1:0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.providers._sync_bridge import bridge_sync_iterator
from app.providers.base import LLMProvider, ToolCall


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model_id: str | None = None, region: str | None = None) -> None:
        import boto3

        from app.config import settings

        self.model_id = model_id or settings.bedrock_model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region or settings.bedrock_region,
        )

    def _to_wire_tools(self, generic_tools: list[dict]) -> list[Any]:
        """OpenAI-function-shaped schema -> Bedrock's {toolSpec: {...}} shape."""
        tool_specs = [
            {
                "toolSpec": {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "inputSchema": {"json": t["function"]["parameters"]},
                }
            }
            for t in generic_tools
        ]
        return [{"tools": tool_specs}] if tool_specs else []

    def _to_wire_history(self, history: list[dict]) -> list[Any]:
        """
        Bedrock Converse messages are {role, content: [{text: ...}]} -
        our generic history is {role, content: "..."} plain strings, so
        we just wrap each string in a content block.
        """
        return [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history]

    async def _stream_once(
        self, system_prompt: str, wire_messages: list[Any], wire_tools: list[Any]
    ) -> AsyncIterator[tuple[str, ToolCall | None]]:
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": wire_messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": 512},
        }
        if wire_tools:
            kwargs["toolConfig"] = wire_tools[0]

        # boto3 is fully synchronous/blocking - run the network call itself
        # off-thread too, not just the iteration, so a slow connection
        # setup can't stall the event loop either.
        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._client.converse_stream(**kwargs))

        # Bedrock streams tool-call *input* JSON as raw string fragments
        # (contentBlockDelta.delta.toolUse.input) that must be concatenated
        # and parsed only once the block is complete - unlike text, you
        # cannot incrementally use partial tool-call JSON.
        pending_tool_name: str | None = None
        pending_tool_id: str | None = None
        pending_tool_json = ""

        async for event in bridge_sync_iterator(response["stream"]):
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]["start"]
                if "toolUse" in start:
                    pending_tool_name = start["toolUse"]["name"]
                    pending_tool_id = start["toolUse"]["toolUseId"]
                    pending_tool_json = ""

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"], None
                elif "toolUse" in delta:
                    pending_tool_json += delta["toolUse"].get("input", "")

            elif "contentBlockStop" in event:
                if pending_tool_name is not None:
                    try:
                        args = json.loads(pending_tool_json) if pending_tool_json else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield "", ToolCall(id=pending_tool_id or "", name=pending_tool_name, arguments=args)
                    pending_tool_name = None

    def _append_tool_exchange(
        self, wire_messages: list[Any], assistant_text: str, tool_call: ToolCall, tool_result: str
    ) -> list[Any]:
        assistant_content: list[dict] = []
        if assistant_text:
            assistant_content.append({"text": assistant_text})
        assistant_content.append(
            {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.name, "input": tool_call.arguments}}
        )

        return [
            *wire_messages,
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_call.id,
                            "content": [{"text": tool_result}],
                        }
                    }
                ],
            },
        ]
