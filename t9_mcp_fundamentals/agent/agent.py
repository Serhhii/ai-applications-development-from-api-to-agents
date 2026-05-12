from typing import Any

from anthropic import AsyncAnthropic

from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient


class AgentMCPFundamentals:
    """Handles Claude model interactions and integrates with MCP client"""

    def __init__(self, api_key: str, model: str, tools: list[dict[str, Any]], mcp_client: MCPClient):
        self.model = model
        self.tools = tools
        self.mcp_client = mcp_client
        self.client = AsyncAnthropic(api_key=api_key)

    def _build_anthropic_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        system = ""
        anthropic_msgs = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            elif msg.role == Role.USER:
                anthropic_msgs.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                anthropic_msgs.append({"role": "assistant", "content": msg.content})
        return system, anthropic_msgs

    async def get_response(self, messages: list[Message]) -> Message:
        system, anthropic_messages = self._build_anthropic_messages(messages)
        return await self._run(system, anthropic_messages)

    async def _run(self, system: str, anthropic_messages: list[dict]) -> Message:
        content_text = ""

        print("🤖: ", end="", flush=True)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=self.tools,
            messages=anthropic_messages,
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
                content_text += text
            final = await stream.get_final_message()
        print()

        tool_uses = [block for block in final.content if block.type == "tool_use"]

        if not tool_uses:
            return Message(role=Role.ASSISTANT, content=content_text)

        anthropic_messages.append({"role": "assistant", "content": final.content})

        tool_results = []
        for tool_use in tool_uses:
            try:
                result = await self.mcp_client.call_tool(tool_use.name, dict(tool_use.input))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": f"Error: {e}",
                })

        anthropic_messages.append({"role": "user", "content": tool_results})
        return await self._run(system, anthropic_messages)
