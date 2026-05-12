import os
import sys
import asyncio
import json
from pathlib import Path

from mcp import Resource
from mcp.types import Prompt

from commons.constants import ANTHROPIC_API_KEY
from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.agent import AgentMCPFundamentals
from t9_mcp_fundamentals.agent.mcp_clients.http import HttpMCPClient
from t9_mcp_fundamentals.agent.mcp_clients.stdio import StdioMCPClient
from t9_mcp_fundamentals.agent.prompts import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).parent.parent.parent
STDIO_SERVER_PATH = PROJECT_ROOT / "t9_mcp_fundamentals" / "mcp_server" / "stdio_server.py"


async def main():
    async with HttpMCPClient(mcp_server_url="http://localhost:8005/mcp") as mcp_client:
        resources: list[Resource] = await mcp_client.get_resources()
        print(f"\nAvailable Resources: {[str(r.uri) for r in resources]}")

        tools = await mcp_client.get_tools_anthropic()
        print(f"Available Tools: {[t['name'] for t in tools]}")

        agent = AgentMCPFundamentals(
            api_key=ANTHROPIC_API_KEY,
            model="claude-sonnet-4-6",
            tools=tools,
            mcp_client=mcp_client,
        )

        messages = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]

        prompts: list[Prompt] = await mcp_client.get_prompts()
        print(f"Available Prompts: {[p.name for p in prompts]}\n")

        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() == "exit":
                break
            messages.append(Message(role=Role.USER, content=user_input))
            ai_message = await agent.get_response(messages)
            messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())
