from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient


class HttpMCPClient(MCPClient):
    """Handles MCP server connection and tool execution via http"""

    def __init__(self, mcp_server_url: str) -> None:
        super().__init__()
        self.mcp_server_url = mcp_server_url
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self):
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamable_http_client(self.mcp_server_url)
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        result = await self.session.initialize()
        print(result)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
