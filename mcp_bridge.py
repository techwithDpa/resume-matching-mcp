"""Bridge between the (synchronous) LangGraph agent and the two MCP servers
(filesystem_mcp_server.py, resume_db_mcp_server.py).

Both servers are spawned over stdio via `langchain-mcp-adapters`'
`MultiServerMCPClient`. Its documented default behavior opens a fresh
session - and, for stdio, a fresh server subprocess - per tool call, so there
is no persistent connection to manage here and no asyncio event-loop
lifetime issue to work around: every call in this module runs its own
`asyncio.run(...)`, start to finish, exactly like the library does
internally under `client.get_tools()`.

Two entry points:
  call(tool_name, **kwargs) -> result   low-level sync call, used directly by
                                         graph nodes (parse_jd, generate_report,
                                         round-2 resume reads) in place of the
                                         direct fs_tools calls Milestone 3 used
  tool_proxies() -> list[MCPToolProxy]  every tool from both servers, wrapped
                                         for the chat-tool-calling node
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = os.path.join(BASE_DIR, ".venv", "bin", "python")
PYTHON = _VENV_PY if os.path.isfile(_VENV_PY) else "python3"

SERVERS: dict[str, dict] = {
    "filesystem": {"command": PYTHON, "args": [os.path.join(BASE_DIR, "filesystem_mcp_server.py")],
                   "transport": "stdio"},
    "resume_db": {"command": PYTHON, "args": [os.path.join(BASE_DIR, "resume_db_mcp_server.py")],
                  "transport": "stdio"},
}


class MCPUnavailable(RuntimeError):
    """An MCP server call failed (couldn't spawn, unknown tool, transport error)."""


@lru_cache(maxsize=1)
def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient(SERVERS)


@lru_cache(maxsize=1)
def _async_tools() -> tuple:
    """All LangChain tools from both MCP servers. Fetching this opens one
    short-lived session per server just to run tools/list; nothing is kept
    open afterward - each tool's own .ainvoke() opens its own session later."""
    return tuple(asyncio.run(_client().get_tools()))


@lru_cache(maxsize=1)
def _tools_by_name() -> dict:
    return {t.name: t for t in _async_tools()}


def _unwrap(content_blocks: Any) -> Any:
    """MCP tool results arrive as a list of content blocks, e.g.
    [{'type': 'text', 'text': '<json>'}] for every tool in this project
    (they all return JSON-serializable dicts/lists). Parse that back into a
    plain Python value.

    A tool-execution failure (isError=True - e.g. our sandbox/writable-dir
    checks raising inside the tool) arrives the same way but with a plain
    English message instead of JSON, since the MCP SDK auto-wraps the raised
    exception as "Error executing tool <name>: <message>". That doesn't
    `json.loads`, so it's normalized into the same {"success": False,
    "error": ...} soft-error shape every tool in this project already uses -
    callers never need to special-case "the MCP layer failed" vs. "the tool
    itself reported failure"."""
    if isinstance(content_blocks, list) and content_blocks:
        first = content_blocks[0]
        text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (TypeError, json.JSONDecodeError):
                return {"success": False, "error": text}
    return content_blocks


def call(tool_name: str, **kwargs) -> Any:
    """Synchronously call one MCP tool by name (from either server). Each call
    spawns a fresh stdio session - the adapters' documented default - runs
    the tool, and tears it down."""
    tool = _tools_by_name().get(tool_name)
    if tool is None:
        raise MCPUnavailable(f"no MCP tool named '{tool_name}' "
                             f"(available: {sorted(_tools_by_name())})")
    try:
        return _unwrap(asyncio.run(tool.ainvoke(kwargs)))
    except Exception as e:
        raise MCPUnavailable(f"MCP call {tool_name}({kwargs}) failed: {type(e).__name__}: {e}") from e


@dataclass
class MCPToolProxy:
    """A sync-callable, Claude-tool-schema-ready wrapper around one MCP tool.
    Exposes the same (.name, .description, .args_schema, .invoke) surface the
    chat node's Python tools use, so matching_agent.chat_agent and
    agent_tools.tool_specs don't need to know a tool came from MCP."""

    name: str
    description: str
    args_schema: dict   # raw JSON Schema, as MCP/langchain-mcp-adapters gives it

    def invoke(self, args: dict) -> Any:
        return call(self.name, **args)


def tool_proxies() -> list[MCPToolProxy]:
    """Every tool from both MCP servers, ready for the chat node's tool loop."""
    return [MCPToolProxy(name=t.name, description=t.description or "",
                         args_schema=dict(t.args_schema))
            for t in _async_tools()]


def is_available() -> bool:
    try:
        _async_tools()
        return True
    except Exception:
        return False
