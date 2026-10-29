"""Protocol-level tests against the two MCP servers directly, over a real
stdio transport (no LangGraph / LangChain / LLM involved) - these exercise
the actual JSON-RPC 2.0 wire behavior: tool/resource discovery, the soft vs.
hard error split, sandboxing, batch_process, and watch_directory.

Each test opens its own `async with stdio_client(...) as (r, w): async with
ClientSession(r, w) as session:` block end to end, matching the SDK's own
documented usage - splitting that across a pytest fixture's yield point trips
an anyio cancel-scope/task-identity check in teardown, so a plain contextlib
helper used inline in each test is both simpler and more robust here.

See tests/test_agent.py / test_tools.py for the agent-level (mcp_bridge /
LangGraph) tests, which run against these same servers one layer up.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")


@asynccontextmanager
async def server(script: str):
    params = StdioServerParameters(command=PYTHON, args=[os.path.join(ROOT, script)], cwd=ROOT)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def fs():
    return server("filesystem_mcp_server.py")


def db():
    return server("resume_db_mcp_server.py")


def _json(result) -> dict | list:
    return json.loads(result.content[0].text)


# --------------------------------------------------------------------------- #
# Discovery endpoints (resources/list, resources/templates/list, tools/list)
# --------------------------------------------------------------------------- #
async def test_filesystem_discovery():
    async with fs() as s:
        tools = {t.name for t in (await s.list_tools()).tools}
        assert tools == {"read_file", "list_files", "write_file", "search_in_file",
                         "batch_process", "watch_directory"}

        resources = {str(r.uri) for r in (await s.list_resources()).resources}
        assert resources == {"resumes://directory", "config://server"}

        templates = [t.uriTemplate for t in (await s.list_resource_templates()).resourceTemplates]
        assert templates == ["resume://{filename}"]


async def test_resume_db_discovery():
    async with db() as s:
        tools = {t.name for t in (await s.list_tools()).tools}
        assert tools == {"query_candidates", "get_candidate_record", "list_roles", "get_job_posting"}
        resources = {str(r.uri) for r in (await s.list_resources()).resources}
        assert resources == {"candidates://all", "jobs://all"}


# --------------------------------------------------------------------------- #
# Tools: soft errors (JSON-RPC success, domain failure in the payload) vs.
# hard errors (isError=True CallToolResult, for security/config violations)
# --------------------------------------------------------------------------- #
async def test_read_file_success_and_soft_error():
    async with fs() as s:
        ok = await s.call_tool("read_file", {"filepath": "resumes/John_Doe.txt"})
        assert ok.isError is False
        data = _json(ok)
        assert data["success"] is True and "John Doe" in data["content"]

        missing = await s.call_tool("read_file", {"filepath": "resumes/NOPE.txt"})
        assert missing.isError is False   # a domain-expected failure, not a protocol error
        data = _json(missing)
        assert data["success"] is False and "not found" in data["error"].lower()


async def test_path_traversal_is_a_hard_error():
    async with fs() as s:
        r = await s.call_tool("read_file", {"filepath": "../../etc/passwd"})
        assert r.isError is True
        assert "outside the project root" in r.content[0].text


async def test_write_file_sandbox_enforced():
    async with fs() as s:
        ok = await s.call_tool("write_file", {"filepath": "reports/__test_write.md", "content": "hi"})
        assert ok.isError is False and _json(ok)["success"] is True
        written = os.path.join(ROOT, "reports", "__test_write.md")
        assert os.path.isfile(written)
        os.remove(written)

        blocked = await s.call_tool("write_file", {"filepath": "resumes/hack.txt", "content": "x"})
        assert blocked.isError is True
        assert "only allowed under" in blocked.content[0].text
        assert not os.path.exists(os.path.join(ROOT, "resumes", "hack.txt"))


async def test_unknown_resource_is_a_json_rpc_error():
    async with fs() as s:
        with pytest.raises(McpError) as exc_info:
            await s.read_resource("resume://NOPE.txt")
        assert "Unknown resume" in str(exc_info.value)


async def test_config_resource():
    async with fs() as s:
        r = await s.read_resource("config://server")
        cfg = json.loads(r.contents[0].text)
        assert cfg["writable_dirs"] == ["reports"]
        assert cfg["resumes_dir"] == "resumes"


# --------------------------------------------------------------------------- #
# batch_process
# --------------------------------------------------------------------------- #
async def test_batch_process_partial_failure():
    async with fs() as s:
        r = await s.call_tool("batch_process", {"filepaths": [
            "resumes/John_Doe.txt", "resumes/Jane_Smith.txt", "resumes/NOPE.txt"]})
        data = _json(r)
        assert data["count"] == 3 and data["succeeded"] == 2
        assert data["elapsed_ms"] >= 0
        by_path = {res["filepath"]: res for res in data["results"]}
        assert by_path["resumes/John_Doe.txt"]["success"] is True
        assert by_path["resumes/NOPE.txt"]["success"] is False


async def test_batch_process_metadata_mode():
    async with fs() as s:
        r = await s.call_tool("batch_process",
                              {"filepaths": ["resumes/John_Doe.txt"], "operation": "metadata"})
        data = _json(r)
        row = data["results"][0]
        assert row["success"] is True and "size_bytes" in row and "content" not in row


# --------------------------------------------------------------------------- #
# watch_directory
# --------------------------------------------------------------------------- #
async def test_watch_directory_detects_new_file():
    """'start' launches a detached worker process (see filesystem_mcp_server.py
    - every tool call here is its own short-lived server subprocess, so an
    in-memory watcher would die with the 'start' call itself). Give the
    worker a moment to actually spin up its Observer before writing the probe
    file - this is a real process launch, not an in-process thread."""
    async with fs() as s:
        started = await s.call_tool("watch_directory", {"action": "start", "directory": "resumes"})
        assert _json(started)["watching"] is True
        await asyncio.sleep(1.0)   # worker process startup

        probe = os.path.join(ROOT, "resumes", "__watch_test_probe.txt")
        try:
            with open(probe, "w") as f:
                f.write("temp\n")
            await asyncio.sleep(2.0)   # let the watcher observe + debounce settle

            events = _json(await s.call_tool("watch_directory", {"action": "events", "directory": "resumes"}))
            names = [e["file"] for e in events["events"]]
            assert "__watch_test_probe.txt" in names
        finally:
            os.remove(probe)
            await s.call_tool("watch_directory", {"action": "stop", "directory": "resumes"})


async def test_watch_directory_status_and_idempotent_start():
    async with fs() as s:
        s1 = _json(await s.call_tool("watch_directory", {"action": "start", "directory": "resumes"}))
        s2 = _json(await s.call_tool("watch_directory", {"action": "start", "directory": "resumes"}))
        assert s1["already_running"] is False and s2["already_running"] is True
        status = _json(await s.call_tool("watch_directory", {"action": "status", "directory": "resumes"}))
        assert status["watching"] is True
        await s.call_tool("watch_directory", {"action": "stop", "directory": "resumes"})


# --------------------------------------------------------------------------- #
# resume_db (the bonus second MCP server)
# --------------------------------------------------------------------------- #
async def test_query_candidates_filters():
    async with db() as s:
        r = _json(await s.call_tool("query_candidates", {"role": "devops", "min_years": 10}))
        assert r["total_matches"] >= 1
        assert all(c["role"] == "devops" and c["years"] >= 10 for c in r["candidates"])


async def test_get_candidate_record_by_name_and_unknown():
    async with db() as s:
        ok = _json(await s.call_tool("get_candidate_record", {"candidate": "John Doe"}))
        assert ok["success"] is True and ok["candidate_id"] == "John_Doe"

        bad = _json(await s.call_tool("get_candidate_record", {"candidate": "Nobody Here"}))
        assert bad["success"] is False


async def test_resume_db_resources():
    async with db() as s:
        all_candidates = json.loads((await s.read_resource("candidates://all")).contents[0].text)
        all_jobs = json.loads((await s.read_resource("jobs://all")).contents[0].text)
        assert len(all_candidates) == 100
        assert len(all_jobs) == 8
