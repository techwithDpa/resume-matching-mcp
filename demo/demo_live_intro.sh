#!/usr/bin/env bash
# The genuinely-live part of the demo: MCP tool/resource discovery and the
# watch_directory protocol test, run fresh every time this script runs.
# Neither needs the LLM, so both work with zero Anthropic credit - only the
# conversational flows (replayed separately; see make_replay_raw.py) need that.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

banner() { printf '\n\033[1;35m━━ %s ━━\033[0m\n' "$1"; sleep 1; }

banner "MCP architecture: agent <-> filesystem + resume_db servers"
cat docs/mcp_architecture.mmd | sed -n '/sequenceDiagram/,$p' | grep -E 'participant|->>|-->>|Note' | sed 's/^\s*/  /'
sleep 2

banner "MCP tool + resource discovery (both servers, live right now)"
MCP_LOG_LEVEL=WARNING $PY -c "
import mcp_bridge as B
tools = B.tool_proxies()
print(f'{len(tools)} tools across both MCP servers:')
for t in tools: print(f'  - {t.name}: {t.description.splitlines()[0]}')
"
sleep 2

banner "watch_directory, live right now: detached worker process, state survives across calls"
rm -rf .mcp_watch
$PY -m pytest -v tests/test_mcp_server.py -k "watch_directory"
rm -rf .mcp_watch
