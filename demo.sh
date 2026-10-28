#!/usr/bin/env bash
# Paced walkthrough of the agent for screen recording (~5-6 min live).
# Prioritizes the Milestone 4 (MCP) story: the filesystem MCP server, its
# batch_process/watch_directory capabilities, and the bonus second (database)
# MCP server, alongside one full 3-round screen and one refinement.
#   ./demo.sh            # live LLM (needs ANTHROPIC_API_KEY in .env)
#   ./demo.sh --offline  # heuristic fallbacks, no API calls (MCP stays live)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
PACE=${PACE:-2}

banner() { printf '\n\033[1;35m━━ %s ━━\033[0m\n' "$1"; sleep 1; }

banner "MCP architecture: agent <-> filesystem + resume_db servers"
cat docs/mcp_architecture.mmd | sed -n '/sequenceDiagram/,$p' | grep -E 'participant|->>|-->>|Note' | sed 's/^\s*/  /'
sleep "$PACE"

banner "MCP tool + resource discovery (both servers, live)"
$PY -c "
import mcp_bridge as B
tools = B.tool_proxies()
print(f'{len(tools)} tools across both MCP servers:')
for t in tools: print(f'  - {t.name}: {t.description.splitlines()[0]}')
"
sleep "$PACE"

banner "Flow 1 — JD file -> 3-round screen (Round 2 batches all resumes via one MCP batch_process call)"
$PY chat_cli.py "$@" --pace "$PACE" --script flows/01_jd_full_report.txt

banner "Flow 8 — MCP watch_directory: live resource monitoring"
$PY chat_cli.py "$@" --pace "$PACE" --script flows/08_watch_directory.txt

banner "Flow 9 — bonus multi-MCP: the resume_db ground-truth server"
$PY chat_cli.py "$@" --pace "$PACE" --script flows/09_batch_and_database.txt

banner "Flow 5 — refine requirements mid-conversation -> re-rank -> explain changes"
$PY chat_cli.py "$@" --pace "$PACE" --script flows/05_refine_rerank.txt

banner "Flow 6 — interview questions + save via the MCP write_file tool"
$PY chat_cli.py "$@" --pace "$PACE" --script flows/06_interview_questions.txt
