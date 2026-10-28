"""Export the agent's diagrams.

  python diagram.py      # writes docs/state_machine.{mmd,png} and
                          # docs/mcp_architecture.{mmd,png}

The state-machine Mermaid source is generated from the compiled LangGraph, so
it always matches the code. The MCP architecture diagram is hand-authored (it
shows process/transport boundaries a graph object doesn't carry) but rendered
through the same langchain_core helper, so both go through the same
mermaid.ink API call. If that's unreachable, the .mmd files are still written
(GitHub renders Mermaid natively).
"""

from __future__ import annotations

import os

from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from matching_agent import build_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE_DIR, "docs")

MCP_ARCHITECTURE_MMD = """\
sequenceDiagram
    participant U as Recruiter
    participant G as LangGraph agent<br/>(matching_agent.py)
    participant B as mcp_bridge.py<br/>(sync <-> async)
    participant FS as filesystem MCP server<br/>(stdio, JSON-RPC 2.0)
    participant DB as resume_db MCP server<br/>(stdio, JSON-RPC 2.0, bonus)
    participant Disk as resumes/ jobs/ reports/
    participant Data as dataset_meta.json

    U->>G: "jobs/jd_ml_engineer.txt"
    G->>B: call("read_file", filepath=...)
    B->>FS: tools/call read_file (spawns server subprocess)
    FS->>Disk: open + parse (.txt/.pdf/.docx)
    Disk-->>FS: text
    FS-->>B: CallToolResult {success, content, metadata}
    B-->>G: parsed dict

    Note over G,FS: Round 2 - one batch_process call instead of<br/>one read_file round trip per shortlisted candidate
    G->>B: call("batch_process", filepaths=[10 resumes])
    B->>FS: tools/call batch_process
    FS->>Disk: ThreadPoolExecutor: read all 10 concurrently
    FS-->>B: {results: [10x {filepath, success, content}]}
    B-->>G: {candidate_id: text, ...}

    G->>B: call("write_file", filepath="reports/...md")
    B->>FS: tools/call write_file (sandbox + writable-dir check)
    FS->>Disk: write
    FS-->>B: {success, bytes_written}

    Note over FS,Disk: watch_directory("start") spawns a watchdog<br/>Observer thread; each change is logged and pushed as a<br/>notifications/resources/list_changed event to the client
    Disk--)FS: filesystem event (new resume dropped)
    FS--)G: notifications/resources/list_changed

    U->>G: "How many devops candidates have 8+ years?"
    G->>B: tool_proxies() -> query_candidates(...)
    B->>DB: tools/call query_candidates
    DB->>Data: filter in-memory records
    Data-->>DB: matches
    DB-->>B: {total_matches, candidates: [...]}
    B-->>G: parsed dict
    G-->>U: answer, grounded in both MCP servers' output
"""


def _write_mermaid(mmd: str, name: str) -> None:
    path = os.path.join(DOCS, f"{name}.mmd")
    with open(path, "w", encoding="utf-8") as f:
        f.write(mmd)
    print(f"wrote docs/{name}.mmd")
    try:
        png = draw_mermaid_png(mmd, max_retries=3, retry_delay=2.0)
        with open(os.path.join(DOCS, f"{name}.png"), "wb") as f:
            f.write(png)
        print(f"wrote docs/{name}.png")
    except Exception as e:
        print(f"PNG render skipped for {name} ({type(e).__name__}: {e})")


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)
    _write_mermaid(build_graph().get_graph().draw_mermaid(), "state_machine")
    _write_mermaid(MCP_ARCHITECTURE_MMD, "mcp_architecture")


if __name__ == "__main__":
    main()
