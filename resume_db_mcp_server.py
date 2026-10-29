"""Second MCP server (bonus multi-MCP integration): a read-only "database"
over the candidate/job ground-truth metadata (dataset_meta.json), independent
of the filesystem server. The agent connects to both at once via
MultiServerMCPClient (see mcp_bridge.py), demonstrating multi-MCP resource use
as the brief suggests ("web search, database").

Run standalone:
    python resume_db_mcp_server.py                      # stdio (default)
    python resume_db_mcp_server.py --transport streamable-http

Configuration: MCP_DATASET_PATH (default: dataset_meta.json next to this file).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ErrorData

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s mcp.resume_db: %(message)s",
)
log = logging.getLogger("resume_db_mcp_server")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("MCP_DATASET_PATH", BASE_DIR / "dataset_meta.json"))

mcp = FastMCP(
    "resume_db",
    instructions=(
        "Read-only ground-truth database of candidates and job postings, independent "
        "of the filesystem server. Use query_candidates to filter by role/skill/years, "
        "get_candidate_record for one candidate's full record, and the candidates:// "
        "and jobs:// resources to browse everything."
    ),
)


def _load() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


_DATA = _load()
log.info("loaded %d candidates, %d jobs from %s",
          len(_DATA["resumes"]), len(_DATA["jobs"]), DATASET_PATH)


def _candidate_id(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _candidate_record(filename: str, rec: dict) -> dict:
    return {"candidate_id": _candidate_id(filename), "resume_file": filename, **rec}


@mcp.tool()
def query_candidates(role: str | None = None, seniority: str | None = None,
                     min_years: int | None = None, skill: str | None = None,
                     limit: int = 20) -> dict:
    """Filter the 100-candidate ground-truth database.
    role: e.g. 'backend', 'ml', 'devops' (see list_roles for the full set).
    seniority: 'Junior' | 'Mid' | 'Senior'. skill: substring match, e.g. 'Kubernetes'.
    Returns up to `limit` matches plus the total match count."""
    matches = []
    for filename, rec in _DATA["resumes"].items():
        if role and rec["role"] != role:
            continue
        if seniority and rec["seniority"] != seniority:
            continue
        if min_years is not None and rec["years"] < min_years:
            continue
        if skill and not any(skill.lower() in s.lower() for s in rec["skills"]):
            continue
        matches.append(_candidate_record(filename, rec))
    return {"total_matches": len(matches), "returned": min(limit, len(matches)),
            "candidates": matches[:limit]}


@mcp.tool()
def get_candidate_record(candidate: str) -> dict:
    """Full ground-truth record for one candidate, by id ('John_Doe'), resume
    filename, or full name ('John Doe')."""
    key = candidate.strip().lower()
    for filename, rec in _DATA["resumes"].items():
        if (_candidate_id(filename).lower() == key or filename.lower() == key
                or rec["name"].lower() == key):
            return {"success": True, **_candidate_record(filename, rec)}
    return {"success": False, "error": f"No candidate matches '{candidate}'"}


@mcp.tool()
def list_roles() -> dict:
    """Distinct candidate roles in the database, with counts (e.g. 'backend': 9)."""
    counts: dict[str, int] = {}
    for rec in _DATA["resumes"].values():
        counts[rec["role"]] = counts.get(rec["role"], 0) + 1
    return {"roles": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


@mcp.tool()
def get_job_posting(job_id: str) -> dict:
    """Ground-truth metadata for one job posting (id, title, target_roles,
    must_have skills, min_years), by id, e.g. 'jd_ml_engineer'."""
    rec = _DATA["jobs"].get(job_id)
    if rec is None:
        return {"success": False, "error": f"No job posting '{job_id}'",
                "available": sorted(_DATA["jobs"])}
    return {"success": True, **rec}


@mcp.resource("candidates://all")
def all_candidates() -> list:
    """Every candidate's ground-truth record."""
    return [_candidate_record(fn, rec) for fn, rec in _DATA["resumes"].items()]


@mcp.resource("jobs://all")
def all_jobs() -> list:
    """Every job posting's ground-truth record."""
    return list(_DATA["jobs"].values())


@mcp.resource("candidate://{candidate_id}")
def one_candidate(candidate_id: str) -> dict:
    """One candidate's ground-truth record, by candidate id."""
    result = get_candidate_record(candidate_id)
    if not result["success"]:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=result["error"],
                                  data={"uri": f"candidate://{candidate_id}"}))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume/job ground-truth database MCP server.")
    ap.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    args = ap.parse_args()
    log.info("starting resume_db MCP server: transport=%s dataset=%s", args.transport, DATASET_PATH)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
