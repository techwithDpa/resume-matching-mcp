"""MCP server exposing the Milestone 1 file-system tools over the Model
Context Protocol (JSON-RPC 2.0), plus two MCP-native capabilities:

  watch_directory() - monitors a directory for new/changed/removed resumes.
                       Each tool call is its own short-lived server process
                       (see mcp_bridge.py), so 'start' launches a small
                       DETACHED worker process that keeps a watchdog Observer
                       alive independently and logs changes to disk; 'status'
                       / 'events' / 'stop' are liveness checks and log reads,
                       correct no matter which call handles them. A call that
                       notices new events also fires a spec-compliant
                       notifications/resources/list_changed - necessarily
                       lazy (checked on the next call), since no session here
                       outlives a single tool call.
  batch_process()    - reads/stats many files in one round trip, concurrently,
                       instead of one read_file call per file.

Run standalone:
    python filesystem_mcp_server.py                          # stdio (default)
    python filesystem_mcp_server.py --transport streamable-http

Configuration is via environment variables (see Config below); MCP_PORT /
MCP_HOST only matter for the HTTP transports.

IMPORTANT: on the stdio transport, stdout IS the JSON-RPC channel. All
logging in this file goes to stderr (never print()).

Error handling: tools return the existing Milestone-1 soft-error shape
({"success": False, "error": ...}) for ordinary domain failures (file not
found, unsupported type) - the agent can read and react to those. A tool
raises only for security/config violations (path escaping the sandbox,
writing outside a writable directory, an oversized file); the MCP SDK turns
that into a CallToolResult with isError=True and a status message, which is
the protocol-correct way to signal a tool-execution failure. A *resource*
read for an unknown URI raises McpError directly, which IS a JSON-RPC
protocol-level error (code -32602 INVALID_PARAMS) - resources and tools use
the two different error channels the spec defines for each.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ErrorData
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import fs_tools

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s mcp.filesystem: %(message)s",
)
log = logging.getLogger("filesystem_mcp_server")


# --------------------------------------------------------------------------- #
# Configuration (environment-overridable)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Server configuration. Every field is overridable via an env var, so the
    same server binary can be pointed at a different sandbox/limits without
    code changes - e.g. `MCP_WRITABLE_DIRS=reports,out python filesystem_mcp_server.py`."""

    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("MCP_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve())
    resumes_dir: str = os.environ.get("MCP_RESUMES_DIR", "resumes")
    writable_dirs: tuple[str, ...] = tuple(
        d.strip() for d in os.environ.get("MCP_WRITABLE_DIRS", "reports").split(",") if d.strip())
    max_file_size_mb: float = float(os.environ.get("MCP_MAX_FILE_SIZE_MB", "25"))
    watch_debounce_seconds: float = float(os.environ.get("MCP_WATCH_DEBOUNCE_SECONDS", "0.5"))
    watch_max_lifetime_seconds: float = float(os.environ.get("MCP_WATCH_MAX_LIFETIME_SECONDS", "1800"))
    max_batch_workers: int = int(os.environ.get("MCP_MAX_BATCH_WORKERS", "8"))
    max_events_kept: int = int(os.environ.get("MCP_MAX_WATCH_EVENTS", "200"))
    host: str = os.environ.get("MCP_HOST", "127.0.0.1")
    port: int = int(os.environ.get("MCP_PORT", "8765"))

    def as_dict(self) -> dict:
        return {"project_root": str(self.project_root), "resumes_dir": self.resumes_dir,
                "writable_dirs": list(self.writable_dirs), "max_file_size_mb": self.max_file_size_mb,
                "watch_debounce_seconds": self.watch_debounce_seconds,
                "watch_max_lifetime_seconds": self.watch_max_lifetime_seconds,
                "max_batch_workers": self.max_batch_workers, "host": self.host, "port": self.port}


CONFIG = Config()

mcp = FastMCP(
    "filesystem",
    instructions=(
        "File-system access for the resume matching agent: read/list/search files "
        "under the project root, batch-read many files in one call, and watch a "
        "directory for new resumes. Writes are restricted to: "
        + ", ".join(CONFIG.writable_dirs)
    ),
    host=CONFIG.host,
    port=CONFIG.port,
)


# --------------------------------------------------------------------------- #
# Path safety: every tool resolves paths through here before touching disk
# --------------------------------------------------------------------------- #
def _resolve(relative_path: str) -> Path:
    """Resolve a path against the project root; raise if it would escape it."""
    candidate = (CONFIG.project_root / relative_path).resolve()
    try:
        candidate.relative_to(CONFIG.project_root)
    except ValueError:
        raise ValueError(
            f"'{relative_path}' resolves outside the project root ({CONFIG.project_root}); refused")
    return candidate


def _check_writable(path: Path) -> None:
    rel_parts = path.relative_to(CONFIG.project_root).parts
    if not rel_parts or rel_parts[0] not in CONFIG.writable_dirs:
        raise ValueError(
            f"writes are only allowed under {list(CONFIG.writable_dirs)}, not "
            f"'{os.sep.join(rel_parts) or '.'}' (configure MCP_WRITABLE_DIRS to change this)")


def _check_size(path: Path) -> None:
    if path.is_file() and path.stat().st_size > CONFIG.max_file_size_mb * 1024 * 1024:
        raise ValueError(f"{path.name} exceeds the {CONFIG.max_file_size_mb} MB limit")


def _stat(path: Path) -> dict:
    s = path.stat()
    return {"name": path.name, "path": str(path), "extension": path.suffix.lower(),
            "size_bytes": s.st_size,
            "modified": datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).isoformat()}


# --------------------------------------------------------------------------- #
# Tools: Milestone 1 parity
# --------------------------------------------------------------------------- #
@mcp.tool()
def read_file(filepath: str) -> dict:
    """Read a resume or text file (.txt/.pdf/.docx) and return its text + metadata.
    filepath is relative to the project root, e.g. 'resumes/John_Doe.txt'."""
    path = _resolve(filepath)
    _check_size(path)
    return fs_tools.read_file(str(path))


@mcp.tool()
def list_files(directory: str = "resumes", extension: str | None = None) -> list:
    """List files in a project directory (e.g. 'resumes', 'jobs', 'reports'),
    optionally filtered by extension (e.g. 'pdf')."""
    return fs_tools.list_files(str(_resolve(directory)), extension)


@mcp.tool()
def write_file(filepath: str, content: str) -> dict:
    """Write text to a file under a writable directory (default: 'reports/').
    Creates parent directories as needed."""
    path = _resolve(filepath)
    _check_writable(path)
    return fs_tools.write_file(str(path), content)


@mcp.tool()
def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict:
    """Case-insensitive keyword search inside one file, with surrounding context."""
    path = _resolve(filepath)
    _check_size(path)
    return fs_tools.search_in_file(str(path), keyword, context_chars)


# --------------------------------------------------------------------------- #
# Tool: batch_process - handle multiple files concurrently in one round trip
# --------------------------------------------------------------------------- #
@mcp.tool()
def batch_process(filepaths: list[str], operation: Literal["read", "metadata"] = "read") -> dict:
    """Process several files in one call instead of one read_file call per file.
    operation='read' extracts full text + metadata for each; 'metadata' only
    stats each file (fast, for a quick directory scan). Returns a per-file
    result list plus timing; one file's failure doesn't abort the batch."""
    started = time.monotonic()

    def one(rel: str) -> dict:
        try:
            path = _resolve(rel)
            _check_size(path)
        except ValueError as e:
            return {"filepath": rel, "success": False, "error": str(e)}
        if operation == "metadata":
            if not path.exists():
                return {"filepath": rel, "success": False, "error": f"File not found: {rel}"}
            return {"filepath": rel, "success": True, **_stat(path)}
        # fs_tools.read_file's own result dict also has a "filepath" key (the
        # absolute path it was called with) - put ours last so the caller's
        # original relative path wins, not the absolute one.
        return {**fs_tools.read_file(str(path)), "filepath": rel}

    workers = min(CONFIG.max_batch_workers, max(1, len(filepaths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, filepaths))

    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    succeeded = sum(1 for r in results if r.get("success"))
    log.info("batch_process(%s): %d/%d ok in %.1fms (%d workers)",
              operation, succeeded, len(results), elapsed_ms, workers)
    return {"operation": operation, "count": len(results), "succeeded": succeeded,
            "elapsed_ms": elapsed_ms, "workers": workers, "results": results}


# --------------------------------------------------------------------------- #
# Tool: watch_directory - monitor for new/changed/removed resumes
#
# Every MCP tool call here runs in its own short-lived server subprocess (see
# mcp_bridge.py: a fresh session - and, over stdio, a fresh subprocess - per
# call, per langchain-mcp-adapters' documented default). An in-memory watcher
# thread would die with the subprocess the instant the 'start' call returns,
# so 'status'/'events' from the NEXT call would always see nothing. Instead,
# 'start' launches a small detached WORKER PROCESS (this same script, run
# with --watch-worker) that keeps a watchdog Observer alive on its own, well
# past any single tool call, and appends what it sees to an on-disk JSONL
# log. 'status'/'events'/'stop' are then just liveness checks and log reads -
# correct no matter which subprocess instance handles them.
# --------------------------------------------------------------------------- #
WATCH_STATE_DIR = CONFIG.project_root / ".mcp_watch"
_last_notified: dict[str, int] = {}   # per-directory count, so list_changed fires once per new batch


def _watch_slug(directory: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", directory.strip("/")) or "root"


def _watch_paths(directory: str) -> tuple[Path, Path]:
    slug = _watch_slug(directory)
    return WATCH_STATE_DIR / f"{slug}.json", WATCH_STATE_DIR / f"{slug}.events.jsonl"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_events(events_path: Path) -> list[dict]:
    if not events_path.is_file():
        return []
    out = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-CONFIG.max_events_kept:]


async def _maybe_notify(ctx: Context | None, directory: str, events_path: Path) -> None:
    """Fire resources/list_changed if new events showed up since this
    directory was last checked. Necessarily lazy (checked when a tool call
    happens), not an instant push - no session outlives one tool call."""
    if ctx is None:
        return
    count = len(_read_events(events_path))
    if count > _last_notified.get(directory, 0):
        _last_notified[directory] = count
        try:
            await ctx.session.send_resource_list_changed()
        except Exception as e:
            log.warning("list_changed notification failed: %s", e)


class _WorkerEventHandler(FileSystemEventHandler):
    """Runs inside the detached worker process, on watchdog's own OS thread."""

    def __init__(self, events_path: Path):
        self.events_path = events_path
        self._last_seen: dict[str, float] = {}

    def _record(self, kind: str, src_path: str) -> None:
        now = time.time()
        if now - self._last_seen.get(src_path, 0) < CONFIG.watch_debounce_seconds:
            return  # debounce duplicate OS events from a single save
        self._last_seen[src_path] = now
        entry = {"type": kind, "file": os.path.basename(src_path), "at": time.strftime("%H:%M:%S")}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        if len(lines) > CONFIG.max_events_kept * 2:
            self.events_path.write_text("\n".join(lines[-CONFIG.max_events_kept:]) + "\n")

    def on_created(self, event):
        if not event.is_directory:
            self._record("created", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._record("modified", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._record("deleted", event.src_path)


def _watch_worker_main(directory: str) -> None:
    """Entry point for the detached worker process (python filesystem_mcp_server.py
    --watch-worker <directory>). No MCP protocol here at all - plain watchdog,
    running until SIGTERM (from watch_directory(action='stop')) or a safety
    time limit, in case a crashed client never calls stop."""
    events_path = _watch_paths(directory)[1]
    watch_dir = (CONFIG.project_root / directory).resolve()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    observer = Observer()
    observer.schedule(_WorkerEventHandler(events_path), str(watch_dir), recursive=False)
    observer.start()
    deadline = time.monotonic() + CONFIG.watch_max_lifetime_seconds
    try:
        while not stop["flag"] and time.monotonic() < deadline:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join(timeout=2)


@mcp.tool()
async def watch_directory(
    action: Literal["start", "stop", "status", "events"] = "status",
    directory: str = "resumes",
    ctx: Context | None = None,
) -> dict:
    """Monitor a directory for new/changed/removed resumes.
    action='start' launches a detached watcher process for the directory
    (idempotent if already running); 'events' returns the changes seen so
    far; 'status' reports whether it's running; 'stop' ends it. The server
    also emits an MCP resources/list_changed notification whenever a call
    notices events that weren't there last time this directory was checked."""
    path = _resolve(directory)
    if not path.is_dir():
        return {"success": False, "error": f"Not a directory: {directory}"}
    WATCH_STATE_DIR.mkdir(exist_ok=True)
    state_path, events_path = _watch_paths(directory)

    def read_state() -> dict | None:
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return state if _pid_alive(state["pid"]) else None

    if action == "start":
        existing = read_state()
        if existing is not None:
            return {"success": True, "watching": True, "directory": directory,
                    "already_running": True, "started_at": existing["started_at"]}
        events_path.unlink(missing_ok=True)   # a fresh watch starts with a clean log
        _last_notified.pop(directory, None)
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--watch-worker", directory],
            cwd=str(CONFIG.project_root), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        state_path.write_text(json.dumps({"pid": proc.pid, "directory": directory,
                                          "started_at": started_at}))
        log.info("watch_directory: started watching %s (worker pid %d)", directory, proc.pid)
        return {"success": True, "watching": True, "directory": directory,
                "already_running": False, "started_at": started_at}

    state = read_state()
    if state is None:
        return {"success": False, "watching": False, "directory": directory,
                "error": f"not watching '{directory}'; call action='start' first"}

    if action == "status":
        await _maybe_notify(ctx, directory, events_path)
        return {"success": True, "watching": True, "directory": directory,
                "started_at": state["started_at"], "events_recorded": len(_read_events(events_path))}
    if action == "events":
        await _maybe_notify(ctx, directory, events_path)
        events = _read_events(events_path)
        return {"success": True, "directory": directory, "count": len(events), "events": events}
    if action == "stop":
        try:
            os.kill(state["pid"], signal.SIGTERM)
        except OSError:
            pass
        state_path.unlink(missing_ok=True)
        log.info("watch_directory: stopped watching %s (worker pid %d)", directory, state["pid"])
        return {"success": True, "watching": False, "directory": directory}

    return {"success": False, "error": f"unknown action: {action}"}


# --------------------------------------------------------------------------- #
# Resources: discovery endpoints (resources/list, resources/templates/list)
# --------------------------------------------------------------------------- #
@mcp.resource("resumes://directory")
def resumes_directory() -> list:
    """All resume files under the configured resumes directory, with metadata."""
    return fs_tools.list_files(str(_resolve(CONFIG.resumes_dir)))


@mcp.resource("resume://{filename}")
def resume_text(filename: str) -> str:
    """The raw text of one resume, by filename (e.g. 'John_Doe.txt')."""
    try:
        path = _resolve(os.path.join(CONFIG.resumes_dir, filename))
    except ValueError as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e),
                                  data={"uri": f"resume://{filename}"})) from e
    if not path.is_file():
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown resume: {filename}",
                                  data={"uri": f"resume://{filename}"}))
    result = fs_tools.read_file(str(path))
    if not result["success"]:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=result["error"],
                                  data={"uri": f"resume://{filename}"}))
    return result["content"]


@mcp.resource("config://server")
def server_config() -> dict:
    """The server's current configuration (read-only)."""
    return CONFIG.as_dict()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Filesystem MCP server for the resume matching agent.")
    ap.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    ap.add_argument("--watch-worker", metavar="DIRECTORY", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.watch_worker:   # internal: spawned by watch_directory(action="start"), not run by hand
        _watch_worker_main(args.watch_worker)
        return
    log.info("starting filesystem MCP server: transport=%s root=%s writable=%s",
              args.transport, CONFIG.project_root, CONFIG.writable_dirs)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
