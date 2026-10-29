#!/usr/bin/env python3
"""Assemble the final demo recording from two kinds of source material:

  1. demo/live_intro_raw.json - genuinely live right now (MCP tool/resource
     discovery, the watch_directory protocol test). Neither needs an LLM, so
     both run for real regardless of Anthropic credit.
  2. Captured transcripts of an earlier FULLY LIVE run of flows 01, 05, 06, 09
     against Claude Sonnet 5.5 (before that session's credit ran out) -
     REPLAYED here verbatim, not re-run. Flow 08 (watch_directory through the
     chat UI) is intentionally not replayed: its fix landed after that run,
     so instead the live protocol test above stands in as the correct,
     verifiable evidence for that capability.

Output: demo/raw.json, in the same {cols, rows, command, chunks} shape
record_live.py produces, so the existing demo/make_cast.py renders it
unchanged into an asciinema cast.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "replay_logs"
OUT = HERE / "raw.json"

NOISE = re.compile(
    r"^(W\d|.*torch/distributed.*redirects|.*Processing request|.*INFO mcp\.|"
    r"Warning: You are sending unauthenticated|Loading weights:)"
)

BANNER = "\033[1;35m\r\n━━ {title} ━━\033[0m\r\n\r\n"

FLOWS = [
    ("01_jd_full_report.log",
     "Flow 1 (replayed) — JD file -> 3-round screen (Round 2 batches all resumes via one MCP batch_process call)"),
    ("05_refine_rerank.log",
     "Flow 5 (replayed) — refine requirements mid-conversation -> re-rank -> explain changes"),
    ("06_interview_questions.log",
     "Flow 6 (replayed) — interview questions + save via the MCP write_file tool"),
    ("09_batch_and_database.log",
     "Flow 9 (replayed) — bonus multi-MCP: the resume_db ground-truth server"),
]


def clean(text: str) -> str:
    """Strip noise lines and convert bare LF to CRLF. The source logs were
    captured via plain `> file 2>&1` redirection (no pty), so rich/Python
    emitted bare '\\n'. Replayed as if it were live terminal output (which
    always carries '\\r\\n'), a lone '\\n' only moves down a line without
    returning to column 0, producing a corrupt staircase render."""
    lines = [l for l in text.splitlines() if not NOISE.match(l)]
    return "\r\n".join(lines) + "\r\n"


def split_turns(text: str) -> list[str]:
    """One chunk per 'you › ...' turn, including its trace/panel/timing."""
    turns, current = [], []
    for line in text.splitlines(keepends=True):
        if line.startswith("you ›") and current:
            turns.append("".join(current))
            current = []
        current.append(line)
    if current:
        turns.append("".join(current))
    return [t for t in turns if t.strip()]


def main() -> None:
    live = json.loads((HERE / "live_intro_raw.json").read_text())
    chunks: list[list] = list(live["chunks"])
    t = chunks[-1][0] + 3.0 if chunks else 0.0

    for filename, title in FLOWS:
        path = LOGS_DIR / filename
        if not path.is_file():
            print(f"skip (missing): {path}")
            continue
        text = clean(path.read_text(encoding="utf-8"))
        chunks.append([round(t, 3), BANNER.format(title=title)])
        t += 1.5
        for turn in split_turns(text):
            chunks.append([round(t, 3), turn])
            t += 2.5   # spacing between turns; make_cast.py adds the real reading pauses
        print(f"added {filename}: {len(split_turns(text))} turns")

    out = {"cols": live["cols"], "rows": live["rows"],
          "command": ["replay", "of", "a", "live", "run"], "chunks": chunks}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({len(chunks)} chunks, {t:.0f}s synthetic timeline)")


if __name__ == "__main__":
    main()
