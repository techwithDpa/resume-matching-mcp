"""Terminal chat interface for the matching agent.

  python chat_cli.py                                # interactive
  python chat_cli.py --script flows/01_jd_full_report.txt
  python chat_cli.py --script flows/05_refine_rerank.txt --pace 1.5   # demo pacing
  python chat_cli.py --offline                      # no LLM: heuristic fallbacks

Each turn streams the agent's reasoning trace (graph node + tool calls) in grey,
then the reply rendered as markdown. Script files hold one user message per
line; blank lines and lines starting with '#' are ignored. A line starting
with '!' runs as a shell command instead of a chat turn (cwd = project root) -
used by the watch_directory demo flow to drop a new resume file onto disk
mid-conversation, so the agent's live watcher has something real to detect.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

console = Console()


def run_turn(session, text: str) -> None:
    console.print(f"\n[bold cyan]you ›[/] {escape(text)}")
    started = time.time()
    # CHAT_NO_SPINNER=1 (set by demo/record_live.py) keeps recordings free of spinner frames
    spinner = (contextlib.nullcontext() if os.environ.get("CHAT_NO_SPINNER")
               else console.status("[dim]thinking…[/]", spinner="dots"))
    with spinner:
        for ev in session.send(text):
            if ev["type"] == "trace":
                console.print(f"  [dim]· {escape(f'[{ev['node']}] ' + ev['text'])}[/]")
            elif ev["type"] == "message":
                console.print(Panel(Markdown(ev["content"]), title="agent", title_align="left",
                                    border_style="green"))
            elif ev["type"] == "end":
                console.print("[bold]Session ended.[/]")
    console.print(f"  [dim]({time.time() - started:.1f}s)[/]")


def load_script(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]


def run_shell_directive(line: str, base_dir: str) -> None:
    cmd = line[1:].strip()
    console.print(f"\n[bold yellow]$ {escape(cmd)}[/]  [dim](simulating a filesystem change for the demo)[/]")
    subprocess.run(cmd, shell=True, cwd=base_dir, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", help="replay user messages from a file")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to pause between scripted turns")
    ap.add_argument("--offline", action="store_true", help="no LLM calls (heuristic fallbacks)")
    args = ap.parse_args()
    if args.offline:
        os.environ["MATCHING_AGENT_OFFLINE"] = "1"

    import agent_tools
    import llm
    from matching_agent import MatchingSession

    mode = "offline heuristics" if llm.is_offline() else llm.model_name()
    console.print(Panel(
        "Resume Matching Agent — LangGraph · 3-round screening over 100 resumes\n"
        f"LLM: {mode}\n"
        "Start with a JD file (e.g. [bold]jobs/jd_ml_engineer.txt[/]) or a request like "
        "[bold]Find me candidates with React and 3+ years experience[/].\n"
        "Then ask questions, refine requirements, or type [bold]done[/].",
        border_style="cyan"))
    session = MatchingSession()

    if args.script:
        for line in load_script(args.script):
            if line.startswith("!"):
                run_shell_directive(line, agent_tools.BASE_DIR)
                continue
            if session.finished:
                break
            run_turn(session, line)
            time.sleep(args.pace)
        return

    while not session.finished:
        try:
            text = console.input("\n[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if text:
            # input() already echoed the line; avoid printing it twice
            sys.stdout.write("\033[F\033[K")
            run_turn(session, text)


if __name__ == "__main__":
    main()
