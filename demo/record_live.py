#!/usr/bin/env python3
"""Record a REAL run of a command under a pseudo-terminal, keeping true timing.

    python demo/record_live.py demo/raw.json -- ./demo.sh

Each output chunk is stored with its monotonic timestamp (seconds since start).
Nothing is altered or synthesized; demo/make_cast.py turns the raw capture into
an asciinema v3 cast (compressing model wait time, adding reading pauses).
"""
import json
import os
import pty
import select
import struct
import sys
import time

COLS, ROWS = 110, 42


def record(cmd: list[str], out: str) -> None:
    chunks = []
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.environ.update(COLUMNS=str(COLS), LINES=str(ROWS), TERM="xterm-256color",
                          CHAT_NO_SPINNER="1")
        os.execvp(cmd[0], cmd)
        os._exit(1)
    try:
        import fcntl
        import termios
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    except Exception:
        pass
    start = time.monotonic()
    while True:
        try:
            r, _, _ = select.select([fd], [], [], 1.0)
        except (OSError, ValueError):
            break
        if fd in r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append([round(time.monotonic() - start, 3), data.decode("utf-8", "replace")])
    os.waitpid(pid, 0)
    with open(out, "w") as f:
        json.dump({"cols": COLS, "rows": ROWS, "command": cmd, "chunks": chunks}, f)
    print(f"wrote {out}: {len(chunks)} chunks, {chunks[-1][0] if chunks else 0:.0f}s real time")


if __name__ == "__main__":
    if "--" not in sys.argv:
        sys.exit("usage: record_live.py <raw.json> -- <command...>")
    record(sys.argv[sys.argv.index("--") + 1:], sys.argv[1])
