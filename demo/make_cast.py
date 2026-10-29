#!/usr/bin/env python3
"""Turn a raw live capture (record_live.py) into an asciinema v3 cast.

    python demo/make_cast.py demo/raw.json demo/demo.cast --target 330

Output text and order are exactly as captured. Only timing is edited:
  * waiting on the model is compressed (gaps capped at --cap seconds); the real
    per-turn latency is still printed by the CLI, e.g. "(39.1s)"
  * after each agent reply / banner a reading pause is inserted, proportional
    to the reply's length, scaled so the video lands near --target seconds
asciinema v3 casts use RELATIVE (inter-event) timestamps.
"""
import argparse
import json
import re

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def build(raw: dict, cap: float, scale: float) -> list[list]:
    events, prev_t, pending, panel_lines, in_panel = [], 0.0, 0.0, 0, False
    for t, text in raw["chunks"]:
        dt = max(min(t - prev_t, cap), pending)
        prev_t, pending = t, 0.0
        events.append([round(dt, 3), "o", text])
        plain = ANSI.sub("", text)
        for line in plain.splitlines():
            if line.startswith("╭"):
                in_panel, panel_lines = True, 0
            elif in_panel:
                panel_lines += 1
            if line.startswith("╰") and in_panel:
                in_panel = False
                pending = max(pending, scale * min(14.0, 2.0 + 0.22 * panel_lines))
            if "━━" in line:  # demo.sh section banner
                pending = max(pending, scale * 3.0)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("cast")
    ap.add_argument("--cap", type=float, default=0.6)
    ap.add_argument("--target", type=float, default=330.0, help="desired length in seconds")
    args = ap.parse_args()
    raw = json.load(open(args.raw))
    lo, hi = 0.05, 6.0      # binary-search the pause scale to hit the target length
    for _ in range(40):
        mid = (lo + hi) / 2
        if sum(e[0] for e in build(raw, args.cap, mid)) < args.target:
            lo = mid
        else:
            hi = mid
    events = build(raw, args.cap, lo)
    header = {"version": 3, "term": {"cols": raw["cols"], "rows": raw["rows"]},
              "title": "Resume Matching Agent - live run (Claude Sonnet 5.5)"}
    with open(args.cast, "w") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
    real = raw["chunks"][-1][0]
    print(f"wrote {args.cast}: {sum(e[0] for e in events):.0f}s (real run {real:.0f}s, pause scale {lo:.2f})")


if __name__ == "__main__":
    main()
