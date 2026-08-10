#!/usr/bin/env python3
"""PreToolUse guard for the autoresearch run — protect the measurement.

Reads the tool-call JSON on stdin and exits 2 (block) when the call would:
  * EDIT the measurement — anything under tests/ or data/, or the benchmark
    harness experiments/bench_training.py; or
  * READ the *target values* — test assertions (tests/test_*.py),
    reference numbers (data/**/truth.json, data/**/ref_ls/**), incl. obvious
    shell reads of them.

It deliberately ALLOWS reading the input data (data/**/obs.csv, rct.csv — the
agent must fit these) and the harness source (it must run it). Read-blocking is
a deterrent, not airtight: pytest and any Python the agent runs read files at the
OS level, which a tool-hook can't intercept. The load-bearing anti-overfitting
guard remains the mission's rule that a win must generalize off the harness.

Activate on the research machine only by copying
.claude/settings.local.example.json -> .claude/settings.local.json (git-ignored),
so it never affects other checkouts. Exit 0 = allow; fails open on any error so a
malformed event can't wedge the agent.
"""

import json
import re
import sys


def block(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


try:
    event = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # fail open — never wedge the agent on a parse error

tool = event.get("tool_name", "")
ti = event.get("tool_input", {}) or {}

WRITE_DENY = (
    re.compile(r"(^|/)tests/"),
    re.compile(r"(^|/)data/"),
    re.compile(r"(^|/)experiments/bench_training\.py$"),
)
READ_DENY = (
    re.compile(r"(^|/)tests/test_[^/]*\.py$"),
    re.compile(r"(^|/)data/.*/truth\.json$"),
    re.compile(r"(^|/)data/.*/ref_ls/"),
)

if tool in ("Write", "Edit", "NotebookEdit"):
    p = str(ti.get("file_path", "")).replace("\\", "/")
    if any(rx.search(p) for rx in WRITE_DENY):
        block(
            "BLOCKED (autoresearch integrity): tests/, data/, and the benchmark "
            "harness are the measurement — you may not edit them. If a test "
            "fails, your change is wrong, not the test. Changing the measurement "
            "is not a result."
        )

if tool == "Read":
    p = str(ti.get("file_path", "")).replace("\\", "/")
    if any(rx.search(p) for rx in READ_DENY):
        block(
            "BLOCKED (autoresearch integrity): reading the target values (test "
            "assertions / truth.json / ref_ls) is disallowed to prevent "
            "overfitting to the harness. Run the tests and trust pass/fail; "
            "optimize for gains that generalize off the benchmark."
        )

if tool == "Bash":
    cmd = str(ti.get("command", ""))
    reads_target = re.search(
        r"tests/test_[^\s]*\.py|data/[^\s]*/truth\.json|data/[^\s]*/ref_ls", cmd
    )
    uses_reader = re.search(r"\b(cat|less|more|head|tail|bat|grep|rg|awk|sed)\b", cmd)
    if reads_target and uses_reader:
        block(
            "BLOCKED (autoresearch integrity): reading the target values via the "
            "shell is disallowed. Trust the test pass/fail signal instead."
        )

sys.exit(0)
