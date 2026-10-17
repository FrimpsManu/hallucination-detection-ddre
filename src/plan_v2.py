"""Pins for docs/research_plan_v2.md.

The plan is append-only: amendments are added at the end, so the approved
text stays pinned as a byte prefix. Scripts written after an amendment also
require the full file to be that amendment's version.
"""

import hashlib
from pathlib import Path

PLAN_PATH = "docs/research_plan_v2.md"
PLAN_BYTES = 16836  # the plan as approved and merged (PR #52)
PLAN_SHA256 = "29a80fa2fcb1e4afd2a7a60d582d6f611ab6fdff3181b8335c82dfa43c1df98c"
AMENDMENT_1_SHA256 = "88e29ec69b064f280c90aa350983e13699f657c38c758140ecea2c8657a21e5e"


def plan_problems(root, require_amendment_1=False):
    """Why the plan file is not the approved text (plus amendment 1 if required)."""
    path = Path(root) / PLAN_PATH
    if not path.exists():
        return [f"{PLAN_PATH} is missing"]
    data = path.read_bytes()
    problems = []
    if hashlib.sha256(data[:PLAN_BYTES]).hexdigest() != PLAN_SHA256:
        problems.append(f"{PLAN_PATH} differs from the approved plan")
    if require_amendment_1 and hashlib.sha256(data).hexdigest() != AMENDMENT_1_SHA256:
        problems.append(f"{PLAN_PATH} is not at amendment 1")
    return problems
