"""Pins for docs/research_plan_v2.md.

The plan is append-only: amendments are added at the end, so the approved
text and each amendment stay pinned as byte prefixes. A script requires the
plan to contain (at least) the amendment it was written under.
"""

import hashlib
from pathlib import Path

PLAN_PATH = "docs/research_plan_v2.md"
PLAN_BYTES = 16836  # the plan as approved and merged (PR #52)
PLAN_SHA256 = "29a80fa2fcb1e4afd2a7a60d582d6f611ab6fdff3181b8335c82dfa43c1df98c"
AMENDMENT_1_BYTES = 20091
AMENDMENT_1_SHA256 = "88e29ec69b064f280c90aa350983e13699f657c38c758140ecea2c8657a21e5e"
AMENDMENT_2_BYTES = 22334
AMENDMENT_2_SHA256 = "71c891ea1ba367c0e692938da34e453107fea4f557bb9b0c80be929e49288a29"

def plan_problems(root, require_amendment_1=False, require_amendment_2=False):
    """Why the plan file does not contain the approved text (and amendments)."""
    path = Path(root) / PLAN_PATH
    if not path.exists():
        return [f"{PLAN_PATH} is missing"]
    data = path.read_bytes()
    wanted = 1 + int(require_amendment_1 or require_amendment_2) + int(require_amendment_2)
    prefixes = ((PLAN_BYTES, PLAN_SHA256, "the approved plan"),
                (AMENDMENT_1_BYTES, AMENDMENT_1_SHA256, "amendment 1"),
                (AMENDMENT_2_BYTES, AMENDMENT_2_SHA256, "amendment 2"))
    problems = []
    for size, sha, name in prefixes[:wanted]:
        if hashlib.sha256(data[:size]).hexdigest() != sha:
            problems.append(f"{PLAN_PATH} differs from the approved plan"
                            if name == "the approved plan"
                            else f"{PLAN_PATH} is not at {name}")
    return problems
