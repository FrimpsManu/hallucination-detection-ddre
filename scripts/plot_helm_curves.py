#!/usr/bin/env python3
"""Draw the HELM confirmatory envelopes as one SVG (two panels: 28/96, 14/24).

Reads the frozen HELM confirmatory result (read-only): step envelopes of
balanced AP against NLI calls per sentence over every (K, C_retrieve) point,
for BSE, DDRE-1D, NB-M, DDRE-M and fixed-k max entailment. Standard library
only; the envelope and panel drawing are those of ``plot_v2_budget``.
"""

import argparse
import hashlib
import json
from pathlib import Path

from plot_v2_budget import H, PAD_L, STYLE, W, panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("~/ddre-artifacts/confirmatory-v2/helm/helm_confirmatory_result_float16.json")
RESULT_SHA256 = "10bbd7306d3710fb4da076d94ebd98df20f624ff7d974a245c49e0639e448151"
OUT = PROJECT_ROOT / "docs" / "figures" / "helm_envelopes.svg"


def render(result):
    total_w, total_h = 2 * W, H + 60
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="10">',
            f'<rect width="{total_w}" height="{total_h}" fill="#fff"/>']
    for i, pair in enumerate(("28/96", "14/24")):
        body += panel(result["curves"], pair, "HELM", i * W, 0)
    x, y = PAD_L, H + 20
    for n, (name, (label, colour, dash)) in enumerate(STYLE.items()):
        if n == 3:
            x, y = PAD_L, y + 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{x}" x2="{x + 22}" y1="{y}" y2="{y}" stroke="{colour}" '
                    f'stroke-width="2"{dash_attr}/>')
        body.append(f'<text x="{x + 27}" y="{y + 4}">{label}</text>')
        x += 140
    body.append("</svg>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", default=str(RESULT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    path = Path(args.result).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RESULT_SHA256:
        raise SystemExit(f"{path} is not the frozen HELM confirmatory result")
    out = Path(args.out)
    out.write_text(render(json.loads(path.read_text())))
    print(f"{out}\n  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
