#!/usr/bin/env python3
"""Draw the confirmatory cost-quality curves (preregistration §4) as one SVG.

Reads the frozen confirmatory replay result (read-only) and writes a two-panel
figure: documents per sentence against balanced PR-AUC for every cell on
ExpertQA and BingCheck, with BSE's official run marked. Both panels share one
y-axis range so ExpertQA's near-chance level is visible. Standard library only,
so the figure is byte-reproducible from the result file.
"""

import argparse
import hashlib
import json
from pathlib import Path

from plot_mddre_curves import H, PAD_B, PAD_L, PAD_R, PAD_T, STYLE, W

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("~/ddre-artifacts/confirmatory/confirmatory_replay_result_float16.json")
RESULT_SHA256 = "6d1c23b2eb9746fbed7c93c9248827f0620f86b439243b8e9a798209926b8301"
OUT = PROJECT_ROOT / "docs" / "figures" / "confirmatory_cost_quality_curves.svg"

PANELS = (("expertqa", "ExpertQA"), ("bingcheck", "BingCheck"))
Y_RANGE = (0.45, 0.75)


def panel(result, key, title, x0):
    boot = result["results"][key]["bootstrap"]
    curves = boot["curves"]
    ref = boot["endpoints"]
    xmax = max(max(curves[c]["documents"]["cost"]) for c in STYLE)
    step = 1 if xmax <= 5 else 5
    xmax = step * (int(xmax // step) + 1)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(d):
        return x0 + PAD_L + pw * d / xmax

    def sy(q):
        q = min(max(q, Y_RANGE[0]), Y_RANGE[1])
        return PAD_T + ph * (1 - (q - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))

    out = [f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="20" text-anchor="middle" '
           f'font-weight="bold">{title}</text>',
           f'<rect x="{x0 + PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" '
           'fill="none" stroke="#333"/>']
    for i in range(7):
        q = Y_RANGE[0] + i * 0.05
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + PAD_L + pw}" y1="{sy(q):.1f}" '
                   f'y2="{sy(q):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{x0 + PAD_L - 5}" y="{sy(q) + 4:.1f}" text-anchor="end">'
                   f'{q:.2f}</text>')
    for d in range(0, xmax + 1, step):
        out.append(f'<text x="{sx(d):.1f}" y="{PAD_T + ph + 16}" text-anchor="middle">'
                   f'{d}</text>')
    out.append(f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">'
               'documents per sentence</text>')
    for cell, (_, colour, dash) in STYLE.items():
        c = curves[cell]["documents"]
        pts = " ".join(f"{sx(d):.1f},{sy(q):.1f}"
                       for d, q in zip(c["cost"], c["balanced_pr_auc"]))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        width = 2.2 if cell == "M" else 1.4
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                   f'stroke-width="{width}"{dash_attr}/>')
    rx = sx(ref["reference.documents"]["observed"])
    ry = sy(ref["reference.balanced_pr_auc"]["observed"])
    out.append(f'<path d="M{rx - 6:.1f},{ry:.1f} L{rx:.1f},{ry - 6:.1f} L{rx + 6:.1f},'
               f'{ry:.1f} L{rx:.1f},{ry + 6:.1f} Z" fill="#000"/>')
    return out


def render(result):
    total_w = W * len(PANELS)
    legend_y = H + 8
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
            f'height="{H + 60}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
            f'<rect width="{total_w}" height="{H + 60}" fill="#fff"/>',
            f'<text transform="translate(14,{PAD_T + (H - PAD_T - PAD_B) / 2}) rotate(-90)" '
            'text-anchor="middle">balanced PR-AUC</text>']
    for i, (key, title) in enumerate(PANELS):
        body += panel(result, key, title, i * W)
    # Two legend rows: the two panels are narrower than the three-panel figure.
    x, y = PAD_L, legend_y + 14
    for cell, (label, colour, dash) in STYLE.items():
        width = 27 + int(5.6 * len(label)) + 16
        if x + width > total_w:
            x, y = PAD_L, y + 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{x}" x2="{x + 22}" y1="{y}" y2="{y}" '
                    f'stroke="{colour}" stroke-width="2"{dash_attr}/>')
        body.append(f'<text x="{x + 27}" y="{y + 4}">{label}</text>')
        x += width
    body.append(f'<path d="M{x},{y} l5,-5 l5,5 l-5,5 Z" fill="#000"/>')
    body.append(f'<text x="{x + 15}" y="{y + 4}">BSE official run</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", default=str(RESULT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    path = Path(args.result).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RESULT_SHA256:
        raise SystemExit(f"{path} is not the frozen confirmatory replay result")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(json.loads(path.read_text())))
    print(f"{out}\n  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
