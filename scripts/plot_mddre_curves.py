#!/usr/bin/env python3
"""Draw the amended cost-quality curves (design amendment 1) as one SVG.

Reads the frozen amended replay result (read-only) and writes a three-panel
figure: documents per sentence against balanced PR-AUC for every cell on every
test set, with BSE's official run marked. No dependency beyond the standard
library, so the figure is byte-reproducible from the result file.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("~/ddre-artifacts/mddre/mddre_replay_result_float16_amendment1.json")
RESULT_SHA256 = "ee531d9fc11f0a52ea5072f0aa6e544e2753d4582194617f61380a3227cb41b5"
OUT = PROJECT_ROOT / "docs" / "figures" / "mddre_cost_quality_curves.svg"

PANELS = (("factcheck_bench", "Factcheck-Bench"),
          ("selfcheckgpt_heldout", "SelfCheckGPT held-out"),
          ("factscore_test", "FActScore test entities"))
STYLE = {  # cell: (label, colour, dash)
    "A": ("A  BSE histogram", "#555555", ""),
    "A_prime": ("A′ histogram, new data", "#8c8c8c", "4 3"),
    "U": ("U  uLSIF on e", "#2b83ba", "6 3"),
    "M": ("M  uLSIF on (e, k, log rank)", "#d7191c", ""),
    "L": ("L  logistic ratio", "#fdae61", "2 2"),
}
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 330, 260, 52, 12, 34, 44
Y_RANGE = (0.55, 0.80)


def panel(result, key, title, x0):
    obs = result["results"][key]["bootstrap"]["observed"]
    xmax = max(max(obs[c]["curve"]["documents"]) for c in STYLE)
    xmax = 5 * (int(xmax // 5) + 1)
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
    for i in range(6):
        q = Y_RANGE[0] + i * 0.05
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + PAD_L + pw}" y1="{sy(q):.1f}" '
                   f'y2="{sy(q):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{x0 + PAD_L - 5}" y="{sy(q) + 4:.1f}" text-anchor="end">'
                   f'{q:.2f}</text>')
    step = 5 if xmax <= 20 else 10
    for d in range(0, xmax + 1, step):
        out.append(f'<text x="{sx(d):.1f}" y="{PAD_T + ph + 16}" text-anchor="middle">'
                   f'{d}</text>')
    out.append(f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">'
               'documents per sentence</text>')
    for cell, (_, colour, dash) in STYLE.items():
        c = obs[cell]["curve"]
        pts = " ".join(f"{sx(d):.1f},{sy(q):.1f}"
                       for d, q in zip(c["documents"], c["balanced_pr_auc"]))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        width = 2.2 if cell == "M" else 1.4
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                   f'stroke-width="{width}"{dash_attr}/>')
    ref = obs["reference"]
    rx, ry = sx(ref["documents"]), sy(ref["balanced_pr_auc"])
    out.append(f'<path d="M{rx - 6:.1f},{ry:.1f} L{rx:.1f},{ry - 6:.1f} L{rx + 6:.1f},'
               f'{ry:.1f} L{rx:.1f},{ry + 6:.1f} Z" fill="#000"/>')
    return out


def render(result):
    total_w = W * len(PANELS)
    legend_y = H + 8
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
            f'height="{H + 40}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
            f'<rect width="{total_w}" height="{H + 40}" fill="#fff"/>',
            f'<text transform="translate(14,{PAD_T + (H - PAD_T - PAD_B) / 2}) rotate(-90)" '
            'text-anchor="middle">balanced PR-AUC</text>']
    for i, (key, title) in enumerate(PANELS):
        body += panel(result, key, title, i * W)
    x = PAD_L
    for cell, (label, colour, dash) in STYLE.items():
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{x}" x2="{x + 22}" y1="{legend_y + 14}" y2="{legend_y + 14}" '
                    f'stroke="{colour}" stroke-width="2"{dash_attr}/>')
        body.append(f'<text x="{x + 27}" y="{legend_y + 18}">{label}</text>')
        x += 27 + int(5.6 * len(label)) + 16
    body.append(f'<path d="M{x},{legend_y + 14} l5,-5 l5,5 l-5,5 Z" fill="#000"/>')
    body.append(f'<text x="{x + 15}" y="{legend_y + 18}">BSE official run</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", default=str(RESULT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    path = Path(args.result).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RESULT_SHA256:
        raise SystemExit(f"{path} is not the frozen amended replay result")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(json.loads(path.read_text())))
    print(f"{out}\n  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
