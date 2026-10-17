#!/usr/bin/env python3
"""Draw the plan-v2 cost-quality curves (Phase 5) as one SVG.

Reads the frozen v2 replay result (read-only) and writes a five-panel figure:
NLI calls per sentence against balanced average precision at C_M/C_FA 28/96,
one point per C_retrieve, for BSE-official, BSE, DDRE-1D, NB-M and DDRE-M,
plus Wang's fixed-k max-entailment baseline (k = 1..max documents). Standard
library only, so the figure is byte-reproducible from the result file.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("~/ddre-artifacts/v2/v2_replay_result_float16.json")
RESULT_SHA256 = "a7fe078169b0625eddffb8991bdd52b113c499ffb26a77b63bcafb9c2531af93"
OUT = PROJECT_ROOT / "docs" / "figures" / "v2_cost_quality_curves.svg"
PANELS = (("selfcheckgpt_heldout", "SelfCheckGPT held-out"), ("factscore_test", "FActScore test"),
          ("factcheck_bench", "Factcheck-Bench"), ("expertqa", "ExpertQA"),
          ("bingcheck", "BingCheck"))
STYLE = {  # series: (label, colour, dash)
    "BSE-official": ("BSE-official (Wang's code)", "#555555", "4 3"),
    "BSE": ("BSE (backward induction)", "#000000", ""),
    "DDRE-1D": ("DDRE-1D", "#d7191c", ""),
    "NB-M": ("NB-M", "#2b83ba", "6 3"),
    "DDRE-M": ("DDRE-M", "#fdae61", ""),
    "fixed-max": ("fixed-k max entailment", "#1a9641", "2 2"),
}
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 300, 250, 48, 10, 32, 42
Y_RANGE = (0.45, 0.85)
SETTING = "28/96"


def series(res, grid):
    out = {}
    for m in ("BSE-official", "BSE", "DDRE-1D", "NB-M", "DDRE-M"):
        pts = [(res["tables"][m][f"{SETTING}/{cr:g}"]["nli_calls_per_sentence"],
                res["tables"][m][f"{SETTING}/{cr:g}"]["balanced_ap"]) for cr in grid]
        out[m] = sorted(set(pts))
    out["fixed-max"] = sorted((v["nli_calls_per_sentence"], v["balanced_ap"])
                              for k, v in res["fixed_k"].items() if k.startswith("max@"))
    return out


def panel(res, grid, title, x0, y0):
    s = series(res, grid)
    xmax = max(c for pts in s.values() for c, _ in pts)
    step = 5 if xmax <= 25 else 10
    xmax = step * (int(xmax // step) + 1)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(c):
        return x0 + PAD_L + pw * c / xmax

    def sy(q):
        q = min(max(q, Y_RANGE[0]), Y_RANGE[1])
        return y0 + PAD_T + ph * (1 - (q - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))

    out = [f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="{y0 + 20}" text-anchor="middle" '
           f'font-weight="bold">{title}</text>',
           f'<rect x="{x0 + PAD_L}" y="{y0 + PAD_T}" width="{pw}" height="{ph}" '
           'fill="none" stroke="#333"/>']
    q = Y_RANGE[0]
    while q <= Y_RANGE[1] + 1e-9:
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + PAD_L + pw}" y1="{sy(q):.1f}" '
                   f'y2="{sy(q):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{x0 + PAD_L - 4}" y="{sy(q) + 4:.1f}" text-anchor="end">{q:.2f}</text>')
        q += 0.1
    for c in range(0, xmax + 1, step):
        out.append(f'<text x="{sx(c):.1f}" y="{y0 + PAD_T + ph + 15}" text-anchor="middle">{c}</text>')
    out.append(f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="{y0 + H - 8}" text-anchor="middle">'
               'NLI calls per sentence</text>')
    for name, (_, colour, dash) in STYLE.items():
        pts = s[name]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        width = 2.2 if name == "DDRE-1D" else 1.4
        path = " ".join(f"{sx(c):.1f},{sy(q):.1f}" for c, q in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" '
                   f'stroke-width="{width}"{dash_attr}/>')
        for c, q in pts:
            out.append(f'<circle cx="{sx(c):.1f}" cy="{sy(q):.1f}" r="1.8" fill="{colour}"/>')
    return out


def render(result):
    grid = result["settings"]["c_retrieve_grid"]
    cols, rows = 3, 2
    total_w, total_h = W * cols, H * rows + 50
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="11">',
            f'<rect width="{total_w}" height="{total_h}" fill="#fff"/>']
    for i, (key, title) in enumerate(PANELS):
        body += panel(result["results"][key], grid, title, (i % cols) * W, (i // cols) * H)
    x, y = 2 * W + 20, H + 50
    for name, (label, colour, dash) in STYLE.items():
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{x}" x2="{x + 22}" y1="{y}" y2="{y}" stroke="{colour}" '
                    f'stroke-width="2"{dash_attr}/>')
        body.append(f'<text x="{x + 27}" y="{y + 4}">{label}</text>')
        y += 20
    for line in ("balanced AP vs NLI calls per sentence,", "C_M/C_FA 28/96; one point per",
                 "C_retrieve (0.125-16) or per k"):
        y += 16
        body.append(f'<text x="{x}" y="{y}" fill="#555">{line}</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", default=str(RESULT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    path = Path(args.result).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RESULT_SHA256:
        raise SystemExit(f"{path} is not the frozen v2 replay result")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(json.loads(path.read_text())))
    print(f"{out}\n  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
