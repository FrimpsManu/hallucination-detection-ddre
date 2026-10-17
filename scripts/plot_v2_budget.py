#!/usr/bin/env python3
"""Draw the amendment-2 budget envelopes as one SVG.

Reads the frozen budget result (read-only) and writes ten panels (five
datasets x C_M/C_FA 28/96 and 14/24): the step envelope of balanced AP against
NLI calls per sentence over every (K, C_retrieve) point, for BSE, DDRE-1D,
NB-M, DDRE-M and the fixed-k max baseline. Standard library only.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = Path("~/ddre-artifacts/v2/v2_budget_result_float16.json")
RESULT_SHA256 = "a02f7bafffe68896d1eebc03bfd4489d657dccf7adc8adb70166209c1f56cdc9"
OUT = PROJECT_ROOT / "docs" / "figures" / "v2_budget_envelopes.svg"
PANELS = (("selfcheckgpt_heldout", "SelfCheckGPT"), ("factscore_test", "FActScore test"),
          ("factcheck_bench", "Factcheck-Bench"), ("expertqa", "ExpertQA"),
          ("bingcheck", "BingCheck"))
STYLE = {"BSE": ("BSE", "#000000", ""), "DDRE-1D": ("DDRE-1D", "#d7191c", ""),
         "NB-M": ("NB-M", "#2b83ba", "6 3"), "DDRE-M": ("DDRE-M", "#fdae61", ""),
         "fixed-max": ("fixed-k max", "#1a9641", "2 2")}
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 230, 210, 44, 8, 30, 38
Y_RANGE = (0.45, 0.85)


def envelope(points):
    pts = sorted((v["nli_calls_per_sentence"], v["balanced_ap"]) for v in points.values())
    out, best = [], None
    for c, q in pts:
        if best is None or q > best:
            if out:
                out.append((c, out[-1][1]))
            best = q
            out.append((c, q))
    return out


def panel(curves, pair, title, x0, y0):
    series = {m: envelope(curves[m if m == "fixed-max" else f"{m}|{pair}"]) for m in STYLE}
    xmax = max(c for s in series.values() for c, _ in s)
    step = 5 if xmax <= 25 else 10
    xmax = step * (int(xmax // step) + 1)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(c):
        return x0 + PAD_L + pw * c / xmax

    def sy(q):
        q = min(max(q, Y_RANGE[0]), Y_RANGE[1])
        return y0 + PAD_T + ph * (1 - (q - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))

    out = [f'<text x="{x0 + PAD_L + pw / 2:.1f}" y="{y0 + 18}" text-anchor="middle" '
           f'font-weight="bold">{title} · {pair}</text>',
           f'<rect x="{x0 + PAD_L}" y="{y0 + PAD_T}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>']
    q = Y_RANGE[0]
    while q <= Y_RANGE[1] + 1e-9:
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + PAD_L + pw}" y1="{sy(q):.1f}" y2="{sy(q):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{x0 + PAD_L - 4}" y="{sy(q) + 4:.1f}" text-anchor="end">{q:.2f}</text>')
        q += 0.1
    for c in range(0, xmax + 1, step):
        out.append(f'<text x="{sx(c):.1f}" y="{y0 + PAD_T + ph + 14}" text-anchor="middle">{c}</text>')
    for name, (_, colour, dash) in STYLE.items():
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        width = 2.2 if name == "DDRE-1D" else 1.3
        path = " ".join(f"{sx(c):.1f},{sy(q):.1f}" for c, q in series[name])
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="{width}"{dash_attr}/>')
    return out


def render(result):
    cols = len(PANELS)
    total_w, total_h = W * cols, 2 * H + 60
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="10">',
            f'<rect width="{total_w}" height="{total_h}" fill="#fff"/>']
    for row, pair in enumerate(("28/96", "14/24")):
        for i, (key, title) in enumerate(PANELS):
            body += panel(result["results"][key]["curves"], pair, title, i * W, row * H)
    x, y = PAD_L, 2 * H + 25
    for name, (label, colour, dash) in STYLE.items():
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{x}" x2="{x + 22}" y1="{y}" y2="{y}" stroke="{colour}" stroke-width="2"{dash_attr}/>')
        body.append(f'<text x="{x + 27}" y="{y + 4}">{label}</text>')
        x += 120
    body.append(f'<text x="{x + 10}" y="{y + 4}" fill="#555">step envelope of balanced AP vs NLI calls '
                'per sentence over all (K, C_retrieve) points</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", default=str(RESULT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    path = Path(args.result).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != RESULT_SHA256:
        raise SystemExit(f"{path} is not the frozen budget result")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(json.loads(path.read_text())))
    print(f"{out}\n  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
