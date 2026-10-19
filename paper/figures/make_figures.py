#!/usr/bin/env python3
"""Draw the paper's data figures (PDF) with matplotlib.

Inputs, all read-only and checked against their pins:

- ``data/helm_example.json`` and ``data/ratio_curves.json``, written by
  ``scripts/paper_figure_data.py`` from the frozen artifacts;
- the frozen HELM confirmatory result (step envelopes, endpoints).

Outputs, next to this file: ``example.pdf`` (Figure 1), ``ratio.pdf`` and
``helm_envelopes.pdf``. Needs only numpy and matplotlib, so it runs in a
plotting environment separate from the experiment one:

    python -m venv plotenv && plotenv/bin/pip install "matplotlib>=3.8,<4"
    plotenv/bin/python paper/figures/make_figures.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULT = Path("~/ddre-artifacts/confirmatory-v2/helm/helm_confirmatory_result_float16.json")
RESULT_SHA256 = "10bbd7306d3710fb4da076d94ebd98df20f624ff7d974a245c49e0639e448151"
ESTIMATORS_SHA256 = "6a007e37"  # prefix; the data files carry the full pin
PREDICTIONS_SHA256 = "994d85f223b45eaa0afe5740ea5cd98e72f8193d05c50266935dd79302d7a8e8"

TEXT_W, COL_W = 6.3, 3.03  # ACL text and column widths, inches
INK, GRID = "#222222", "#e6e6e6"
STYLE = {  # Okabe-Ito colours
    "BSE": dict(label="BSE", color="#000000", ls="-", lw=1.2),
    "DDRE-1D": dict(label="DDRE-1D", color="#D55E00", ls="-", lw=1.8),
    "NB-M": dict(label="NB-M", color="#0072B2", ls="--", lw=1.0),
    "DDRE-M": dict(label="DDRE-M", color="#E69F00", ls="-.", lw=1.0),
    "fixed-max": dict(label="fixed-$k$ max entailment", color="#009E73", ls=":", lw=1.3),
}
OPERATING = "K10/cr1"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    out = HERE / name
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(out)


# ---------------------------------------------------------------- Figure 1


def latex_escape(text):
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")):
        text = text.replace(a, b)
    return text


def example_table(ex):
    """Panel (a) of Figure 1 as a LaTeX fragment: the sentence and the documents read."""
    bse, ddre = ex["methods"]["BSE"], ex["methods"]["DDRE-1D"]
    read = max(bse["documents"], ddre["documents"])
    gen = {"falcon40b": "Falcon-40B"}.get(ex["generator"], ex["generator"])
    verdict = "hallucinated" if ex["label"] == 0 else "factual"
    lines = ["% Written by make_figures.py from data/helm_example.json; do not edit.",
             r"\textbf{(a)}~{\color{black!65}HELM sentence, %s; human label: %s}\\[2pt]"
             % (gen, verdict),
             r"\textit{``%s''}\\[4pt]" % latex_escape(ex["sentence"]),
             r"\begin{tabular}{@{}r@{\hspace{6pt}}l r r r@{}}",
             r"\toprule",
             r" & & & \multicolumn{2}{c}{$\log \hat r(e_n)$} \\ \cmidrule(l{3pt}){4-5}",
             r"$n$ & retrieved passage (title) & $e_n$ & BSE & DDRE \\",
             r"\midrule"]
    for i, d in enumerate(ex["documents"][:read]):
        n = d["rank"]
        cell = lambda r, c: (r"\textcolor{%s}{$%+.2f$}" % (c, r["log_ratios"][i])  # noqa: E731
                             if n <= r["documents"] else r"\textcolor{black!40}{--}")
        lines.append(r"%d & %s & %.1f & %s & %s \\" % (
            n, latex_escape(d["title"]), d["entailment"], cell(bse, "bse"), cell(ddre, "ddre")))
    lines += [r"\bottomrule", r"\end{tabular}\\[2pt]",
              r"{\footnotesize\color{black!65}$e_n$: NLI entailment of passage $n$ (0--100); "
              r"passages %d--%d were read by neither method.}" % (read + 1, len(ex["documents"]))]
    out = HERE / "example_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out)


def example_figure(ex):
    """Panel (b) of Figure 1: the posterior after each document, and the continue region."""
    bse, ddre = ex["methods"]["BSE"], ex["methods"]["DDRE-1D"]
    fig, ax = plt.subplots(figsize=(3.05, 2.2))
    ax.set_yscale("logit")
    regions = bse["continue_regions"]
    stages = sorted(int(n) for n in regions)
    lo = [regions[str(n)][0][0] for n in stages]
    hi = [regions[str(n)][0][1] for n in stages]
    xs = np.array(stages + [stages[-1] + 1]) - 0.5
    ax.fill_between(xs, lo + lo[-1:], hi + hi[-1:], step="post", color="#e3e3e3", lw=0,
                    label="continue: read the next document")
    ax.axhline(ex["tau"], color="#777777", lw=0.7, ls=(0, (4, 2)))
    ax.text(10.4, ex["tau"] * 1.07, r"$\tau$", color="#555555", va="bottom", ha="right")
    for m, r, mk in (("BSE", bse, "o"), ("DDRE-1D", ddre, "s")):
        st = STYLE[m]
        ax.plot(range(len(r["posterior"])), r["posterior"], marker=mk, ms=3.0,
                color=st["color"], lw=st["lw"] * 0.85, label=st["label"])
        ax.plot([r["documents"]], [r["p_factual"]], marker=mk, ms=7, mfc="none",
                mec=st["color"], mew=0.9)
    ax.annotate(f"BSE stops after {bse['documents']}", (bse["documents"], bse["p_factual"]),
                xytext=(1.2, 0.0125), fontsize=7, color=STYLE["BSE"]["color"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666", shrinkB=4))
    ax.annotate(f"DDRE-1D stops after {ddre['documents']}",
                (ddre["documents"], ddre["p_factual"]), xytext=(6.2, 0.11), fontsize=7,
                color=STYLE["DDRE-1D"]["color"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666", shrinkB=4))
    ax.set_xlim(-0.4, 10.5)
    ax.set_ylim(0.008, 0.985)
    ax.set_xticks(range(0, 11))
    ax.set_yticks([0.01, 0.05, 0.2, 0.5, 0.8, 0.95])
    ax.set_yticklabels(["0.01", "0.05", "0.2", "0.5", "0.8", "0.95"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.0, labelspacing=0.3)
    ax.set_xlabel("documents read $n$")
    ax.set_ylabel(r"posterior $\pi_n = P(\mathrm{factual}\mid e_{1:n})$")
    ax.text(-0.2, 1.02, "(b)", fontweight="bold", va="bottom", transform=ax.transAxes)
    save(fig, "example_plot.pdf")


# ---------------------------------------------------------------- ratio figure


def ratio_figure(curves):
    grid = np.array(curves["grid"])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.08))
    counts = curves["bse_training_counts"]
    f, h = np.array(counts["factual"]), np.array(counts["hallucinated"])
    empty = (f + h) == 0
    for b in np.flatnonzero(empty):
        for a in (ax, bx):
            a.axvspan(10 * b, 10 * b + 10, color="#f2f2f2", lw=0)
    ax.axhline(0, color="#999999", lw=0.5)
    for m in ("BSE", "DDRE-1D"):
        st = STYLE[m]
        ax.plot(grid, curves["log_ratio"][m], color=st["color"], lw=st["lw"],
                label={"BSE": "BSE: log of a ratio of two histograms",
                       "DDRE-1D": r"DDRE-1D: $\log\hat r(e)$ fitted directly"}[m])
    ax.set_ylabel(r"log likelihood ratio")
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.1)
    ax.set_ylim(-2.3, 4.9)
    ax.text(90, -2.1, "no training\ndocuments", fontsize=6.5, ha="center", va="bottom",
            color="#777777")
    centres = np.arange(10) * 10 + 5
    bx.bar(centres - 2.1, f, width=4.0, color="#7f7f7f", label="factual")
    bx.bar(centres + 2.1, h, width=4.0, color="#ffffff", edgecolor=INK, lw=0.5,
           label="hallucinated")
    bx.set_ylabel("training\ndocuments")
    bx.set_xlabel("entailment score $e$ of a document (0–100)")
    bx.legend(loc="upper right", ncol=2, handlelength=1.0, borderaxespad=0.1)
    bx.set_xlim(0, 100)
    bx.set_xticks(range(0, 101, 10))
    save(fig, "ratio.pdf")


# ---------------------------------------------------------------- envelopes


def envelope(points):
    """The step envelope Q(c) = max{q_i : c_i <= c}, as (costs, qualities)."""
    pts = sorted(((v["nli_calls_per_sentence"], -v["balanced_ap"]) for v in points.values()))
    cs, qs, best = [], [], -np.inf
    for c, nq in pts:
        if -nq > best:
            best = -nq
            cs.append(c)
            qs.append(best)
    return np.array(cs), np.array(qs)


def cost_for(env, q):
    hit = np.flatnonzero(env[1] >= q)
    return float(env[0][hit[0]])


def envelope_figure(result):
    curves, endpoints = result["curves"], result["endpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, pair in zip(axes, ("28/96", "14/24")):
        env = {m: envelope(curves[m if m == "fixed-max" else f"{m}|{pair}"]) for m in STYLE}
        xmax = 8.0
        for m, (c, q) in env.items():
            st = STYLE[m]
            ax.step(np.append(c, xmax), np.append(q, q[-1]), where="post", color=st["color"],
                    ls=st["ls"], lw=st["lw"], label=st["label"], zorder=3 if m == "DDRE-1D" else 2)
        # area between DDRE-1D and BSE over their shared cost range (the AUC-gap)
        (cd, qd), (cb, qb) = env["DDRE-1D"], env["BSE"]
        lo, hi = max(cd[0], cb[0]), min(cd[-1], cb[-1])
        xs = np.linspace(lo, hi, 2000)
        step = lambda c, q, x: q[np.searchsorted(c, x, side="right") - 1]  # noqa: E731
        ax.fill_between(xs, step(cb, qb, xs), step(cd, qd, xs), color="#D55E00", alpha=0.13,
                        lw=0, zorder=1)
        # BSE's operating point and the cost at which each envelope reaches it
        op = curves[f"BSE|{pair}"][OPERATING]
        c_op, q_op = op["nli_calls_per_sentence"], op["balanced_ap"]
        ax.plot([c_op], [q_op], marker="o", ms=4.5, mfc="white", mec=INK, mew=0.9, zorder=4)
        c_b, c_d = cost_for(env["BSE"], q_op), cost_for(env["DDRE-1D"], q_op)
        saved = endpoints[f"DDRE-1D|BSE|{pair}|e_cost_nli"]
        if abs((c_b - c_d) - saved["observed"]) > 1e-9:
            raise SystemExit(f"{pair}: the drawn saving differs from the frozen endpoint")
        ax.plot([c_d, c_op], [q_op, q_op], color=INK, lw=0.6, ls=(0, (1, 1.5)), zorder=4)
        for c in (c_b, c_d):
            ax.plot([c], [q_op], marker="|", ms=6, mew=1.0, color=INK, zorder=5)
        ax.annotate("", xy=(c_d, q_op - 0.009), xytext=(c_b, q_op - 0.009), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))
        ax.text(c_b + 0.12, q_op - 0.009,
                f"{c_b:.2f} \u2192 {c_d:.2f} NLI calls:\n"
                f"{1 - c_d / c_b:.0%} fewer at the quality\nof BSE's operating point ($\\circ$)",
                ha="left", va="top", fontsize=7, linespacing=1.3)
        gap = endpoints[f"DDRE-1D|BSE|{pair}|auc_gap"]
        ax.text(0.97, 0.05, f"AUC-gap vs BSE: {gap['observed']:+.3f}\n"
                f"95% CI [{gap['ci_lower']:+.3f}, {gap['ci_upper']:+.3f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=STYLE["DDRE-1D"]["color"])
        ax.set_title(f"$C_M/C_{{FA}}$ = {pair}")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.60, 0.86)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("NLI calls per sentence")
    axes[0].set_ylabel("balanced AP")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="#D55E00", alpha=0.13, lw=0))
    labels.append("DDRE-1D above BSE")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=6,
               handlelength=2.0, columnspacing=1.2)
    save(fig, "helm_envelopes.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", default=str(RESULT))
    args = ap.parse_args()
    result_path = Path(args.result).expanduser()
    if sha256(result_path) != RESULT_SHA256:
        raise SystemExit(f"{result_path} is not the frozen HELM confirmatory result")
    ex = json.loads((HERE / "data/helm_example.json").read_text())
    curves = json.loads((HERE / "data/ratio_curves.json").read_text())
    for d in (ex, curves):
        if not d["pins"]["estimators"].startswith(ESTIMATORS_SHA256):
            raise SystemExit("the figure data were not made from the frozen estimators")
    if ex["pins"]["predictions"] != PREDICTIONS_SHA256:
        raise SystemExit("the example was not chosen from the frozen predictions")
    example_table(ex)
    example_figure(ex)
    ratio_figure(curves)
    envelope_figure(json.loads(result_path.read_text()))


if __name__ == "__main__":
    main()
