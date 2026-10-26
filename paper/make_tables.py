#!/usr/bin/env python3
"""Write the paper's result tables (LaTeX fragments) from the frozen artifacts.

Reads, read-only and checked against their SHA-256 pins, the HELM
confirmatory result and the exploratory budget result, and writes
``tables/helm_tests.tex``, ``tables/helm_methods.tex`` and
``tables/explore.tex``, which ``main.tex`` inputs. Standard library only.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = Path("~/ddre-artifacts").expanduser()
HELM = (ART / "confirmatory-v2/helm/helm_confirmatory_result_float16.json",
        "10bbd7306d3710fb4da076d94ebd98df20f624ff7d974a245c49e0639e448151")
BUDGET = (ART / "v2/v2_budget_result_float16.json", "a02f7baf")  # prefix, as in the docs
PAIRS = ("28/96", "14/24")
PAIR_TEX = {"28/96": "(28, 96)", "14/24": "(14, 24)"}
HEADER = "% Written by make_tables.py from the frozen artifacts; do not edit.\n"


def load(path, sha):
    if not hashlib.sha256(path.read_bytes()).hexdigest().startswith(sha):
        raise SystemExit(f"{path} does not match its pin")
    return json.loads(path.read_text())


def signed(x, digits=3, lead=True):
    s = f"{x:+.{digits}f}"
    return s if lead else s.replace("0.", ".", 1)


def num(x, digits=3, lead=True):
    s = f"{x:.{digits}f}".replace("-", "$-$")
    return s if lead else s.replace("0.", ".", 1)


def tex_signed(x, digits=3, lead=True):
    return signed(x, digits, lead).replace("-", "$-$")


def ci(e, digits=3, lead=False, plus=True):
    lo, hi = e["ci_lower"], e["ci_upper"]
    f = tex_signed if plus else (lambda x, d, l: num(x, d, l))
    return f"[{f(lo, digits, lead)}, {f(hi, digits, lead)}]"


def bold_if(text, cond):
    return f"\\textbf{{{text}}}" if cond else text


def helm_tests(r):
    tests = r["decision"]["tests"]
    ends = r["endpoints"]
    rows = []
    for fam, ref, label in (("H1", "BSE", "BSE"), ("H2", "fixed-max", "fixed-$k$ max entailment")):
        if fam == "H2":
            rows.append("\\addlinespace[3pt]")
        rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{fam}: DDRE-1D vs.\\ {label}}}}} \\\\")
        for t, pair in ((fam + "a", "28/96"), (fam + "b", "14/24")):
            s = tests[t]
            e = ends[f"DDRE-1D|{ref}|{pair}|auc_gap"]
            p = s["p_one_sided"]
            p_tex = "\\textless{}0.0001" if p == 0 else f"{p:.3f}"
            rows.append(f"\\quad {t} & {PAIR_TEX[pair]} & "
                        f"{bold_if(tex_signed(s['observed']), s['supported'])} & "
                        f"{ci(e, lead=True, plus=False)} & {tex_signed(s['q05'])} & {p_tex} \\\\")
    return HEADER + "\n".join([
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Test & $(\\CM, \\CFA)$ & $\\Delta_{\\mathrm{AUC}}$ & 95\\% CI & 5th pct. & $p$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n"


def helm_methods(r):
    ot, ends = r["operating_tables"], r["endpoints"]
    methods = ("BSE-official", "BSE", "DDRE-1D", "NB-M", "DDRE-M")
    names = {"BSE-official": "BSE, released look-ahead", "BSE": "BSE",
             "DDRE-1D": "DDRE-1D", "NB-M": "NB-M", "DDRE-M": "DDRE-M"}
    best = {}
    for pair in PAIRS:
        rows = [ot[f"{m}|{pair}"] for m in methods]
        best[pair] = {"bap": max(x["balanced_ap"] for x in rows),
                      "calls": min(x["nli_calls_per_sentence"] for x in rows),
                      "brier": min(x["calibration"]["brier"] for x in rows),
                      "gap": max(ends[f"{m}|fixed-max|{pair}|auc_gap"]["observed"]
                                 for m in methods if m != "BSE-official")}
    lines = []
    for m in methods:
        if m == "NB-M":
            lines.append("\\addlinespace[2pt]")
        cells = [names[m]]
        for pair in PAIRS:
            x, b = ot[f"{m}|{pair}"], best[pair]
            if m == "BSE-official":
                cells.append("--")
            else:
                g = ends[f"{m}|fixed-max|{pair}|auc_gap"]
                cells.append(bold_if(tex_signed(g["observed"]), g["observed"] == b["gap"])
                             + f" {{\\scriptsize {ci(g)}}}")
            cells += [bold_if(num(x["balanced_ap"]), x["balanced_ap"] == b["bap"]),
                      bold_if(num(x["nli_calls_per_sentence"], 2),
                              x["nli_calls_per_sentence"] == b["calls"]),
                      bold_if(num(x["calibration"]["brier"]),
                              x["calibration"]["brier"] == b["brier"])]
        lines.append(" & ".join(cells) + " \\\\")
    head = " & ".join(["$\\Delta_{\\mathrm{AUC}}$ vs.\\ fixed-$k$", "bAP",
                       "NLI calls", "Brier"])
    return HEADER + "\n".join([
        "\\begin{tabular}{@{}l lccc @{\\hspace{14pt}} lccc@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{$(\\CM, \\CFA) = (28, 96)$} & "
        "\\multicolumn{4}{c}{$(\\CM, \\CFA) = (14, 24)$} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
        f"Method & {head} & {head} \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def explore(b):
    names = {"selfcheckgpt_heldout": "SelfCheckGPT", "factscore_test": "FActScore",
             "factcheck_bench": "Factcheck-B.", "expertqa": "ExpertQA",
             "bingcheck": "BingCheck"}
    lines = []
    for key, name in names.items():
        e = b["results"][key]["endpoints"]
        vals, cis = [name], [""]
        for ref in ("BSE", "fixed-max"):
            for pair in PAIRS:
                g = e[f"DDRE-1D|{ref}|{pair}|auc_gap"]
                vals.append(bold_if(tex_signed(g["observed"], lead=False),
                                    g["ci_lower"] > 0 or g["ci_upper"] < 0))
                cis.append(f"{{\\scriptsize\\color{{black!60}}{ci(g, plus=False)}}}")
        lines.append(" & ".join(vals) + " \\\\")
        lines.append(" & ".join(cis) + " \\\\[1pt]")
    return HEADER + "\n".join([
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{vs.\\ BSE} & \\multicolumn{2}{c}{vs.\\ fixed-$k$} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
        "Dataset & (28, 96) & (14, 24) & (28, 96) & (14, 24) \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def main():
    helm, budget = load(*HELM), load(*BUDGET)
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    for name, text in (("helm_tests", helm_tests(helm)), ("helm_methods", helm_methods(helm)),
                       ("explore", explore(budget))):
        (out / f"{name}.tex").write_text(text)
        print(out / f"{name}.tex")


if __name__ == "__main__":
    main()
