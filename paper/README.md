# Paper

ACL-format LaTeX source (`acl.sty` and `acl_natbib.bst` from
acl-org/acl-style-files @ `d5adc82`).

**Build:** `tectonic main.tex` (tested with Tectonic 0.17.0), or
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

**Figures and tables** are generated from the frozen, hash-pinned artifacts;
each script checks its pins before drawing anything:

| Output | Made by | Environment |
|---|---|---|
| `figures/data/helm_example.json`, `figures/data/ratio_curves.json` | `scripts/paper_figure_data.py` | the project's |
| `figures/example_plot.pdf`, `figures/example_table.tex`, `figures/ratio.pdf`, `figures/helm_envelopes.pdf` | `figures/make_figures.py` | numpy + matplotlib only |
| `figures/example.pdf` (Figure 1) | `tectonic figures/example.tex` | |
| `figures/overview.pdf` | `tectonic figures/overview.tex` | |
| `tables/*.tex` | `make_tables.py` | standard library |

`make_figures.py` runs in a plotting environment kept apart from the
experiment one:

```bash
python3 -m venv plotenv && plotenv/bin/pip install "matplotlib>=3.8,<4"
plotenv/bin/python paper/figures/make_figures.py
```

Figure 1's example is chosen by the fixed rule stated in its caption (and in
`paper_figure_data.py`), and its replay must reproduce the frozen predictions.

**Sources of the numbers:**
- `docs/helm_results_summary.md`: the confirmatory test;
- `docs/v2_budget_results.md` and `docs/v2_results_summary.md`: the
  exploratory results;
- `docs/research_plan_v2.md` amendment 1: the stopping-rule finding.

Every figure in those documents is generated from a hash-pinned artifact.

**Still to supply:** the red `[...]` marker (`\pending`) for the public
repository URL.

**Word version:** `python paper/make_docx.py OUT.docx` (needs tectonic, pandoc
and pdftoppm). It builds a .docx laid out like ACL's own Word template:
- A4 page, 2.5 cm margins;
- Times New Roman 11 pt;
- two 7.7 cm columns with a 0.6 cm gap, below a one-column title block;
- page-wide figures and tables in one-column sections between continuous
  section breaks;
- ACL citation style.

The LaTeX PDF is authoritative. Apple Pages starts the two-column body on a
new page; Microsoft Word and Google Docs keep it on page 1.
