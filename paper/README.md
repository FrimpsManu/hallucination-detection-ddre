# Paper

ACL-format LaTeX source (`acl.sty` and `acl_natbib.bst` from
acl-org/acl-style-files @ `d5adc82`).

**Build:** `tectonic main.tex` (tested with Tectonic 0.17.0), or
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

**Figures:** `figures/*.pdf` are the SVGs of the same name in `docs/figures/`,
converted with `rsvg-convert -f pdf`.

**Sources of the numbers:**
- `docs/helm_results_summary.md`: the confirmatory test;
- `docs/v2_budget_results.md` and `docs/v2_results_summary.md`: the
  exploratory results;
- `docs/research_plan_v2.md` amendment 1: the stopping-rule finding.

Every figure in those documents is generated from a hash-pinned artifact.

**Still to supply:** the red `[...]` marker (`\pending`) for the public
repository URL.

**Word version:** `python paper/make_docx.py OUT.docx` (needs tectonic, pandoc
and rsvg-convert). It builds a .docx laid out like ACL's own Word template:
- A4 page, 2.5 cm margins;
- Times New Roman 11 pt;
- two 7.7 cm columns with a 0.6 cm gap, below a one-column title block;
- ACL citation style.

The LaTeX PDF is authoritative. Apple Pages starts the two-column body on a
new page; Microsoft Word and Google Docs keep it on page 1.
