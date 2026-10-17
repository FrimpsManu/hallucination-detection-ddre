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

**Still to supply:** red `[...]` markers (`\pending`) mark the contact email
and the public repository URL.
