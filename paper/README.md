# Paper draft

ACL-format LaTeX source (`acl.sty` and `acl_natbib.bst` from
acl-org/acl-style-files @ `d5adc82`).

Build: `tectonic main.tex` (tested with Tectonic 0.17.0), or `pdflatex main && bibtex main && pdflatex main && pdflatex main`,
or upload this folder to Overleaf.

Red `[...]` markers (`\pending`) are items still to be supplied: contact email,
repository URL and wall-clock time. Every number comes from the results
summaries in `docs/` (`mddre_results_summary.md`, `confirmatory_results_summary.md`).
`figures/mddre_cost_quality_curves.pdf` and `figures/confirmatory_cost_quality_curves.pdf`
are the SVGs of the same name in `docs/figures/` converted with `rsvg-convert -f pdf`.
Bibliography entries for ACL Anthology papers are the Anthology's own BibTeX.
