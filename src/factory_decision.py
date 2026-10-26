"""The FACTORY preregistration's decision rule (docs/factory_preregistration.md §5).

HELM's rule (``src.helm_decision``) unchanged, plus §5.1: the test is
informative only if fixed-k max entailment at K = 10 has a balanced AP whose
95% bootstrap interval lies above 0.5. If it is not, every statement reads
"uninformative" and the four tests are kept only as descriptive numbers.
The check uses no DDRE output, so it cannot turn a failed hypothesis into a
supported one.
"""

from src import helm_decision as D
from src import metrics_v2 as V

INFORMATIVE_ABOVE = 0.5
INFORMATIVE_KEY = "fixed_k10_balanced_ap"
UNINFORMATIVE = ("uninformative: the evidence does not separate the labels "
                 "(fixed-k max entailment at K = 10 is not above chance); "
                 "the hypothesis tests are descriptive only")


def informativeness(observed, draws):
    """§5.1 from the bootstrap of fixed-k K = 10 balanced AP."""
    s = V.summarise(observed, draws)
    s["threshold"] = INFORMATIVE_ABOVE
    s["informative"] = bool(s.get("ci_lower") is not None and s["ci_lower"] > INFORMATIVE_ABOVE)
    return s


def decide(observed, draws, informative):
    """HELM's decision, then §5.1. ``informative`` is ``informativeness(...)``."""
    out = D.decide(observed, draws)
    out["informativeness"] = informative
    if not informative["informative"]:
        out["hypothesis_statements_if_informative"] = out["statements"]
        out["statements"] = {fam: UNINFORMATIVE for fam in out["statements"]}
        for s in out["tests"].values():
            s["supported"] = False
    return out
