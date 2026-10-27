#!/usr/bin/env python3
"""Exploratory follow-ups to the FACTORY confirmatory test (not preregistered).

Three descriptive questions, from the frozen features and results; nothing
frozen is rerun, refitted or overwritten:

1. **Where the curves cross.** DDRE-1D's and fixed-k max entailment's
   envelopes (balanced AP against NLI calls per sentence), on FACTORY and
   HELM, evaluated on a grid over their shared cost range.
2. **Evidence strength.** The H2 AUC-gap within tertiles of a label-free
   evidence-strength score, s = the largest entailment or contradiction score
   over a sentence's ten passages (each dataset's own tertiles). It asks
   whether DDRE-1D's edge over fixed-k depends on how informative the
   evidence is.
3. **Why the pooled gaps exceed both datasets'.** Balanced AP on the pooled
   units also rewards ranking HELM sentences against FACTORY sentences,
   which per-dataset AP never sees. The script compares each method's pooled
   balanced AP at the operating point with the unit-weighted mean of its two
   per-dataset values.

The output is one JSON file, written once, read-only, under
``~/ddre-artifacts/confirmatory-v3/factory/exploratory/``.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_factory_confirmatory as FC  # noqa: E402
import run_frozen_heldout as frozen  # noqa: E402
import run_helm_confirmatory as HC  # noqa: E402
import run_v2_budget as RB  # noqa: E402
import run_v2_replay as RV  # noqa: E402
from src import factory_data as F  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import metrics_v2 as V  # noqa: E402
from src import v2_budget as B  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

RESULT = (F.ROOT / "factory_confirmatory_result_float16.json",
          "55efea0501cf78e57bcbca157fa15f03f387ef261b16b471229ba9e2788eaeb6")
OUT = F.ROOT / "exploratory" / "factory_followups.json"
N_RESAMPLES, SEED = 2_000, 42
GRID = 200


def envelope(curves, labels, method, pair, idx):
    import numpy as np

    pts = curves[(method, None if method == B.FIXED else pair)]
    keys = sorted(pts)
    return V.Envelope([float(np.mean(pts[k]["nli_calls"][idx])) for k in keys],
                      [V.balanced_ap_fast(labels[idx], pts[k]["p_rounded"][idx]) for k in keys])


def crossing(curves, labels, idx):
    """Q_DDRE(c) - Q_fixed(c) on a grid over the shared cost range, per cost pair."""
    import numpy as np

    out = {}
    for pair in R.COST_PAIRS:
        m = envelope(curves, labels, "DDRE-1D", pair, idx)
        f = envelope(curves, labels, B.FIXED, pair, idx)
        lo, hi = max(m.cost_range[0], f.cost_range[0]), min(m.cost_range[1], f.cost_range[1])
        cs = np.linspace(lo, hi, GRID)
        d = np.array([m.quality_at(c) - f.quality_at(c) for c in cs])
        tag = f"{pair[0]:g}/{pair[1]:g}"
        out[tag] = {"shared_range": [float(lo), float(hi)],
                    "fraction_above": float(np.mean(d > 0)),
                    "fraction_below": float(np.mean(d < 0)),
                    "gap_by_cost_band": {
                        f"{a:g}-{b:g}": float(np.mean(d[(cs >= a) & (cs < b)]))
                        for a, b in ((1, 2), (2, 4), (4, 7), (7, 10.01))
                        if np.any((cs >= a) & (cs < b))},
                    "fixed_k_quality": {f"K{k}": f.quality_at(float(k)) for k in (1, 2, 5, 10)}}
    return out


def strength(prepared):
    """Label-free evidence strength: max over passages of max(entailment, contradiction);
    0 for a sentence without passages."""
    import numpy as np

    return np.array([max((max(d["e"], d["k"]) for d in s["subclaims"][0]), default=0.0)
                     for s in prepared])


def by_strength(curves, labels, clusters, prepared):
    import numpy as np

    s = strength(prepared)
    cuts = np.quantile(s, [1 / 3, 2 / 3])
    out = {"cuts": [float(c) for c in cuts]}
    for name, mask in (("weak", s <= cuts[0]), ("middle", (s > cuts[0]) & (s <= cuts[1])),
                       ("strong", s > cuts[1])):
        idx = np.flatnonzero(mask)
        obs, draws, n = HC.bootstrap(curves, labels, clusters, idx, N_RESAMPLES)
        row = {"units": int(idx.size), "clusters": n,
               "hallucinated": int(np.sum(labels[idx] == 0)),
               "fixed_k10_balanced_ap": V.balanced_ap(labels[idx],
                                                      curves[(B.FIXED, None)]["K10"]["p_rounded"][idx])}
        for ref in ("fixed-max", "BSE"):
            for pair in ("28/96", "14/24"):
                k = f"DDRE-1D|{ref}|{pair}|auc_gap"
                row[f"vs {ref} {pair}"] = V.summarise(obs[k], draws[k])
        out[name] = row
    return out


def pooled_vs_mean(helm, factory, helm_labels, factory_labels):
    """Balanced AP at the operating point: pooled vs unit-weighted per-dataset mean."""
    import numpy as np

    op = B.operating_key(HC.MAX_DOCS)
    nh, nf = len(helm_labels), len(factory_labels)
    y = np.concatenate([helm_labels, factory_labels])
    out = {}
    for m in ("BSE", "DDRE-1D"):
        for pair in R.COST_PAIRS:
            ph, pf = helm[(m, pair)][op]["p_rounded"], factory[(m, pair)][op]["p_rounded"]
            per = (nh * V.balanced_ap(helm_labels, ph) + nf * V.balanced_ap(factory_labels, pf)) \
                / (nh + nf)
            out[f"{m}|{pair[0]:g}/{pair[1]:g}"] = {
                "pooled": V.balanced_ap(y, np.concatenate([ph, pf])),
                "weighted_mean_of_datasets": per,
                "mean_p_factual": {"helm": float(np.mean(ph)), "factory": float(np.mean(pf))}}
    out["factual_rate"] = {"helm": float(np.mean(helm_labels)),
                           "factory": float(np.mean(factory_labels))}
    return out


def main():
    import numpy as np

    from src.mddre_features import FeatureReader

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; written once")
    if sha256_path(RESULT[0]) != RESULT[1]:
        raise SystemExit("the FACTORY result differs from its pin")
    FC.verify_inputs({}, FC.N_RESAMPLES, dry_run=True)
    est, sup = RV.load_estimators(json.loads(FC.PINS["estimators"][0].read_text()))
    policies = RB.build_policies(sup)

    units, _, _ = FC.load_units()
    fm = json.loads(FC.FACTORY_FEATURES_MANIFEST.read_text())
    reader = FeatureReader(FC.PINS["factory_features"][0], frozen.OFFICIAL_MODEL, fm["dtype"])
    try:
        prepared = HC.prepare_helm(units, FC.PINS["factory_evidence"][0], reader)
    finally:
        reader.close()
    curves = RB.budget_curves(prepared, est, sup, policies, HC.MAX_DOCS)
    labels = np.array([u.label for u in units])
    clusters = [u.cluster for u in units]
    frozen_result = json.loads(RESULT[0].read_text())
    for k in ("DDRE-1D|fixed-max|28/96|auc_gap", "DDRE-1D|BSE|14/24|auc_gap"):
        got = B.contrast_endpoints(curves, labels, np.arange(len(units)), HC.MAX_DOCS,
                                   R.COST_PAIRS)[k]
        if abs(got - frozen_result["endpoints"][k]["observed"]) > 1e-12:
            raise SystemExit(f"{k}: recomputed {got} differs from the frozen result")

    hunits, hcurves = FC.helm_curves(est, sup, policies)
    hlabels = np.array([u.label for u in hunits])
    hclusters = [u.cluster for u in hunits]
    hreader = FeatureReader(HC.PINS["helm_features"][0], frozen.OFFICIAL_MODEL,
                            json.loads(HC.HELM_FEATURES_MANIFEST.read_text())["dtype"])
    try:
        hprepared = HC.prepare_helm(hunits, HC.PINS["helm_evidence"][0], hreader)
    finally:
        hreader.close()

    result = {
        "artifact": "factory-exploratory-followups",
        "status": "exploratory, not preregistered",
        "factory_result_sha256": RESULT[1],
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": SEED, "unit": "cluster"},
        "crossing": {"factory": crossing(curves, labels, np.arange(len(units))),
                     "helm": crossing(hcurves, hlabels, np.arange(len(hunits)))},
        "evidence_strength": {
            "factory": by_strength(curves, labels, clusters, prepared),
            "helm": by_strength(hcurves, hlabels, hclusters, hprepared)},
        "pooled_vs_per_dataset": pooled_vs_mean(hcurves, curves, hlabels, labels),
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    out.chmod(0o444)
    print(json.dumps(result, indent=1, default=float))
    print(f"wrote {out}\n  sha256 {sha256_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
