"""Running time on HELM (docs/runtime_protocol.md).

Pure parts of the measurement: the sentence subset, the eight cells, the live
detector loop with its three timers, the summaries and the cluster bootstrap.
The scorer and the clock are passed in, so everything here runs without a
model; ``scripts/run_runtime.py`` supplies DeBERTa, the artifacts and the
checks.
"""

import numpy as np

from src import sequential as S

PROTOCOL_PATH = "docs/runtime_protocol.md"
N_UNITS, N_SUBSET, SUBSET_SEED = 3736, 200, 42
PASS_SEEDS = (1, 2, 3)
N_RESAMPLES, BOOTSTRAP_SEED = 10_000, 42
WARMUP_CALLS = 20
TIMERS = ("nli", "estimator", "decide_update")

# (cell, C_M, C_FA, method, K, C_retrieve), protocol §4.
CELLS = (
    ("D1", 28.0, 96.0, "BSE", 10, 1.0),
    ("D2", 28.0, 96.0, "DDRE-1D", 10, 1.0),
    ("D3", 14.0, 24.0, "BSE", 10, 1.0),
    ("D4", 14.0, 24.0, "DDRE-1D", 10, 1.0),
    ("E1", 28.0, 96.0, "BSE", 5, 0.125),
    ("E2", 28.0, 96.0, "DDRE-1D", 2, 0.25),
    ("E3", 14.0, 24.0, "BSE", 3, 1.0),
    ("E4", 14.0, 24.0, "DDRE-1D", 1, 0.125),
)
CELL = {c[0]: c for c in CELLS}
# (contrast, reference cell, method cell), protocol §9: reference - method.
CONTRASTS = (
    ("T1 28/96 equal quality", "E1", "E2"),
    ("T1 14/24 equal quality", "E3", "E4"),
    ("T2 28/96 default", "D1", "D2"),
    ("T2 14/24 default", "D3", "D4"),
)


class RuntimeMeasurementError(RuntimeError):
    pass


def subset(n_units=N_UNITS, n=N_SUBSET, seed=SUBSET_SEED):
    """Protocol §3: indices into the loader order, sorted."""
    return sorted(int(i) for i in np.random.default_rng(seed).choice(n_units, n, replace=False))


def pass_order(n_sentences, seed):
    """Protocol §6: the sentence order of one pass and, per sentence, the cell order."""
    rng = np.random.default_rng(seed)
    sentences = [int(i) for i in rng.permutation(n_sentences)]
    cells = [[CELLS[int(i)][0] for i in rng.permutation(len(CELLS))] for _ in sentences]
    return sentences, cells


def setting_key(cell):
    _, cm, cfa, _, k, cr = CELL[cell]
    return f"{cm:g}/{cfa:g}", f"K{k}/cr{cr:g}"


def policies(supports, cells=None):
    """One ``PolicyByHorizon`` per cell, every horizon built before any timing."""
    out = {}
    for name, cm, cfa, method, k, cr in CELLS:
        if cells is not None and name not in cells:
            continue
        p = S.PolicyByHorizon(supports[method], S.Costs(cm, cfa, cr), 10)
        for h in range(1, k + 1):
            p.for_documents(h)
        out[name] = p
    return out


def run_live(claim, documents, cell, estimator, policy, score_document, clock):
    """One sentence under one cell, as the HELM test runs it, timed.

    ``documents``: the saved top-10 texts in rank order. ``score_document(claim,
    text)`` returns ``(features, span_rows)`` with features ``{"e", "k", "spans"}``.
    The loop is ``S.run_subclaim_bi``'s, with each document scored when it is
    read. Returns the trajectory and the time in each part, in seconds.
    """
    _, cm, cfa, _, k, _ = CELL[cell]
    t = dict.fromkeys(TIMERS, 0.0)
    start = clock()
    p, log_odds = 0.5, S._logit(0.5)
    docs = nli = 0
    spans_scored = []
    horizon = policy.for_documents(min(len(documents), k))
    if horizon is not None:
        for rank, text in enumerate(documents[:k], start=1):
            a = clock()
            go = horizon.should_continue(p, rank)
            t["decide_update"] += clock() - a
            if not go:
                break
            a = clock()
            features, rows = score_document(claim, text)
            b = clock()
            lr = estimator.log_ratio(dict(features, rank=rank))
            c = clock()
            log_odds = min(max(log_odds + float(lr), -S.LOG_ODDS_CLIP), S.LOG_ODDS_CLIP)
            p = S._sigmoid(log_odds)
            d = clock()
            t["nli"] += b - a
            t["estimator"] += c - b
            t["decide_update"] += d - c
            docs += 1
            nli += int(features["spans"])
            spans_scored.append(rows)
    total = clock() - start
    return {"cell": cell, "p_factual": float(p), "documents": docs, "nli_calls": nli,
            "prediction": S.decision(p, cm, cfa), "total": total, **t,
            "spans": spans_scored}


def medians(rows):
    """Protocol §6: per (sentence, cell), the median of each timer over passes."""
    by = {}
    for r in rows:
        by.setdefault((r["sentence"], r["cell"]), []).append(r)
    out = {}
    for key, rs in by.items():
        if len(rs) != len(PASS_SEEDS):
            raise RuntimeMeasurementError(f"{key}: {len(rs)} passes, expected {len(PASS_SEEDS)}")
        if len({(r["documents"], r["nli_calls"]) for r in rs}) != 1:
            raise RuntimeMeasurementError(f"{key}: passes read different documents")
        out[key] = {name: float(np.median([r[name] for r in rs]))
                    for name in ("total", *TIMERS)}
        out[key].update(documents=rs[0]["documents"], nli_calls=rs[0]["nli_calls"])
    return out


def cell_summary(med, sentences):
    """Protocol §9, per cell: means per sentence and milliseconds per NLI call."""
    out = {}
    for cell, *_ in CELLS:
        rs = [med[(s, cell)] for s in sentences]
        calls = sum(r["nli_calls"] for r in rs)
        out[cell] = {
            "seconds_per_sentence": float(np.mean([r["total"] for r in rs])),
            **{f"{n}_seconds_per_sentence": float(np.mean([r[n] for r in rs]))
               for n in TIMERS},
            "documents_per_sentence": float(np.mean([r["documents"] for r in rs])),
            "nli_calls_per_sentence": float(np.mean([r["nli_calls"] for r in rs])),
            "ms_per_nli_call": (1000.0 * sum(r["nli"] for r in rs) / calls) if calls else None,
        }
    return out


def cluster_bootstrap(values, clusters, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Mean of ``values`` with a 95% percentile interval, resampling clusters."""
    values = np.asarray(values, dtype=float)
    labels = sorted(set(clusters))
    index = {c: i for i, c in enumerate(labels)}
    g = np.array([index[c] for c in clusters])
    sums = np.bincount(g, weights=values, minlength=len(labels))
    counts = np.bincount(g, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(n_resamples, len(labels)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"observed": float(values.mean()), "ci_lower": float(lo), "ci_upper": float(hi),
            "replicates": int(n_resamples), "clusters": len(labels)}


def wording(e):
    """Protocol §9: faster or slower only when the interval excludes 0."""
    if e["ci_lower"] > 0:
        return "faster"
    if e["ci_upper"] < 0:
        return "slower"
    return "no measurable difference"


def contrasts(med, sentences, clusters):
    """Reference minus method seconds per sentence (positive: the method is faster)."""
    out = {}
    for name, ref, m in CONTRASTS:
        diff = [med[(s, ref)]["total"] - med[(s, m)]["total"] for s in sentences]
        e = cluster_bootstrap(diff, clusters)
        ref_mean = float(np.mean([med[(s, ref)]["total"] for s in sentences]))
        e.update(reference=ref, method=m, reference_seconds=ref_mean,
                 percent_saved=100.0 * e["observed"] / ref_mean if ref_mean else None,
                 wording=f"{CELL[m][3]} is {wording(e)} than {CELL[ref][3]}"
                 if wording(e) != "no measurable difference" else wording(e))
        out[name] = e
    return out
