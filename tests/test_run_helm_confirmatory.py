"""The HELM confirmatory script (scripts/run_helm_confirmatory.py), without artifacts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_helm_confirmatory as HC  # noqa: E402
import run_v2_budget as RB  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import helm_decision as D  # noqa: E402
from test_run_v2_replay import histogram_everything  # noqa: E402
from test_v2_replay import synthetic  # noqa: E402


class FakeReader:
    def document(self, claim, text, rank):
        return {"e": float(len(text) % 90 + 5), "k": 10.0, "rank": rank, "spans": 1}


def units():
    return [H.Unit("g1:7:0", "7", "g1", "Fort Bayard ( Washington , D.C. )", "S0 .", 1),
            H.Unit("mpt7b:7:0", "7", "mpt7b", "Fort Bayard ( Washington , D.C. )", "S1 .", 0)]


def evidence(path):
    rows = []
    for u in units():
        docs = [{"title": "Fort Bayard (Washington, D.C.)" if i in (0, 3) else f"Other {i}",
                 "text": f"passage {i} " * (i + 1)} for i in range(10)]
        rows.append({"id": u.id, "documents": docs})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class Prepare(unittest.TestCase):
    def test_primary_keeps_ten_documents_in_rank_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            evidence(p)
            prepared = HC.prepare_helm(units(), p, FakeReader())
        self.assertEqual([d["rank"] for d in prepared[0]["subclaims"][0]], list(range(1, 11)))
        self.assertEqual(prepared[1]["label"], 0)

    def test_source_article_exclusion_renumbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            evidence(p)
            prepared = HC.prepare_helm(units(), p, FakeReader(), exclude_source=True)
        docs = prepared[0]["subclaims"][0]
        self.assertEqual(len(docs), 8)
        self.assertEqual([d["rank"] for d in docs], list(range(1, 9)))

    def test_order_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            evidence(p)
            with self.assertRaises(HC.HelmConfirmatoryRefused):
                HC.prepare_helm(units()[::-1], p, FakeReader())

    def test_title_normalisation(self):
        self.assertEqual(HC.normalise_title("Fort Bayard ( Washington , D.C. )"),
                         HC.normalise_title("Fort Bayard (Washington, D.C.)"))


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = synthetic(60, seed=8, max_docs=4)
        for s in cls.prepared:   # single subclaim per unit, as in HELM
            s["subclaims"] = s["subclaims"][:1]
        cls.est, cls.sup = histogram_everything()
        cls.curves = RB.budget_curves(cls.prepared, cls.est, cls.sup,
                                      RB.build_policies(cls.sup), 4)
        cls.labels = np.array([s["label"] for s in cls.prepared])
        cls.clusters = [s["cluster"] for s in cls.prepared]

    def test_bootstrap_maps_subset_indices(self):
        with mock.patch.object(HC, "MAX_DOCS", 4):
            sub = list(range(0, 60, 2))
            obs, draws, n = HC.bootstrap(self.curves, self.labels, self.clusters, sub, 20)
        self.assertEqual(n, len({self.clusters[i] for i in sub}))
        self.assertIn(D.endpoint_key("H1a"), obs)
        self.assertEqual(len(draws[D.endpoint_key("H2b")]), 20)
        d = D.decide(obs, draws)
        self.assertEqual(set(d["statements"]), {"H1", "H2"})

    def test_language_guard_accepts_every_statement_form(self):
        from src import ablation_replay as AR
        for centers in ({"H1a": 1}, {"H1a": 1, "H1b": 1}, {"H1a": -1, "H1b": -1}, {}):
            obs, dr = {}, {}
            for t in D.TESTS:
                c = 0.02 * centers.get(t, 0)
                obs[D.endpoint_key(t)] = c
                dr[D.endpoint_key(t)] = list(np.random.default_rng(1).normal(c, 0.004, 500))
            AR.check_language(json.dumps(D.decide(obs, dr)["statements"]))


class Inputs(unittest.TestCase):
    def test_refusals_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = HC.output_paths(Path(tmp))
            out["result"].write_text("{}")
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in HC.PINS.items()}
            with mock.patch.object(HC, "PINS", pins), \
                    mock.patch.object(HC, "HELM_FEATURES_MANIFEST", Path(tmp) / "none.json"):
                with self.assertRaises(HC.HelmConfirmatoryRefused) as ctx:
                    HC.verify_inputs(out, 7)
            msg = str(ctx.exception)
            self.assertIn("helm_features:", msg)
            self.assertIn("feature manifest", msg)
            self.assertIn("already exists", msg)
            self.assertIn("n_resamples is fixed", msg)


if __name__ == "__main__":
    unittest.main()
