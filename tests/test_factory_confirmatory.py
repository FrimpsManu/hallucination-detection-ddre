"""The FACTORY confirmatory script and decision rule, without artifacts
(scripts/run_factory_confirmatory.py, src/factory_decision.py)."""

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

import run_factory_confirmatory as FC  # noqa: E402
import run_helm_confirmatory as HC  # noqa: E402
import run_v2_budget as RB  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import factory_data as F  # noqa: E402
from src import factory_decision as FD  # noqa: E402
from src import helm_decision as D  # noqa: E402
from test_factory_data import fake_release  # noqa: E402
from test_run_v2_replay import histogram_everything  # noqa: E402
from test_v2_replay import synthetic  # noqa: E402


def draws(centres, n=500, sd=0.004):
    obs, dr = {}, {}
    for t in D.TESTS:
        c = centres.get(t, 0.0)
        obs[D.endpoint_key(t)] = c
        dr[D.endpoint_key(t)] = list(np.random.default_rng(1).normal(c, sd, n))
    return obs, dr


class Decision(unittest.TestCase):
    def test_informative_keeps_helm_rule(self):
        obs, dr = draws({t: 0.03 for t in D.TESTS})
        info = FD.informativeness(0.7, list(np.random.default_rng(2).normal(0.7, 0.01, 500)))
        self.assertTrue(info["informative"])
        d = FD.decide(obs, dr, info)
        self.assertEqual(d["statements"], D.decide(obs, dr)["statements"])
        self.assertTrue(all(s["supported"] for s in d["tests"].values()))

    def test_uninformative_overrides_every_statement(self):
        obs, dr = draws({t: 0.03 for t in D.TESTS})
        info = FD.informativeness(0.51, list(np.random.default_rng(2).normal(0.51, 0.02, 500)))
        self.assertFalse(info["informative"])
        d = FD.decide(obs, dr, info)
        self.assertEqual(set(d["statements"].values()), {FD.UNINFORMATIVE})
        self.assertFalse(any(s["supported"] for s in d["tests"].values()))
        self.assertIn("confirmed", d["hypothesis_statements_if_informative"]["H1"])

    def test_threshold_is_strict(self):
        self.assertFalse(FD.informativeness(0.5, [0.5] * 100)["informative"])

    def test_language_guard_accepts_every_statement(self):
        obs, dr = draws({})
        for info in (FD.informativeness(0.7, [0.7] * 50), FD.informativeness(0.5, [0.5] * 50)):
            AR.check_language(json.dumps(FD.decide(obs, dr, info)["statements"]))


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = synthetic(60, seed=8, max_docs=4)
        for s in cls.prepared:
            s["subclaims"] = s["subclaims"][:1]
        cls.est, cls.sup = histogram_everything()
        cls.curves = RB.budget_curves(cls.prepared, cls.est, cls.sup,
                                      RB.build_policies(cls.sup), 4)
        cls.labels = np.array([s["label"] for s in cls.prepared])
        cls.clusters = [s["cluster"] for s in cls.prepared]

    def test_analyse_and_informativeness_share_draws(self):
        with mock.patch.object(HC, "MAX_DOCS", 4), mock.patch.object(FC, "MAX_DOCS", 4):
            sub = np.arange(0, 60, 2)
            a = FC.analyse(self.curves, self.labels, self.clusters, sub, 25)
            info = FC.informativeness_bootstrap(self.curves, self.labels, self.clusters,
                                                sub, 25)
        self.assertEqual(a["units"], 30)
        self.assertEqual(a["clusters"], len({self.clusters[i] for i in sub}))
        self.assertEqual(a["decision"]["informativeness"], info)
        self.assertEqual(info["replicates"], 25)
        self.assertEqual(set(a["decision"]["statements"]), {"H1", "H2"})

    def test_concat_curves(self):
        both = FC.concat_curves(self.curves, self.curves)
        key = next(k for k in self.curves if k[1] is not None)
        point = next(iter(self.curves[key]))
        self.assertEqual(len(both[key][point]["p_rounded"]), 120)
        np.testing.assert_array_equal(both[key][point]["nli_calls"][60:],
                                      self.curves[key][point]["nli_calls"])
        with self.assertRaises(FC.FactoryConfirmatoryRefused):
            FC.concat_curves(self.curves, {})


class Units(unittest.TestCase):
    def test_rule_b_units_are_rule_a_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, sha = fake_release(tmp)
            with mock.patch.object(F, "RELEASE_SHA256", sha):
                a = F.load_factory(path)
                b = F.load_factory(path, F.rule_b)
            exp = {**F.totals(*a)}
            exp_b = {k: v for k, v in F.totals(*b).items() if k in F.EXPECTED_RULE_B}
            with mock.patch.object(F, "RELEASE_FILE", path), \
                    mock.patch.object(F, "RELEASE_SHA256", sha), \
                    mock.patch.object(F, "EXPECTED", exp), \
                    mock.patch.object(F, "EXPECTED_RULE_B", exp_b):
                units, idx_b, g0 = FC.load_units()
        self.assertEqual([units[i].id for i in idx_b], [u.id for u in b[0]])
        self.assertEqual(g0["rule_b"], exp_b)


class Inputs(unittest.TestCase):
    def test_refusals_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = FC.output_paths(Path(tmp))
            out["result"].write_text("{}")
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in FC.PINS.items()}
            with mock.patch.object(FC, "PINS", pins), \
                    mock.patch.object(FC, "FACTORY_FEATURES_MANIFEST", Path(tmp) / "none.json"):
                with self.assertRaises(FC.FactoryConfirmatoryRefused) as ctx:
                    FC.verify_inputs(out, 7)
        msg = str(ctx.exception)
        for needle in ("factory_features:", "helm_features:", "feature manifest",
                       "already exists", "n_resamples is fixed"):
            self.assertIn(needle, msg)


if __name__ == "__main__":
    unittest.main()
