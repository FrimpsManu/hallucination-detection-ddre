"""HELM loader and retrieval script (src/helm_data.py, scripts/retrieve_helm_evidence.py)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src import helm_data as H  # noqa: E402

P = H.PROMPT_PREFIX


class Titles(unittest.TestCase):
    def test_preregistered_abbreviation_titles(self):
        cases = {
            "U.S. Route 491. U.S. Route 491 ( US 491 ) is a highway .": "U.S. Route 491",
            "Davis v. Ayala. Davis v. Ayala , 576 U.S. was a case .": "Davis v. Ayala",
            "Fort Bayard ( Washington , D.C. ). Fort Bayard was a fort .":
                "Fort Bayard ( Washington , D.C. )",
            "Haywood S. Hansell. Haywood Shepherd Hansell Jr .": "Haywood S. Hansell",
            "Voltaire P. Twombly. Voltaire Paine Twombly was a veteran .": "Voltaire P. Twombly",
            "Arthur W. Radford. Arthur William Radford was an admiral .": "Arthur W. Radford",
        }
        for rest, title in cases.items():
            self.assertEqual(H.parse_title(P + rest), title)

    def test_ordinary_titles_and_a_lowercase_letter(self):
        self.assertEqual(H.parse_title(P + "Gliese 876 b. Gliese 876 b is a planet ."),
                         "Gliese 876 b")
        self.assertEqual(H.parse_title(P + "Nick Mitchell. Nicholas Cole Mitchell is ."),
                         "Nick Mitchell")

    def test_refusals(self):
        with self.assertRaises(H.HelmDataRefused):
            H.parse_title("A Wikipedia passage about X. Y")
        with self.assertRaises(H.HelmDataRefused):
            H.parse_title(P + "no boundary at all")


def fake_release(tmp, generators=("g1", "g2"), bad_label=False, clash=False):
    release = Path(tmp) / "release"
    lines = []
    for gi, g in enumerate(generators):
        d = release / "helm" / "data" / g
        d.mkdir(parents=True)
        title = "Other" if (clash and gi == 1) else "Alpha"
        data = {"7": {"prompt": P + f"{title}. Alpha is a thing .",
                      "sentences": [{"sentence": "Alpha is red .", "label": 1},
                                    {"sentence": "Alpha is big .", "label": 2 if bad_label else 0}]},
                "9": {"prompt": P + "Beta. Beta was .",
                      "sentences": [{"sentence": "Beta was born .", "label": 0}]}}
        path = d / "data.json"
        path.write_text(json.dumps(data))
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./helm/data/{g}/data.json")
    listing = Path(tmp) / "listing.txt"
    listing.write_text("\n".join(lines) + "\n")
    return release, listing, hashlib.sha256(listing.read_bytes()).hexdigest()


class Loader(unittest.TestCase):
    def load(self, **kw):
        tmp = tempfile.mkdtemp()
        release, listing, sha = fake_release(tmp, **kw)
        with mock.patch.object(H, "LISTING_SHA256", sha), \
                mock.patch.object(H, "GENERATORS", ("g1", "g2")):
            return H.load_helm(release, listing)

    def test_units_labels_clusters(self):
        units, titles = self.load()
        self.assertEqual(len(units), 6)
        self.assertEqual(titles, {"7": "Alpha", "9": "Beta"})
        first = units[0]
        self.assertEqual((first.id, first.cluster, first.generator), ("g1:7:0", "7", "g1"))
        self.assertEqual(first.label, 0)          # released 1 = hallucination
        self.assertEqual(units[1].label, 1)
        self.assertEqual(H.totals(units, titles)["clusters"], 2)

    def test_queries_are_distinct_and_carry_no_labels(self):
        units, _ = self.load()
        q = H.queries(units)
        self.assertEqual(len(q), 3)               # identical sentences across generators
        self.assertTrue(all(len(x) == 2 and all(isinstance(v, str) for v in x) for x in q))

    def test_bad_label_and_inconsistent_prompts_refused(self):
        with self.assertRaises(H.HelmDataRefused):
            self.load(bad_label=True)
        with self.assertRaises(H.HelmDataRefused):
            self.load(clash=True)

    def test_a_changed_file_is_refused(self):
        tmp = tempfile.mkdtemp()
        release, listing, sha = fake_release(tmp)
        (release / "helm" / "data" / "g1" / "data.json").write_text("{}")
        with mock.patch.object(H, "LISTING_SHA256", sha):
            with self.assertRaises(H.HelmDataRefused):
                H.load_helm(release, listing)

    def test_gate_g0(self):
        units, titles = self.load()
        with self.assertRaises(H.HelmDataRefused):
            H.check_expected(units, titles)        # the real totals differ
        want = dict(H.totals(units, titles))
        with mock.patch.object(H, "EXPECTED", want), \
                mock.patch.object(H, "ABBREVIATION_TITLES", {"9": "Beta"}):
            self.assertEqual(H.check_expected(units, titles), want)

    def test_preregistration_pin(self):
        self.assertEqual(H.prereg_problems(ROOT), [])


class RetrievalScript(unittest.TestCase):
    def test_refusals_without_artifacts(self):
        import retrieve_helm_evidence as RH
        with tempfile.TemporaryDirectory() as tmp:
            out = RH.output_paths(Path(tmp))
            out["evidence"].write_text("")
            with mock.patch.object(H, "LISTING", Path(tmp) / "absent.txt"):
                with self.assertRaises(RH.HelmRetrievalRefused) as ctx:
                    RH.verify_inputs(Path(tmp), out, dry_run=False)
            msg = str(ctx.exception)
            self.assertIn("file listing", msg)
            self.assertIn("BM25 index manifest", msg)
            self.assertIn("already exists", msg)
            with mock.patch.object(H, "LISTING", Path(tmp) / "absent.txt"):
                with self.assertRaises(RH.HelmRetrievalRefused) as ctx:
                    RH.verify_inputs(Path(tmp), out, dry_run=True)
            self.assertNotIn("already exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
