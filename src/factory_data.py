"""FACTORY data for the second confirmatory test (docs/factory_preregistration.md §2).

One unit per annotated sentence that Rule A labels: hallucinated if any claim
is ``nonfactual`` or ``inconclusive``, factual if otherwise at least one claim
is ``factual``. A unit's cluster is its question; its only subclaim is the
sentence itself, and the question plays the role of HELM's article title in
the retrieval query. Units have the fields of ``src.helm_data.Unit``, so
HELM's replay helpers apply unchanged; ``generator`` is empty because the
release does not record it.

Every loader fails closed unless the release file, the preregistration and
the preregistered totals (gate G0) match. Labels live only on units;
``queries`` returns retrieval queries without them.
"""

import hashlib
import json
import re
from pathlib import Path

from src.factscore_data import sha256_path
from src.helm_data import Unit

PREREG_PATH = "docs/factory_preregistration.md"
PREREG_BYTES = 10997  # amendments only append, so this prefix stays pinned
PREREG_SHA256 = "eb8bd921d24d43bd017bc97d2b8368104143b4daf6f330c6c7093a9feef47057"

ROOT = Path("~/ddre-artifacts/confirmatory-v3/factory").expanduser()
RELEASE_FILE = ROOT / "release" / "fact_checking" / "human_annotations.jsonl"
RELEASE_SHA256 = "4d1904629e6f4570e3d59efb1d674cddeb2a76921bb223937acf9b7f63e0598f"
RELEASE_REVISION = "c820453a0c8637ad0eea2692b7e03e79acc591d1"
TAGS = {"factual": "supported", "nonfactual": "unsupported",
        "inconclusive": "inconclusive", "no verifiable facts": "no claim"}
EXPECTED = {"records": 2999, "units": 2983, "hallucinated": 1045, "clusters": 1373,
            "excluded_no_labelled_claim": 10, "excluded_only_no_verifiable_facts": 6}
EXPECTED_RULE_B = {"units": 2243, "hallucinated": 305, "clusters": 1187}
_TAG_KEY = re.compile(r"Claim (\d+) Tag$")


class FactoryDataRefused(RuntimeError):
    pass


def prereg_problems(root):
    path = Path(root) / PREREG_PATH
    if not path.exists():
        return [f"{PREREG_PATH} is missing"]
    if hashlib.sha256(path.read_bytes()[:PREREG_BYTES]).hexdigest() != PREREG_SHA256:
        return ["the FACTORY preregistration differs from its pinned digest"]
    return []


def claim_tags(record):
    """The record's non-empty claim tags, stripped and case-folded, in claim order."""
    found = sorted((int(m.group(1)), v) for k, v in record.items()
                   if (m := _TAG_KEY.match(k)))
    tags = [str(v).strip().casefold() for _, v in found if str(v).strip()]
    unknown = sorted(set(tags) - set(TAGS))
    if unknown:
        raise FactoryDataRefused(f"gate G0: unknown claim tags {unknown}")
    return tags


def rule_a(tags):
    """1 factual, 0 hallucinated, None excluded (§2.2)."""
    if "nonfactual" in tags or "inconclusive" in tags:
        return 0
    return 1 if "factual" in tags else None


def rule_b(tags):
    """Sensitivity labels (§6): hallucinated only if a claim is ``nonfactual``."""
    if "nonfactual" in tags:
        return 0
    if "inconclusive" in tags:
        return None
    return 1 if "factual" in tags else None


def read_records(path=RELEASE_FILE):
    if not Path(path).exists() or sha256_path(path) != RELEASE_SHA256:
        raise FactoryDataRefused(f"{path} is not the pinned FACTORY release file")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_factory(path=RELEASE_FILE, rule=rule_a):
    """(units, counts). Unit IDs are 0-based line numbers; clusters number the
    questions in order of first appearance."""
    records = read_records(path)
    clusters, units = {}, []
    excluded = {"excluded_no_labelled_claim": 0, "excluded_only_no_verifiable_facts": 0}
    for i, r in enumerate(records):
        tags = claim_tags(r)
        label = rule(tags)
        if label is None:
            if not tags:
                excluded["excluded_no_labelled_claim"] += 1
            elif set(tags) == {"no verifiable facts"}:
                excluded["excluded_only_no_verifiable_facts"] += 1
            continue
        question, sentence = r["question"], r["target_sentence"]
        if not question.strip() or not sentence.strip():
            raise FactoryDataRefused(f"record {i} has an empty question or sentence")
        cluster = clusters.setdefault(question, f"q{len(clusters)}")
        units.append(Unit(str(i), cluster, "", question, sentence, label))
    return units, {"records": len(records), **excluded}


def totals(units, counts):
    return {**counts, "units": len(units), "hallucinated": sum(u.label == 0 for u in units),
            "clusters": len({u.cluster for u in units})}


def check_expected(units, counts, expected=EXPECTED):
    """Gate G0 (§7)."""
    got = totals(units, counts)
    bad = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    if bad:
        raise FactoryDataRefused(f"gate G0: totals {bad}")
    return got


def queries(units):
    """Distinct (question, sentence) retrieval queries; no labels."""
    return sorted({(u.title, u.sentence) for u in units})
