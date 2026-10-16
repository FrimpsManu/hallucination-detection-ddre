"""Data for the confirmatory test (``docs/confirmatory_preregistration.md`` §2).

ExpertQA (primary)
    One unit per labelled sentence (``claim_string``). Its subclaims are the
    released ``atomic_claims``. Each atomic claim's documents are its released
    ``atomic_evidences`` passages, split at ``Passage ID i: `` markers and kept in
    ID order. Factual = Definitely/Probably correct; hallucinated = Likely/
    Definitely incorrect; Unsure and missing labels, and sentences with no
    atomic claims, are excluded. Cluster: the question author.

BingCheck (second)
    One unit per claim, the keys of ``subconclusions``; the claim is its own
    single subclaim. Its documents are ``search_results[claim]`` in stored
    order, premise = the page ``content`` joined by spaces. Factual = supported;
    hallucinated = refuted or partially supported; "not supported" is
    excluded. Cluster: the response.

Never read: BingCheck's human-selected ``evidence`` and ExpertQA's
``revised_*`` fields, which encode the annotator's judgement. Labels live only
on units. Every loader fails closed unless the pinned hashes and the
preregistered totals match.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from src.factscore_data import sha256_path

PREREG_PATH = "docs/confirmatory_preregistration.md"
PREREG_SHA256 = "db4ff73ae0946b8cb8c57bf98a48763385ab9b1f469b12d4890b5e1d2d063232"
PREREG_MERGE_COMMIT = "c554eb6807798c8b220b6cadef632683a5def3af"
PREREG_BYTES = 10369  # amendments only append, so this prefix stays pinned
PREREG_AMENDMENT_1_SHA256 = "89dd2f6463ccc98303809a01d5380d2918397010af88af3f7a8ccd5e1cea29fb"

ARTIFACTS = Path("~/ddre-artifacts/confirmatory").expanduser()
EXPERTQA_DIR = ARTIFACTS / "expertqa" / "release"
BINGCHECK_DIR = ARTIFACTS / "bingcheck" / "release"

EXPERTQA_COMMIT = "1aa7ba81bd4c083c12ef1aa2fee338f6cb60fa87"
EXPERTQA_FILE = "r2_compiled_out_corrected_revised_atomic_w_evidences_factscores_anon.jsonl"
EXPERTQA_SHA256 = "5e6afb0453bc8c5fabfe43fbef0b205ceb3175694640cfd1f5d47a97900d0ac7"
EXPERTQA_MAX_DOCS = 5

BINGCHECK_COMMIT = "d33157db37c4475ae1b9a1df4907e7368b35e995"
BINGCHECK_LISTING_SHA256 = "0432979f4c7ca6533a624cd8bc425b4b31b777a50f3a9af7800cff9cb606d5f0"
BINGCHECK_FILES = 397
BINGCHECK_PLACEHOLDER = "examples.json"  # 134-byte placeholder, not a response
BINGCHECK_MAX_DOCS = 3

EXPERTQA_FACTUAL = ("Definitely correct", "Probably correct")
EXPERTQA_HALLUCINATED = ("Likely incorrect", "Definitely incorrect")

# label -> 1 factual, 0 hallucinated, None excluded. "primary" is preregistered
# for the decision; the others are the §6 sensitivity analyses.
EXPERTQA_MAPPINGS = {
    "primary": {**{k: 1 for k in EXPERTQA_FACTUAL}, **{k: 0 for k in EXPERTQA_HALLUCINATED}},
    "unsure_hallucinated": {**{k: 1 for k in EXPERTQA_FACTUAL},
                            **{k: 0 for k in EXPERTQA_HALLUCINATED}, "Unsure": 0},
}
BINGCHECK_MAPPINGS = {
    "primary": {"supported": 1, "refuted": 0, "partially supported": 0},
    "refuted_only": {"supported": 1, "refuted": 0},
    "not_supported_hallucinated": {"supported": 1, "refuted": 0,
                                   "partially supported": 0, "not supported": 0},
}

# Preregistered totals for the primary mapping (§2, gate G0).
EXPECTED = {
    "expertqa": {"units": 11389, "hallucinated": 596, "clusters": 483},  # amendment 1
    "bingcheck": {"units": 3623, "hallucinated": 942, "clusters": 396},
}

PASSAGE_MARKER = re.compile(r"Passage ID (\d+): ")


class ConfirmatoryDataRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Subclaim:
    text: str
    documents: Tuple[str, ...]  # in rank order; rank = position + 1


@dataclass(frozen=True)
class Unit:
    id: str
    cluster: str
    label: int  # 1 factual, 0 hallucinated
    subclaims: Tuple[Subclaim, ...]


def prereg_problems(root):
    """Why the preregistration file is not the merged text plus pinned amendments."""
    data = (Path(root) / PREREG_PATH).read_bytes()
    problems = []
    if hashlib.sha256(data[:PREREG_BYTES]).hexdigest() != PREREG_SHA256:
        problems.append("the confirmatory preregistration differs from its pinned digest")
    if hashlib.sha256(data).hexdigest() != PREREG_AMENDMENT_1_SHA256:
        problems.append("the preregistration is not at amendment 1")
    return problems


def split_passages(evidence):
    """Passages of one ``atomic_evidences`` string, in ID order.

    Refuses unless the markers run 1, 2, ..., n. Empty strings mean no passage.
    """
    if not evidence:
        return ()
    parts = PASSAGE_MARKER.split(evidence)
    if parts[0].strip():
        raise ConfirmatoryDataRefused("text before the first passage marker")
    ids = [int(x) for x in parts[1::2]]
    if ids != list(range(1, len(ids) + 1)):
        raise ConfirmatoryDataRefused(f"passage markers out of order: {ids}")
    return tuple(p.strip() for p in parts[2::2])


def load_expertqa(release_dir=EXPERTQA_DIR, mapping="primary"):
    path = Path(release_dir) / EXPERTQA_FILE
    if sha256_path(path) != EXPERTQA_SHA256:
        raise ConfirmatoryDataRefused(f"{path} is not the pinned ExpertQA release")
    labels = EXPERTQA_MAPPINGS[mapping]
    units = []
    with open(path) as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            if len(row["answers"]) != 1:
                raise ConfirmatoryDataRefused(f"row {row_index} has {len(row['answers'])} answers")
            (answer,) = row["answers"].values()
            for claim_index, claim in enumerate(answer["claims"]):
                label = labels.get(claim.get("correctness"))
                atoms = claim.get("atomic_claims") or []
                if label is None or not atoms:
                    continue
                evidences = claim.get("atomic_evidences") or []
                if len(evidences) != len(atoms):
                    raise ConfirmatoryDataRefused(
                        f"row {row_index} claim {claim_index}: {len(atoms)} atomic claims, "
                        f"{len(evidences)} evidence lists")
                subclaims = tuple(
                    Subclaim(a, tuple(p for p in split_passages(e) if p)[:EXPERTQA_MAX_DOCS])
                    for a, e in zip(atoms, evidences))
                units.append(Unit(f"{row_index}:{claim_index}", str(row["annotator_id"]),
                                  label, subclaims))
    return units


BINGCHECK_LISTING = ARTIFACTS / "bingcheck" / "file_sha256.txt"


def bingcheck_release_matches(release_dir, listing=BINGCHECK_LISTING):
    """Every file matches the pinned ``shasum -a 256`` listing (order-free).

    The listing itself is pinned by BINGCHECK_LISTING_SHA256 (preregistration §2).
    """
    if sha256_path(listing) != BINGCHECK_LISTING_SHA256:
        return False
    want = set(Path(listing).read_text().splitlines())
    got = {f"{sha256_path(p)}  {p.name}" for p in Path(release_dir).glob("*.json")}
    return got == want


def load_bingcheck(release_dir=BINGCHECK_DIR, mapping="primary", listing=BINGCHECK_LISTING):
    release_dir = Path(release_dir)
    files = sorted(release_dir.glob("*.json"))
    if len(files) != BINGCHECK_FILES or not bingcheck_release_matches(release_dir, listing):
        raise ConfirmatoryDataRefused(f"{release_dir} is not the pinned BingCheck release")
    labels = BINGCHECK_MAPPINGS[mapping]
    units = []
    for f in files:
        if f.name == BINGCHECK_PLACEHOLDER:
            continue
        data = json.loads(f.read_text())
        if len(data) != 1:
            raise ConfirmatoryDataRefused(f"{f.name} holds {len(data)} responses")
        (response_id, item), = data.items()
        checking = item["fact-checking"]
        results = checking.get("search_results", {})
        for claim_index, (claim, verdict) in enumerate(checking["subconclusions"].items()):
            label = labels.get(verdict)
            if label is None:
                continue
            pages = []
            for r in results.get(claim, [])[:BINGCHECK_MAX_DOCS]:
                content = r["content"]
                text = " ".join(content) if isinstance(content, list) else str(content)
                if text.strip():
                    pages.append(text)
            units.append(Unit(f"{response_id}:{claim_index}", response_id, label,
                              (Subclaim(claim, tuple(pages)),)))
    return units


LOADERS = {"expertqa": load_expertqa, "bingcheck": load_bingcheck}
MAX_DOCS = {"expertqa": EXPERTQA_MAX_DOCS, "bingcheck": BINGCHECK_MAX_DOCS}


def totals(units):
    return {"units": len(units), "hallucinated": sum(u.label == 0 for u in units),
            "clusters": len({u.cluster for u in units}),
            "subclaims": sum(len(u.subclaims) for u in units),
            "subclaims_without_documents": sum(not s.documents for u in units
                                               for s in u.subclaims),
            "documents": sum(len(s.documents) for u in units for s in u.subclaims)}


def check_expected(name, units):
    """Gate G0 for the primary mapping."""
    got = totals(units)
    want = EXPECTED[name]
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise ConfirmatoryDataRefused(f"{name} totals differ from the preregistration: {bad}")
    return got
