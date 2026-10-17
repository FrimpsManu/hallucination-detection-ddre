"""HELM data for the confirmatory test (docs/helm_preregistration.md §2).

One unit per released sentence of every generator. The released ``label`` is
1 for hallucination, so the factual label used here is 1 - label. A unit's
cluster is its Wikipedia article (the passage ID shared across generators);
its only subclaim is the sentence itself. The article title is read from the
prompt by the preregistered rule (``parse_title``).

Every loader fails closed unless the pinned file listing, the preregistration
and the preregistered totals (gate G0) match. Labels live only on units;
``queries`` returns retrieval queries without them.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.factscore_data import sha256_path

PREREG_PATH = "docs/helm_preregistration.md"
PREREG_BYTES = 11084  # amendments only append, so this prefix stays pinned
PREREG_SHA256 = "b191e221f3c10d8615a06143d5e1a8d7bb1497c9a53483bb77349faa50952ce1"

ROOT = Path("~/ddre-artifacts/confirmatory-v2/helm").expanduser()
RELEASE = ROOT / "release"
LISTING = ROOT / "file_sha256.txt"
LISTING_SHA256 = "3a315449f46e2e21feff715308c21c48b9dc50ac2a05e0dedbe8f078f9fe33bc"
RELEASE_COMMIT = "3573342d20db0feb6658a26347337ea8a4217526"
GENERATORS = ("falcon40b", "gptj7b", "llamabase7b", "llamachat13b", "llamachat7b",
              "mpt7b", "opt7b")
PROMPT_PREFIX = "This is a Wikipedia passage about "
EXPECTED = {"units": 3736, "hallucinated": 1978, "clusters": 229, "generators": 7,
            "titles": 229}
# The six titles containing abbreviations (preregistration §2.3).
ABBREVIATION_TITLES = {
    "86": "U.S. Route 491", "132": "Davis v. Ayala",
    "140": "Fort Bayard ( Washington , D.C. )", "270": "Haywood S. Hansell",
    "360": "Voltaire P. Twombly", "368": "Arthur W. Radford",
}


class HelmDataRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Unit:
    id: str          # "<generator>:<passage id>:<sentence index>"
    cluster: str     # passage (article) id
    generator: str
    title: str
    sentence: str
    label: int       # 1 factual, 0 hallucinated


def prereg_problems(root):
    path = Path(root) / PREREG_PATH
    if not path.exists():
        return [f"{PREREG_PATH} is missing"]
    if hashlib.sha256(path.read_bytes()[:PREREG_BYTES]).hexdigest() != PREREG_SHA256:
        return ["the HELM preregistration differs from its pinned digest"]
    return []


def parse_title(prompt):
    """The article title in "This is a Wikipedia passage about {title}. {opening}".

    The title ends at the first ". " whose preceding token is not a single
    capital letter, not "v", and does not itself contain "." (§2.3).
    """
    if not prompt.startswith(PROMPT_PREFIX):
        raise HelmDataRefused(f"prompt without the expected prefix: {prompt[:60]!r}")
    rest = prompt[len(PROMPT_PREFIX):]
    start = 0
    while True:
        i = rest.find(". ", start)
        if i < 0:
            raise HelmDataRefused(f"no title boundary in prompt: {prompt[:80]!r}")
        token = rest[:i].split()[-1] if rest[:i].split() else ""
        initial = len(token) == 1 and token.isalpha() and token.isupper()
        if not (initial or token == "v" or "." in token):
            return rest[:i]
        start = i + 1


def release_matches(release_dir=RELEASE, listing=LISTING):
    """Every listed file matches the pinned ``shasum -a 256`` listing."""
    if not Path(listing).exists() or sha256_path(listing) != LISTING_SHA256:
        return False
    for line in Path(listing).read_text().splitlines():
        sha, _, rel = line.partition("  ")
        if sha256_path(Path(release_dir) / rel) != sha:
            return False
    return True


def load_helm(release_dir=RELEASE, listing=LISTING):
    if not release_matches(release_dir, listing):
        raise HelmDataRefused(f"{release_dir} is not the pinned HELM release")
    units, titles = [], {}
    for gen in GENERATORS:
        data = json.loads((Path(release_dir) / "helm" / "data" / gen / "data.json").read_text())
        for pid, item in data.items():
            title = parse_title(item["prompt"])
            if titles.setdefault(pid, title) != title:
                raise HelmDataRefused(f"passage {pid} has different prompts across generators")
            for i, s in enumerate(item["sentences"]):
                if s["label"] not in (0, 1):
                    raise HelmDataRefused(f"{gen}:{pid}:{i} has label {s['label']!r}")
                units.append(Unit(f"{gen}:{pid}:{i}", str(pid), gen, title,
                                  s["sentence"], 1 - int(s["label"])))
    return units, titles


def totals(units, titles):
    return {"units": len(units), "hallucinated": sum(u.label == 0 for u in units),
            "clusters": len({u.cluster for u in units}),
            "generators": len({u.generator for u in units}), "titles": len(titles)}


def check_expected(units, titles):
    """Gate G0 (§7)."""
    got = totals(units, titles)
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    wrong = {pid: titles.get(pid) for pid, t in ABBREVIATION_TITLES.items()
             if titles.get(pid) != t}
    if bad or wrong:
        raise HelmDataRefused(f"gate G0: totals {bad}, abbreviation titles {wrong}")
    return got


def queries(units):
    """Distinct (title, sentence) retrieval queries; no labels."""
    return sorted({(u.title, u.sentence) for u in units})
