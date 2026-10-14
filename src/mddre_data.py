"""Data for the multivariate DDRE study (``docs/multivariate_ddre_design.md`` §4).

FActScore entity split
    The 183 FActScore entities (all three generators, "Bridget Moynahan"
    excluded) are sorted by the SHA-256 of the entity name. The first half are
    training entities, the rest test entities.

    - Training: every generator's facts on training entities (InstructGPT,
      ChatGPT, PerplexityAI), parsed with the transfer preregistration's
      exclusions.
    - FActScore test: the transfer analysis set (ChatGPT and PerplexityAI)
      restricted to test entities.

Factcheck-Bench
    ``factcheck-GPT-benchmark.jsonl`` at Factcheck-GPT commit ``a9e6a04``.

    - A claim is included iff its checkworthiness is "factual" and its
      factuality label is True or False. "unknown" claims are excluded, as are
      opinions and non-claims.
    - A sentence is included iff it keeps at least one claim. It is factual iff
      every included claim is True.
    - A claim's documents are its released ``auto_evidence`` passages, in the
      released order. Human-added evidence is never used.

Labels are returned only on sentences and claims, never on documents.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.factscore_data import (
    EXCLUDED_ENTITIES,
    LABEL_FILE_SHA256,
    FactScoreDataRefused,
    load_analysis_set,
    parse_generator,
    sha256_path,
)

DESIGN_PATH = "docs/multivariate_ddre_design.md"
DESIGN_SHA256 = "fd96b04f2a16dfaedfb3139ec221bb0dfab2a13cf6a00738b6415676198f0df3"
DESIGN_MERGE_COMMIT = "e5c8bda8295d9569da0222a982cc051eac49d9c1"

TRAINING_GENERATORS = ("InstructGPT", "ChatGPT", "PerplexityAI")
INSTRUCTGPT_SHA256 = "4ec1fa538a9f99606130c76db63d4af7f203ca460954a3edd7c4d0e385cd2d58"
ALL_LABEL_SHA256 = {**LABEL_FILE_SHA256, "InstructGPT": INSTRUCTGPT_SHA256}
EXPECTED_ENTITIES = 182

FACTCHECK_COMMIT = "a9e6a04a953ad880806529c504a679dd1ad06528"
FACTCHECK_FILE = "factcheck-GPT-benchmark.jsonl"
FACTCHECK_SHA256 = "8bbf41a58ee1b7431fecd2a23f231f764e378e04edeb4d2a7a83949a7f40cf4b"
FACTCHECK_RESPONSES = 94
FACTCHECK_DOCS = 5


class MDDREDataRefused(RuntimeError):
    pass


def entity_hash(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def split_entities(entities):
    """(training, test) entity sets: sorted by SHA-256, first half trains."""
    ordered = sorted(entities, key=entity_hash)
    half = len(ordered) // 2
    return frozenset(ordered[:half]), frozenset(ordered[half:])


def _generator_lines(release_dir, generator):
    path = Path(release_dir) / "labeled" / f"{generator}.jsonl"
    got = sha256_path(path) if path.exists() else None
    if got != ALL_LABEL_SHA256[generator]:
        raise MDDREDataRefused(f"{path}: sha256 {got}, pinned {ALL_LABEL_SHA256[generator]}")
    return path.read_text(encoding="utf-8").splitlines()


def all_entities(release_dir):
    out = set()
    for generator in TRAINING_GENERATORS:
        for line in _generator_lines(release_dir, generator):
            if line.strip():
                row = json.loads(line)
                if row.get("annotations") and row["topic"] not in EXCLUDED_ENTITIES:
                    out.add(row["topic"])
    if len(out) != EXPECTED_ENTITIES:
        raise MDDREDataRefused(f"{len(out)} entities, expected {EXPECTED_ENTITIES}")
    return out


def factscore_split(release_dir):
    """(training sentences, FActScore test sentences, split record)."""
    train_entities, test_entities = split_entities(all_entities(release_dir))
    training = [s for g in TRAINING_GENERATORS
                for s in parse_generator(_generator_lines(release_dir, g), g)
                if s.entity in train_entities]
    try:
        analysis = load_analysis_set(release_dir)
    except FactScoreDataRefused as exc:
        raise MDDREDataRefused(str(exc)) from exc
    test = [s for s in analysis if s.entity in test_entities]
    overlap = {s.entity for s in training} & {s.entity for s in test}
    if overlap:
        raise MDDREDataRefused(f"entities in both splits: {sorted(overlap)[:5]}")
    record = {
        "rule": "entities sorted by sha256(name); first half training",
        "training_entities": len(train_entities),
        "test_entities": len(test_entities),
        "training_entities_sha256": hashlib.sha256(
            "\n".join(sorted(train_entities)).encode()).hexdigest(),
    }
    return training, test, record


@dataclass(frozen=True)
class Claim:
    text: str
    label: int  # 1 = true, 0 = false
    documents: Tuple[str, ...]


@dataclass(frozen=True)
class FactcheckSentence:
    response_index: int
    sentence_key: str
    text: str
    claims: Tuple[Claim, ...]

    @property
    def label(self):
        return int(all(c.label == 1 for c in self.claims))


def parse_factcheck(lines):
    """Factcheck-Bench sentences under the inclusion rules above. Pure."""
    out = []
    for response_index, line in enumerate(l for l in lines if l.strip()):
        row = json.loads(line)
        for key, s in row["sentences"].items():
            claims = []
            for i, text in enumerate(s["claims"]):
                label = s["claims_factuality_label"][i]
                if s["claim_checkworthiness"][i] != "factual" or label not in (True, False):
                    continue
                docs = tuple(str(d) for d in s["auto_evidence"][i])
                if len(docs) != FACTCHECK_DOCS or not all(d.strip() for d in docs):
                    raise MDDREDataRefused(f"response {response_index} {key} claim {i}: "
                                           f"{len(docs)} evidence passages")
                claims.append(Claim(str(text).strip(), int(label), docs))
            if claims:
                out.append(FactcheckSentence(response_index, key, s["text"], tuple(claims)))
    return out


def load_factcheck(release_dir) -> List[FactcheckSentence]:
    path = Path(release_dir) / FACTCHECK_FILE
    got = sha256_path(path) if path.exists() else None
    if got != FACTCHECK_SHA256:
        raise MDDREDataRefused(f"{path}: sha256 {got}, pinned {FACTCHECK_SHA256}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if sum(1 for l in lines if l.strip()) != FACTCHECK_RESPONSES:
        raise MDDREDataRefused("Factcheck-Bench response count differs")
    return parse_factcheck(lines)


def sentence_counts(sentences, claims_of):
    claims = [c for s in sentences for c in claims_of(s)]
    return {"sentences": len(sentences), "factual": sum(s.label for s in sentences),
            "hallucinated": sum(1 - s.label for s in sentences), "subclaims": len(claims)}
