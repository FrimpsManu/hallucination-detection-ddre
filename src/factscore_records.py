"""FActScore sentences + frozen evidence -> the detectors' input records.

Turns the preregistered analysis set and the frozen top-10 evidence file into
the same ``SentenceRecord`` / ``Subclaim`` / ``EvidenceDocument`` objects the
SelfCheckGPT pipeline uses, so every frozen detector runs on FActScore
unchanged:

- a subclaim's documents are its retrieved passages in rank order, with
  premise text ``title + "\\n" + passage`` exactly as written by retrieval;
- ``passage_index`` is the entity's position in the sorted entity list: the
  bootstrap cluster (prereg §2). One entity's sentences from both generators
  share it;
- ``sentence_index`` encodes generator, response and sentence, so every
  (cluster, sentence) identity is unique.

Every subclaim must have exactly one evidence row whose entity and fact text
match; anything else refuses.
"""

import json

from src.factscore_data import GENERATORS
from src.wang_data import EvidenceDocument, SentenceRecord, Subclaim


class EvidenceMismatch(RuntimeError):
    """The evidence file does not cover the analysis set exactly."""


def read_evidence(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_key(row):
    return (row["generator"], row["response_index"], row["sentence_index"], row["fact_index"])


def sentence_identity(sentence):
    return (GENERATORS.index(sentence.generator) * 1_000_000
            + sentence.response_index * 1_000 + sentence.sentence_index)


def build_records(sentences, evidence_rows):
    by_key = {}
    for row in evidence_rows:
        key = evidence_key(row)
        if key in by_key:
            raise EvidenceMismatch(f"duplicate evidence row {key}")
        by_key[key] = row
    clusters = {entity: i for i, entity in enumerate(sorted({s.entity for s in sentences}))}
    records, used = [], set()
    for s in sentences:
        if s.sentence_index >= 1_000 or s.response_index >= 1_000:
            raise EvidenceMismatch(f"index out of range for {s.key}")
        subclaims = []
        for fact_index, fact in enumerate(s.facts):
            key = (*s.key, fact_index)
            row = by_key.get(key)
            if row is None:
                raise EvidenceMismatch(f"no evidence for subclaim {key}")
            if row["entity"] != s.entity or row["fact"] != fact.text:
                raise EvidenceMismatch(f"evidence for {key} names a different entity or fact")
            ranks = [d["rank"] for d in row["documents"]]
            if ranks != list(range(1, len(ranks) + 1)):
                raise EvidenceMismatch(f"evidence for {key} is not in rank order")
            used.add(key)
            subclaims.append(Subclaim(text=fact.text, documents=[
                EvidenceDocument(url=f"{d['title']}#{d['pidx']}", page_content=d["text"])
                for d in row["documents"]]))
        records.append(SentenceRecord(
            passage_index=clusters[s.entity],
            sentence_index=sentence_identity(s),
            sentence=s.text,
            label=s.label,
            raw_label="factual" if s.label else "hallucinated",
            subclaims=subclaims,
        ))
    extra = set(by_key) - used
    if extra:
        raise EvidenceMismatch(f"{len(extra)} evidence rows match no subclaim")
    identities = [(r.passage_index, r.sentence_index) for r in records]
    if len(set(identities)) != len(identities):
        raise EvidenceMismatch("sentence identities are not unique")
    return records, clusters
