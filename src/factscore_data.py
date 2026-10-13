"""FActScore analysis set, exactly as fixed in the transfer preregistration.

``docs/factscore_transfer_preregistration.md`` §1-§2 fix the data: the
ChatGPT and PerplexityAI human-annotated files (pinned by SHA-256), with the
InstructGPT subset and the entity "Bridget Moynahan" excluded, declined
responses, irrelevant sentences and IR facts removed, and every remaining
sentence's S/NS facts as its subclaims. A sentence is factual iff every
subclaim is S.

The loader refuses to return anything if the files are not the pinned ones or
the counts differ from the preregistered table.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

PREREGISTRATION_PATH = "docs/factscore_transfer_preregistration.md"
PREREGISTRATION_SHA256 = "5aa2a2a6d3ec07fa6f2ed2fb294ff590510c614efdf67c4d91d231b93f43a385"
PREREGISTRATION_MERGE_COMMIT = "ac872e6d7f830aac12e6e64654b0f7d7a2a8860c"

GENERATORS = ("ChatGPT", "PerplexityAI")
LABEL_FILE_SHA256 = {
    "ChatGPT": "b58fe7113acc28dee8d3e6ee7e426bfdedaa293bba270f0977f568b9319e92d4",
    "PerplexityAI": "81ada151f2de8b7c205e3c15491e54b92d89bfb37b05366dca8b7a0486f109ff",
}
EXCLUDED_ENTITIES = frozenset({"Bridget Moynahan"})

# Preregistered analysis-set counts (§1).
EXPECTED = {
    "ChatGPT": {"entities": 156, "sentences": 1202, "factual": 438,
                "hallucinated": 764, "facts": 4854, "S": 3167, "NS": 1687},
    "PerplexityAI": {"entities": 155, "sentences": 1371, "factual": 963,
                     "hallucinated": 408, "facts": 5532, "S": 4777, "NS": 755},
}
EXPECTED_POOLED_ENTITIES = 172


class FactScoreDataRefused(RuntimeError):
    """The release files or the analysis set are not the preregistered ones."""


@dataclass(frozen=True)
class Fact:
    text: str
    label: str  # "S" or "NS"


@dataclass(frozen=True)
class FactSentence:
    generator: str
    entity: str
    response_index: int
    sentence_index: int
    text: str
    facts: Tuple[Fact, ...]

    @property
    def label(self):
        """1 = factual (every fact supported), 0 = hallucinated."""
        return int(all(f.label == "S" for f in self.facts))

    @property
    def key(self):
        return (self.generator, self.response_index, self.sentence_index)


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_generator(lines, generator):
    """Analysis-set sentences from one generator's JSONL lines. Pure."""
    out = []
    for response_index, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("annotations") or row["topic"] in EXCLUDED_ENTITIES:
            continue
        for sentence_index, ann in enumerate(row["annotations"]):
            if not ann.get("is-relevant"):
                continue
            facts = tuple(
                Fact(text=f["text"].strip(), label=f["label"])
                for f in (ann.get("human-atomic-facts") or [])
                if f.get("label") in ("S", "NS") and f.get("text", "").strip()
            )
            if not facts:
                continue
            out.append(FactSentence(generator, row["topic"], response_index,
                                    sentence_index, ann["text"], facts))
    return out


def counts(sentences):
    facts = [f for s in sentences for f in s.facts]
    return {
        "entities": len({s.entity for s in sentences}),
        "sentences": len(sentences),
        "factual": sum(s.label for s in sentences),
        "hallucinated": sum(1 - s.label for s in sentences),
        "facts": len(facts),
        "S": sum(f.label == "S" for f in facts),
        "NS": sum(f.label == "NS" for f in facts),
    }


def load_analysis_set(release_dir) -> List[FactSentence]:
    """The preregistered analysis set, or FactScoreDataRefused."""
    release_dir = Path(release_dir)
    problems, sentences = [], []
    for generator in GENERATORS:
        path = release_dir / "labeled" / f"{generator}.jsonl"
        got = sha256_path(path) if path.exists() else None
        if got != LABEL_FILE_SHA256[generator]:
            problems.append(f"{path}: sha256 {got}, pinned {LABEL_FILE_SHA256[generator]}")
            continue
        part = parse_generator(path.read_text(encoding="utf-8").splitlines(), generator)
        observed = counts(part)
        if observed != EXPECTED[generator]:
            problems.append(f"{generator}: counts {observed} != preregistered {EXPECTED[generator]}")
        sentences.extend(part)
    if not problems and len({s.entity for s in sentences}) != EXPECTED_POOLED_ENTITIES:
        problems.append("pooled entity count differs from the preregistration")
    if problems:
        raise FactScoreDataRefused("FActScore analysis set:\n  " + "\n  ".join(problems))
    return sentences
