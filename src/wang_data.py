import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class EvidenceDocument:
    url: str
    page_content: str


@dataclass
class Subclaim:
    text: str
    documents: List[EvidenceDocument]


@dataclass
class SentenceRecord:
    passage_index: int
    sentence_index: int
    sentence: str
    label: int
    raw_label: str
    subclaims: List[Subclaim]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def label_to_int(label: str) -> int:
    return 1 if str(label).strip().lower() == "accurate" else 0


def validate_wang_data_root(data_root: str) -> Path:
    root = Path(data_root)
    required = [
        root / "selfcheckgpt" / "dataset.json",
        root / "decomposed",
        root / "webpage",
        root / "NBC" / "NBC_positive.json",
        root / "NBC" / "NBC_negative.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Wang et al. data are not prepared. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRun: python scripts/prepare_wang_data.py"
        )
    return root


def load_sentence_records(data_root: str, strict: bool = True) -> List[SentenceRecord]:
    """Load the exact decomposed claims and retrieved webpages released by Wang et al.

    The paper evaluates 1,908 generated sentences. Each sentence is mapped to its
    released decomposed subclaims and the ordered web documents for each subclaim.
    We preserve that order because retrieval position is part of the sequential
    decision problem.
    """
    root = validate_wang_data_root(data_root)
    dataset = _read_json(root / "selfcheckgpt" / "dataset.json")

    records: List[SentenceRecord] = []
    problems = []

    for passage_index, passage in enumerate(dataset):
        sentences = passage["gpt3_sentences"]
        annotations = passage["annotation"]

        for sentence_index, (sentence, raw_label) in enumerate(zip(sentences, annotations)):
            decomposition_path = (
                root
                / "decomposed"
                / f"decomposed_{passage_index}_{sentence_index}.txt"
            )
            webpage_path = root / "webpage" / f"{passage_index}_{sentence_index}.json"

            if not decomposition_path.exists() or not webpage_path.exists():
                problems.append(
                    f"passage={passage_index} sentence={sentence_index}: missing released evidence"
                )
                if strict:
                    continue
                else:
                    continue

            subclaim_texts = _read_nonempty_lines(decomposition_path)
            webpage_map: Dict[str, List[dict]] = _read_json(webpage_path)

            subclaims: List[Subclaim] = []
            for subclaim_text in subclaim_texts:
                raw_documents = webpage_map.get(subclaim_text)
                if raw_documents is None:
                    problems.append(
                        f"passage={passage_index} sentence={sentence_index}: "
                        f"no webpage key for subclaim {subclaim_text!r}"
                    )
                    if strict:
                        continue
                    raw_documents = []

                documents = [
                    EvidenceDocument(
                        url=str(doc.get("url", "")),
                        page_content=str(doc.get("page_content", "")),
                    )
                    for doc in raw_documents
                    if str(doc.get("page_content", "")).strip()
                ]
                subclaims.append(Subclaim(text=subclaim_text, documents=documents))

            if not subclaims:
                problems.append(
                    f"passage={passage_index} sentence={sentence_index}: no usable subclaims"
                )
                continue

            records.append(
                SentenceRecord(
                    passage_index=passage_index,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    label=label_to_int(raw_label),
                    raw_label=str(raw_label),
                    subclaims=subclaims,
                )
            )

    if strict and problems:
        preview = "\n".join(problems[:10])
        raise ValueError(
            f"Found {len(problems)} integrity problem(s) in Wang data. "
            f"First problems:\n{preview}"
        )

    return records


def load_nbc_pairs(data_root: str, per_class: int = 200):
    """Load the paper's separate positive/negative NBC evidence pairs.

    Wang et al. report sampling s=200 factual and s=200 nonfactual examples to
    estimate their feature likelihoods. We use the same released data for both
    BSE and the proposed direct density-ratio estimator.
    """
    root = validate_wang_data_root(data_root)
    pos = _read_json(root / "NBC" / "NBC_positive.json")
    neg = _read_json(root / "NBC" / "NBC_negative.json")

    if per_class is not None:
        pos = pos[:per_class]
        neg = neg[:per_class]

    return pos, neg


def group_split_records(
    records: List[SentenceRecord],
    validation_fraction: float = 0.20,
    random_state: int = 42,
):
    """Split by passage/article for DDRE stopping-rule tuning and final testing.

    The density estimator itself is trained only on the separate Wang NBC data.
    SelfCheckGPT validation data tune the DDRE stopping thresholds. The final test
    passages remain untouched. BSE and DDRE are evaluated on the same test set.
    """
    import numpy as np

    passage_ids = sorted({record.passage_index for record in records})
    rng = np.random.default_rng(random_state)
    shuffled = np.asarray(passage_ids, dtype=int)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    val_ids = set(int(x) for x in shuffled[:n_val])
    test_ids = set(int(x) for x in shuffled[n_val:])

    validation = [r for r in records if r.passage_index in val_ids]
    test = [r for r in records if r.passage_index in test_ids]

    assert val_ids.isdisjoint(test_ids)
    return validation, test, {
        "random_state": random_state,
        "validation_fraction": validation_fraction,
        "validation_passages": len(val_ids),
        "test_passages": len(test_ids),
        "validation_sentences": len(validation),
        "test_sentences": len(test),
        "validation_passage_ids": sorted(val_ids),
        "test_passage_ids": sorted(test_ids),
    }
