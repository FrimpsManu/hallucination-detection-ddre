"""NLI feature scorer for multivariate DDRE (preregistration §2, §8 step 2).

Scores premise/hypothesis spans exactly as ``EntailmentScorer`` does: same
tokenizer call, same forward pass, same temperature-5 softmax, probabilities
taken off the tensor before the * 100 scaling. It returns and caches all three
class probabilities plus the raw logits per span, so any feature or
temperature can be derived later without rescoring.

The entailment value must equal ``EntailmentScorer``'s score bit for bit on the
same pair, model and dtype. ``entailment_mismatches`` checks that against an
existing cache before any feature is used.

The forward pass is duplicated here rather than refactored out of
``src/utils.py``, so the frozen scorer is not touched. The equality check is
what ties the two together.
"""

import hashlib
import os
import sqlite3

FEATURE_VERSION = "mddre-nli3-temp5-seg400-overlap100-v1"
SEGMENT_LENGTH = 400
OVERLAP_LENGTH = 100
TEMPERATURE = 5.0
FIELDS = ("entailment", "neutral", "contradiction", "logit_entailment",
          "logit_neutral", "logit_contradiction")


class FeatureScoreError(RuntimeError):
    pass


def label_indices(id2label):
    """Indices of entailment, neutral and contradiction in ``id2label``."""
    found = {}
    for index, label in id2label.items():
        name = str(label).lower()
        for target in ("entailment", "neutral", "contradiction"):
            if name == target:
                found[target] = int(index)
    if set(found) != {"entailment", "neutral", "contradiction"}:
        raise FeatureScoreError(f"cannot identify the three NLI labels in {id2label}")
    return found


def feature_key(model_name, dtype, premise, hypothesis):
    payload = "\0".join(
        [model_name, FEATURE_VERSION, str(dtype), str(premise), str(hypothesis)]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def document_features(span_rows):
    """Per-document features from its span rows (preregistration §2).

    e = max over spans of the entailment score, k = max over spans of the
    contradiction score, both on Wang's 0-100 scale.
    """
    if not span_rows:
        raise FeatureScoreError("a document with no spans has no features")
    return {
        "e": max(r["entailment"] for r in span_rows),
        "k": max(r["contradiction"] for r in span_rows),
        "spans": len(span_rows),
    }


def entailment_mismatches(feature_rows, reference_scores):
    """Pairs whose feature entailment differs from the reference score.

    ``feature_rows`` and ``reference_scores`` map the same pair identifiers to a
    feature row and to an ``EntailmentScorer`` score. Exact float equality.
    """
    missing = sorted(set(reference_scores) - set(feature_rows))
    if missing:
        raise FeatureScoreError(f"{len(missing)} reference pairs have no feature row")
    return [k for k, score in reference_scores.items()
            if feature_rows[k]["entailment"] != score]


class FeatureScorer:
    """Batched NLI feature scorer with a persistent cache (needs torch)."""

    def __init__(self, tokenizer, model, model_name, *, dtype, cache_path, batch_size=1):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.dtype = str(dtype)
        self.batch_size = max(1, int(batch_size))
        self.device = next(model.parameters()).device
        self.index = label_indices(model.config.id2label)
        os.makedirs(os.path.dirname(str(cache_path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(str(cache_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nli_features (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                dtype TEXT NOT NULL,
                entailment REAL NOT NULL,
                neutral REAL NOT NULL,
                contradiction REAL NOT NULL,
                logit_entailment REAL NOT NULL,
                logit_neutral REAL NOT NULL,
                logit_contradiction REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def _infer_batch(self, pairs):
        import torch

        premises = [premise for premise, _ in pairs]
        hypotheses = [hypothesis for _, hypothesis in pairs]
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits / TEMPERATURE, dim=-1)
        # Off the tensor before scaling, as EntailmentScorer does.
        p = {name: probs[:, i].detach().cpu().tolist() for name, i in self.index.items()}
        logits = {name: outputs.logits[:, i].detach().float().cpu().tolist()
                  for name, i in self.index.items()}
        return [{
            "entailment": float(p["entailment"][j]) * 100.0,
            "neutral": float(p["neutral"][j]) * 100.0,
            "contradiction": float(p["contradiction"][j]) * 100.0,
            "logit_entailment": logits["entailment"][j],
            "logit_neutral": logits["neutral"][j],
            "logit_contradiction": logits["contradiction"][j],
        } for j in range(len(pairs))]

    def _lookup(self, keys):
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            rows = self.conn.execute(
                f"SELECT cache_key, {', '.join(FIELDS)} FROM nli_features "
                f"WHERE cache_key IN ({','.join('?' for _ in chunk)})", chunk).fetchall()
            found.update({r[0]: dict(zip(FIELDS, map(float, r[1:]))) for r in rows})
        return found

    def score_pairs(self, pairs, *, progress=None):
        keys = [feature_key(self.model_name, self.dtype, p, h) for p, h in pairs]
        found = self._lookup(list(dict.fromkeys(keys)))
        todo = list(dict.fromkeys(k for k in keys if k not in found))
        first = {k: pair for k, pair in zip(reversed(keys), reversed(pairs))}
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            rows = self._infer_batch([first[k] for k in batch])
            self.conn.executemany(
                f"INSERT INTO nli_features (cache_key, model_name, feature_version, dtype, "
                f"{', '.join(FIELDS)}) VALUES (?, ?, ?, ?, {', '.join('?' for _ in FIELDS)})",
                [(k, self.model_name, FEATURE_VERSION, self.dtype, *(r[f] for f in FIELDS))
                 for k, r in zip(batch, rows)])
            self.conn.commit()
            found.update(zip(batch, rows))
            if progress is not None:
                progress(len(batch))
        return [found[k] for k in keys]

    def score_document(self, claim, page_content):
        from src.utils import split_text

        segments = split_text(page_content, segment_length=SEGMENT_LENGTH,
                              overlap_length=OVERLAP_LENGTH)
        if not segments:
            raise FeatureScoreError("a document with no text has no features")
        return document_features(self.score_pairs([(s, claim) for s in segments]))

    def cache_size(self):
        return int(self.conn.execute("SELECT COUNT(*) FROM nli_features").fetchone()[0])

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None
