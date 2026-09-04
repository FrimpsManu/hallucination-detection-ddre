import hashlib
import os
import sqlite3

import torch
from tqdm import tqdm


# v2 (Gate 1 Step 2): entailment probabilities now leave the tensor BEFORE the
# * 100 scaling, matching Wang's released utils.py:62. The bump is mandatory,
# not cosmetic: v1 cache rows were scaled inside the tensor, and on a
# half-precision run that rounds a score by up to half of 0.015625 in the 16-32
# range. Reusing them under the corrected scorer would silently mix the two
# conventions. Historical caches and result artifacts are left untouched; they
# simply no longer match a v2 key.
SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"


def split_text(text, segment_length=400, overlap_length=100):
    """Match Wang et al.'s document segmentation defaults (m=400, step=300)."""
    words = str(text).split()[:4000]
    if not words:
        return []

    segments = []
    start = 0
    step = segment_length - overlap_length
    if step <= 0:
        raise ValueError("segment_length must be greater than overlap_length")

    while start < len(words):
        end = start + segment_length
        if end >= len(words):
            # The released implementation uses the final segment_length words.
            segment = words[-segment_length:]
        else:
            segment = words[start:end]
        segments.append(" ".join(segment))
        start += step

    # The released split_text can create a duplicate tail span. Removing exact
    # duplicates does not change max-document entailment but avoids redundant NLI.
    return list(dict.fromkeys(segments))


class EntailmentScorer:
    """Batched DeBERTa NLI scorer with a persistent deterministic cache."""

    def __init__(
        self,
        tokenizer,
        model,
        model_name,
        cache_path="results/nli_cache.sqlite",
        batch_size=8,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = max(1, int(batch_size))
        self.device = next(model.parameters()).device
        self._memory_cache = {}
        self.entailment_index = self._find_entailment_index()

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nli_scores (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                score_version TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def _find_entailment_index(self):
        id2label = self.model.config.id2label
        for index, label in id2label.items():
            normalized = str(label).lower()
            if "entail" in normalized and "not" not in normalized and "contra" not in normalized:
                return int(index)

        # The exact model used by Wang et al. exposes entailment as label 0.
        # We only use this fallback when labels are generic LABEL_0/LABEL_1/...
        generic = all(str(label).upper().startswith("LABEL_") for label in id2label.values())
        if generic and 0 in [int(i) for i in id2label.keys()]:
            return 0
        raise ValueError("Could not identify entailment class from model.config.id2label")

    def _cache_key(self, premise, hypothesis):
        payload = "\0".join(
            [self.model_name, SCORE_VERSION, str(premise), str(hypothesis)]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _lookup_db(self, keys):
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT cache_key, score FROM nli_scores WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update({key: float(score) for key, score in rows})
        return found

    def _infer_batch(self, pairs):
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

        probs = torch.softmax(outputs.logits / 5.0, dim=-1)
        # Leave the tensor BEFORE scaling, matching Wang's released
        # utils.py:59-62, which calls .tolist() on the probabilities and only
        # then multiplies by 100 in Python float64.
        #
        # Scaling inside the tensor rounds the product to the tensor dtype. The
        # Gate 1 Step 2 diagnostic measured this on the real T4 run: the forward
        # path and the softmax shape were bit-identical across all 398 NBC
        # pairs, and the scaling order alone reproduced the recorded
        # 19.9462890625 -> 19.953125 discrepancy and the exact Step 1 positive
        # NBC histogram movement.
        #
        # No rounding is applied here. Scores stay continuous for DDRE; BSE's
        # one-decimal rounding and bucketing stay in src/baseline_core.py where
        # the released implementation puts them.
        entailment = probs[:, self.entailment_index].detach().cpu().tolist()
        return [float(x) * 100.0 for x in entailment]

    def score_pairs(
        self,
        pairs,
        *,
        use_cache=True,
        write_cache=True,
        show_progress=False,
        description="NLI inference",
    ):
        if not pairs:
            return []

        keys = [self._cache_key(p, h) for p, h in pairs]
        results = [None] * len(pairs)

        if use_cache:
            unresolved = []
            for i, key in enumerate(keys):
                if key in self._memory_cache:
                    results[i] = self._memory_cache[key]
                else:
                    unresolved.append(key)
            db_hits = self._lookup_db(list(dict.fromkeys(unresolved)))
            self._memory_cache.update(db_hits)
            for i, key in enumerate(keys):
                if results[i] is None and key in db_hits:
                    results[i] = db_hits[key]

        missing = [i for i, value in enumerate(results) if value is None]
        starts = range(0, len(missing), self.batch_size)
        if show_progress and missing:
            starts = tqdm(
                starts,
                total=(len(missing) + self.batch_size - 1) // self.batch_size,
                desc=description,
                unit="batch",
            )

        for start in starts:
            indices = missing[start : start + self.batch_size]
            batch_pairs = [pairs[i] for i in indices]
            batch_scores = self._infer_batch(batch_pairs)
            rows = []
            for i, score in zip(indices, batch_scores):
                key = keys[i]
                results[i] = score
                self._memory_cache[key] = score
                if write_cache:
                    rows.append((key, self.model_name, SCORE_VERSION, score))
            if rows:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO nli_scores
                    (cache_key, model_name, score_version, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                self.conn.commit()

        return [float(x) for x in results]

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        """Return Wang et al.'s document score: max entailment across text spans."""
        segments = split_text(page_content, segment_length=400, overlap_length=100)
        if not segments:
            return 0.0, 0
        pairs = [(segment, claim) for segment in segments]
        scores = self.score_pairs(
            pairs,
            use_cache=use_cache,
            write_cache=write_cache,
        )
        return max(scores), len(segments)

    def warm_pairs(self, pairs, description="Precomputing NLI scores"):
        before = self.cache_size()
        self.score_pairs(
            pairs,
            use_cache=True,
            write_cache=True,
            show_progress=True,
            description=description,
        )
        after = self.cache_size()
        return {
            "pairs_requested": len(pairs),
            "cache_rows_before": before,
            "cache_rows_after": after,
            "new_scores_computed": max(0, after - before),
        }

    def cache_size(self):
        row = self.conn.execute("SELECT COUNT(*) FROM nli_scores").fetchone()
        return int(row[0]) if row else 0

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None
