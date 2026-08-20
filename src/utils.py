import hashlib
import os
import sqlite3
from contextlib import nullcontext

import torch
from tqdm import tqdm


SCORE_VERSION = "entailment-softmax-temp5-v1"


def split_text(text, segment_length=300, overlap_length=50):
    words = text.split()[:4000]
    segments = []

    start = 0
    step = segment_length - overlap_length

    while start < len(words):
        end = start + segment_length
        segment = words[start:end]

        if not segment:
            break

        segments.append(" ".join(segment))

        if end >= len(words):
            break

        start += step

    return segments


class EntailmentScorer:
    """Batched NLI scorer with an optional persistent SQLite cache.

    Training and validation may safely reuse cached NLI scores because those
    scores are deterministic features of the claim/evidence pair. Final test
    latency measurements should set use_cache=False and write_cache=False so
    wall-clock measurements reflect real model inference rather than cache I/O.
    """

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
        for index, label in self.model.config.id2label.items():
            normalized = str(label).lower()
            if "entail" in normalized and "not" not in normalized and "contra" not in normalized:
                return int(index)
        raise ValueError(
            "Could not identify the entailment class from model.config.id2label."
        )

    def _cache_key(self, premise, hypothesis):
        payload = "\0".join(
            [self.model_name, SCORE_VERSION, premise, hypothesis]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _lookup_db(self, keys):
        found = {}
        if not keys:
            return found

        # Stay well below SQLite's parameter-count limit.
        chunk_size = 500
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start : start + chunk_size]
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
        scores = probs[:, self.entailment_index] * 100.0
        return [float(value) for value in scores.detach().cpu().tolist()]

    def score_pairs(
        self,
        pairs,
        *,
        use_cache=True,
        write_cache=True,
        batch_size=None,
        show_progress=False,
        description="NLI inference",
    ):
        """Score (premise, hypothesis) pairs while preserving input order."""
        if not pairs:
            return []

        batch_size = max(1, int(batch_size or self.batch_size))
        keys = [self._cache_key(premise, hypothesis) for premise, hypothesis in pairs]
        results = [None] * len(pairs)

        if use_cache:
            unresolved_keys = []
            for index, key in enumerate(keys):
                if key in self._memory_cache:
                    results[index] = self._memory_cache[key]
                else:
                    unresolved_keys.append(key)

            db_hits = self._lookup_db(list(dict.fromkeys(unresolved_keys)))
            self._memory_cache.update(db_hits)
            for index, key in enumerate(keys):
                if results[index] is None and key in db_hits:
                    results[index] = db_hits[key]

        missing_indices = [i for i, value in enumerate(results) if value is None]
        batches = range(0, len(missing_indices), batch_size)
        if show_progress and missing_indices:
            batches = tqdm(
                batches,
                total=(len(missing_indices) + batch_size - 1) // batch_size,
                desc=description,
                unit="batch",
            )

        for start in batches:
            batch_indices = missing_indices[start : start + batch_size]
            batch_pairs = [pairs[i] for i in batch_indices]
            batch_scores = self._infer_batch(batch_pairs)

            rows_to_write = []
            for index, score in zip(batch_indices, batch_scores):
                key = keys[index]
                results[index] = score
                if use_cache or write_cache:
                    self._memory_cache[key] = score
                if write_cache:
                    rows_to_write.append(
                        (key, self.model_name, SCORE_VERSION, float(score))
                    )

            # Commit each completed batch. If a long run is interrupted, all
            # previously finished batches remain available on the next run.
            if rows_to_write:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO nli_scores
                    (cache_key, model_name, score_version, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows_to_write,
                )
                self.conn.commit()

        return [float(value) for value in results]

    def score_evidence(
        self,
        sentence,
        evidence,
        *,
        use_cache=True,
        write_cache=True,
    ):
        segments = split_text(evidence)
        pairs = [(segment, sentence) for segment in segments]
        scores = self.score_pairs(
            pairs,
            use_cache=use_cache,
            write_cache=write_cache,
        )
        return scores, segments

    def warm_cache(self, data, description="Precomputing train/validation NLI features"):
        """Batch all unique NLI pairs from a dataset and persist them.

        This is the expensive stage on a first run. Subsequent runs resume from
        the existing SQLite cache and infer only pairs that are still missing.
        """
        pairs = []
        seen = set()

        for item in tqdm(data, desc="Indexing NLI pairs", unit="sample"):
            sentence = item["sentence"]
            for segment in split_text(item["wiki_bio_text"]):
                key = self._cache_key(segment, sentence)
                if key not in seen:
                    seen.add(key)
                    pairs.append((segment, sentence))

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
            "unique_pairs_requested": len(pairs),
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


def get_entailment_score(premise, hypothesis, tokenizer, model):
    """Backward-compatible one-pair scorer without persistent caching."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits[0] / 5.0, dim=-1).detach().cpu()

    for index, label in model.config.id2label.items():
        normalized = str(label).lower()
        if "entail" in normalized and "not" not in normalized and "contra" not in normalized:
            return float(probs[int(index)].item() * 100.0)

    raise ValueError("Could not identify entailment class in model.config.id2label.")
