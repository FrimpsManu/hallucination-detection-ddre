"""Read-only document scoring from an existing NLI cache.

The NBC sensitivity analysis re-runs ``bse_official`` 200 times (100 histogram
combinations x 2 cost configurations). Doing that with a real scorer would mean
hundreds of thousands of NLI forward passes. It is unnecessary: every score the
analysis needs was already computed by the Gate 1 run and stored in
``results/wang_nli_cache.sqlite``.

This module exposes those scores as a drop-in for ``EntailmentScorer`` in the
one method ``BSEDetector`` calls, ``score_document``. It constructs no model,
downloads nothing, and opens the cache **read-only** (SQLite ``mode=ro``), so a
formal cache cannot be modified even by accident.

Coverage is the one real hazard. Retrieval is adaptive: which documents get
scored depends on the histogram *and* on the cost configuration, so a different
histogram can require a document the recorded run never consumed. Rather than
silently substituting a default, a missing score raises
:class:`MissingDocumentScore`, and the caller marks only the cost configuration
that hit it incomplete -- a result the other configuration already produced for
the same placement is preserved. A sensitivity analysis that quietly invented
scores for the documents it had not seen would be worthless.
"""

import hashlib
import sqlite3


class MissingDocumentScore(LookupError):
    """A span score required by the replay is absent from the cache."""

    def __init__(self, claim, span_index, total_spans):
        super().__init__(
            f"span {span_index + 1}/{total_spans} for claim {claim[:60]!r} "
            "is not in the cache"
        )
        self.claim = claim
        self.span_index = span_index
        self.total_spans = total_spans


class CachedDocumentScorer:
    """Serve Wang's max-over-spans document score from a cache, never a model.

    The cache key convention is transcribed from
    ``src.utils.EntailmentScorer._cache_key``: a SHA-256 over
    ``model_name``, ``score_version``, premise and hypothesis, NUL-separated.
    Both are supplied explicitly so a v1 cache and a v2 cache can each be
    replayed against deliberately, rather than whichever the installed source
    happens to define.
    """

    def __init__(self, cache_path, model_name, score_version, split_text):
        self.cache_path = str(cache_path)
        self.model_name = model_name
        self.score_version = score_version
        self._split_text = split_text
        # mode=ro is load-bearing: this must not be able to write to a formal
        # cache under any code path.
        self._connection = sqlite3.connect(
            f"file:{self.cache_path}?mode=ro", uri=True
        )
        self._span_cache = {}
        self._document_cache = {}
        self.lookups = 0
        self.misses = 0

    def _cache_key(self, premise, hypothesis):
        payload = "\0".join(
            [self.model_name, self.score_version, str(premise), str(hypothesis)]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _span_score(self, premise, hypothesis):
        key = self._cache_key(premise, hypothesis)
        if key in self._span_cache:
            return self._span_cache[key]
        row = self._connection.execute(
            "SELECT score FROM nli_scores WHERE cache_key = ?", (key,)
        ).fetchone()
        self.lookups += 1
        if row is None:
            self.misses += 1
            self._span_cache[key] = None
            return None
        score = float(row[0])
        self._span_cache[key] = score
        return score

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        """Wang's document score: the maximum entailment across text spans.

        Signature matches ``EntailmentScorer.score_document`` so ``BSEDetector``
        needs no change. ``use_cache``/``write_cache`` are accepted and ignored:
        this scorer is cache-only by construction and never writes.
        """
        memo_key = (claim, page_content)
        cached = self._document_cache.get(memo_key)
        if cached is not None:
            return cached

        segments = self._split_text(page_content, segment_length=400, overlap_length=100)
        if not segments:
            result = (0.0, 0)
            self._document_cache[memo_key] = result
            return result

        scores = []
        for index, segment in enumerate(segments):
            score = self._span_score(segment, claim)
            if score is None:
                raise MissingDocumentScore(claim, index, len(segments))
            scores.append(score)

        result = (max(scores), len(segments))
        self._document_cache[memo_key] = result
        return result

    def cache_rows(self):
        row = self._connection.execute("SELECT COUNT(*) FROM nli_scores").fetchone()
        return int(row[0]) if row else 0

    def stats(self):
        return {
            "cache_path": self.cache_path,
            "model_name": self.model_name,
            "score_version": self.score_version,
            "span_lookups": self.lookups,
            "span_misses": self.misses,
            "documents_memoized": len(self._document_cache),
        }

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
