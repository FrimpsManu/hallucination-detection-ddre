"""Read per-document features from the frozen feature cache (read-only).

A document's features are e = max entailment and k = max contradiction over
its spans (design §2), looked up by the feature scorer's key. A missing span
is an error, never a default.
"""

import sqlite3

from src.feature_scorer import FIELDS, document_features, feature_key
from src.utils import split_text

SEGMENT_LENGTH = 400
OVERLAP_LENGTH = 100


class MissingFeature(LookupError):
    pass


class FeatureReader:
    def __init__(self, cache_path, model_name, dtype):
        self.model_name, self.dtype = model_name, dtype
        self.conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        self._memo = {}

    def _rows(self, keys):
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            rows = self.conn.execute(
                f"SELECT cache_key, {', '.join(FIELDS)} FROM nli_features WHERE cache_key IN "
                f"({','.join('?' for _ in chunk)})", chunk).fetchall()
            found.update({r[0]: dict(zip(FIELDS, map(float, r[1:]))) for r in rows})
        return found

    def document(self, claim, text, rank):
        memo_key = (claim, text)
        if memo_key not in self._memo:
            spans = split_text(text, segment_length=SEGMENT_LENGTH,
                               overlap_length=OVERLAP_LENGTH)
            keys = [feature_key(self.model_name, self.dtype, s, claim) for s in spans]
            rows = self._rows(keys)
            if len(rows) != len(set(keys)):
                raise MissingFeature(f"{len(set(keys)) - len(rows)} spans missing for "
                                     f"claim {claim[:60]!r}")
            self._memo[memo_key] = document_features([rows[k] for k in keys])
        return dict(self._memo[memo_key], rank=rank)

    def pair(self, premise, hypothesis):
        key = feature_key(self.model_name, self.dtype, premise, hypothesis)
        rows = self._rows([key])
        if key not in rows:
            raise MissingFeature(f"pair missing for hypothesis {hypothesis[:60]!r}")
        return rows[key]

    def close(self):
        self.conn.close()
