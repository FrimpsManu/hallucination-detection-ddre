#!/usr/bin/env python3
"""Retrieve the ordered top-10 evidence for every FActScore subclaim, once.

Implements ``docs/factscore_transfer_preregistration.md`` §3. For each of the
10,386 subclaims of the preregistered analysis set, the query is
``entity name + " " + fact`` (stopwords removed, OR-joined) against the frozen
BM25 index. The top 10 passages, ties broken by page title then passage index,
become that subclaim's reading order for every cell.

Labels are never passed to retrieval: the analysis set is loaded only to know
which subclaims exist, and the evidence file records no label. The output is
written once, made read-only and hashed.
"""

import argparse
import json
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import wikipedia_bm25 as W  # noqa: E402
from src.factscore_data import (  # noqa: E402
    PREREGISTRATION_MERGE_COMMIT,
    PREREGISTRATION_PATH,
    PREREGISTRATION_SHA256,
    load_analysis_set,
    sha256_path,
)

RELEASE = Path("~/ddre-artifacts/factscore-release").expanduser()
SNAPSHOT_NAME = "enwiki-20230401.db"
SNAPSHOT_SHA256 = "31cf7b6b4465459844bb00f3a6ac75560fc7d1525112205a21859323dc5d33d7"
INDEX_NAME = "wikipedia_bm25_index.sqlite"
INDEX_MANIFEST_NAME = "wikipedia_bm25_index_manifest.json"
EVIDENCE_NAME = "factscore_evidence_top10.jsonl"
EVIDENCE_MANIFEST_NAME = "factscore_evidence_manifest.json"
EXPECTED_SUBCLAIMS = 10_386


class EvidenceRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release-dir", default=str(RELEASE))
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel read-only queries. Results do not depend on it.")
    return parser.parse_args()


def subclaim_queries(sentences):
    """(subclaim key, entity, fact text) for every subclaim. No labels leave this function."""
    out = []
    for s in sentences:
        for fact_index, fact in enumerate(s.facts):
            out.append(((s.generator, s.response_index, s.sentence_index, fact_index),
                        s.entity, fact.text))
    return out


_INDEX = None
_SNAPSHOT = None


def _init(index_path, snapshot_path):
    global _INDEX, _SNAPSHOT
    _INDEX = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    _SNAPSHOT = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)


def _retrieve_one(query):
    entity, fact = query
    terms = W.query_terms(entity, fact)
    hits = W.retrieve(_INDEX, terms)
    docs = [{"rank": r + 1, "title": title, "pidx": pidx, "bm25": score,
             "text": W.document_text(title, W.passage_text(_SNAPSHOT, title, pidx))}
            for r, (score, title, pidx) in enumerate(hits)]
    return terms, docs


def git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def verify_inputs(release):
    problems = []
    prereg = PROJECT_ROOT / PREREGISTRATION_PATH
    if sha256_path(prereg) != PREREGISTRATION_SHA256:
        problems.append("the preregistration differs from its pinned digest")
    manifest_path = release / INDEX_MANIFEST_NAME
    if not manifest_path.exists():
        problems.append("the index manifest is missing; build the index first")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("snapshot_sha256") != SNAPSHOT_SHA256:
            problems.append("the index was not built from the pinned snapshot")
        if sha256_path(release / INDEX_NAME) != manifest.get("index_sha256"):
            problems.append("the index file differs from its manifest digest")
    if sha256_path(release / SNAPSHOT_NAME) != SNAPSHOT_SHA256:
        problems.append("the snapshot is not the pinned one")
    for name in (EVIDENCE_NAME, EVIDENCE_MANIFEST_NAME):
        if (release / name).exists():
            problems.append(f"{name} already exists; retrieval runs once")
    if problems:
        raise EvidenceRefused("inputs:\n  " + "\n  ".join(problems))
    return manifest


def main():
    args = parse_args()
    release = Path(args.release_dir).expanduser()
    index_manifest = verify_inputs(release)
    queries = subclaim_queries(load_analysis_set(release / "data"))
    if len(queries) != EXPECTED_SUBCLAIMS:
        raise EvidenceRefused(f"{len(queries)} subclaims, preregistered {EXPECTED_SUBCLAIMS}")

    unique = sorted({(entity, fact) for _, entity, fact in queries})
    print(f"{len(queries):,} subclaims, {len(unique):,} distinct queries")
    start = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(release / INDEX_NAME, release / SNAPSHOT_NAME)) as pool:
        results = dict(zip(unique, pool.map(_retrieve_one, unique, chunksize=16)))
    seconds = time.time() - start

    evidence_path = release / EVIDENCE_NAME
    short = 0
    with evidence_path.open("w", encoding="utf-8") as out:
        for key, entity, fact in queries:
            terms, docs = results[(entity, fact)]
            short += len(docs) < W.TOP_K
            out.write(json.dumps({
                "generator": key[0], "response_index": key[1], "sentence_index": key[2],
                "fact_index": key[3], "entity": entity, "fact": fact,
                "query_terms": terms, "documents": docs,
            }, ensure_ascii=False) + "\n")
    evidence_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    record = {
        "artifact": "factscore-evidence-top10",
        "protocol": PREREGISTRATION_PATH + " §3",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_merge_commit": PREREGISTRATION_MERGE_COMMIT,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "index_sha256": index_manifest["index_sha256"],
        "subclaims": len(queries),
        "distinct_queries": len(unique),
        "subclaims_with_fewer_than_10_documents": short,
        "top_k": W.TOP_K,
        "labels_recorded": False,
        "retrieval_seconds": round(seconds, 1),
        "sqlite_version": sqlite3.sqlite_version,
        "evidence_sha256": sha256_path(evidence_path),
        "code_commit": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (release / EVIDENCE_MANIFEST_NAME).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EvidenceRefused as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
