#!/usr/bin/env python3
"""BM25 top-10 Wikipedia evidence for every FACTORY unit, once
(docs/factory_preregistration.md §2.3, §8 step 2).

HELM's retrieval exactly (``scripts/retrieve_helm_evidence.py``): SQLite FTS5
BM25 over every passage of the enwiki-20230401 snapshot, query =
``query_terms(question, sentence)`` (the question in the place of HELM's
article title), top 10 passages in rank order, premise = page title +
passage.

Gates, any failure stops before anything is written:

- **P:** the FACTORY preregistration, the FACTORY release file, the snapshot
  and the BM25 index (via its manifest) match their pins; outputs do not
  exist.
- **G0:** the loader reproduces the preregistered totals.
- **R:** every unit gets one list; units with fewer than 10 hits are counted.

``--dry-run`` runs P and G0 and counts queries; nothing is retrieved or
written. The evidence carries no labels. It is written once, read-only.
"""

import argparse
import json
import sqlite3
import stat
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import retrieve_factscore_evidence as RF  # noqa: E402
from src import factory_data as H  # noqa: E402
from src import wikipedia_bm25 as W  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

INDEX_SHA256 = "79d0e38df6414235731e4c469b951e9ec884a07bce4746680f28f01f6d89925f"
EVIDENCE_NAME = "factory_evidence_top10.jsonl"
MANIFEST_NAME = "factory_evidence_manifest.json"


class FactoryRetrievalRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wiki-dir", default=str(RF.RELEASE))
    parser.add_argument("--out-dir", default=str(H.ROOT))
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel read-only queries. Results do not depend on it.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Gates P and G0 and query counts only. Nothing written.")
    return parser.parse_args()


def output_paths(out_dir):
    return {"evidence": out_dir / EVIDENCE_NAME, "manifest": out_dir / MANIFEST_NAME}


def verify_inputs(wiki, out, dry_run):
    problems = list(H.prereg_problems(PROJECT_ROOT))
    if not H.RELEASE_FILE.exists() or sha256_path(H.RELEASE_FILE) != H.RELEASE_SHA256:
        problems.append("the FACTORY release file differs from its pinned digest")
    manifest_path = wiki / RF.INDEX_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("index_sha256") != INDEX_SHA256:
        problems.append("the BM25 index manifest does not certify the pinned index")
    if manifest.get("snapshot_sha256") != RF.SNAPSHOT_SHA256:
        problems.append("the BM25 index was not built from the pinned snapshot")
    for name, sha in ((RF.INDEX_NAME, INDEX_SHA256), (RF.SNAPSHOT_NAME, RF.SNAPSHOT_SHA256)):
        path = wiki / name
        if not path.exists():
            problems.append(f"{path} is missing")
        elif not dry_run and sha256_path(path) != sha:
            problems.append(f"{name} differs from its pinned digest")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; retrieval runs once")
    if problems:
        raise FactoryRetrievalRefused("inputs:\n  " + "\n  ".join(problems))
    return manifest


def main():
    args = parse_args()
    wiki, out = Path(args.wiki_dir).expanduser(), output_paths(Path(args.out_dir).expanduser())
    verify_inputs(wiki, out, args.dry_run)
    units, counts = H.load_factory()
    g0 = H.check_expected(units, counts)
    unique = H.queries(units)
    print(f"gate G0: {g0}; {len(unique):,} distinct queries")
    if args.dry_run:
        print("--dry-run: gates P and G0 passed. Nothing retrieved or written.")
        return 0

    start = time.time()
    with Pool(args.workers, initializer=RF._init,
              initargs=(wiki / RF.INDEX_NAME, wiki / RF.SNAPSHOT_NAME)) as pool:
        results = dict(zip(unique, pool.map(RF._retrieve_one, unique, chunksize=16)))
    seconds = time.time() - start

    short = 0
    with out["evidence"].open("w", encoding="utf-8") as handle:
        for u in units:
            terms, docs = results[(u.title, u.sentence)]
            short += len(docs) < W.TOP_K
            handle.write(json.dumps({"id": u.id, "cluster": u.cluster,
                                     "question": u.title, "sentence": u.sentence,
                                     "query_terms": terms, "documents": docs},
                                    ensure_ascii=False) + "\n")
    out["evidence"].chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    record = {
        "artifact": "factory-evidence-top10",
        "protocol": H.PREREG_PATH + " §2.3",
        "preregistration_sha256": H.PREREG_SHA256,
        "factory_release_sha256": H.RELEASE_SHA256,
        "factory_release_revision": H.RELEASE_REVISION,
        "snapshot_sha256": RF.SNAPSHOT_SHA256,
        "index_sha256": INDEX_SHA256,
        "gate_G0": g0,
        "units": len(units),
        "distinct_queries": len(unique),
        "units_with_fewer_than_10_documents": short,
        "top_k": W.TOP_K,
        "labels_recorded": False,
        "retrieval_seconds": round(seconds, 1),
        "sqlite_version": sqlite3.sqlite_version,
        "evidence_sha256": sha256_path(out["evidence"]),
        "code_commit": RF.git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(RF.git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["manifest"].write_text(json.dumps(record, indent=1) + "\n")
    out["manifest"].chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FactoryRetrievalRefused, H.FactoryDataRefused, RF.EvidenceRefused) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
