#!/usr/bin/env python3
"""Build the BM25 index over the pinned FActScore Wikipedia snapshot, once.

Implements the corpus side of ``docs/factscore_transfer_preregistration.md`` §3:
every passage of every page in ``enwiki-20230401.db``, indexed by SQLite FTS5
(unicode61 remove_diacritics 2; title and passage columns). The snapshot is
opened read-only and must match its pinned SHA-256; the index is written once,
made read-only, hashed, and described in a manifest. No FActScore data is
read here.
"""

import argparse
import json
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import wikipedia_bm25 as W  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

RELEASE = Path("~/ddre-artifacts/factscore-release").expanduser()
SNAPSHOT_NAME = "enwiki-20230401.db"
SNAPSHOT_SHA256 = "31cf7b6b4465459844bb00f3a6ac75560fc7d1525112205a21859323dc5d33d7"
SNAPSHOT_PAGES = 6_187_531
INDEX_NAME = "wikipedia_bm25_index.sqlite"
MANIFEST_NAME = "wikipedia_bm25_index_manifest.json"


class IndexRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release-dir", default=str(RELEASE))
    return parser.parse_args()


def git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    release = Path(args.release_dir).expanduser()
    snapshot, index, manifest = (release / SNAPSHOT_NAME, release / INDEX_NAME,
                                 release / MANIFEST_NAME)
    for path in (index, manifest):
        if path.exists():
            raise IndexRefused(f"{path} already exists; the index is built once")
    print("verifying the snapshot digest (22 GB, about a minute)...")
    if sha256_path(snapshot) != SNAPSHOT_SHA256:
        raise IndexRefused("the snapshot is not the pinned one")

    start = time.time()
    counts = W.build_index(snapshot, index, progress=lambda pages, passages: print(
        f"  {pages:>9,} pages  {passages:>11,} passages  {time.time() - start:7.0f}s", flush=True))
    if counts["pages"] != SNAPSHOT_PAGES:
        raise IndexRefused(f"indexed {counts['pages']} pages, snapshot has {SNAPSHOT_PAGES}")
    index.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    record = {
        "artifact": "factscore-wikipedia-bm25-index",
        "protocol": "docs/factscore_transfer_preregistration.md §3",
        "snapshot_sha256": SNAPSHOT_SHA256,
        "pages": counts["pages"],
        "passages": counts["passages"],
        "tokenizer": W.TOKENIZER,
        "column_weights": list(W.COLUMN_WEIGHTS),
        "bm25": "SQLite FTS5 bm25(), k1 = 1.2, b = 0.75",
        "sqlite_version": sqlite3.sqlite_version,
        "index_sha256": sha256_path(index),
        "index_bytes": index.stat().st_size,
        "build_seconds": round(time.time() - start, 1),
        "code_commit": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    print(f"manifest sha256 {sha256_path(manifest)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IndexRefused as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
