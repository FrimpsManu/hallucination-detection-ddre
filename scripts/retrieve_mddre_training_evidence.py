#!/usr/bin/env python3
"""Top-10 BM25 evidence for the InstructGPT training facts, once
(``docs/multivariate_ddre_design.md`` §4, §8 step 3).

The ChatGPT and PerplexityAI training facts already have evidence in the
frozen ``factscore_evidence_top10.jsonl``. Only the InstructGPT facts on
training entities need retrieval. It uses the same index, query rule, top-k and
tie-breaking as that file, through the same functions. No label leaves this
script's input; the output records documents only.
"""

import argparse
import json
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
from src import mddre_data as M  # noqa: E402
from src import wikipedia_bm25 as W  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

RELEASE = Path("~/ddre-artifacts/factscore-release").expanduser()
OUT_DIR = Path("~/ddre-artifacts/mddre").expanduser()
EVIDENCE_NAME = "instructgpt_training_evidence_top10.jsonl"
MANIFEST_NAME = "instructgpt_training_evidence_manifest.json"


class TrainingEvidenceRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release-dir", default=str(RELEASE))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def verify_inputs(release, out_dir):
    problems = []
    problems.extend(M.design_problems(PROJECT_ROOT))
    manifest = json.loads((release / RF.INDEX_MANIFEST_NAME).read_text())
    if manifest.get("snapshot_sha256") != RF.SNAPSHOT_SHA256:
        problems.append("the index was not built from the pinned snapshot")
    if sha256_path(release / RF.INDEX_NAME) != manifest.get("index_sha256"):
        problems.append("the index file differs from its manifest digest")
    if sha256_path(release / RF.SNAPSHOT_NAME) != RF.SNAPSHOT_SHA256:
        problems.append("the snapshot is not the pinned one")
    for name in (EVIDENCE_NAME, MANIFEST_NAME):
        if (out_dir / name).exists():
            problems.append(f"{name} already exists; retrieval runs once")
    if problems:
        raise TrainingEvidenceRefused("inputs:\n  " + "\n  ".join(problems))
    return manifest


def main():
    args = parse_args()
    release = Path(args.release_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    index_manifest = verify_inputs(release, out_dir)
    training, _, split = M.factscore_split(release / "data")
    sentences = [s for s in training if s.generator == "InstructGPT"]
    queries = RF.subclaim_queries(sentences)
    unique = sorted({(entity, fact) for _, entity, fact in queries})
    print(f"{len(queries):,} InstructGPT training facts, {len(unique):,} distinct queries")

    start = time.time()
    with Pool(args.workers, initializer=RF._init,
              initargs=(release / RF.INDEX_NAME, release / RF.SNAPSHOT_NAME)) as pool:
        results = dict(zip(unique, pool.map(RF._retrieve_one, unique, chunksize=16)))
    seconds = time.time() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EVIDENCE_NAME
    short = 0
    with path.open("w", encoding="utf-8") as out:
        for key, entity, fact in queries:
            terms, docs = results[(entity, fact)]
            short += len(docs) < W.TOP_K
            out.write(json.dumps({
                "generator": key[0], "response_index": key[1], "sentence_index": key[2],
                "fact_index": key[3], "entity": entity, "fact": fact,
                "query_terms": terms, "documents": docs,
            }, ensure_ascii=False) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    record = {
        "artifact": "mddre-instructgpt-training-evidence-top10",
        "protocol": M.DESIGN_PATH + " §4",
        "design_sha256": M.DESIGN_SHA256,
        "design_merge_commit": M.DESIGN_MERGE_COMMIT,
        "split": split,
        "snapshot_sha256": RF.SNAPSHOT_SHA256,
        "index_sha256": index_manifest["index_sha256"],
        "subclaims": len(queries),
        "distinct_queries": len(unique),
        "subclaims_with_fewer_than_10_documents": short,
        "top_k": W.TOP_K,
        "labels_recorded": False,
        "retrieval_seconds": round(seconds, 1),
        "evidence_sha256": sha256_path(path),
        "code_commit": RF.git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(RF.git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TrainingEvidenceRefused, M.MDDREDataRefused) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
