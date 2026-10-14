#!/usr/bin/env python3
"""Score every document the multivariate DDRE study needs, once
(``docs/multivariate_ddre_design.md`` §2, §8 step 2).

One NLI pass over every (span, subclaim) pair of:

- the training set: FActScore training-entity facts, all three generators;
- the SelfCheckGPT held-out set: the frozen 190-passage split, positions 1-10;
- the FActScore test set: test-entity ChatGPT and PerplexityAI facts;
- Factcheck-Bench: the 5 released passages per included claim;
- Wang's 398 NBC pairs, which cell A's histogram needs.

Each pair is scored with ``FeatureScorer``: three class probabilities plus the
raw logits per span, in the precision chosen by the design's §10.4 rule.

Equality checks, which must both pass for the cache to be marked complete:

1. **Before** any bulk scoring: the 398 NBC pairs' entailment must equal the
   precision-matched reference cache bit for bit.
2. **After** scoring: every scored pair present in a precision-matched
   reference cache must equal it bit for bit.

Every input is opened read-only. The output cache and manifest are written
once and made read-only. No detector is run and no label is read beyond what
the loaders need to build sentences.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import complete_ablation_heldout_cache as CA  # noqa: E402
import run_frozen_heldout as frozen  # noqa: E402
from src import ablation_cache as AC  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

ARTIFACTS = Path("~/ddre-artifacts").expanduser()
GATE1 = ARTIFACTS / "gate1-2026-09-20"
FACTSCORE = ARTIFACTS / "factscore-release"
FACTCHECK = ARTIFACTS / "factcheck-bench" / "release"
MDDRE = ARTIFACTS / "mddre"
FP32_REFERENCE = ARTIFACTS / "precision-fp32" / "wang_nli_cache_fp32.sqlite"

FACTSCORE_EVIDENCE = ("factscore_evidence_top10.jsonl",
                      "44f479128d41818fe5a056f125407dbde75a8735669c398a3b945cfeaa310d09")
TRAINING_EVIDENCE = "instructgpt_training_evidence_top10.jsonl"
TRAINING_MANIFEST = "instructgpt_training_evidence_manifest.json"

# Precision-matched reference caches, all keyed by EntailmentScorer's key.
FLOAT16_REFERENCES = {
    "d03_cache": (GATE1 / "d03_diagnostic_cache_main_2661b04.sqlite",
                  "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"),
    "heldout_full_cache": (GATE1 / "ablation_heldout_full_cache.sqlite",
                           "2497487664e2df552ba9bf28947c4ccae7b5f66d2feccbada05ffa5aaea88f28"),
    "factscore_cache": (FACTSCORE / "factscore_nli_cache.sqlite",
                        "7bd69b0c4a2e809311be9a9748cb1fba53ed09fee1c98d4e5fe7c4c2bb2edff0"),
}
DTYPES = ("float16", "float32")


class ScoringRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtype", choices=DTYPES, required=True,
                        help="Fixed by the design's §10.4 rule from the float32 Gate 1 check.")
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(MDDRE))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs and enumerate pairs. No model, nothing written.")
    return parser.parse_args()


def output_paths(out_dir, dtype):
    return {"cache": out_dir / f"mddre_features_{dtype}.sqlite",
            "manifest": out_dir / f"mddre_features_{dtype}_manifest.json"}


def references(dtype):
    """{name: (path, pinned sha or None)} for the precision-matched caches."""
    if dtype == "float16":
        return dict(FLOAT16_REFERENCES)
    return {"fp32_gate1_cache": (FP32_REFERENCE, None)}


def verify_inputs(out, dtype):
    problems, digests = [], {}
    if sha256_path(PROJECT_ROOT / M.DESIGN_PATH) != M.DESIGN_SHA256:
        problems.append("the design preregistration differs from its pinned digest")
    name, sha = FACTSCORE_EVIDENCE
    if sha256_path(FACTSCORE / name) != sha:
        problems.append(f"{name} is not the frozen FActScore evidence")
    tm = json.loads((MDDRE / TRAINING_MANIFEST).read_text())
    got = sha256_path(MDDRE / TRAINING_EVIDENCE)
    if got != tm.get("evidence_sha256") or tm.get("design_sha256") != M.DESIGN_SHA256:
        problems.append("the InstructGPT training evidence does not match its manifest")
    digests["training_evidence"] = got
    for ref, (path, pinned) in references(dtype).items():
        if not path.exists():
            problems.append(f"reference {ref} missing at {path}")
            continue
        digests[ref] = sha256_path(path)
        if pinned is not None and digests[ref] != pinned:
            problems.append(f"reference {ref}: sha256 {digests[ref]}, pinned {pinned}")
    result_path = GATE1 / CA.RESULT_NAME
    if sha256_path(result_path) != CA.RESULT_SHA256:
        problems.append("the frozen held-out result differs from its pinned digest")
    for path in out.values():
        if path.exists():
            problems.append(f"{path} already exists; scoring runs once")
    if problems:
        raise ScoringRefused("inputs:\n  " + "\n  ".join(problems))
    return digests, json.loads(result_path.read_text())


def evidence_rows(path):
    with open(path, encoding="utf-8") as handle:
        return {(r["generator"], r["response_index"], r["sentence_index"], r["fact_index"]):
                [d["text"] for d in r["documents"]]
                for r in map(json.loads, handle)}


def factscore_records(sentences, evidence):
    """Sentence-shaped records (subclaims with documents) for enumeration."""
    out = []
    for s in sentences:
        subs = []
        for i, fact in enumerate(s.facts):
            docs = evidence[(s.generator, s.response_index, s.sentence_index, i)]
            subs.append(SimpleNamespace(
                text=fact.text, documents=[SimpleNamespace(page_content=t) for t in docs]))
        out.append(SimpleNamespace(subclaims=subs))
    return out


def factcheck_records(sentences):
    return [SimpleNamespace(subclaims=[
        SimpleNamespace(text=c.text, documents=[SimpleNamespace(page_content=t)
                                                for t in c.documents])
        for c in s.claims]) for s in sentences]


def enumerate_datasets(wang_root, frozen_result):
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    training, test, split = M.factscore_split(FACTSCORE / "data")
    fs_evidence = evidence_rows(FACTSCORE / FACTSCORE_EVIDENCE[0])
    fs_evidence.update(evidence_rows(MDDRE / TRAINING_EVIDENCE))
    heldout, _ = CA.held_out_records(wang_root, frozen_result)
    positive, negative = load_nbc_pairs(wang_root, per_class=None)
    nbc = [(x["premise"], x["hypothesis"]) for x in positive + negative]

    datasets = {
        "training": factscore_records(training, fs_evidence),
        "selfcheckgpt_heldout": heldout,
        "factscore_test": factscore_records(test, fs_evidence),
        "factcheck_bench": factcheck_records(M.load_factcheck(FACTCHECK)),
    }
    coverage = {name: AC.enumerate_coverage(records, split_text,
                                            model_name=frozen.OFFICIAL_MODEL,
                                            score_version=SCORE_VERSION)
                for name, records in datasets.items()}
    return coverage, nbc, split


def reference_scores(paths, keys):
    """{entailment-scorer key: score} found in any reference cache (read-only)."""
    found = {}
    for path in paths:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                rows = con.execute(
                    f"SELECT cache_key, score FROM nli_scores WHERE cache_key IN "
                    f"({','.join('?' for _ in chunk)})", chunk).fetchall()
                for key, score in rows:
                    found.setdefault(key, float(score))
        finally:
            con.close()
    return found


def load_model(dtype):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import select_device

    device = select_device(torch)
    if device != "mps":
        raise ScoringRefused(f"selected device is {device!r}; every frozen run used 'mps'")
    kwargs = {"revision": CA.REVISION}
    if dtype == "float32":
        kwargs["dtype"] = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(frozen.OFFICIAL_MODEL, revision=CA.REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        frozen.OFFICIAL_MODEL, **kwargs).to(torch.device(device))
    model.eval()
    loaded = str(next(model.parameters()).dtype)
    if loaded != f"torch.{dtype}":
        raise ScoringRefused(f"requested {dtype}, model loaded as {loaded}")
    return tokenizer, model, device, loaded


def main():
    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser(), args.dtype)
    digests, frozen_result = verify_inputs(out, args.dtype)
    coverage, nbc, split = enumerate_datasets(args.wang_data_root, frozen_result)

    all_pairs, all_keys, seen = [], [], set()
    for cov in coverage.values():
        for key, pair in zip(cov["keys"], cov["pairs"]):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)
                all_pairs.append(pair)
    print(f"precision {args.dtype}; split {split}")
    for name, cov in coverage.items():
        print(f"  {name:22s} {cov['counts']}")
    print(f"  {'nbc':22s} {len(nbc)} pairs\n  unique pairs overall  {len(all_pairs):,}")
    if args.dry_run:
        print("--dry-run: inputs verified, pairs enumerated. No model loaded, nothing written.")
        return 0

    from tqdm import tqdm

    from src.feature_scorer import FEATURE_VERSION, FeatureScorer, entailment_mismatches
    from src.utils import SCORE_VERSION

    ref_paths = [path for path, _ in references(args.dtype).values()]
    tokenizer, model, device, loaded = load_model(args.dtype)
    scorer = FeatureScorer(tokenizer, model, frozen.OFFICIAL_MODEL, dtype=loaded,
                           cache_path=out["cache"], batch_size=1)
    status, mismatches = "complete", {}
    try:
        # 1. NBC equality before any bulk scoring.
        nbc_keys = [AC.production_cache_key(frozen.OFFICIAL_MODEL, SCORE_VERSION, p, h)
                    for p, h in nbc]
        nbc_ref = reference_scores(ref_paths, nbc_keys)
        nbc_rows = dict(zip(nbc_keys, scorer.score_pairs(nbc)))
        mismatches["nbc"] = entailment_mismatches(nbc_rows, nbc_ref)
        if len(nbc_ref) != len(set(nbc_keys)) or mismatches["nbc"]:
            raise ScoringRefused(
                f"NBC equality failed: {len(nbc_ref)} reference rows, "
                f"{len(mismatches['nbc'])} mismatches. No bulk scoring was done.")
        print(f"NBC equality: {len(nbc_ref)} / {len(nbc_ref)} exact")

        # 2. Bulk scoring.
        with tqdm(total=len(all_pairs), desc="feature scoring", unit="pair") as bar:
            for start in range(0, len(all_pairs), 1000):
                scorer.score_pairs(all_pairs[start:start + 1000], progress=bar.update)

        # 3. Equality on every overlapping pair.
        rows = dict(zip(all_keys, scorer.score_pairs(all_pairs)))
        ref = reference_scores(ref_paths, all_keys)
        mismatches["overlap"] = entailment_mismatches({k: rows[k] for k in ref}, ref)
        if mismatches["overlap"]:
            status = "failed_equality"
        cache_rows = scorer.cache_size()
    finally:
        scorer.close()

    for ref_name, (path, _) in references(args.dtype).items():
        if sha256_path(path) != digests[ref_name]:
            raise ScoringRefused(f"reference {ref_name} changed during scoring")
    out["cache"].chmod(0o444)
    manifest = {
        "artifact": "mddre-nli-features",
        "protocol": M.DESIGN_PATH + " §2, §8 step 2",
        "design_sha256": M.DESIGN_SHA256,
        "status": status,
        "dtype": loaded,
        "device": device,
        "batch_size": 1,
        "model_name": frozen.OFFICIAL_MODEL,
        "checkpoint_revision": CA.REVISION,
        "feature_version": FEATURE_VERSION,
        "split": split,
        "coverage": {name: dict(cov["counts"], coverage_sha256=cov["coverage_sha256"])
                     for name, cov in coverage.items()},
        "unique_pairs": len(all_pairs),
        "nbc_pairs": len(nbc),
        "cache_rows": cache_rows,
        "equality": {
            "nbc_exact": len(nbc) - len(mismatches["nbc"]),
            "overlap_pairs_checked": len(ref),
            "overlap_mismatches": len(mismatches["overlap"]),
            "reference_caches": {k: {"path": str(p), "sha256": digests[k]}
                                 for k, (p, _) in references(args.dtype).items()},
        },
        "inputs": {"training_evidence_sha256": digests["training_evidence"],
                   "factscore_evidence_sha256": FACTSCORE_EVIDENCE[1],
                   "factcheck_sha256": M.FACTCHECK_SHA256},
        "detectors_run": False,
        "cache_sha256": sha256_path(out["cache"]),
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["manifest"].write_text(json.dumps(manifest, indent=1) + "\n")
    out["manifest"].chmod(0o444)
    print(f"status {status}; {len(ref):,} overlapping pairs checked, "
          f"{len(mismatches['overlap'])} mismatches")
    print(f"cache {out['cache']}\n  sha256 {manifest['cache_sha256']}")
    print(f"manifest {out['manifest']}\n  sha256 {sha256_path(out['manifest'])}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ScoringRefused, M.MDDREDataRefused, AC.CompletionAborted) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
