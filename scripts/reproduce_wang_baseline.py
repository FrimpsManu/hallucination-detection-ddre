import argparse
import json
import sys
from pathlib import Path

# Allow this file to be executed directly from the repository root with:
#   python scripts/reproduce_wang_baseline.py
# Python otherwise puts only scripts/ on sys.path, so the sibling src/ package
# is not importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_core import BSEDetector, build_nbc_histograms
from src.evaluation import evaluate_detector
from src.utils import EntailmentScorer
from src.wang_data import load_nbc_pairs, load_sentence_records


OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Wang et al. BSE on their full released SelfCheckGPT evidence set."
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument("--cache-path", default="results/wang_nli_cache.sqlite")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--nbc-per-class", type=int, default=200)
    args = parser.parse_args()

    records = load_sentence_records(args.data_root, strict=True)
    pos_pairs, neg_pairs = load_nbc_pairs(args.data_root, args.nbc_per_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or (8 if torch.cuda.is_available() else 2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()
    scorer = EntailmentScorer(
        tokenizer,
        model,
        args.model_name,
        cache_path=args.cache_path,
        batch_size=batch_size,
    )

    try:
        pos_hist, neg_hist, _, _ = build_nbc_histograms(pos_pairs, neg_pairs, scorer)
        outputs = {}
        for c_miss, c_false_alarm in ((14, 24), (28, 96)):
            name = f"CM_{c_miss}_CFA_{c_false_alarm}"
            detector = BSEDetector(
                pos_hist,
                neg_hist,
                mode="official",
                p0=0.5,
                c_miss=c_miss,
                c_false_alarm=c_false_alarm,
                c_retrieve=1,
                max_docs=10,
            )
            metrics, _ = evaluate_detector(
                detector,
                records,
                scorer,
                description=f"Wang reproduction {name}",
                use_cache=True,
            )
            outputs[name] = metrics

        result = {
            "purpose": "sanity-check reproduction of Wang et al. released BSE implementation",
            "model_name": args.model_name,
            "sentences": len(records),
            "nbc_examples_per_class": args.nbc_per_class,
            "published_table1_reference": {
                "CM_14_CFA_24": {
                    "nonfactual_auc_pr": 0.8242,
                    "factual_auc_pr": 0.5701,
                    "accuracy": 0.8024,
                    "pearson": 0.7137,
                    "spearman": 0.6455,
                    "evidence_num": 3.05,
                },
                "CM_28_CFA_96": {
                    "nonfactual_auc_pr": 0.8645,
                    "factual_auc_pr": 0.6196,
                    "accuracy": 0.8239,
                    "pearson": 0.8118,
                    "spearman": 0.7420,
                    "evidence_num": 6.22,
                },
            },
            "reproduction": outputs,
        }

        Path("results").mkdir(exist_ok=True)
        path = Path("results/wang_reproduction.json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wang reproduction results written to {path}")
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
