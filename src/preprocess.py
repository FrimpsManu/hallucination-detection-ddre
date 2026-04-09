import json

def convert_label(label: str) -> int:
    return 1 if label.lower() == "accurate" else 0


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dataset(dataset):
    processed = []

    for item in dataset:
        sentences = item["gpt3_sentences"]
        labels = item["annotation"]

        for sent, label in zip(sentences, labels):
            processed.append({
                "sentence": sent,
                "label_raw": label,
                "label": convert_label(label),
                "gpt3_text": item["gpt3_text"],
                "wiki_bio_text": item["wiki_bio_text"],
                "gpt3_text_samples": item["gpt3_text_samples"],
                "wiki_bio_test_idx": item["wiki_bio_test_idx"],
            })

    return processed


def save_processed(data, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    raw_path = "data/raw/dataset_v3.json"
    out_path = "data/processed/processed_sentences.json"

    dataset = load_dataset(raw_path)
    processed = flatten_dataset(dataset)
    save_processed(processed, out_path)

    print(f"Saved {len(processed)} sentence-level samples to {out_path}")