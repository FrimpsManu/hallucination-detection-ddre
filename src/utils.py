import torch


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


def get_entailment_score(premise, hypothesis, tokenizer, model):
    device = next(model.parameters()).device

    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits[0] / 5, dim=-1).detach().cpu()

    id2label = model.config.id2label
    label_probs = {id2label[i].lower(): probs[i].item() * 100 for i in range(len(probs))}

    entailment_prob = 0.0
    for label, prob in label_probs.items():
        if "entail" in label and "not" not in label and "contra" not in label:
            entailment_prob = prob
            break

    return entailment_prob