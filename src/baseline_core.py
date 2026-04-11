from src.utils import split_text, get_entailment_score


def discretize_score(score):
    bucket = int((score - 0.1) / 10)
    return max(0, min(bucket, 9))


def bayes_update(P, p_given_1, p_given_0):
    denominator = (1 - P) * p_given_0 + P * p_given_1
    if denominator == 0:
        return P
    return (P * p_given_1) / denominator


def max_entailment_score(sentence, evidence, tokenizer, model):
    segments = split_text(evidence)
    max_score = 0.0

    for seg in segments:
        score = get_entailment_score(seg, sentence, tokenizer, model)
        if score > max_score:
            max_score = score

    return max_score


def build_nbc_features(data, tokenizer, model, max_samples=20):
    pos_features = [0] * 10
    neg_features = [0] * 10

    subset = data[:max_samples]

    for idx, item in enumerate(subset, start=1):
        print(f"Processing sample {idx}/{len(subset)}")    

        sentence = item["sentence"]
        evidence = item["wiki_bio_text"]
        label = item["label"]

        score = max_entailment_score(sentence, evidence, tokenizer, model)
        bucket = discretize_score(score)

        if label == 1:
            pos_features[bucket] += 1
        else:
            neg_features[bucket] += 1

    # Laplace smoothing
    pos_features = [x + 1 for x in pos_features]
    neg_features = [x + 1 for x in neg_features]

    return pos_features, neg_features


def predict_one_sentence(sentence, evidence, tokenizer, model, pos_features, neg_features, P0=0.5):
    P = P0

    max_score = max_entailment_score(sentence, evidence, tokenizer, model)
    feature = discretize_score(max_score)

    p_given_1 = pos_features[feature] / sum(pos_features)
    p_given_0 = neg_features[feature] / sum(neg_features)

    P = bayes_update(P, p_given_1, p_given_0)

    return P, max_score, feature
