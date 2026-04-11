from src.utils import split_text, get_entailment_score


def discretize_score(score):
    bucket = int((score - 0.1) / 10)
    return max(0, min(bucket, 9))


def bayes_update(P, p_given_1, p_given_0):
    denominator = (1 - P) * p_given_0 + P * p_given_1
    if denominator == 0:
        return P
    return (P * p_given_1) / denominator

def min_cost(P, C_M, C_FA):
    return min((1 - P) * C_M, P * C_FA)

def expected_next__posterior(P, pos_features, neg_features):
    expected = 0.0
    total_pos = sum(pos_features)
    total_neg = sum(neg_features)

    for bucket in range(10):
        p_given_1 = pos_features[bucket] / total_pos
        p_given_0 = neg_features[bucket] / total_neg

        denominator = (1 - P) * p_given_0 + P * p_given_1
        if denominator == 0:
            next_P = P
        else:
            next_P = (P * p_given_1) / denominator

    return expected / 10.0

def should_continue(P, pos_features, neg_features, C_M=28, C_FA=96 ,C_retrieve=1):
    stop_cost = min_cost(P, C_M, C_FA)
    next_P = expected_next__posterior(P, pos_features, neg_features)
    continue_cost = C_retrieve + min_cost(next_P, C_M, C_FA)

    return stop_cost > continue_cost , stop_cost, continue_cost

def build_nbc_features(data, tokenizer, model, max_samples=50):
    pos_features = [0] * 10
    neg_features = [0] * 10

    subset = data[:max_samples]

    for idx, item in enumerate(subset, start=1):
        print(f"Processing sample {idx}/{len(subset)}")    

        sentence = item["sentence"]
        evidence = item["wiki_bio_text"]
        label = item["label"]

        segments = split_text(evidence)
        max_score = 0.0

        for seg in segments:
            score = get_entailment_score(seg, sentence, tokenizer, model)
            if score > max_score:
                max_score = score

        bucket = discretize_score(score)

        if label == 1:
            pos_features[bucket] += 1
        else:
            neg_features[bucket] += 1

    # Laplace smoothing
    pos_features = [x + 1 for x in pos_features]
    neg_features = [x + 1 for x in neg_features]

    return pos_features, neg_features

def predict_one_sentence_iterative(
    sentence,
    evidence,
    tokenizer,
    model,
    pos_features,
    neg_features,
    P0=0.5,
    C_M=28,
    C_FA=96,
    C_retrieve=1,
    max_steps=None,
):
    """
    Iterative version closer to the paper:
    - split evidence into segments
    - score each segment
    - discretize each score
    - update posterior sequentially
    - stop/continue after each update
    """
    P = P0
    segments = split_text(evidence)

    if max_steps is None:
        max_steps = len(segments)

    total_pos = sum(pos_features)
    total_neg = sum(neg_features)

    used_steps = 0
    history = []

    for seg in segments[:max_steps]:
        score = get_entailment_score(seg, sentence, tokenizer, model)
        bucket = discretize_score(score)

        p_given_1 = pos_features[bucket] / total_pos
        p_given_0 = neg_features[bucket] / total_neg

        P = bayes_update(P, p_given_1, p_given_0)
        used_steps += 1

        continue_flag, stop_cost, continue_cost = should_continue(
            P, pos_features, neg_features, C_M=C_M, C_FA=C_FA, C_retrieve=C_retrieve
        )

        history.append({
            "score": score,
            "bucket": bucket,
            "posterior": P,
            "stop_cost": stop_cost,
            "continue_cost": continue_cost,
            "continue": continue_flag,
        })

        if not continue_flag:
            break

    return {
        "posterior": P,
        "prediction": 1 if P >= 0.5 else 0,
        "steps_used": used_steps,
        "history": history,
    }
