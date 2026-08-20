from tqdm import tqdm

from src.utils import split_text


def discretize_score(score):
    bucket = int((score - 0.1) / 10)
    return max(0, min(bucket, 9))


def bayes_update(P, p_given_1, p_given_0):
    denominator = (1 - P) * p_given_0 + P * p_given_1
    if denominator <= 0:
        return P
    return (P * p_given_1) / denominator


def min_cost(P, C_M, C_FA):
    """Minimum expected misclassification cost at posterior P=P(factual)."""
    return min((1 - P) * C_M, P * C_FA)


def expected_next_risk(P, pos_features, neg_features, C_M=28, C_FA=96):
    """Expected Bayes risk after observing one additional evidence segment."""
    total_pos = sum(pos_features)
    total_neg = sum(neg_features)

    expected_risk = 0.0
    predictive_mass = 0.0

    for bucket in range(10):
        p_given_1 = pos_features[bucket] / total_pos
        p_given_0 = neg_features[bucket] / total_neg

        predictive_prob = P * p_given_1 + (1 - P) * p_given_0
        if predictive_prob <= 0:
            continue

        next_P = bayes_update(P, p_given_1, p_given_0)
        expected_risk += predictive_prob * min_cost(next_P, C_M, C_FA)
        predictive_mass += predictive_prob

    if predictive_mass <= 0:
        return min_cost(P, C_M, C_FA)

    return expected_risk / predictive_mass


def should_continue(P, pos_features, neg_features, C_M=28, C_FA=96, C_retrieve=1):
    stop_cost = min_cost(P, C_M, C_FA)
    continue_cost = C_retrieve + expected_next_risk(
        P,
        pos_features,
        neg_features,
        C_M=C_M,
        C_FA=C_FA,
    )
    return stop_cost > continue_cost, stop_cost, continue_cost


def build_nbc_features(data, scorer, max_samples=None):
    """Estimate score-bucket likelihoods using cached training NLI features."""
    pos_features = [0] * 10
    neg_features = [0] * 10

    subset = data if max_samples is None else data[:max_samples]

    for item in tqdm(subset, desc="Building Bayesian training distributions", unit="sample"):
        sentence = item["sentence"]
        evidence = item["wiki_bio_text"]
        label = item["label"]

        scores, _ = scorer.score_evidence(
            sentence,
            evidence,
            use_cache=True,
            write_cache=True,
        )
        max_score = max(scores) if scores else 0.0
        bucket = discretize_score(max_score)

        if label == 1:
            pos_features[bucket] += 1
        else:
            neg_features[bucket] += 1

    # Laplace smoothing prevents zero-probability buckets.
    pos_features = [x + 1 for x in pos_features]
    neg_features = [x + 1 for x in neg_features]

    return pos_features, neg_features


def predict_one_sentence_iterative(
    sentence,
    evidence,
    scorer,
    pos_features,
    neg_features,
    P0=0.5,
    C_M=28,
    C_FA=96,
    C_retrieve=1,
    max_steps=None,
    use_cache=False,
):
    """Sequential Bayesian hallucination detector.

    Test-time scoring is uncached by default so latency represents real NLI
    inference. The method remains sequential because the stop/continue decision
    after one evidence segment determines whether the next segment is evaluated.
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
        score = scorer.score_pairs(
            [(seg, sentence)],
            use_cache=use_cache,
            write_cache=use_cache,
        )[0]
        bucket = discretize_score(score)

        p_given_1 = pos_features[bucket] / total_pos
        p_given_0 = neg_features[bucket] / total_neg

        P = bayes_update(P, p_given_1, p_given_0)
        used_steps += 1

        continue_flag, stop_cost, continue_cost = should_continue(
            P,
            pos_features,
            neg_features,
            C_M=C_M,
            C_FA=C_FA,
            C_retrieve=C_retrieve,
        )

        history.append(
            {
                "score": score,
                "bucket": bucket,
                "posterior": P,
                "stop_cost": stop_cost,
                "continue_cost": continue_cost,
                "continue": continue_flag,
            }
        )

        if not continue_flag:
            break

    return {
        "posterior": P,
        "prediction": 1 if P >= 0.5 else 0,
        "steps_used": used_steps,
        "nli_calls": used_steps,
        "history": history,
    }
