from dataclasses import dataclass


@dataclass
class DetectionResult:
    p_factual: float
    prediction: int
    documents_used: int
    nli_calls: int


def discretize_nbc_score(score):
    """Match NBC_feature.py: int(round(score, 1) / 10), clamped defensively."""
    bucket = int(round(float(score), 1) / 10.0)
    return max(0, min(bucket, 9))


def discretize_document_score(score):
    """Match released main.py: int((round(score, 1) - 0.1) / 10)."""
    bucket = int((round(float(score), 1) - 0.1) / 10.0)
    return max(0, min(bucket, 9))


def build_nbc_histograms(pos_pairs, neg_pairs, scorer):
    """Estimate P(f|factual) and P(f|hallucinated) from Wang's NBC data."""
    pos_hist = [0] * 10
    neg_hist = [0] * 10

    pos_scores = scorer.score_pairs(
        [(item["premise"], item["hypothesis"]) for item in pos_pairs],
        use_cache=True,
        write_cache=True,
        show_progress=True,
        description="NBC factual pairs",
    )
    neg_scores = scorer.score_pairs(
        [(item["premise"], item["hypothesis"]) for item in neg_pairs],
        use_cache=True,
        write_cache=True,
        show_progress=True,
        description="NBC hallucinated pairs",
    )

    for score in pos_scores:
        pos_hist[discretize_nbc_score(score)] += 1
    for score in neg_scores:
        neg_hist[discretize_nbc_score(score)] += 1

    # Wang et al. use Laplace smoothing (+1 per bin).
    pos_hist = [x + 1 for x in pos_hist]
    neg_hist = [x + 1 for x in neg_hist]
    return pos_hist, neg_hist, pos_scores, neg_scores


def bayes_update(p_factual, p_feature_given_factual, p_feature_given_hallucinated):
    denominator = (
        p_factual * p_feature_given_factual
        + (1.0 - p_factual) * p_feature_given_hallucinated
    )
    if denominator <= 0:
        return p_factual
    return (p_factual * p_feature_given_factual) / denominator


def stop_cost(p_factual, c_miss, c_false_alarm):
    return min(
        (1.0 - p_factual) * c_miss,
        p_factual * c_false_alarm,
    )


def cost_based_prediction(p_factual, c_miss, c_false_alarm):
    factual_cost = (1.0 - p_factual) * c_miss
    hallucination_cost = p_factual * c_false_alarm
    return 1 if factual_cost < hallucination_cost else 0


class BSEDetector:
    """Wang et al. Bayesian sequential estimation over retrieved documents.

    mode="official" reproduces the released repository's one-step look-ahead:
    it averages the ten possible next posteriors and then evaluates stop risk.

    mode="eq8" implements Equation 8 from the paper literally by weighting the
    next-step stop risk by the predictive probability of each feature bucket.
    """

    def __init__(
        self,
        pos_hist,
        neg_hist,
        *,
        mode="official",
        p0=0.5,
        c_miss=28,
        c_false_alarm=96,
        c_retrieve=1,
        max_docs=10,
    ):
        if mode not in {"official", "eq8"}:
            raise ValueError("mode must be 'official' or 'eq8'")
        self.pos_hist = list(pos_hist)
        self.neg_hist = list(neg_hist)
        self.mode = mode
        self.p0 = float(p0)
        self.c_miss = float(c_miss)
        self.c_false_alarm = float(c_false_alarm)
        self.c_retrieve = float(c_retrieve)
        self.max_docs = int(max_docs)
        self.total_pos = float(sum(self.pos_hist))
        self.total_neg = float(sum(self.neg_hist))

    def _bucket_likelihoods(self, bucket):
        return (
            self.pos_hist[bucket] / self.total_pos,
            self.neg_hist[bucket] / self.total_neg,
        )

    def _official_expected_next_posterior(self, p_factual):
        values = []
        for bucket in range(10):
            p1, p0 = self._bucket_likelihoods(bucket)
            values.append(bayes_update(p_factual, p1, p0))
        return sum(values) / len(values)

    def _eq8_expected_next_stop_risk(self, p_factual):
        expected = 0.0
        total_mass = 0.0
        for bucket in range(10):
            p1, p0 = self._bucket_likelihoods(bucket)
            predictive = p_factual * p1 + (1.0 - p_factual) * p0
            if predictive <= 0:
                continue
            next_p = bayes_update(p_factual, p1, p0)
            expected += predictive * stop_cost(
                next_p, self.c_miss, self.c_false_alarm
            )
            total_mass += predictive
        if total_mass <= 0:
            return stop_cost(p_factual, self.c_miss, self.c_false_alarm)
        return expected / total_mass

    def continue_cost(self, p_factual):
        if self.mode == "official":
            expected_p = self._official_expected_next_posterior(p_factual)
            future_risk = stop_cost(expected_p, self.c_miss, self.c_false_alarm)
        else:
            future_risk = self._eq8_expected_next_stop_risk(p_factual)
        return self.c_retrieve + future_risk

    def should_continue(self, p_factual):
        return stop_cost(
            p_factual, self.c_miss, self.c_false_alarm
        ) > self.continue_cost(p_factual)

    def detect_subclaim(self, subclaim, scorer, *, use_cache=True):
        p_factual = self.p0
        documents_used = 0
        nli_calls = 0

        for document in subclaim.documents[: self.max_docs]:
            if not self.should_continue(p_factual):
                break

            score, segment_calls = scorer.score_document(
                subclaim.text,
                document.page_content,
                use_cache=use_cache,
                write_cache=use_cache,
            )
            documents_used += 1
            nli_calls += segment_calls

            bucket = discretize_document_score(score)
            p1, p0 = self._bucket_likelihoods(bucket)
            p_factual = bayes_update(p_factual, p1, p0)

        return DetectionResult(
            p_factual=float(p_factual),
            prediction=cost_based_prediction(
                p_factual, self.c_miss, self.c_false_alarm
            ),
            documents_used=documents_used,
            nli_calls=nli_calls,
        )

    def detect_sentence_with_trace(self, record, scorer, *, use_cache=True):
        """The sentence result AND the exact subclaim results that produced it.

        Audit finding D-10: ``detect_sentence`` collapsed the per-subclaim
        retrieval depths into one sentence total and discarded them, so the
        distribution of stopping depths could not be recovered afterwards.

        The trace is returned from the SAME pass that computes the result, in
        ``record.subclaims`` order. It is deliberately not obtainable by
        re-running ``detect_subclaim`` afterwards: that would duplicate NLI
        work, distort the wall-clock accounting, touch the cache a second time,
        and -- most importantly -- produce a trace of a *different* computation
        from the one that produced the sentence result.

        This is instrumentation. No decision in ``detect_subclaim`` observes it,
        and the aggregation below is the one ``detect_sentence`` always used.
        """
        subclaim_results = tuple(
            self.detect_subclaim(subclaim, scorer, use_cache=use_cache)
            for subclaim in record.subclaims
        )
        p_factual = min(result.p_factual for result in subclaim_results)
        sentence_result = DetectionResult(
            p_factual=float(p_factual),
            prediction=cost_based_prediction(
                p_factual, self.c_miss, self.c_false_alarm
            ),
            documents_used=sum(r.documents_used for r in subclaim_results),
            nli_calls=sum(r.nli_calls for r in subclaim_results),
        )
        return sentence_result, subclaim_results

    def detect_sentence(self, record, scorer, *, use_cache=True):
        """Unchanged behaviour: the sentence result alone.

        Delegates so there is exactly ONE aggregation implementation. Two copies
        would be free to drift, and a traced run could then disagree with an
        untraced one on the same input.
        """
        sentence_result, _ = self.detect_sentence_with_trace(
            record, scorer, use_cache=use_cache
        )
        return sentence_result
