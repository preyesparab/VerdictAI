"""Small numeric helpers shared by the evaluation layer (Phase 17).

Kept deliberately tiny and dependency-free: `evaluation.retrieval_eval` and
`evaluation.ablation` both need "average of these numbers" and "fraction of
these that clear a threshold" - defining them once here means both modules
compute Precision@k and mean scores identically instead of drifting apart.
"""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Arithmetic mean of `values`.

    Args:
        values: Numbers to average.

    Returns:
        The mean, or 0.0 for an empty list (rather than raising a
        `ZeroDivisionError` - an evaluation run with nothing to average
        is a valid, if uninteresting, result).
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def precision_at_k(scores: list[int], k: int, threshold: int) -> float:
    """Fraction of the top `k` graded `scores` that meet `threshold`.

    Args:
        scores: Relevance grades (e.g. 1-5 from an LLM-as-judge), in
            ranked order (most relevant first).
        k: How many of the leading `scores` to consider. If `scores` has
            fewer than `k` entries, every available score is used instead
            (a repository with fewer than `k` retrievable chunks is not
            treated as an error).
        threshold: Minimum score, inclusive, to count as "relevant".

    Returns:
        `(# of the top-k scores >= threshold) / (# scores considered)`,
        or 0.0 if `scores` is empty.
    """
    considered = scores[:k]
    if not considered:
        return 0.0
    relevant = sum(1 for score in considered if score >= threshold)
    return relevant / len(considered)
