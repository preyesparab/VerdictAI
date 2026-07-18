"""Tests for evaluation.metrics."""

from __future__ import annotations

from evaluation.metrics import mean, precision_at_k


class TestMean:
    def test_averages_values(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_list_is_zero(self) -> None:
        assert mean([]) == 0.0


class TestPrecisionAtK:
    def test_computes_fraction_meeting_threshold(self) -> None:
        assert precision_at_k([5, 4, 3, 2, 1], k=5, threshold=4) == 0.4

    def test_only_considers_top_k(self) -> None:
        assert precision_at_k([5, 5, 1, 1, 1], k=2, threshold=4) == 1.0

    def test_fewer_scores_than_k_uses_what_is_available(self) -> None:
        assert precision_at_k([5, 5], k=5, threshold=4) == 1.0

    def test_empty_scores_is_zero(self) -> None:
        assert precision_at_k([], k=5, threshold=4) == 0.0

    def test_none_meeting_threshold_is_zero(self) -> None:
        assert precision_at_k([1, 2, 3], k=3, threshold=4) == 0.0
