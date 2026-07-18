"""Tests for evaluation.visualization.generate_ablation_plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import EvaluationError
from evaluation.ablation import AblationReport, AblationResult
from evaluation.visualization import generate_ablation_plots


def _report() -> AblationReport:
    return AblationReport(
        results=[
            AblationResult(
                config_name=f"Config {i}", faithfulness=0.5 + i * 0.1, answer_relevancy=0.6,
                context_precision=0.7, retrieval_precision_at_5=0.4 + i * 0.1, average_latency_ms=100.0 + i * 10,
            )
            for i in range(5)
        ]
    )


class TestGenerateAblationPlots:
    def test_writes_three_png_files(self, tmp_path: Path) -> None:
        paths = generate_ablation_plots(_report(), output_dir=tmp_path)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()
            assert path.suffix == ".png"
            assert path.stat().st_size > 0

    def test_uses_expected_filenames(self, tmp_path: Path) -> None:
        paths = generate_ablation_plots(_report(), output_dir=tmp_path)

        names = {path.name for path in paths}
        assert names == {"retrieval_precision.png", "faithfulness.png", "latency.png"}

    def test_creates_output_directory_if_missing(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "plots"

        generate_ablation_plots(_report(), output_dir=output_dir)

        assert output_dir.exists()

    def test_empty_report_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError):
            generate_ablation_plots(AblationReport(results=[]), output_dir=tmp_path)
