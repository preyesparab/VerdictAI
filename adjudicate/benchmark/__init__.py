"""Benchmark harness (Phase 31): measures Adjudicate's review quality against a baseline.

Runs three conditions - (a) a single-agent baseline reviewer, (b) the
real Defender/Prosecutor debate with claims trusted at face value (no
Verifier), (c) the full Phases 25-30 pipeline - across a curated,
ground-truth-labeled case set (`cases/*.json`), and reports catch rate,
false-positive rate, cost, rounds-to-verdict, and claim-flip rate (how
often a claim that looked concerning without verification was actually
REFUTED once checked for real).

Entry point: `python -m adjudicate.benchmark.run`. See
`adjudicate.benchmark.harness`'s own docstring for exactly how each
condition is run and what "flagged" means for each.
"""

from __future__ import annotations
