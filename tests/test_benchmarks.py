"""Generality benchmark (``pytest -m benchmark``): every case must meet its expectation."""

from __future__ import annotations

import pytest

from mlkit.benchmarks import CASES, run_case


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_benchmark_case(case):
    row = run_case(case)
    assert row["passed"], row["detail"]


def test_cli_rejects_unknown_case():
    from mlkit.cli import cmd_benchmark

    assert cmd_benchmark(["no_such_case"]) == 1


def test_cli_runs_a_fast_case(capsys):
    from mlkit.cli import cmd_benchmark

    assert cmd_benchmark(["leaky_feature"], with_eda=False) == 0
    assert "1/1 passed" in capsys.readouterr().out
