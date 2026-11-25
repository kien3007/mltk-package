"""v0.7: large data — sampled profiling, stratified EDA sampling, chunked scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.qc_eda import ProfileOptions, profile_dataframe, stratified_sample

rng = np.random.default_rng(7)


class TestSampledProfile:
    def _frame(self, n: int = 6000) -> pd.DataFrame:
        a = rng.normal(size=n)
        df = pd.DataFrame(
            {
                "a": a,
                "a_copy": a,
                "almost": a.copy(),
                "cat": rng.choice(["x", "y", "z"], n),
                "miss": np.where(rng.uniform(size=n) < 0.3, np.nan, 1.0) + rng.normal(size=n),
            }
        )
        df.loc[n - 1, "almost"] = 99.0  # differs from 'a' on a single row
        return pd.concat([df, df.head(250)], ignore_index=True)  # 250 exact duplicate rows

    def test_sample_with_exact_counts(self):
        df = self._frame()
        full = profile_dataframe(df, options=ProfileOptions(max_rows=None))
        prof = profile_dataframe(df, options=ProfileOptions(max_rows=1000))
        assert full.sampled_rows is None and prof.sampled_rows == 1000
        assert prof.n_rows == len(df) and prof.n_duplicate_rows == full.n_duplicate_rows == 250
        assert prof.columns["miss"].missing_rate == pytest.approx(full.columns["miss"].missing_rate)
        assert {c: p.semantic_type for c, p in prof.columns.items()} == {
            c: p.semantic_type for c, p in full.columns.items()
        }
        assert "sample" in prof.columns["a"].notes[-1]
        # a column identical only on the sampled rows is not reported as a duplicate
        groups = [set(g) for g in prof.duplicate_columns]
        assert {"a", "a_copy"} in groups and not any("almost" in g for g in groups)
        assert prof.to_dict()["sampled_rows"] == 1000

    def test_small_frames_unchanged(self):
        prof = profile_dataframe(pd.DataFrame({"x": range(10)}))
        assert prof.sampled_rows is None


def test_stratified_sample_keeps_rare_classes():
    n = 50_000
    df = pd.DataFrame({"y": np.where(np.arange(n) < 60, "rare", "common"), "x": np.arange(n)})
    s = stratified_sample(df, 2000, by="y", min_per_group=100)
    assert (s["y"] == "rare").sum() == 60 and len(s) <= 2000 + 100
    assert s.index.is_monotonic_increasing
    assert (
        len(stratified_sample(df, 500)) == 500 and stratified_sample(df.head(10), 500) is not None
    )


def test_run_eda_samples_by_target():
    from mlkit.qc_eda import run_eda

    n = 30_000
    df = pd.DataFrame({"x": rng.normal(size=n), "y": (np.arange(n) < 40).astype(int)})
    rep = run_eda(df, target="y", sample_rows=3000, with_qc=False)
    assert rep.profile.target is not None
    assert rep.profile.target.class_counts.get("1") == 40


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from mlkit.runner import train

    tmp = tmp_path_factory.mktemp("scale")
    n = 1500
    X = pd.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q"], n)})
    y = ((X["a"] + (X["c"] == "p")) > 0.5).astype(int)
    train(
        TaskConfig(target="y", cv_folds=3, modelling={"candidates": ["linear"]}),
        X.assign(y=y),
        output_dir=tmp / "b",
    )
    return tmp, X


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_predict_file_matches_in_memory(bundle, fmt):
    from mlkit.runner import predict_bundle, predict_file, read_table

    tmp, X = bundle
    src = tmp / f"in.{fmt}"
    X.to_csv(src, index=False) if fmt == "csv" else X.to_parquet(src, index=False)
    out = tmp / f"out.{fmt}"
    assert predict_file(tmp / "b", src, out, chunksize=400) == len(X)
    chunked = read_table(out)
    whole = predict_bundle(tmp / "b", X)
    assert len(chunked) == len(X)
    assert np.allclose(chunked["score"], whole["score"])


def test_cli_chunksize(bundle, tmp_path):
    from mlkit.cli import cmd_predict

    tmp, X = bundle
    src = tmp_path / "in.csv"
    X.to_csv(src, index=False)
    assert cmd_predict(str(tmp / "b"), str(src), str(tmp_path / "o.csv"), chunksize=500) == 0
    assert len(pd.read_csv(tmp_path / "o.csv")) == len(X)
    assert cmd_predict(str(tmp / "b"), str(src), None, chunksize=500) == 1  # needs --out


def test_read_table_pyarrow_engine(tmp_path):
    pytest.importorskip("pyarrow")
    from mlkit.runner import read_table

    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    assert read_table(p, engine="pyarrow").equals(read_table(p))


def test_run_scale_small():
    from mlkit.benchmarks import run_scale

    row = run_scale(n_rows=300_000, n_cols=20)
    assert row["passed"] and row["sampled_rows"] is None and row["profile_s"] < 60
