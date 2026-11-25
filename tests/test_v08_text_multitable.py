"""v0.8: semantic text features and point-in-time multi-table aggregation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from mlkit.core import Pipeline, TaskConfig
from mlkit.core.errors import ValidationError
from mlkit.features import PointInTimeAggregator, TextFeatures, point_in_time_features

rng = np.random.default_rng(8)


def _events(n_cust: int = 40, n_ev: int = 800) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cid": rng.integers(0, n_cust, n_ev),
            "t": rng.uniform(0, 200, n_ev),
            "amount": rng.lognormal(2, 0.5, n_ev),
        }
    )


def _brute(primary, ev, window):
    rows = []
    for _, r in primary.iterrows():
        e = ev[(ev.cid == r.cid) & (ev.t < r.as_of)]
        if window is not None:
            e = e[e.t >= r.as_of - window]
        rows.append(
            (
                len(e),
                e.amount.sum(),
                e.amount.mean() if len(e) else np.nan,
                e.amount.std() if len(e) > 1 else np.nan,
            )
        )
    return np.array(rows, dtype=float)


class TestPointInTime:
    def test_matches_brute_force(self):
        ev = _events()
        primary = pd.DataFrame({"cid": list(range(40)) + [999], "as_of": rng.uniform(0, 220, 41)})
        f = point_in_time_features(primary, ev, "cid", "as_of", "t", windows=(30,))
        for tag, w in (("all", None), ("30d", 30)):
            ref = _brute(primary, ev, w)
            got = f[
                [
                    f"pit_count_{tag}",
                    f"pit_amount_sum_{tag}",
                    f"pit_amount_mean_{tag}",
                    f"pit_amount_std_{tag}",
                ]
            ].to_numpy()
            assert np.allclose(got, ref, equal_nan=True, rtol=1e-6, atol=1e-6)
        # unknown entity: no history
        assert f.iloc[-1]["pit_count_all"] == 0 and np.isnan(f.iloc[-1]["pit_days_since_last"])
        last = ev[(ev.cid == primary.cid[0]) & (ev.t < primary.as_of[0])].t.max()
        assert f.iloc[0]["pit_days_since_last"] == pytest.approx(primary.as_of[0] - last)

    def test_future_events_never_leak(self):
        ev = _events()
        primary = pd.DataFrame({"cid": range(40), "as_of": rng.uniform(50, 150, 40)})
        full = point_in_time_features(primary, ev, "cid", "as_of", "t")
        cutoff = primary.set_index("cid")["as_of"]
        past_only = ev[ev.t < ev.cid.map(cutoff)]
        # adding / removing events at or after each cutoff must not change a single value
        assert full.equals(point_in_time_features(primary, past_only, "cid", "as_of", "t"))
        shifted = ev.assign(amount=np.where(ev.t >= ev.cid.map(cutoff), 1e9, ev.amount))
        assert full.equals(point_in_time_features(primary, shifted, "cid", "as_of", "t"))

    def test_datetimes_and_validation(self):
        ev = _events().assign(t=lambda d: pd.Timestamp("2024-01-01") + pd.to_timedelta(d.t, "D"))
        primary = pd.DataFrame(
            {"cid": [1, 2], "as_of": pd.to_datetime(["2024-04-01", "2024-06-01"])}
        )
        f = point_in_time_features(primary, ev, "cid", "as_of", "t", windows=(30,))
        assert (f["pit_count_all"] >= f["pit_count_30d"]).all()
        with pytest.raises(ValidationError, match="not found"):
            point_in_time_features(primary, ev, "nope", "as_of", "t")

    def test_pipeline_step(self):
        from mlkit.modelling import ModelArena
        from mlkit.preprocessing import MissingImputer

        ev = _events(200, 6000)
        primary = pd.DataFrame({"cid": range(200), "as_of": rng.uniform(100, 180, 200)})
        cnt = point_in_time_features(primary, ev, "cid", "as_of", "t")["pit_count_30d"]
        y = (cnt > cnt.median()).astype(int).to_numpy()
        pipe = Pipeline(
            [
                (
                    "pit",
                    PointInTimeAggregator(ev, "cid", "as_of", "t", windows=[30], drop_as_of=True),
                ),
                ("imp", MissingImputer()),
                ("arena", ModelArena(cv=3, candidates=["linear"], enforce_baseline_gate=False)),
            ]
        )
        pipe.fit(primary, y, cfg=TaskConfig(target="y"))
        assert "as_of" not in pipe.named_steps["arena"].feature_medians_
        assert pipe.predict_proba(primary.head(5)).shape == (5, 2)


class TestSemanticText:
    def _data(self):
        from mlkit.benchmarks import _semantic_text

        return _semantic_text(np.random.default_rng(0))

    def test_semantic_columns_and_signal(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        df = self._data()
        X, y = df[["review"]], df["target"].to_numpy()
        stats = TextFeatures(semantic=None).fit_transform(X, y)
        sem = TextFeatures().fit_transform(X, y)
        assert not any("_sem" in c for c in stats.columns)
        assert sum(c.startswith("review_sem") for c in sem.columns) == 32

        def auc(Z):
            model = LogisticRegression(max_iter=2000)
            return cross_val_score(model, Z, y, cv=3, scoring="roc_auc").mean()

        assert auc(sem) > auc(stats) + 0.2

    def test_transform_unseen_rows_and_degenerate_text(self):
        df = self._data()
        tf = TextFeatures(n_components=4).fit(df[["review"]].head(500))
        out = tf.transform(df[["review"]].tail(10))
        assert out.filter(like="_sem").shape == (10, 4)
        tiny = pd.DataFrame({"note": [f"unique words number {i} " * 3 for i in range(3)]})
        assert TextFeatures(columns=["note"]).fit_transform(tiny).shape[0] == 3

    def test_embedding_falls_back_without_sentence_transformers(self):
        if importlib.util.find_spec("sentence_transformers") is not None:
            pytest.skip("sentence-transformers installed")
        df = self._data().head(300)
        tf = TextFeatures(semantic="embedding", n_components=8).fit(df[["review"]])
        assert any(i.status == "warn" for i in tf.report_.items)
        assert tf.transform(df[["review"]]).filter(like="_sem").shape[1] == 8

    def test_invalid_semantic(self):
        with pytest.raises(ValueError, match="semantic"):
            TextFeatures(semantic="bert").fit(self._data().head(50)[["review"]])
