"""Point-in-time feature builder (mlkit.features.temporal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit.core.errors import ValidationError
from mlkit.features import TemporalSpec, build_features, build_labels, build_panel

SPEC = TemporalSpec(
    entity_col="hh",
    time_col="day",
    windows=(7, 28),
    sum_cols=("amount",),
    nunique_cols=("basket",),
)


def _events() -> pd.DataFrame:
    # hh 1: days 1, 5, 20, 30 ; hh 2: day 2 only ; hh 3: day 40 (after as_of=30)
    return pd.DataFrame(
        {
            "hh": [1, 1, 1, 1, 1, 2, 3],
            "day": [1, 5, 20, 30, 30, 2, 40],
            "amount": [10.0, 5.0, 2.0, 1.0, 4.0, 7.0, 99.0],
            "basket": ["a", "b", "c", "d", "d", "e", "f"],
        }
    )


def test_window_and_lifetime_values():
    f = build_features(_events(), as_of=30, spec=SPEC)
    r = f.loc[1]
    assert r["n_events_7d"] == 2 and r["sum_amount_7d"] == 5.0 and r["nunique_basket_7d"] == 1
    # 28-day window is (2, 30]: days 5, 20, 30, 30
    assert r["n_events_28d"] == 4 and r["sum_amount_28d"] == 12.0 and r["n_active_days_28d"] == 3
    assert r["recency_days"] == 0 and r["tenure_days"] == 29 and r["n_active_days_life"] == 4
    assert r["ipi_mean"] == pytest.approx((4 + 15 + 10) / 3)
    # hh 2 last bought on day 2 (28 days before as_of): outside the 28-day eligibility window
    assert 2 not in f.index and 3 not in f.index


def test_future_events_never_change_features():
    ev = _events()
    base = build_features(ev, as_of=30, spec=SPEC)
    future = pd.DataFrame(
        {"hh": [1, 2, 1], "day": [31, 35, 60], "amount": [1e6] * 3, "basket": ["z"] * 3}
    )
    with_future = build_features(pd.concat([ev, future]), as_of=30, spec=SPEC)
    pd.testing.assert_frame_equal(base, with_future)


def test_labels_use_only_the_prediction_window():
    ev = _events()
    y = build_labels(ev, as_of=20, horizon=10, spec=SPEC, entities=[1, 2, 3])
    assert y.to_dict() == {1: 1, 2: 0, 3: 0}  # hh1 buys on day 30 (in (20,30]); hh3 on day 40
    assert build_labels(ev, as_of=30, horizon=10, spec=SPEC, entities=[1, 3]).to_dict() == {
        1: 0,
        3: 1,
    }
    with pytest.raises(ValueError):
        build_labels(ev, as_of=30, horizon=0, spec=SPEC)


def test_panel_rows_and_truncation_guard():
    ev = _events()
    panel = build_panel(ev, cutoffs=[10, 20], horizon=10, spec=SPEC)
    assert set(panel["cutoff"]) == {10, 20}
    row = panel[(panel["cutoff"] == 20) & (panel["hh"] == 1)].iloc[0]
    assert row["target"] == 1 and row["recency_days"] == 0
    with pytest.raises(ValidationError, match="truncated"):
        build_panel(
            ev, cutoffs=[35], horizon=10, spec=SPEC
        )  # window (35,45] beyond last event (40)


def test_datetime_time_column_matches_numeric():
    ev = _events()
    ev_dt = ev.assign(day=pd.Timestamp("2024-01-01") + pd.to_timedelta(ev["day"] - 1, unit="D"))
    num = build_features(ev, as_of=30, spec=SPEC)
    dt = build_features(ev_dt, as_of="2024-01-30", spec=SPEC)
    pd.testing.assert_frame_equal(num, dt, check_dtype=False)


def test_errors():
    with pytest.raises(ValidationError, match="missing columns"):
        build_features(_events().drop(columns="amount"), as_of=30, spec=SPEC)
    with pytest.raises(ValueError, match="windows"):
        TemporalSpec(entity_col="hh", time_col="day", windows=(0,))


def test_panel_trains_end_to_end():
    from mlkit.core import TaskConfig
    from mlkit.runner import train

    rng = np.random.default_rng(3)
    n_hh, days = 150, 200
    rate = rng.uniform(0.02, 0.5, n_hh)  # heterogeneous purchase frequency
    hh, day = np.nonzero(rng.uniform(size=(n_hh, days)) < rate[:, None])
    ev = pd.DataFrame(
        {
            "hh": hh,
            "day": day + 1,
            "amount": rng.gamma(2.0, 10.0, len(hh)),
            "basket": np.arange(len(hh)),
        }
    )
    spec = TemporalSpec(entity_col="hh", time_col="day", windows=(7, 28), sum_cols=("amount",))
    panel = build_panel(ev, cutoffs=range(40, 190, 7), horizon=7, spec=spec)
    cfg = TaskConfig(
        target="target",
        id_col="hh",
        time_col="cutoff",
        cv_folds=3,
        modelling={"candidates": ["linear"]},
    )
    res = train(cfg, panel)
    assert (
        res.metrics["holdout"]["roc_auc"] > 0.75
    )  # frequency is learnable from point-in-time features
