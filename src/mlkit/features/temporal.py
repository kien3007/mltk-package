"""Point-in-time (as-of) behavioural features from an event log.

The same function builds features for training panels (many cutoffs) and for scoring
(one ``as_of``), so training and serving use identical logic and only events with
``time <= as_of`` are ever read.

Example (dunnhumby-style transactions, day index as time)::

    spec = TemporalSpec(
        entity_col="household_key", time_col="DAY",
        windows=(7, 28, 90),
        sum_cols=("SALES_VALUE",), nunique_cols=("BASKET_ID", "PRODUCT_ID"),
    )
    panel = build_panel(tx, cutoffs=range(120, 705, 7), horizon=7, spec=spec)   # training
    X_now = build_features(tx, as_of=711, spec=spec)                            # scoring

Time may be numeric (e.g. day index) or datetime; windows and horizons are in days.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlkit.core.errors import ValidationError


@dataclass(frozen=True)
class TemporalSpec:
    """What to aggregate from the event log.

    entity_col    : entity key (household, customer, account, ...)
    time_col      : event time (numeric day index or datetime)
    windows       : look-back windows in days, each giving ``(as_of - w, as_of]`` features
    sum_cols      : numeric columns summed per window (``sum_<col>_<w>d``)
    nunique_cols  : columns counted as distinct values per window (``nunique_<col>_<w>d``)
    eligibility_days : entity must have >= 1 event in ``(as_of - eligibility_days, as_of]``
                       (default: the largest window)
    """

    entity_col: str
    time_col: str
    windows: tuple[int, ...] = (7, 28, 90)
    sum_cols: tuple[str, ...] = ()
    nunique_cols: tuple[str, ...] = ()
    eligibility_days: int | None = None
    lifetime: bool = True

    def __post_init__(self) -> None:
        if not self.windows or any(w <= 0 for w in self.windows):
            raise ValueError("windows must be positive day counts")


def _days(series: pd.Series) -> np.ndarray:
    """Event time as float days (numeric as-is; datetimes as days since epoch)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series.astype("int64") / 86_400e9).to_numpy(dtype=float)
    return pd.to_numeric(series, errors="raise").to_numpy(dtype=float)


def _as_of_days(as_of: float | int | pd.Timestamp | str, times: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(times):
        return float(pd.Timestamp(as_of).value / 86_400e9)
    return float(as_of)


def _validate(events: pd.DataFrame, spec: TemporalSpec) -> None:
    needed = {spec.entity_col, spec.time_col, *spec.sum_cols, *spec.nunique_cols}
    missing = sorted(needed - set(events.columns))
    if missing:
        raise ValidationError(f"Event log is missing columns: {missing}")


def build_features(
    events: pd.DataFrame, as_of: float | int | pd.Timestamp | str, spec: TemporalSpec
) -> pd.DataFrame:
    """Features per eligible entity using only events with ``time <= as_of``.

    Returns a DataFrame indexed by ``spec.entity_col``.
    """
    _validate(events, spec)
    t_all = _days(events[spec.time_col])
    t0 = _as_of_days(as_of, events[spec.time_col])
    past_mask = t_all <= t0
    past = events.loc[past_mask]
    age = t0 - t_all[past_mask]  # days before as_of, >= 0
    key = past[spec.entity_col].to_numpy()
    day = np.floor(t_all[past_mask])

    elig_days = spec.eligibility_days or max(spec.windows)
    eligible = pd.Index(pd.unique(key[age < elig_days]), name=spec.entity_col)
    feats = pd.DataFrame(index=eligible)

    for w in sorted(spec.windows):
        m = age < w
        g = pd.DataFrame({"k": key[m], "day": day[m]})
        for c in spec.sum_cols:
            g[c] = past[c].to_numpy()[m]
        for c in spec.nunique_cols:
            g["u_" + c] = past[c].to_numpy()[m]
        grp = g.groupby("k")
        feats[f"n_events_{w}d"] = grp.size()
        feats[f"n_active_days_{w}d"] = grp["day"].nunique()
        for c in spec.sum_cols:
            feats[f"sum_{c}_{w}d"] = grp[c].sum()
        for c in spec.nunique_cols:
            feats[f"nunique_{c}_{w}d"] = grp["u_" + c].nunique()

    count_cols = [c for c in feats.columns if c.startswith(("n_", "sum_", "nunique_"))]
    feats[count_cols] = feats[count_cols].fillna(0)

    if spec.lifetime:
        life = pd.DataFrame({"k": key, "day": day})
        active = life.drop_duplicates().sort_values(["k", "day"])
        gl = active.groupby("k")["day"]
        feats["recency_days"] = t0 - gl.max()
        feats["tenure_days"] = t0 - gl.min()
        feats["n_active_days_life"] = gl.size()
        gaps = active.assign(gap=gl.diff()).dropna(subset=["gap"]).groupby("k")["gap"]
        feats["ipi_mean"] = gaps.mean()
        feats["ipi_std"] = gaps.std()
        feats["recency_over_ipi"] = feats["recency_days"] / feats["ipi_mean"].clip(lower=1.0)

    feats.index.name = spec.entity_col
    return feats


def build_labels(
    events: pd.DataFrame,
    as_of: float | int | pd.Timestamp | str,
    horizon: int,
    spec: TemporalSpec,
    entities: Iterable | None = None,
) -> pd.Series:
    """1 if the entity has >= 1 event in ``(as_of, as_of + horizon]`` else 0."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive number of days")
    t_all = _days(events[spec.time_col])
    t0 = _as_of_days(as_of, events[spec.time_col])
    future = events.loc[(t_all > t0) & (t_all <= t0 + horizon), spec.entity_col].unique()
    idx = pd.Index(
        entities if entities is not None else events[spec.entity_col].unique(), name=spec.entity_col
    )
    return pd.Series(idx.isin(future).astype(int), index=idx, name="target")


def build_panel(
    events: pd.DataFrame,
    cutoffs: Iterable[float | int | pd.Timestamp | str],
    horizon: int,
    spec: TemporalSpec,
    cutoff_col: str = "cutoff",
    target_col: str = "target",
) -> pd.DataFrame:
    """Training panel: one row per (eligible entity, cutoff) with features and a label.

    Cutoffs whose prediction window extends beyond the last observed event are rejected,
    because their labels would be silently truncated (false negatives).
    """
    _validate(events, spec)
    last = float(np.nanmax(_days(events[spec.time_col])))
    frames = []
    for c in cutoffs:
        t0 = _as_of_days(c, events[spec.time_col])
        if t0 + horizon > last:
            raise ValidationError(
                f"Cutoff {c}: prediction window ends after the last observed event; labels would be truncated."
            )
        feats = build_features(events, c, spec)
        labels = build_labels(events, c, horizon, spec, entities=feats.index)
        frames.append(feats.assign(**{target_col: labels.to_numpy(), cutoff_col: c}).reset_index())
    if not frames:
        raise ValidationError("No cutoffs given.")
    return pd.concat(frames, ignore_index=True)
