"""Point-in-time aggregation of a related event table (no look-ahead leakage).

Each primary row has an entity key and an ``as_of`` time (the moment the prediction is made).
Only events of that entity **strictly before** ``as_of`` contribute to its features, so a
transaction that happens after the cutoff can never leak into training. This differs from
:class:`~mlkit.features.MultiTableAggregator`, which aggregates every event of an entity.

Fully vectorised (sort + cumulative sums + binary search): O((n_events + n_rows) log n).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.errors import ValidationError
from mlkit.core.registry import register
from mlkit.core.step import BaseStep


def _to_days(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(values):
        dt = pd.to_datetime(values)
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float) / 86400
    num = pd.to_numeric(values, errors="coerce")
    if num.notna().any():
        return num.to_numpy(dtype=float)
    dt = pd.to_datetime(values, errors="coerce")
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float) / 86400


def point_in_time_features(
    primary: pd.DataFrame,
    events: pd.DataFrame,
    key: str,
    as_of: str,
    event_time: str,
    value_cols: Sequence[str] | None = None,
    windows: Sequence[float] = (30, 90),
    prefix: str = "pit_",
) -> pd.DataFrame:
    """Features of ``events`` before each primary row's ``as_of``, aligned to ``primary.index``.

    Per entity and window (``all`` + each of ``windows``, in days or in the unit of a numeric
    time column): ``count``, and for each numeric ``value_cols``: ``sum``, ``mean``, ``std``.
    Plus ``days_since_last`` and ``days_since_first`` (NaN when there is no prior event).
    """
    for col, frame, name in (
        (key, primary, "primary"),
        (as_of, primary, "primary"),
        (key, events, "events"),
        (event_time, events, "events"),
    ):
        if col not in frame.columns:
            raise ValidationError(f"Column '{col}' not found in the {name} table.")
    if value_cols is None:
        value_cols = [
            c
            for c in events.select_dtypes(include=[np.number]).columns
            if c not in (key, event_time)
        ]
    t_ev = _to_days(events[event_time])
    t_q = _to_days(primary[as_of])
    keep = np.isfinite(t_ev) & events[key].notna().to_numpy()
    ev = events.loc[keep, [key, *value_cols]]
    t_ev = t_ev[keep]

    # one integer code per entity, shared by both tables; unknown entities get no events
    codes_ev, uniques = pd.factorize(ev[key])
    lookup = pd.Index(uniques)
    codes_q = lookup.get_indexer(pd.Index(primary[key]))

    lo = float(np.nanmin(np.concatenate([t_ev, t_q[np.isfinite(t_q)]]))) if len(t_ev) else 0.0
    hi = float(np.nanmax(np.concatenate([t_ev, t_q[np.isfinite(t_q)]]))) if len(t_ev) else 0.0
    span = hi - lo + max([0.0, *map(float, windows)]) + 1.0
    # entity blocks laid end to end on one axis: position = code * span + (t - lo)
    pos_ev = codes_ev * span + (t_ev - lo)
    order = np.argsort(pos_ev, kind="stable")
    pos_sorted = pos_ev[order]
    t_sorted = t_ev[order]
    block_start = np.searchsorted(pos_sorted, np.arange(len(lookup)) * span, side="left")

    known = (codes_q >= 0) & np.isfinite(t_q)
    q = np.where(known, np.maximum(codes_q, 0) * span + (t_q - lo), 0.0)
    end = np.searchsorted(pos_sorted, q, side="left")  # strictly before as_of
    start = np.where(known, block_start[np.maximum(codes_q, 0)], 0)
    end = np.where(known, end, 0)

    out: dict[str, np.ndarray] = {}
    # cumulative sums restart at every entity, so an entity's values (and any event after its
    # cutoff) can never affect another entity's features, not even through float rounding
    block = codes_ev[order]

    def local_cumsum(values: np.ndarray) -> np.ndarray:
        return pd.Series(values).groupby(block, sort=False).cumsum().to_numpy(dtype=float)

    def range_sum(cl: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # sum of the sorted events [a, b) of one entity block starting at `start`
        if len(cl) == 0:
            return np.zeros(len(a))
        upper = np.where(b > start, cl[np.maximum(b - 1, 0)], 0.0)
        lower = np.where(a > start, cl[np.maximum(a - 1, 0)], 0.0)
        return upper - lower

    cums = {}
    for c in value_cols:
        v = pd.to_numeric(ev[c], errors="coerce").to_numpy(dtype=float)[order]
        ok = np.isfinite(v)
        v0 = np.where(ok, v, 0.0)
        cums[c] = (
            local_cumsum(v0),
            local_cumsum(v0 * v0),
            local_cumsum(ok.astype(float)),
        )
    spans: list[tuple[str, float | None]] = [("all", None)]
    spans += [(f"{float(w):g}d", float(w)) for w in windows]
    for tag, width in spans:
        if width is None:
            a = start
        else:
            a = np.searchsorted(pos_sorted, q - width, side="left")
            a = np.where(known, np.maximum(a, start), 0)
        n = (end - a).astype(float)
        out[f"{prefix}count_{tag}"] = n
        for c, (cs, cs2, cn) in cums.items():
            s = range_sum(cs, a, end)
            s2 = range_sum(cs2, a, end)
            m = np.round(range_sum(cn, a, end))
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(m > 0, s / np.where(m > 0, m, 1), np.nan)
                var = np.where(m > 1, (s2 - m * mean**2) / np.where(m > 1, m - 1, 1), np.nan)
            out[f"{prefix}{c}_sum_{tag}"] = s
            out[f"{prefix}{c}_mean_{tag}"] = mean
            out[f"{prefix}{c}_std_{tag}"] = np.sqrt(np.clip(var, 0, None))
    has = end > start
    last = (
        np.where(has, t_sorted[np.maximum(end - 1, 0)], np.nan)
        if len(t_sorted)
        else np.full(len(t_q), np.nan)
    )
    first = (
        np.where(has, t_sorted[np.minimum(start, max(len(t_sorted) - 1, 0))], np.nan)
        if len(t_sorted)
        else np.full(len(t_q), np.nan)
    )
    out[f"{prefix}days_since_last"] = t_q - last
    out[f"{prefix}days_since_first"] = t_q - first
    return pd.DataFrame(out, index=primary.index)


@register("features.point_in_time")
class PointInTimeAggregator(BaseStep):
    """Pipeline step: add :func:`point_in_time_features` of ``events`` to every row.

    The primary frame must contain ``key`` and ``as_of``; ``drop_as_of`` removes the cutoff
    column afterwards (it is a timestamp, not a feature).
    """

    def __init__(
        self,
        events: pd.DataFrame,
        key: str,
        as_of: str,
        event_time: str,
        value_cols: list[str] | None = None,
        windows: list[float] | None = None,
        prefix: str = "pit_",
        drop_as_of: bool = False,
    ) -> None:
        super().__init__()
        self.events = events  # never mutated; not copied so clone() works
        self.key = key
        self.as_of = as_of
        self.event_time = event_time
        self.value_cols = value_cols
        self.windows = windows
        self.prefix = prefix
        self.drop_as_of = drop_as_of

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> PointInTimeAggregator:
        feats = self._features(X)
        self.feature_names_ = list(feats.columns)
        self.report_.add(
            "point_in_time",
            "ok",
            f"{len(self.feature_names_)} point-in-time features from events before '{self.as_of}'.",
        )
        return self

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        return point_in_time_features(
            X,
            self.events,
            self.key,
            self.as_of,
            self.event_time,
            self.value_cols,
            self.windows if self.windows is not None else (30, 90),
            self.prefix,
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([X, self._features(X)], axis=1)
        return out.drop(columns=[self.as_of]) if self.drop_as_of else out
