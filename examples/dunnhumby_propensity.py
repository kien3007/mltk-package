"""Household purchase propensity on dunnhumby "The Complete Journey".

Builds a point-in-time correct panel (one row per household x cutoff day) and writes
a CSV plus a YAML config that ``mlkit run`` can consume directly.

    uv run python examples/dunnhumby_propensity.py --data-dir "<.../dunnhumby_The-Complete-Journey CSV>"
    uv run mlkit run artifacts/dunnhumby/config.yaml --out artifacts/dunnhumby/bundle

Design (observed-retailer propensity):
- Cutoff T: features use only transactions with DAY <= T.
- Eligibility: at least one valid purchase in the observation window (T-obs_days, T].
- Target: >= 1 valid purchase in the prediction window (T, T+pred_days] (default 7 days).
- Valid purchase: QUANTITY > 0 and SALES_VALUE > 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAX_DAY = 711


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(
        data_dir / "transaction_data.csv",
        usecols=[
            "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
            "STORE_ID", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC",
        ],
    )
    tx = tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)].copy()
    tx["disc"] = -(tx["RETAIL_DISC"] + tx["COUPON_DISC"] + tx["COUPON_MATCH_DISC"])
    tx["coupon"] = -(tx["COUPON_DISC"] + tx["COUPON_MATCH_DISC"])
    prod = pd.read_csv(data_dir / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])
    tx = tx.merge(prod, on="PRODUCT_ID", how="left")
    return {
        "tx": tx.sort_values("DAY").reset_index(drop=True),
        "demo": pd.read_csv(data_dir / "hh_demographic.csv"),
        "camp_desc": pd.read_csv(data_dir / "campaign_desc.csv"),
        "camp_hh": pd.read_csv(data_dir / "campaign_table.csv"),
        "redeem": pd.read_csv(data_dir / "coupon_redempt.csv"),
    }


def build_cutoff(
    t: int, d: dict[str, pd.DataFrame], hh_day: pd.DataFrame, obs_days: int, pred_days: int
) -> pd.DataFrame:
    tx = d["tx"]
    past = tx[tx["DAY"] <= t]
    w = past[past["DAY"] > t - obs_days]
    w28 = w[w["DAY"] > t - 28]
    w7 = w[w["DAY"] > t - 7]

    g = w.groupby("household_key")
    feats = pd.DataFrame(
        {
            "n_baskets_90": g["BASKET_ID"].nunique(),
            "spend_90": g["SALES_VALUE"].sum(),
            "qty_90": g["QUANTITY"].sum(),
            "n_products_90": g["PRODUCT_ID"].nunique(),
            "n_commodities_90": g["COMMODITY_DESC"].nunique(),
            "n_departments_90": g["DEPARTMENT"].nunique(),
            "n_stores_90": g["STORE_ID"].nunique(),
            "disc_90": g["disc"].sum(),
            "coupon_90": g["coupon"].sum(),
        }
    )
    feats["disc_share_90"] = feats["disc_90"] / (feats["spend_90"] + feats["disc_90"]).clip(lower=1e-6)
    feats["spend_per_basket_90"] = feats["spend_90"] / feats["n_baskets_90"]
    feats["n_shop_days_90"] = w.groupby("household_key")["DAY"].nunique()
    g28 = w28.groupby("household_key")
    feats["n_baskets_28"] = g28["BASKET_ID"].nunique()
    feats["spend_28"] = g28["SALES_VALUE"].sum()
    feats["n_shop_days_28"] = g28["DAY"].nunique()
    g7 = w7.groupby("household_key")
    feats["n_baskets_7"] = g7["BASKET_ID"].nunique()
    feats["spend_7"] = g7["SALES_VALUE"].sum()
    short_cols = ["n_baskets_28", "spend_28", "n_shop_days_28", "n_baskets_7", "spend_7"]
    feats[short_cols] = feats[short_cols].fillna(0)

    # Lifetime rhythm up to T (shopping-day level)
    hd = hh_day[hh_day["DAY"] <= t]
    life = hd.groupby("household_key")["DAY"].agg(first_day="min", last_day="max", n_shop_days_life="nunique")
    gaps = hd.groupby("household_key")["DAY"].diff().dropna()
    gap_stats = gaps.groupby(hd.loc[gaps.index, "household_key"]).agg(ipi_mean="mean", ipi_std="std")
    feats = feats.join(life).join(gap_stats)
    feats["recency"] = t - feats["last_day"]
    feats["tenure"] = t - feats["first_day"]
    feats["recency_over_ipi"] = feats["recency"] / feats["ipi_mean"].clip(lower=1.0)
    feats = feats.drop(columns=["first_day", "last_day"])

    # Campaign / coupon context known at T
    ch = d["camp_hh"].merge(d["camp_desc"][["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN")
    ended = ch[ch["END_DAY"] <= t].groupby("household_key").size().rename("n_campaigns_ended")
    active = (
        ch[(ch["START_DAY"] <= t) & (ch["END_DAY"] > t)].groupby("household_key").size().rename("in_active_campaign")
    )
    red = d["redeem"][d["redeem"]["DAY"] <= t].groupby("household_key").size().rename("n_redemptions_before")
    feats = feats.join(ended).join(active).join(red)
    feats[["n_campaigns_ended", "in_active_campaign", "n_redemptions_before"]] = feats[
        ["n_campaigns_ended", "in_active_campaign", "n_redemptions_before"]
    ].fillna(0)

    # Target: any valid purchase in (T, T+28]
    future = tx[(tx["DAY"] > t) & (tx["DAY"] <= t + pred_days)]["household_key"].unique()
    feats["target"] = feats.index.isin(future).astype(int)
    feats["cutoff"] = t
    return feats.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("artifacts/dunnhumby"), type=Path)
    ap.add_argument("--pred-days", type=int, default=7, help="Prediction window length")
    ap.add_argument("--obs-days", type=int, default=90, help="Observation window length")
    ap.add_argument("--step", type=int, default=None, help="Days between cutoffs (default: pred-days)")
    args = ap.parse_args()

    d = load(args.data_dir)
    hh_day = d["tx"].groupby(["household_key", "DAY"], as_index=False).size()[["household_key", "DAY"]]
    hh_day = hh_day.sort_values(["household_key", "DAY"]).reset_index(drop=True)

    step = args.step or args.pred_days
    cutoffs = list(range(120, MAX_DAY - args.pred_days + 1, step))
    panel = pd.concat(
        [build_cutoff(t, d, hh_day, args.obs_days, args.pred_days) for t in cutoffs], ignore_index=True
    )
    panel = panel.merge(d["demo"], on="household_key", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_dir / "panel.csv", index=False)
    (args.out_dir / "config.yaml").write_text(
        f"""task_type: classification
target: target
id_col: household_key
time_col: cutoff
cv_folds: 4
test_size: 0.2
data_path: {(args.out_dir / "panel.csv").as_posix()}
metric_thresholds:   # release gate on the time-based holdout
  roc_auc: 0.80
  brier: 0.19
  ece: 0.02
monitoring:
  # cumulative-since-start features grow with time by construction: drift alarms there are noise
  exclude: [tenure, n_shop_days_life, n_campaigns_ended]
modelling:
  candidates: [linear, random_forest, lightgbm]
  handle_imbalance: false   # keep raw class priors so scores are usable as probabilities
""",
        encoding="utf-8",
    )
    print(f"cutoffs={len(cutoffs)} rows={len(panel):,} households={panel['household_key'].nunique():,}")
    print(f"positive rate={panel['target'].mean():.3f}")
    print(panel.groupby("cutoff")["target"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
