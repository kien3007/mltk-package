"""Household purchase propensity on dunnhumby "The Complete Journey".

Builds a point-in-time correct panel (one row per household x cutoff day) with
``mlkit.features.build_panel`` and writes a CSV plus a YAML config for ``mlkit run``.

    uv run python examples/dunnhumby_propensity.py --data-dir "<.../dunnhumby_The-Complete-Journey CSV>"
    uv run mlkit run artifacts/dunnhumby/config.yaml --out artifacts/dunnhumby/bundle
    uv run mlkit evaluate artifacts/dunnhumby/bundle <labelled panel rows> --by cutoff

Design (observed-retailer propensity):
- Cutoff T: features use only transactions with DAY <= T (enforced by build_features).
- Eligibility: at least one valid purchase in (T - obs_days, T].
- Target: >= 1 valid purchase in the prediction window (T, T + pred_days] (default 7 days).
- Valid purchase: QUANTITY > 0 and SALES_VALUE > 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlkit.features import TemporalSpec, build_panel

MAX_DAY = 711


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(
        data_dir / "transaction_data.csv",
        usecols=[
            "household_key",
            "BASKET_ID",
            "DAY",
            "PRODUCT_ID",
            "QUANTITY",
            "SALES_VALUE",
            "STORE_ID",
            "RETAIL_DISC",
            "COUPON_DISC",
            "COUPON_MATCH_DISC",
        ],
    )
    tx = tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)].copy()
    tx["disc"] = -(tx["RETAIL_DISC"] + tx["COUPON_DISC"] + tx["COUPON_MATCH_DISC"])
    tx["coupon"] = -(tx["COUPON_DISC"] + tx["COUPON_MATCH_DISC"])
    prod = pd.read_csv(
        data_dir / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]
    )
    tx = tx.merge(prod, on="PRODUCT_ID", how="left")
    return {
        "tx": tx,
        "demo": pd.read_csv(data_dir / "hh_demographic.csv"),
        "camp_desc": pd.read_csv(data_dir / "campaign_desc.csv"),
        "camp_hh": pd.read_csv(data_dir / "campaign_table.csv"),
        "redeem": pd.read_csv(data_dir / "coupon_redempt.csv"),
    }


def campaign_context(t: int, d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Campaign / coupon facts known at cutoff ``t`` (no future information)."""
    ch = d["camp_hh"].merge(d["camp_desc"][["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN")
    ctx = pd.DataFrame(
        {
            "n_campaigns_ended": ch[ch["END_DAY"] <= t].groupby("household_key").size(),
            "in_active_campaign": ch[(ch["START_DAY"] <= t) & (ch["END_DAY"] > t)]
            .groupby("household_key")
            .size(),
            "n_redemptions_before": d["redeem"][d["redeem"]["DAY"] <= t]
            .groupby("household_key")
            .size(),
        }
    )
    ctx.index.name = "household_key"
    return ctx.fillna(0).assign(cutoff=t).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("artifacts/dunnhumby"), type=Path)
    ap.add_argument("--pred-days", type=int, default=7, help="Prediction window length")
    ap.add_argument("--obs-days", type=int, default=90, help="Observation window length")
    ap.add_argument(
        "--step", type=int, default=None, help="Days between cutoffs (default: pred-days)"
    )
    args = ap.parse_args()

    d = load(args.data_dir)
    spec = TemporalSpec(
        entity_col="household_key",
        time_col="DAY",
        windows=(7, 28, args.obs_days),
        sum_cols=("SALES_VALUE", "QUANTITY", "disc", "coupon"),
        nunique_cols=("BASKET_ID", "PRODUCT_ID", "COMMODITY_DESC", "DEPARTMENT", "STORE_ID"),
        eligibility_days=args.obs_days,
    )
    step = args.step or args.pred_days
    cutoffs = list(range(120, MAX_DAY - args.pred_days + 1, step))
    panel = build_panel(d["tx"], cutoffs=cutoffs, horizon=args.pred_days, spec=spec)

    w = args.obs_days
    panel[f"spend_per_basket_{w}d"] = (
        panel[f"sum_SALES_VALUE_{w}d"] / panel[f"nunique_BASKET_ID_{w}d"]
    )
    panel[f"disc_share_{w}d"] = panel[f"sum_disc_{w}d"] / (
        panel[f"sum_SALES_VALUE_{w}d"] + panel[f"sum_disc_{w}d"]
    ).clip(lower=1e-6)

    ctx = pd.concat([campaign_context(t, d) for t in cutoffs], ignore_index=True)
    panel = panel.merge(ctx, on=["household_key", "cutoff"], how="left")
    ctx_cols = ["n_campaigns_ended", "in_active_campaign", "n_redemptions_before"]
    panel[ctx_cols] = panel[ctx_cols].fillna(0)
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
  exclude: [tenure_days, n_active_days_life, n_campaigns_ended]
modelling:
  candidates: [linear, random_forest, lightgbm]
  handle_imbalance: false   # keep raw class priors so scores are usable as probabilities
""",
        encoding="utf-8",
    )
    print(
        f"cutoffs={len(cutoffs)} rows={len(panel):,} households={panel['household_key'].nunique():,}"
    )
    print(f"positive rate={panel['target'].mean():.3f}  features={panel.shape[1] - 3}")


if __name__ == "__main__":
    main()
