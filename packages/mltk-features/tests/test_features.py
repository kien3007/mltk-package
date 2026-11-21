"""Unit tests for mltk-features."""

import pandas as pd
from mltk_features import DatetimeFeatures, GroupAggFeatures, RatioFeatures


def test_datetime_features():
    df = pd.DataFrame({"signup_date": ["2026-01-01", "2026-07-15", "2026-12-31"]})
    dt = DatetimeFeatures(columns=["signup_date"], features=["year", "month", "day", "is_weekend"])
    res = dt.fit_transform(df)

    assert "signup_date_year" in res.columns
    assert "signup_date_month" in res.columns
    assert "signup_date_day" in res.columns
    assert "signup_date_is_weekend" in res.columns
    assert res["signup_date_year"].iloc[0] == 2026
    assert res["signup_date_month"].iloc[1] == 7


def test_datetime_features_graceful_skip():
    df = pd.DataFrame({"num": [1, 2, 3]})
    dt = DatetimeFeatures(columns="auto")
    res = dt.fit_transform(df)
    assert list(res.columns) == ["num"]


def test_ratio_features():
    df = pd.DataFrame({"charges": [100.0, 200.0, 0.0], "tenure": [10.0, 0.0, 5.0]})
    ratio = RatioFeatures(pairs=[("charges", "tenure")], fill_inf=0.0)
    res = ratio.fit_transform(df)

    assert "charges_ratio_tenure" in res.columns
    assert "charges_diff_tenure" in res.columns
    assert res["charges_ratio_tenure"].iloc[0] == 10.0
    assert res["charges_ratio_tenure"].iloc[1] == 0.0  # Safe division by zero
    assert res["charges_diff_tenure"].iloc[0] == 90.0


def test_group_agg_features():
    train_df = pd.DataFrame(
        {"city": ["NY", "NY", "SF", "SF"], "sales": [100.0, 200.0, 300.0, 400.0]}
    )
    test_df = pd.DataFrame({"city": ["NY", "SF", "LA"], "sales": [50.0, 60.0, 70.0]})

    agg = GroupAggFeatures(groupby="city", agg_cols=["sales"], aggs=["mean"])
    agg.fit(train_df)

    res_train = agg.transform(train_df)
    assert "city_grp_sales_mean" in res_train.columns
    assert res_train["city_grp_sales_mean"].iloc[0] == 150.0

    res_test = agg.transform(test_df)
    assert res_test["city_grp_sales_mean"].iloc[0] == 150.0  # Uses train stats
    assert res_test["city_grp_sales_mean"].iloc[1] == 350.0
    assert pd.isna(res_test["city_grp_sales_mean"].iloc[2])  # LA was unseen on train
