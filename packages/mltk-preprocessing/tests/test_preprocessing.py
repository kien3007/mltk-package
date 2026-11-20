"""Unit tests for mltk-preprocessing."""

import numpy as np
import pandas as pd
import pytest

from mltk_preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)


def test_type_caster():
    df = pd.DataFrame({"a": [" 1 ", " 2 ", " 3 "], "b": ["x ", " y", "z"]})
    caster = TypeCaster(dtypes={"a": int})
    res = caster.fit_transform(df)
    assert res["a"].dtype == int
    assert list(res["b"]) == ["x", "y", "z"]


def test_missing_imputer():
    train_df = pd.DataFrame(
        {
            "num": [10.0, 20.0, np.nan, 40.0],
            "cat": ["a", "b", "b", None],
        }
    )
    test_df = pd.DataFrame(
        {
            "num": [np.nan, 50.0],
            "cat": [None, "a"],
        }
    )
    imputer = MissingImputer(numeric_strategy="median", categorical_strategy="most_frequent")
    imputer.fit(train_df)

    res_train = imputer.transform(train_df)
    assert res_train["num"].iloc[2] == 20.0  # Median of 10, 20, 40 is 20.0
    assert res_train["cat"].iloc[3] == "b"  # Mode of 'a', 'b', 'b' is 'b'

    res_test = imputer.transform(test_df)
    assert res_test["num"].iloc[0] == 20.0  # Uses train median, no leakage
    assert res_test["cat"].iloc[0] == "b"


def test_outlier_clipper():
    df = pd.DataFrame({"num": [-100.0, 1.0, 2.0, 3.0, 1000.0]})
    clipper = OutlierClipper(lower_quantile=0.1, upper_quantile=0.9)
    res = clipper.fit_transform(df)
    assert res["num"].min() > -100.0
    assert res["num"].max() < 1000.0


def test_rare_category_grouper():
    df = pd.DataFrame({"cat": ["A"] * 100 + ["B"] * 100 + ["Rare1", "Rare2"]})
    grouper = RareCategoryGrouper(threshold=0.05, rare_label="__rare__")
    res = grouper.fit_transform(df)
    assert "__rare__" in res["cat"].values
    assert "Rare1" not in res["cat"].values


def test_categorical_encoder_target_and_onehot():
    df = pd.DataFrame(
        {
            "low_card": ["cat1", "cat2", "cat1", "cat2", "cat1", "cat2"],
            "high_card": [f"id_{i}" for i in range(6)],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    encoder = CategoricalEncoder(
        low_card_threshold=2,
        high_card_threshold=3,
        low_strategy="onehot",
        high_strategy="target",
        cv=2,
        random_state=42,
    )
    res = encoder.fit_transform(df, y)

    # low_card should be one-hot encoded
    assert "low_card_cat1" in res.columns
    assert "low_card_cat2" in res.columns
    assert "low_card" not in res.columns

    # high_card should be target encoded
    assert "high_card" in res.columns
    assert pd.api.types.is_numeric_dtype(res["high_card"].dtype)


def test_feature_scaler():
    df = pd.DataFrame({"num": [10.0, 20.0, 30.0], "cat": ["x", "y", "z"]})
    scaler = FeatureScaler(method="standard")
    res = scaler.fit_transform(df)
    assert isinstance(res, pd.DataFrame)
    assert np.isclose(res["num"].mean(), 0.0)
    assert list(res["cat"]) == ["x", "y", "z"]
