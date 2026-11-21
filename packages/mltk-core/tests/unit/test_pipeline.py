"""Unit tests for MLTK Pipeline and split_data."""

import pandas as pd
from mltk_core.pipeline import Pipeline, split_data
from mltk_preprocessing import MissingImputer


def test_split_data():
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test = split_data(
        df, target="target", test_size=0.2, random_state=42
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_pipeline_execution():
    df = pd.DataFrame({"num": [1.0, None, 3.0, 4.0]})
    pipe = Pipeline([("imputer", MissingImputer(numeric_strategy="median"))])

    res = pipe.fit_transform(df)
    assert isinstance(res, pd.DataFrame)
    assert not res["num"].isnull().any()
