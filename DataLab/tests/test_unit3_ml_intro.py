import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datalab import unit3_ml_intro as ml
from datalab.generate_datasets import generate_car_data
from datalab import unit1_eda as eda


def test_explain_supervised_learning_covers_both_kinds():
    explanation = ml.explain_supervised_learning()
    assert "regression" in explanation
    assert "classification" in explanation
    assert "supervised_learning" in explanation
    assert len(explanation["regression"]) > 10  # real text, not a stub


def test_encode_categorical_columns_produces_integers_and_reversible_encoders():
    df, _ = eda.clean_car_data(generate_car_data())
    encoded, encoders = ml.encode_categorical_columns(df, columns=["fuel", "transmission"])

    assert encoded["fuel"].dtype.kind in "iu"  # integer dtype now
    assert "fuel" in encoders

    # the encoder must be able to reverse itself back to the real labels
    decoded = encoders["fuel"].inverse_transform(encoded["fuel"])
    assert set(decoded) == set(df["fuel"].unique())


def test_engineer_car_age_feature_computes_correctly():
    df = generate_car_data()
    engineered = ml.engineer_car_age_feature(df, current_year=2026)
    assert (engineered["car_age"] == 2026 - df["year"]).all()


def test_select_feature_subset_excludes_target_and_non_numeric():
    df, _ = eda.clean_car_data(generate_car_data())
    df = ml.engineer_car_age_feature(df)
    encoded, _ = ml.encode_categorical_columns(df, columns=["fuel", "seller_type", "transmission", "owner"])

    X, y = ml.select_feature_subset(encoded, target_column="selling_price", exclude_columns=["year"])

    assert "selling_price" not in X.columns
    assert "year" not in X.columns
    assert "name" not in X.columns  # non-numeric, correctly dropped
    assert (y == encoded["selling_price"]).all()


def test_train_test_split_data_produces_correct_proportions():
    df, _ = eda.clean_car_data(generate_car_data())
    X, y = ml.select_feature_subset(
        ml.engineer_car_age_feature(df), target_column="selling_price", exclude_columns=["year", "name", "fuel", "seller_type", "transmission", "owner"]
    )
    X_train, X_test, y_train, y_test = ml.train_test_split_data(X, y, test_size=0.25)

    total = len(X)
    assert abs(len(X_test) / total - 0.25) < 0.02
    assert len(X_train) + len(X_test) == total
    assert len(y_train) == len(X_train)


def test_train_test_split_is_reproducible_with_same_seed():
    df, _ = eda.clean_car_data(generate_car_data())
    X, y = ml.select_feature_subset(
        ml.engineer_car_age_feature(df), target_column="selling_price", exclude_columns=["year", "name", "fuel", "seller_type", "transmission", "owner"]
    )
    split_a = ml.train_test_split_data(X, y, random_state=1)
    split_b = ml.train_test_split_data(X, y, random_state=1)
    assert list(split_a[0].index) == list(split_b[0].index)


def test_basic_cross_validation_returns_scores_and_summary():
    from sklearn.linear_model import LinearRegression

    df, _ = eda.clean_car_data(generate_car_data())
    df = ml.engineer_car_age_feature(df)
    X, y = ml.select_feature_subset(
        df, target_column="selling_price", exclude_columns=["year", "name", "fuel", "seller_type", "transmission", "owner"]
    )

    result = ml.basic_cross_validation(LinearRegression(), X, y, cv=5)
    assert len(result["scores"]) == 5
    assert result["mean"] == result["scores"].mean()


def test_prepare_car_data_for_modeling_end_to_end():
    data = ml.prepare_car_data_for_modeling()

    assert set(data["X_train"].columns) == set(data["feature_columns"])
    assert len(data["X_train"]) + len(data["X_test"]) > 0
    assert data["X_train"].isnull().sum().sum() == 0  # cleaning actually happened
    assert "fuel" in data["encoders"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
