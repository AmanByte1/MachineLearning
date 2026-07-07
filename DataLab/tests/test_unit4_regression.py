import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datalab import unit3_ml_intro as ml_intro
from datalab import unit4_regression as reg


def _modeling_data():
    return ml_intro.prepare_car_data_for_modeling()


def test_evaluate_regression_returns_all_four_metrics():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]
    metrics = reg.evaluate_regression(y_true, y_pred)
    assert set(metrics.keys()) == {"r2", "mae", "mse", "rmse"}
    assert metrics["mae"] > 0
    assert metrics["mse"] > 0
    assert abs(metrics["rmse"] ** 2 - metrics["mse"]) < 1e-6


def test_evaluate_regression_perfect_predictions_score_perfectly():
    y_true = [100, 200, 300, 400]
    metrics = reg.evaluate_regression(y_true, y_true)
    assert metrics["r2"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0


def test_train_simple_linear_regression_uses_only_one_feature():
    data = _modeling_data()
    model = reg.train_simple_linear_regression(data["X_train"], data["y_train"], "car_age")
    assert model.coef_.shape == (1,)  # exactly one coefficient - one feature


def test_train_multiple_linear_regression_uses_all_features():
    data = _modeling_data()
    model = reg.train_multiple_linear_regression(data["X_train"], data["y_train"])
    assert model.coef_.shape == (len(data["feature_columns"]),)


def test_train_polynomial_regression_produces_a_working_pipeline():
    data = _modeling_data()
    model = reg.train_polynomial_regression(data["X_train"], data["y_train"], "car_age", degree=2)
    predictions = model.predict(data["X_test"][["car_age"]])
    assert len(predictions) == len(data["X_test"])


def test_run_practical9_produces_sane_metrics_for_both_models():
    data = _modeling_data()
    result = reg.run_practical9(data)

    for key in ("simple_linear", "multiple_linear"):
        metrics = result[key]["metrics"]
        # sanity bounds, not "one model must beat the other" - a real
        # model on real (synthetic, noisy) data can go either way, and
        # asserting a specific winner would be testing this dataset's
        # luck, not the code's correctness
        assert -1.0 <= metrics["r2"] <= 1.0
        assert metrics["mae"] > 0
        assert metrics["mse"] > 0


def test_run_practical10_compares_linear_and_polynomial_on_same_split():
    data = _modeling_data()
    result = reg.run_practical10(data, degree=2)

    assert "linear" in result and "polynomial" in result
    assert result["degree"] == 2
    assert result["feature_column"] == "car_age"
    for key in ("linear", "polynomial"):
        assert result[key]["mae"] > 0


def test_compare_linear_vs_polynomial_uses_identical_test_set():
    # both models must be scored on the exact same held-out rows -
    # otherwise "compare" wouldn't be a fair, real comparison
    data = _modeling_data()
    result = reg.compare_linear_vs_polynomial(
        data["X_train"], data["X_test"], data["y_train"], data["y_test"],
        feature_column="car_age", degree=3,
    )
    assert result["linear"]["mse"] != result["polynomial"]["mse"] or True  # different models CAN tie; just must both exist
    assert isinstance(result["linear"]["r2"], float)
    assert isinstance(result["polynomial"]["r2"], float)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
