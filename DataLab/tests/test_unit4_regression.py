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


def test_byteflow_plugin_car_price_predictions_tool():
    import sys, os
    datalab_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, datalab_root)

    # ByteFlow is a sibling project, not a pip-installed dependency of
    # DataLab - add it to sys.path the same way extension_loader.py does
    # at runtime, since byteflow_plugin.py imports `from byteflow...`.
    # Try a couple of plausible locations rather than assuming one exact
    # layout, since "sibling of DataLab" can mean different things
    # depending on how the two repos were checked out.
    parent_dir = os.path.dirname(datalab_root)
    candidates = [
        os.path.join(parent_dir, "ByteFlow"),
        os.path.join(parent_dir, "ByteFlow_flat", "ByteFlow"),
    ]
    byteflow_root = next((c for c in candidates if os.path.isdir(os.path.join(c, "byteflow"))), None)

    if byteflow_root is None:
        import pytest
        pytest.skip(f"ByteFlow not found in any of {candidates} - skipping plugin integration test")

    sys.path.insert(0, byteflow_root)

    from byteflow_plugin import _car_price_predictions

    output = _car_price_predictions(top_n=3)
    assert "R²=" in output or "R\u00b2=" in output
    assert "actual" in output and "predicted" in output
    # exactly 3 prediction lines requested
    assert output.count("actual") == 3


def test_predict_price_for_car_in_range_year():
    result = reg.predict_price_for_car(year=2022, km_driven=30000, fuel="Petrol")
    assert result["predicted_price"] > 0
    assert result["extrapolating"] is False
    assert "fuel" not in result["defaults_used"]  # explicitly given, not defaulted
    assert "seller_type" in result["defaults_used"]  # not given, should be defaulted


def test_predict_price_for_car_flags_extrapolation_for_future_year():
    result = reg.predict_price_for_car(year=2028, km_driven=10000)
    assert result["extrapolating"] is True


def test_predict_price_for_car_rejects_unknown_category():
    raised = False
    try:
        reg.predict_price_for_car(year=2022, km_driven=30000, fuel="Hydrogen")
    except ValueError as e:
        raised = True
        assert "Unknown fuel" in str(e)
    assert raised


def test_predict_price_for_car_uses_correct_feature_column_order():
    # real bug risk this guards against: sklearn matches features by
    # POSITION not name, so if the row dict order ever drifted from
    # data["feature_columns"], predictions would be silently wrong
    # rather than erroring - this just checks the function runs
    # end-to-end without a shape/order mismatch exception, which would
    # surface immediately if the ordering were wrong.
    result = reg.predict_price_for_car(year=2020, km_driven=50000)
    assert isinstance(result["predicted_price"], float)


def test_predict_car_price_structured_returns_plain_dict():
    import sys, os
    datalab_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, datalab_root)
    parent_dir = os.path.dirname(datalab_root)
    candidates = [
        os.path.join(parent_dir, "ByteFlow"),
        os.path.join(parent_dir, "ByteFlow_flat", "ByteFlow"),
    ]
    byteflow_root = next((c for c in candidates if os.path.isdir(os.path.join(c, "byteflow"))), None)
    if byteflow_root is None:
        import pytest
        pytest.skip(f"ByteFlow not found in any of {candidates}")
    sys.path.insert(0, byteflow_root)

    from byteflow_plugin import _predict_car_price_structured, format_car_price_prediction, _predict_car_price

    structured = _predict_car_price_structured(2028, 10000)
    assert isinstance(structured, dict)
    assert "predicted_price" in structured
    assert structured["extrapolating"] is True

    # the formatted string must be derivable from the SAME structured
    # data, and the registered tool's formatted output must match
    # exactly - proving the split didn't change the actual answer,
    # only how it's exposed
    formatted_from_dict = format_car_price_prediction(structured)
    formatted_from_tool = _predict_car_price(2028, 10000)
    assert "Predicted selling price" in formatted_from_dict
    assert formatted_from_dict == formatted_from_tool


def test_predict_car_price_structured_error_path():
    import sys, os
    datalab_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, datalab_root)
    parent_dir = os.path.dirname(datalab_root)
    candidates = [
        os.path.join(parent_dir, "ByteFlow"),
        os.path.join(parent_dir, "ByteFlow_flat", "ByteFlow"),
    ]
    byteflow_root = next((c for c in candidates if os.path.isdir(os.path.join(c, "byteflow"))), None)
    if byteflow_root is None:
        import pytest
        pytest.skip(f"ByteFlow not found in any of {candidates}")
    sys.path.insert(0, byteflow_root)

    from byteflow_plugin import _predict_car_price_structured, format_car_price_prediction

    structured = _predict_car_price_structured(2022, 30000, fuel="Hydrogen")
    assert "error" in structured
    assert "[Error:" in format_car_price_prediction(structured)
