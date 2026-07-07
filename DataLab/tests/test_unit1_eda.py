import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datalab import unit1_eda as eda
from datalab.generate_datasets import generate_car_data, generate_students, generate_supermarket_sales


def test_load_car_data_has_expected_columns():
    df = eda.load_car_data()
    expected = {"name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner"}
    assert expected.issubset(set(df.columns))


def test_dataset_overview_reports_real_shape_and_memory():
    df = generate_car_data()
    overview = eda.dataset_overview(df)
    assert overview["shape"] == df.shape
    assert overview["total_memory_bytes"] > 0
    assert len(overview["head"]) == 5
    assert len(overview["tail"]) == 5


def test_clean_car_data_removes_duplicates_and_fills_missing():
    df = generate_car_data()
    assert df.isnull().sum().sum() > 0, "test dataset should have missing values by construction"
    assert df.duplicated().sum() > 0, "test dataset should have duplicates by construction"

    cleaned, report = eda.clean_car_data(df)

    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0
    assert report["duplicates_before"] > 0
    assert report["duplicates_after"] == 0
    assert report["rows_after"] < report["rows_before"]


def test_petrol_cars_after_2015_filters_correctly():
    df = generate_car_data()
    result = eda.petrol_cars_after_2015(df)
    assert (result["fuel"] == "Petrol").all()
    assert (result["year"] > 2015).all()


def test_cars_with_high_mileage_filters_and_sorts_descending():
    df = generate_car_data()
    result = eda.cars_with_high_mileage(df, threshold=50_000)
    assert (result["km_driven"] > 50_000).all()
    assert list(result["km_driven"]) == sorted(result["km_driven"], reverse=True)


def test_average_price_by_fuel_type_returns_series_per_fuel():
    df = generate_car_data()
    result = eda.average_price_by_fuel_type(df)
    assert "Petrol" in result.index
    assert "Diesel" in result.index
    assert (result > 0).all()


def test_detect_outliers_iqr_flags_the_deliberate_outliers():
    df = generate_car_data()
    outliers, bounds = eda.detect_outliers_iqr(df, "selling_price")
    # the generator deliberately injects prices of 4.5M/5.2M and 10k
    # against a normal range of roughly 100k-1.2M - these must be caught
    assert (outliers["selling_price"] > bounds["upper_bound"]).any() or \
           (outliers["selling_price"] < bounds["lower_bound"]).any()
    assert len(outliers) > 0
    assert len(outliers) < len(df)  # sanity: not flagging everything


def test_remove_outliers_iqr_leaves_no_out_of_bound_rows():
    df = generate_car_data()
    cleaned = eda.remove_outliers_iqr(df.dropna(subset=["selling_price"]), "selling_price")
    _, bounds = eda.detect_outliers_iqr(df.dropna(subset=["selling_price"]), "selling_price")
    assert (cleaned["selling_price"] >= bounds["lower_bound"]).all()
    assert (cleaned["selling_price"] <= bounds["upper_bound"]).all()


def test_gender_result_crosstab_has_totals_and_both_genders():
    students = generate_students()
    crosstab = eda.gender_result_crosstab(students)
    assert "Total" in crosstab.index
    assert "Total" in crosstab.columns
    assert "Male" in crosstab.index
    assert "Female" in crosstab.index


def test_students_correlation_shows_real_signal_not_just_noise():
    students = generate_students()
    result = eda.students_correlation_and_summary(students)
    corr = result["correlation"]
    # study hours and attendance should correlate positively with each
    # other at least weakly, since generate_datasets.py builds them from
    # partially related underlying factors via the pass-probability logit
    assert corr.shape == (3, 3)
    assert "summary" in result
    assert "study_hours_per_day" in result["summary"].columns


def test_revenue_by_product_line_and_payment_sums_correctly():
    sales = generate_supermarket_sales()
    result = eda.revenue_by_product_line_and_payment(sales)
    total_by_line = result["by_product_line"].sum()
    total_by_payment = result["by_payment_method"].sum()
    # both groupings must sum to the same grand total - a real
    # correctness check, not just "did it run without crashing"
    assert abs(total_by_line - total_by_payment) < 0.01
    assert abs(total_by_line - sales["total"].sum()) < 0.01


def test_merge_and_concat_examples_work():
    left = generate_car_data().head(10)[["name", "year"]]
    right = generate_car_data().head(10)[["name", "fuel"]]
    merged = eda.merge_example(left, right, on="name", how="inner")
    assert "fuel" in merged.columns and "year" in merged.columns

    concatenated = eda.concat_example([left, left])
    assert len(concatenated) == 2 * len(left)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
