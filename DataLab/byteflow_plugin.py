"""
ByteFlow extension entry point for DataLab (see the sibling DataLab
project's PLAN.md for what it covers - currently Unit 1: Pandas & EDA).

Kept intentionally modest: only wires up what's actually built and
tested in DataLab right now (Unit 1's car-data cleaning/EDA
functions), rather than pretending all 7 units are connected. As more
DataLab units get built, more tools get added here - see DataLab's
PLAN.md for what's next.

Requires DataLab's own dependencies (pandas, numpy) - if they're
missing, or DataLab can't be found at all, this reports a clear error
via the extension_loader's failure handling rather than crashing
ByteFlow (see byteflow/extension_loader.py's module docstring).
"""

from byteflow.plugin import Plugin
from byteflow.tools import Tool


def _car_data_overview():
    """Load car_data.csv and summarize it: shape, dtypes, memory
    usage, and summary statistics - DataLab's dataset_overview() via
    unit1_eda.py, formatted as readable text for a chat reply."""
    from datalab import unit1_eda as eda

    df = eda.load_car_data()
    overview = eda.dataset_overview(df)
    lines = [
        f"Shape: {overview['shape'][0]} rows x {overview['shape'][1]} columns",
        f"Total memory usage: {overview['total_memory_bytes']:,} bytes",
        "",
        "Column types:",
        overview["dtypes"].to_string(),
    ]
    return "\n".join(lines)


def _car_data_clean_report():
    """Clean car_data.csv (missing values + duplicates) and report
    exactly what changed - DataLab's clean_car_data() via unit1_eda.py."""
    from datalab import unit1_eda as eda

    df = eda.load_car_data()
    _, report = eda.clean_car_data(df)
    lines = [
        f"Rows before cleaning: {report['rows_before']}",
        f"Rows after cleaning: {report['rows_after']}",
        f"Duplicate rows removed: {report['duplicates_before']}",
        f"Missing values before: {report['missing_before']}",
        f"Missing values after: {report['missing_after']}",
    ]
    return "\n".join(lines)


def _car_data_outliers(column="selling_price"):
    """Detect IQR-based outliers in a numeric column of car_data.csv -
    DataLab's detect_outliers_iqr() via unit1_eda.py."""
    from datalab import unit1_eda as eda

    df = eda.load_car_data()
    outliers, bounds = eda.detect_outliers_iqr(df.dropna(subset=[column]), column)
    if outliers.empty:
        return f"No outliers found in '{column}' (bounds: {bounds['lower_bound']:.0f} to {bounds['upper_bound']:.0f})."
    lines = [f"{len(outliers)} outlier(s) found in '{column}' "
              f"(bounds: {bounds['lower_bound']:.0f} to {bounds['upper_bound']:.0f}):"]
    for _, row in outliers.iterrows():
        lines.append(f"  {row.get('name', '?')}: {column}={row[column]}")
    return "\n".join(lines)


def _car_price_predictions(top_n=10):
    """
    Trains a real regression model (DataLab's Unit 4 multiple linear
    regression, via unit3_ml_intro's preprocessing pipeline) on
    car_data.csv, and reports actual vs predicted selling price for a
    sample of cars from the held-out test set - a real prediction, not
    a canned/fake answer, using the model's genuine test-set accuracy
    (see the metrics line) so the person can judge how much to trust it.
    """
    from datalab import unit3_ml_intro as ml_intro
    from datalab import unit4_regression as reg

    data = ml_intro.prepare_car_data_for_modeling()
    model = reg.train_multiple_linear_regression(data["X_train"], data["y_train"])
    predictions = model.predict(data["X_test"])
    metrics = reg.evaluate_regression(data["y_test"], predictions)

    lines = [
        f"Multiple linear regression trained on car_data.csv "
        f"(R\u00b2={metrics['r2']:.2f}, avg error \u2248 \u20b9{metrics['mae']:,.0f} - "
        f"treat predictions with that margin of error in mind):",
        "",
    ]
    n = min(top_n, len(data["X_test"]))
    for i in range(n):
        actual = data["y_test"].iloc[i]
        predicted = predictions[i]
        lines.append(f"  actual \u20b9{actual:,.0f}  ->  predicted \u20b9{predicted:,.0f}")

    return "\n".join(lines)


def _predict_car_price_structured(year, km_driven, fuel=None):
    """
    The deterministic core: returns a plain dict, never a formatted
    string and never an LLM call - {"error": "..."} on failure, or
    {"predicted_price": ..., "model_r2": ..., "model_mae": ...,
    "extrapolating": ..., "defaults_used": ..., "year": ..., "km_driven": ...}
    on success.

    Kept separate from _predict_car_price() (the formatted-string
    version registered as the actual Tool) so structured data is
    available to anything that wants to REASON about a prediction -
    compare two predictions, check confidence before deciding whether
    to trust it, extract just the number - without needing to re-parse
    formatted English text back into numbers. This is what lets the
    LLM be used only for explaining/interpreting a result when that's
    actually useful, rather than being the only way to get the number
    out of the tool at all.
    """
    try:
        year = int(year)
        km_driven = float(km_driven)
    except (TypeError, ValueError):
        return {"error": f"year and km_driven must be numbers - got year={year!r}, km_driven={km_driven!r}"}

    from datalab import unit4_regression as reg
    try:
        result = reg.predict_price_for_car(year=year, km_driven=km_driven, fuel=fuel)
    except ValueError as e:
        return {"error": str(e)}

    result["year"] = year
    result["km_driven"] = km_driven
    return result


def format_car_price_prediction(result):
    """
    Deterministic, no-LLM formatting of a structured prediction result
    (see _predict_car_price_structured()) into readable text. Pure
    string formatting - given the same dict, always produces the same
    text, so this never needs to go through the LLM at all for the
    common case of just wanting to see the answer.
    """
    if "error" in result:
        return f"[Error: {result['error']}]"

    lines = [
        f"Predicted selling price: \u20b9{result['predicted_price']:,.0f}",
        f"(model accuracy on real held-out data: R\u00b2={result['model_r2']:.2f}, "
        f"avg error \u2248 \u20b9{result['model_mae']:,.0f} - treat this prediction with "
        f"that margin of error in mind)",
    ]
    if result["defaults_used"]:
        defaults_str = ", ".join(f"{k}={v}" for k, v in result["defaults_used"].items())
        lines.append(f"(assumed typical values for unspecified details: {defaults_str})")
    if result["extrapolating"]:
        lines.append(
            f"\u26a0 Year {result['year']} is outside the range of years in the training data - "
            f"this prediction is an EXTRAPOLATION and is less reliable than a "
            f"prediction for a year the model has real examples of."
        )
    return "\n".join(lines)


def _predict_car_price(year, km_driven, fuel=None):
    """
    Predicts selling price for a HYPOTHETICAL car with the given year
    and km_driven (fuel optional) - not an existing row in the
    dataset, a genuine new prediction from the trained regression
    model. Reports the model's real accuracy and flags if the given
    year is outside what the model was actually trained on, so the
    answer is honest about its own reliability rather than presenting
    every prediction with the same false confidence.

    Returns readable formatted text (the deterministic prediction +
    formatting, no LLM involved in either step) - see
    _predict_car_price_structured() for the same prediction as a plain
    dict, if you need the raw numbers instead of formatted text.
    """
    result = _predict_car_price_structured(year, km_driven, fuel=fuel)
    return format_car_price_prediction(result)


class DataLabPlugin(Plugin):
    def setup(self, agent):
        agent.register_tool(Tool(
            "car_data_overview",
            _car_data_overview,
            "summarizes the car_data.csv dataset: shape, dtypes, memory usage",
            example='"give me an overview of the car data" -> car_data_overview()',
        ))
        agent.register_tool(Tool(
            "predict_car_price_raw",
            _predict_car_price_structured,
            "same as predict_car_price, but returns the raw structured numbers as a dict "
            "(predicted_price, model_r2, model_mae, extrapolating, defaults_used) instead of "
            "formatted text - use this when the prediction needs to be reasoned about "
            "or compared, not just displayed",
            example='"give me the raw prediction numbers for a 2028 car with 10000km" -> predict_car_price_raw(2028, 10000)',
        ))
        agent.register_tool(Tool(
            "car_data_clean_report",
            _car_data_clean_report,
            "cleans car_data.csv (missing values, duplicates) and reports what changed",
            example='"clean the car data and tell me what changed" -> car_data_clean_report()',
        ))
        agent.register_tool(Tool(
            "car_data_outliers",
            _car_data_outliers,
            "detects statistical outliers in a numeric column of car_data.csv (default: selling_price)",
            example='"find outliers in the car price data" -> car_data_outliers("selling_price")',
        ))
        agent.register_tool(Tool(
            "car_price_predictions",
            _car_price_predictions,
            "trains a real regression model on car_data.csv and shows actual vs predicted "
            "selling price for a sample of EXISTING cars in the dataset, with the model's "
            "real accuracy (R2, MAE); takes one optional argument, how many cars to show (default 10)",
            example='"show me price predictions for some cars in the data" -> car_price_predictions(10)',
        ))
        agent.register_tool(Tool(
            "predict_car_price",
            _predict_car_price,
            "predicts the selling price for a NEW hypothetical car (not one already in the "
            "dataset) given its year and km_driven, and optionally its fuel type",
            # Two differently-phrased examples, not one - a real observed
            # gap: the planner reliably handled "predict car price 2028
            # which runs 10000km" but failed (and hallucinated a fake
            # error) on "predict the price of a 2022 petrol car with
            # 30000 km", a simple rephrasing of the exact same request.
            # More varied examples give a small local model more surface
            # area to pattern-match against.
            example=(
                '"predict car price 2028 which runs 10000km" -> predict_car_price(2028, 10000)\n'
                '    "predict the price of a 2022 petrol car with 30000 km" -> '
                'predict_car_price(2022, 30000, "Petrol")'
            ),
        ))


def get_plugin():
    return DataLabPlugin("datalab")
