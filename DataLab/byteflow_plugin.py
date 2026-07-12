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


class DataLabPlugin(Plugin):
    def setup(self, agent):
        agent.register_tool(Tool(
            "car_data_overview",
            _car_data_overview,
            "summarizes the car_data.csv dataset: shape, dtypes, memory usage",
        ))
        agent.register_tool(Tool(
            "car_data_clean_report",
            _car_data_clean_report,
            "cleans car_data.csv (missing values, duplicates) and reports what changed",
        ))
        agent.register_tool(Tool(
            "car_data_outliers",
            _car_data_outliers,
            "detects statistical outliers in a numeric column of car_data.csv (default: selling_price)",
        ))
        agent.register_tool(Tool(
            "car_price_predictions",
            _car_price_predictions,
            "trains a real regression model on car_data.csv and shows actual vs predicted "
            "selling price for a sample of cars, with the model's real accuracy (R2, MAE); "
            "takes one optional argument, how many cars to show (default 10)",
        ))


def get_plugin():
    return DataLabPlugin("datalab")
