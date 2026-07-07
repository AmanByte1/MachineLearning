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


def get_plugin():
    return DataLabPlugin("datalab")
