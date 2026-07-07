import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datalab import unit1_eda as eda
from datalab import unit2_visualization as viz
from datalab.generate_datasets import generate_car_data


def _cleaned_car_df():
    df = generate_car_data()
    cleaned, _ = eda.clean_car_data(df)
    return cleaned


def test_seaborn_boxplot_returns_a_real_figure():
    fig = viz.seaborn_boxplot(_cleaned_car_df(), x="fuel", y="selling_price")
    assert fig is not None
    assert len(fig.axes) == 1


def test_seaborn_scatterplot_returns_a_real_figure():
    fig = viz.seaborn_scatterplot(_cleaned_car_df(), x="year", y="selling_price", hue="fuel")
    assert len(fig.axes) == 1


def test_seaborn_correlation_heatmap_returns_a_real_figure():
    df = _cleaned_car_df()
    corr = eda.numeric_correlation_matrix(df, columns=["year", "selling_price", "km_driven"])
    fig = viz.seaborn_correlation_heatmap(corr)
    assert len(fig.axes) == 2  # the heatmap axis + its colorbar axis


def test_interpret_correlation_finds_the_real_signal():
    df = _cleaned_car_df()
    corr = eda.numeric_correlation_matrix(df, columns=["year", "selling_price", "km_driven"])
    findings = viz.interpret_correlation(corr, threshold=0.3)
    # by construction (see generate_datasets.py) price rises with year
    # and falls with km_driven - both should be picked up as findings
    joined = " | ".join(findings)
    assert "year and selling_price" in joined
    assert "year and km_driven" in joined


def test_interpret_correlation_handles_no_strong_signal():
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(0)
    noise_df = pd.DataFrame({"a": rng.random(100), "b": rng.random(100)})
    findings = viz.interpret_correlation(noise_df.corr(), threshold=0.99)
    assert len(findings) == 1
    assert "No pair" in findings[0]


def test_generate_practical5_outputs_saves_real_png_files(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "_OUTPUT_DIR", str(tmp_path))
    result = viz.generate_practical5_outputs(_cleaned_car_df(), save=True)

    for filename in ("practical5_boxplot.png", "practical5_scatterplot.png", "practical5_heatmap.png"):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 1000  # a real image, not an empty/corrupt file

    assert "correlation_matrix" in result
    assert len(result["interpretation"]) >= 1


def test_plotly_price_vs_year_returns_a_real_figure_with_data():
    fig = viz.plotly_price_vs_year(_cleaned_car_df())
    assert len(fig.data) > 0  # at least one trace was actually plotted
    assert fig.data[0].x is not None and len(fig.data[0].x) > 0


def test_build_car_dashboard_compiles_with_expected_components():
    app = viz.build_car_dashboard(_cleaned_car_df())
    assert app.layout is not None
    # dropdown + graph must both be present for the filter-and-view flow
    layout_str = str(app.layout)
    assert "fuel-filter" in layout_str
    assert "price-year-graph" in layout_str


def test_build_relationship_graph_and_degrees():
    edges = [("Ahmedabad", "Surat"), ("Ahmedabad", "Vadodara"), ("Surat", "Vadodara"), ("Surat", "Rajkot")]
    graph = viz.build_relationship_graph(edges)
    degrees = viz.node_degrees(graph)

    assert degrees["Surat"] == 3  # connected to Ahmedabad, Vadodara, Rajkot
    assert degrees["Rajkot"] == 1  # only connected to Surat
    assert set(graph.nodes()) == {"Ahmedabad", "Surat", "Vadodara", "Rajkot"}


def test_draw_relationship_graph_returns_a_real_figure():
    edges = [("A", "B"), ("B", "C")]
    graph = viz.build_relationship_graph(edges)
    fig = viz.draw_relationship_graph(graph)
    assert len(fig.axes) == 1


def test_generate_practical7_outputs_saves_real_png(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "_OUTPUT_DIR", str(tmp_path))
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    result = viz.generate_practical7_outputs(edges, save=True)

    path = tmp_path / "practical7_graph.png"
    assert path.exists()
    assert path.stat().st_size > 1000
    assert result["degrees"] == {"A": 2, "B": 2, "C": 2}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
