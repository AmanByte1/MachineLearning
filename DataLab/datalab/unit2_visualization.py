"""
Unit 2: Data Visualization with Python.

Covers syllabus topics 2.1-2.4 (Seaborn, Plotly, Dash, NetworkX) and
implements practicals 5-7:

  Practical 5: Seaborn boxplot/scatterplot/heatmap + interpret correlation.
  Practical 6: Plotly interactive selling-price-vs-year chart, plus a
               simple Dash dashboard filtering records by fuel type.
  Practical 7: NetworkX graph of relationships (cities/students), with
               degree of each node.

Every plotting function returns the actual figure object (matplotlib
Figure / plotly Figure / networkx Graph / Dash App) rather than calling
.show() - so callers (including tests) can save, inspect, or embed the
result instead of it only working in an interactive session. Uses the
non-interactive "Agg" matplotlib backend so this works headlessly (no
display needed), which is what makes it possible to test at all.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend - must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from . import unit1_eda as eda

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_HERE, "outputs")


def _ensure_output_dir():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


# ---------------------------------------------------------------------
# 2.1 - Seaborn: box plots, scatter plots, heatmaps  (Practical 5)
# ---------------------------------------------------------------------

def seaborn_boxplot(df, x, y, title=None):
    """Practical 5: boxplot of `y` grouped by `x` (e.g. selling_price by fuel)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title or f"{y} by {x}")
    fig.tight_layout()
    return fig


def seaborn_scatterplot(df, x, y, hue=None, title=None):
    """Practical 5: scatterplot of `x` vs `y`, optionally colored by `hue`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title or f"{y} vs {x}")
    fig.tight_layout()
    return fig


def seaborn_correlation_heatmap(corr_matrix, title="Correlation Heatmap"):
    """Practical 5: heatmap of a correlation matrix (see
    unit1_eda.numeric_correlation_matrix() for building the input)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def interpret_correlation(corr_matrix, threshold=0.3):
    """
    Practical 5's "interpret the correlation results" step - turns the
    numeric matrix into a plain-English list of the notable
    relationships (|correlation| >= threshold, excluding the diagonal),
    so "interpret" produces an actual interpretation, not just a plot.
    """
    findings = []
    columns = corr_matrix.columns
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if abs(value) >= threshold:
                direction = "positive" if value > 0 else "negative"
                strength = "strong" if abs(value) >= 0.7 else "moderate"
                findings.append(
                    f"{col_a} and {col_b}: {strength} {direction} correlation ({value:.2f})"
                )
    if not findings:
        return [f"No pair of variables shows a correlation of |r| >= {threshold}."]
    return findings


def generate_practical5_outputs(car_df, save=True):
    """
    Run all of practical 5 in one call: boxplot (price by fuel),
    scatterplot (price vs year), heatmap (numeric correlation), plus
    the plain-English interpretation. Returns a dict with every figure
    and the interpretation text; saves PNGs if save=True.
    """
    corr = eda.numeric_correlation_matrix(car_df, columns=["year", "selling_price", "km_driven"])

    box_fig = seaborn_boxplot(car_df, x="fuel", y="selling_price", title="Selling Price by Fuel Type")
    scatter_fig = seaborn_scatterplot(car_df, x="year", y="selling_price", hue="fuel",
                                       title="Selling Price vs Year")
    heatmap_fig = seaborn_correlation_heatmap(corr)
    interpretation = interpret_correlation(corr)

    if save:
        out_dir = _ensure_output_dir()
        box_fig.savefig(os.path.join(out_dir, "practical5_boxplot.png"), dpi=100)
        scatter_fig.savefig(os.path.join(out_dir, "practical5_scatterplot.png"), dpi=100)
        heatmap_fig.savefig(os.path.join(out_dir, "practical5_heatmap.png"), dpi=100)

    return {
        "boxplot": box_fig, "scatterplot": scatter_fig, "heatmap": heatmap_fig,
        "correlation_matrix": corr, "interpretation": interpretation,
    }


# ---------------------------------------------------------------------
# 2.2 - Interactive visualization using Plotly  (Practical 6, part 1)
# ---------------------------------------------------------------------

def plotly_price_vs_year(car_df):
    """Practical 6: interactive scatter of selling price vs year, colored
    by fuel type - hovering shows the exact values, which is the whole
    point of "interactive" over a static matplotlib plot."""
    fig = px.scatter(
        car_df, x="year", y="selling_price", color="fuel",
        hover_data=["name", "km_driven"],
        title="Selling Price vs Year (interactive)",
    )
    return fig


# ---------------------------------------------------------------------
# 2.3 - Introduction to Dash  (Practical 6, part 2)
# ---------------------------------------------------------------------

def build_car_dashboard(car_df):
    """
    Practical 6: a simple Dash dashboard letting the user filter
    records by fuel type and see the selling-price-vs-year scatter
    update live. Returns the Dash App object - call app.run(debug=True)
    to actually serve it locally; kept unrun here so this is importable/
    testable without starting a web server as a side effect of import.
    """
    app = Dash(__name__)
    fuel_options = [{"label": f, "value": f} for f in sorted(car_df["fuel"].dropna().unique())]

    app.layout = html.Div([
        html.H2("Car Data Dashboard"),
        html.Label("Filter by fuel type:"),
        dcc.Dropdown(
            id="fuel-filter",
            options=fuel_options,
            value=[opt["value"] for opt in fuel_options],
            multi=True,
        ),
        dcc.Graph(id="price-year-graph"),
    ])

    @app.callback(Output("price-year-graph", "figure"), Input("fuel-filter", "value"))
    def _update_graph(selected_fuels):
        filtered = car_df[car_df["fuel"].isin(selected_fuels)] if selected_fuels else car_df
        return px.scatter(filtered, x="year", y="selling_price", color="fuel",
                           title="Selling Price vs Year (filtered)")

    return app


# ---------------------------------------------------------------------
# 2.4 - Graph visualization using NetworkX  (Practical 7)
# ---------------------------------------------------------------------

def build_relationship_graph(edges):
    """
    Practical 7: build a graph from a list of (node_a, node_b) pairs
    representing a relationship (e.g. cities connected by a travel
    route, or students who are classmates/friends).
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def node_degrees(graph):
    """Practical 7: degree of each node - dict of {node: degree}."""
    return dict(graph.degree())


def draw_relationship_graph(graph, title="Relationship Graph"):
    """Render the graph to a matplotlib figure (spring layout - the
    standard general-purpose layout for a graph with no inherent
    geometry, like city names or student names)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)  # seeded for reproducible layout
    nx.draw(
        graph, pos, ax=ax, with_labels=True, node_color="lightblue",
        node_size=1200, font_size=9, font_weight="bold", edge_color="gray",
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def generate_practical7_outputs(edges, save=True):
    """Run practical 7 in one call: build the graph, compute degrees,
    render it. Returns dict with graph, degrees, and figure."""
    graph = build_relationship_graph(edges)
    degrees = node_degrees(graph)
    fig = draw_relationship_graph(graph)

    if save:
        out_dir = _ensure_output_dir()
        fig.savefig(os.path.join(out_dir, "practical7_graph.png"), dpi=100)

    return {"graph": graph, "degrees": degrees, "figure": fig}
