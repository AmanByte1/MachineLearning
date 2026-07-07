# DataLab - Companion project for SEM IV FCSP-2 Syllabus

Standalone project covering syllabus Units 1-7 (Pandas/EDA, Visualization,
ML intro, Regression, Classification, Deep Learning, Web Scraping & APIs) -
the data-science half of the course, which is ~63% of it by hour-weighting.
Units 8-10 (Django backend) are intentionally NOT covered here - see the
main README for why.

Kept as a fully separate package from ByteFlow (heavy deps: pandas, numpy,
scikit-learn, matplotlib/seaborn, plotly, networkx, tensorflow, requests,
beautifulsoup4) so ByteFlow itself stays lightweight. Connects to ByteFlow
as a Plugin (see `byteflow_plugin.py`), the same extension mechanism
ByteFlow already has for exactly this purpose (see ByteFlow's
`math_plugin.py` for the existing precedent) - not a new integration
mechanism invented for this.

## Status

| Unit | Topic | Module | Status |
|---|---|---|---|
| 1 | Pandas & EDA | `unit1_eda.py` | ✅ Built, 12 tests passing |
| 2 | Data Visualization | `unit2_visualization.py` | ✅ Built, 11 tests passing |
| 3 | Intro to ML | `unit3_ml_intro.py` | ✅ Built, 8 tests passing |
| 4 | Regression | `unit4_regression.py` | ✅ Built, 8 tests passing |
| 5 | Classification | `unit5_classification.py` | ✅ Built, 12 tests passing |
| 6 | Deep Learning | `unit6_deep_learning.py` | ⏳ Next (needs TensorFlow - heavier install) |
| 7 | Web Scraping & APIs | `unit7_scraping_apis.py` | ⏳ Planned |
| - | ByteFlow connector | `byteflow_plugin.py` | ✅ Built (Unit 1 tools only so far - see below) |

**51 tests passing across Units 1-5.** Two real bugs were caught and
fixed while building, not just theoretical concerns:
  - IQR-based outlier detection doesn't catch the deliberately-injected
    LOW price outlier because the price distribution is right-skewed,
    pushing the lower bound negative - a real, explainable limitation
    of the method on skewed data, not a code bug.
  - kNN and SVM (distance-based) were initially trained on unscaled
    features, unfairly disadvantaging them against scale-invariant
    tree models. Fixed with `scale_features()` (StandardScaler) -
    kNN's real accuracy went from 66% to 80% once fixed, confirming
    this was a genuine bug, not a nitpick.
  - Also honestly worth knowing: multiple linear regression and
    polynomial regression both scored *slightly worse* than simple
    linear regression (car_age alone) on this dataset - not a bug,
    but a real result worth understanding if asked about it (label-
    encoded nominal categories imposing a false ordinal relationship,
    and the underlying price-vs-age relationship being genuinely
    close to linear so a curve doesn't help).

The ByteFlow connector (`byteflow_plugin.py`) currently only wires up
Unit 1's three functions - deliberately not overstating what's
connected. Extending it with Unit 2-5 tools (plots, model training,
classification) is straightforward now that the discovery mechanism
is proven (see ByteFlow's `extension_loader.py`), just needs the
`Tool(...)` registrations added.

## Datasets (`datalab/datasets/`)

The syllabus practicals reference `car_data.csv`, `students.csv`, and
`supermarket_sales.csv` without providing them. Since the real course
files weren't available in usable form, these are **synthetically
generated** (see `datalab/generate_datasets.py`) to match exactly what
each practical needs to demonstrate (realistic value ranges, some missing
values and duplicates on purpose so the cleaning practicals have real work
to do, a plausible fuel-type/price/year distribution for the outlier-
detection practical, etc). Generation is seeded for reproducibility.

**Be upfront about this in your submission**: these are synthetic
stand-ins for the real practical datasets, not the originals - fine for
learning/demonstrating the techniques, but say so rather than imply they
came from the instructor.

## Practical -> Module mapping

| Practical # | What it needs | Covered by |
|---|---|---|
| 1, 2 | car_data.csv cleaning + EDA questions | `unit1_eda.py` |
| 3 | students.csv cross-tab + correlation | `unit1_eda.py` |
| 4 | supermarket_sales.csv groupby/aggregation | `unit1_eda.py` |
| 5, 6, 7 | Seaborn/Plotly/Dash/NetworkX visuals | `unit2_visualization.py` |
| 8 | train/test split, encoding | `unit3_ml_intro.py` |
| 9, 10 | Linear/Polynomial Regression + metrics | `unit4_regression.py` |
| 11, 12, 13, 14 | kNN/DT/RF/SVM + confusion matrix | `unit5_classification.py` |
| 15, 16, 17 | Neural net, CNN, transfer learning | `unit6_deep_learning.py` |
| 18 | BeautifulSoup scraping | `unit7_scraping_apis.py` |
| 19, 20 | Django (NOT covered - see above) | - |
