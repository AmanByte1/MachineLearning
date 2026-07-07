# DataLab

Companion project to [ByteFlow](../ByteFlow), covering Units 1-7 of the
SEM IV FCSP-2 syllabus (Pandas/EDA, Visualization, ML intro, Regression,
Classification, Deep Learning, Web Scraping & APIs) - the data-science
half of the course (~63% by hour-weighting). See `PLAN.md` for the full
unit-by-unit roadmap and status.

Kept as a separate project from ByteFlow on purpose: this needs pandas,
numpy, scikit-learn, matplotlib/seaborn, plotly, networkx, tensorflow,
requests, and beautifulsoup4 - a genuinely heavy stack that shouldn't be
forced onto every ByteFlow install. Connects to ByteFlow as a Plugin
(ByteFlow already has this exact extension mechanism - see ByteFlow's
`math_plugin.py` for the existing pattern) once enough of this is built
to be worth exposing as tools.

## Status: Unit 1 complete, Units 2-7 planned (see PLAN.md)

## Setup

```bash
pip install pandas numpy
python -m datalab.generate_datasets   # generates the 3 practical datasets
python -m pytest tests/ -v
```

## What's real vs honest limitations

- The three datasets (`car_data.csv`, `students.csv`,
  `supermarket_sales.csv`) are **synthetically generated**, not the
  actual course files - say so in any submission (see PLAN.md).
- The outlier detection in Unit 1 uses the standard IQR method. On the
  car price data (which is right-skewed), the computed lower bound goes
  negative, so a deliberately-injected implausibly-*low* price doesn't
  get flagged - only the high ones do. That's a real, known limitation
  of IQR on skewed distributions, not a bug - worth understanding if
  asked about it rather than just noticing the "missing" outlier.
