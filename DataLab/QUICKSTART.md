# QUICKSTART - run this yourself and check the output

## 1. Setup

```bash
cd DataLab
pip install -r requirements.txt
python -m datalab.generate_datasets
```

Expected output:
```
car_data.csv: 405 rows, 8 columns
students.csv: 250 rows, 6 columns
supermarket_sales.csv: 500 rows, 9 columns
```

## 2. Run the full test suite (fastest way to check everything works)

```bash
python -m pytest tests/ -v
```

Expected: `51 passed` (2 harmless deprecation warnings from seaborn are fine).
If anything fails, that's a real problem to look at - don't ignore it.

## 3. Try each unit yourself, and compare against the output shown here

### Unit 1 - Pandas & EDA

```python
from datalab import unit1_eda as eda

df = eda.load_car_data()
cleaned, report = eda.clean_car_data(df)
print(report)
# expect: duplicates_before > 0, duplicates_after == 0

print(eda.average_price_by_fuel_type(cleaned).round(0))
# expect: a price per fuel type (Petrol/Diesel/CNG/Electric), no errors

outliers, bounds = eda.detect_outliers_iqr(cleaned, "selling_price")
print(len(outliers), "outliers found")
# expect: a small number (2-3), not 0 and not most of the dataset
```

### Unit 2 - Visualization

```python
from datalab import unit2_visualization as viz

result = viz.generate_practical5_outputs(cleaned)
print(result["interpretation"])
# expect: real sentences like "year and selling_price: strong positive correlation (0.71)"
```
Check `datalab/outputs/practical5_*.png` afterward - open them, they should be real,
readable charts (not blank/corrupt images).

### Unit 3 - ML preprocessing

```python
from datalab import unit3_ml_intro as ml

data = ml.prepare_car_data_for_modeling()
print(data["X_train"].shape, data["X_test"].shape)
# expect: roughly 80/20 split of however many rows survived cleaning
```

### Unit 4 - Regression

```python
from datalab import unit4_regression as reg

p9 = reg.run_practical9(data)
print(p9["multiple_linear"]["metrics"])
# expect: r2 roughly 0.4-0.5, mae/mse are positive numbers (not 0, not NaN)
```

### Unit 5 - Classification

```python
from datalab import unit5_classification as clf

student_data = clf.prepare_students_for_classification()
p13 = clf.run_practical13(student_data)
print(p13["accuracy_comparison"])
# expect: {'random_forest': ~0.75-0.80, 'svm': ~0.75-0.80}
```

If your numbers are close to what's shown above (not identical - only
the seeded parts like the dataset generation are exactly reproducible;
some model internals can vary slightly by scikit-learn version), it's
working correctly.

## 4. Try it through ByteFlow directly (the "connected" experience)

```bash
cd ../ByteFlow
python -m byteflow.cli run "give me an overview of the car data" --extension-path ../DataLab
```

Expected: real shape/dtypes/memory output from the actual car_data.csv,
via the DataLab connector - proves the whole pipeline (ByteFlow ->
extension loader -> DataLab -> pandas) is actually wired together, not
just two separate projects sitting next to each other.
