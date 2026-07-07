"""
Generates the three datasets the syllabus practicals reference
(car_data.csv, students.csv, supermarket_sales.csv) but doesn't
provide. These are SYNTHETIC - built to have the right shape and the
right kinds of problems (missing values, duplicates, outliers) for the
practicals to be meaningful, not real course data. See PLAN.md for why.

Seeded (random_state=42 throughout) so re-running this produces
identical files - useful for grading/reproducibility, and so unit
tests that check specific values don't break on regeneration.

Run directly to (re)generate all three CSVs into datalab/datasets/:
    python -m datalab.generate_datasets
"""

import os
import numpy as np
import pandas as pd

_RNG_SEED = 42
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "datasets")


def generate_car_data(n=400, seed=_RNG_SEED):
    """
    Used car listings - name, year, selling price, kms driven, fuel
    type, seller type, transmission, owner count. Matches what
    practicals 1, 2, 8, 9 need: dtypes/shape/summary stats, missing
    values + duplicates to clean, a fuel-type/year/price-based EDA
    question set, and enough numeric spread for outlier detection and
    later regression practice.
    """
    rng = np.random.default_rng(seed)

    brands_models = [
        "Maruti Swift", "Maruti Baleno", "Hyundai i20", "Hyundai Creta",
        "Honda City", "Honda Amaze", "Toyota Innova", "Toyota Fortuner",
        "Tata Nexon", "Tata Tiago", "Mahindra XUV500", "Ford EcoSport",
        "Renault Kwid", "Volkswagen Polo", "Skoda Rapid",
    ]
    fuel_types = rng.choice(
        ["Petrol", "Diesel", "CNG", "Electric"], size=n, p=[0.55, 0.33, 0.10, 0.02]
    )
    seller_types = rng.choice(["Individual", "Dealer", "Trustmark Dealer"], size=n, p=[0.6, 0.3, 0.1])
    transmissions = rng.choice(["Manual", "Automatic"], size=n, p=[0.78, 0.22])
    owners = rng.choice(["First Owner", "Second Owner", "Third Owner", "Fourth & Above"],
                         size=n, p=[0.55, 0.28, 0.12, 0.05])

    years = rng.integers(2008, 2025, size=n)
    age = 2026 - years
    # base price depends on how new the car is, with fuel/brand noise so
    # it isn't a perfectly clean linear relationship (real data never is,
    # and later regression practicals should show real, imperfect error)
    base_price = 1_200_000 - age * 65_000
    noise = rng.normal(0, 80_000, size=n)
    fuel_premium = np.where(fuel_types == "Diesel", 60_000, 0) + np.where(fuel_types == "Electric", 300_000, 0)
    selling_price = np.clip(base_price + noise + fuel_premium, 45_000, None).round(-3)

    kms_driven = np.clip((age * rng.normal(11_000, 3_000, size=n)), 500, None).round(0)

    df = pd.DataFrame({
        "name": rng.choice(brands_models, size=n),
        "year": years,
        "selling_price": selling_price.astype(int),
        "km_driven": kms_driven.astype(int),
        "fuel": fuel_types,
        "seller_type": seller_types,
        "transmission": transmissions,
        "owner": owners,
    })

    # Deliberate outliers for the outlier-detection practical: a few
    # implausibly high selling prices and one implausibly high mileage.
    outlier_idx = rng.choice(n, size=4, replace=False)
    df.loc[outlier_idx[:2], "selling_price"] = [4_500_000, 5_200_000]
    df.loc[outlier_idx[2], "km_driven"] = 480_000
    df.loc[outlier_idx[3], "selling_price"] = 10_000  # implausibly low too

    # Deliberate missing values (a handful of cells across a few columns)
    # so dropna()/fillna() practicals have real work to do.
    for col in ["selling_price", "km_driven", "fuel"]:
        missing_idx = rng.choice(n, size=max(3, n // 60), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Deliberate exact-duplicate rows for drop_duplicates() practice.
    dup_rows = df.sample(n=5, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_students(n=250, seed=_RNG_SEED):
    """
    Student records - gender, study hours, attendance %, previous score,
    result (Pass/Fail). Matches practical 3: two-way cross-tabulation
    (Gender x Result) and a correlation matrix that should show a real,
    visible relationship between study hours/attendance and outcome
    (not random noise - the pass probability is deliberately built from
    those features so describe()/corr() have something real to find).
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    study_hours = np.clip(rng.normal(5, 2.2, size=n), 0, 14).round(1)
    attendance = np.clip(rng.normal(78, 12, size=n), 40, 100).round(1)
    previous_score = np.clip(rng.normal(62, 15, size=n), 20, 100).round(1)

    # Pass probability genuinely driven by the three features above,
    # plus noise - so correlation analysis finds a real, honest signal
    # (moderate, not a suspiciously perfect 1.0 - real data never is).
    z_study = (study_hours - study_hours.mean()) / study_hours.std()
    z_attend = (attendance - attendance.mean()) / attendance.std()
    z_prev = (previous_score - previous_score.mean()) / previous_score.std()
    logit = 0.35 + 1.1 * z_study + 0.9 * z_attend + 0.7 * z_prev
    pass_prob = 1 / (1 + np.exp(-logit))
    result = np.where(rng.random(n) < pass_prob, "Pass", "Fail")

    df = pd.DataFrame({
        "student_id": [f"S{1000+i}" for i in range(n)],
        "gender": gender,
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score,
        "result": result,
    })
    return df


def generate_supermarket_sales(n=500, seed=_RNG_SEED):
    """
    Retail transaction records - branch, product line, payment method,
    unit price, quantity, total. Matches practical 4: groupby/aggregation
    for total revenue by product line and payment method.
    """
    rng = np.random.default_rng(seed)

    product_lines = [
        "Health and beauty", "Electronic accessories", "Home and lifestyle",
        "Sports and travel", "Food and beverages", "Fashion accessories",
    ]
    payment_methods = ["Cash", "Credit card", "Ewallet"]
    branches = ["A", "B", "C"]
    cities = {"A": "Ahmedabad", "B": "Surat", "C": "Vadodara"}

    branch = rng.choice(branches, size=n)
    product_line = rng.choice(product_lines, size=n)
    payment = rng.choice(payment_methods, size=n, p=[0.35, 0.35, 0.30])
    unit_price = np.round(rng.uniform(10, 100, size=n), 2)
    quantity = rng.integers(1, 11, size=n)
    total = np.round(unit_price * quantity * 1.05, 2)  # +5% tax, like the classic dataset this mirrors

    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 90, size=n), unit="D")

    df = pd.DataFrame({
        "invoice_id": [f"INV-{5000+i}" for i in range(n)],
        "branch": branch,
        "city": [cities[b] for b in branch],
        "product_line": product_line,
        "unit_price": unit_price,
        "quantity": quantity,
        "total": total,
        "payment": payment,
        "date": dates,
    })
    return df.sort_values("date").reset_index(drop=True)


def generate_all(out_dir=None):
    """Generate and write all three datasets as CSVs. Returns the dict
    of {name: dataframe} generated, in case a caller wants them without
    a disk round-trip (e.g. tests)."""
    out_dir = out_dir or _OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    datasets = {
        "car_data": generate_car_data(),
        "students": generate_students(),
        "supermarket_sales": generate_supermarket_sales(),
    }
    for name, df in datasets.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    return datasets


if __name__ == "__main__":
    generated = generate_all()
    for name, df in generated.items():
        print(f"{name}.csv: {df.shape[0]} rows, {df.shape[1]} columns")
