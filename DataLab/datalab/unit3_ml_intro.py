"""
Unit 3: Introduction to Machine Learning with Python.

Covers syllabus topics 3.1-3.3 (supervised learning concepts,
data cleaning/feature engineering, train-validation split with basic
cross-validation) and implements practical 8:

  Practical 8: using car_data.csv, preprocess (encode categorical
               variables) and split into training/testing sets.

This module is deliberately the shared preprocessing step for units 4
and 5 (regression, classification) - both need "clean data in, encoded
features, a train/test split" before they can do anything, so it's
built once here and imported by both rather than duplicated.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

from . import unit1_eda as eda


# ---------------------------------------------------------------------
# 3.1 - What is ML? Supervised Learning - Regression vs Classification
# ---------------------------------------------------------------------

def explain_supervised_learning():
    """
    Not a plot or a model - a plain-text explanation, returned as a
    dict so it can be displayed or tested for content, matching what
    topic 3.1 actually asks for (a conceptual understanding, not code).
    """
    return {
        "supervised_learning": (
            "Learning a mapping from labeled input features to a known "
            "output, using examples where the correct answer is already "
            "known (e.g. past car sales with their actual selling price)."
        ),
        "regression": (
            "Predicts a continuous numeric value - e.g. predicting a car's "
            "exact selling price in rupees (see unit4_regression.py)."
        ),
        "classification": (
            "Predicts a discrete category - e.g. predicting Pass/Fail, or "
            "which of several classes a data point belongs to "
            "(see unit5_classification.py)."
        ),
    }


# ---------------------------------------------------------------------
# 3.2 - Data Cleaning/Pre-processing, Feature Engineering
# ---------------------------------------------------------------------

def encode_categorical_columns(df, columns):
    """
    Practical 8: encode categorical variables (fuel, seller_type,
    transmission, owner) as integers via LabelEncoder, so a model that
    only understands numbers can use them.

    Returns (encoded_df, encoders) where `encoders` is a dict of
    {column: fitted LabelEncoder} - kept so a caller can
    encoder.inverse_transform(...) later to get back the original
    category names from a prediction, rather than being stuck with
    unlabeled integers.
    """
    encoded = df.copy()
    encoders = {}
    for col in columns:
        if col not in encoded.columns:
            continue
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col].astype(str))
        encoders[col] = encoder
    return encoded, encoders


def engineer_car_age_feature(df, current_year=2026):
    """
    Practical 8's "feature engineering" half: derive `car_age` from
    `year`, since age-of-car is a more directly useful predictor for
    price than the raw year value (a model can learn this relationship
    either way, but an engineered feature makes it explicit and is
    exactly what the syllabus topic 3.2 asks students to practice).
    """
    engineered = df.copy()
    engineered["car_age"] = current_year - engineered["year"]
    return engineered


def select_feature_subset(df, target_column, exclude_columns=None):
    """
    Practical 8's "feature subset selection": returns (X, y) - the
    numeric feature matrix and the target column - dropping any
    non-numeric leftovers (like `name`, which is too high-cardinality
    to be a useful raw feature) and any explicitly excluded columns.
    """
    exclude_columns = set(exclude_columns or [])
    exclude_columns.add(target_column)

    numeric_df = df.select_dtypes(include=[np.number])
    feature_columns = [c for c in numeric_df.columns if c not in exclude_columns]

    X = numeric_df[feature_columns]
    y = df[target_column]
    return X, y


# ---------------------------------------------------------------------
# 3.3 - Train-validation split, basic cross-validation
# ---------------------------------------------------------------------

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Practical 8: train/test split. Seeded by default for reproducibility."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def basic_cross_validation(model, X, y, cv=5, scoring=None):
    """
    Topic 3.3's "basic cross-validation" - k-fold cross-validation
    scores for a given (unfitted) model. Returns the raw scores array
    plus mean/std, so a caller can see both the average performance
    and how much it varies across folds (a single held-out test score
    can be lucky/unlucky; cross-validation is what actually shows that).
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


def prepare_car_data_for_modeling(car_df=None, target_column="selling_price"):
    """
    Convenience wrapper tying this whole unit together: load (or accept)
    car_data.csv, clean it (unit1_eda), engineer the car_age feature,
    encode categoricals, select the numeric feature subset, and split
    into train/test. This is what unit4_regression.py and
    unit5_classification.py both call to get modeling-ready data,
    rather than each reimplementing "load + clean + encode + split".
    """
    if car_df is None:
        car_df = eda.load_car_data()

    cleaned, _ = eda.clean_car_data(car_df)
    engineered = engineer_car_age_feature(cleaned)
    encoded, encoders = encode_categorical_columns(
        engineered, columns=["fuel", "seller_type", "transmission", "owner"]
    )

    X, y = select_feature_subset(encoded, target_column=target_column, exclude_columns=["year"])
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "encoders": encoders, "feature_columns": list(X.columns),
    }
