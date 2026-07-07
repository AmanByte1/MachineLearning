"""
Unit 4: Regression - Model Training and Evaluation.

Covers syllabus topics 4.1-4.3 (Simple/Multiple Linear Regression,
Polynomial Regression, evaluation via R-squared/MAE/MSE) and
implements practicals 9-10:

  Practical 9: Simple and Multiple Linear Regression to predict
               selling price. Evaluate using R^2, MAE, MSE.
  Practical 10: Polynomial Regression, compare performance with Linear
                Regression, interpret evaluation metrics.

Uses unit3_ml_intro.prepare_car_data_for_modeling() for the train/test
split rather than reloading/re-cleaning data here - the whole point of
building unit 3 first was so units 4 and 5 don't each redo that work.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import unit3_ml_intro as ml_intro


def evaluate_regression(y_true, y_pred):
    """
    Practicals 9 & 10's evaluation step: R^2, MAE, MSE - the three
    metrics the syllabus explicitly names. Returned together since a
    real evaluation should report all three (R^2 alone can look good
    while MAE/MSE reveal the actual error magnitude in rupees, which
    matters more to someone deciding whether the model is useful).
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---------------------------------------------------------------------
# 4.1 - Simple and Multiple Linear Regression  (Practical 9)
# ---------------------------------------------------------------------

def train_simple_linear_regression(X_train, y_train, feature_column):
    """
    Practical 9's "simple" half: linear regression using exactly ONE
    feature (e.g. car_age alone) - kept as its own function rather
    than just "multiple regression with one column" so the distinction
    the syllabus draws (simple vs multiple) is explicit in the code,
    not just implicit in how many columns happen to be passed in.
    """
    model = LinearRegression()
    model.fit(X_train[[feature_column]], y_train)
    return model


def train_multiple_linear_regression(X_train, y_train):
    """Practical 9's "multiple" half: linear regression using every
    feature column in X_train."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# 4.2 - Polynomial Regression  (Practical 10)
# ---------------------------------------------------------------------

def train_polynomial_regression(X_train, y_train, feature_column, degree=2):
    """
    Practical 10: polynomial regression on a single feature (fitting a
    curve, not just a straight line) - built as a Pipeline
    (PolynomialFeatures -> LinearRegression) so degree is a real,
    inspectable hyperparameter rather than hand-building polynomial
    columns manually.
    """
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train[[feature_column]], y_train)
    return model


# ---------------------------------------------------------------------
# 4.3 - Evaluation using R-squared, MAE, MSE  (Practicals 9 & 10)
# ---------------------------------------------------------------------

def compare_linear_vs_polynomial(X_train, X_test, y_train, y_test, feature_column, degree=2):
    """
    Practical 10's "compare performance with Linear Regression" step:
    trains both a simple linear model and a polynomial model on the
    SAME single feature, evaluates both on the same held-out test set,
    and returns both metric sets side by side so the comparison is a
    real, direct one (same data, same split, same evaluation function) -
    not an apples-to-oranges comparison across different setups.
    """
    linear_model = train_simple_linear_regression(X_train, y_train, feature_column)
    poly_model = train_polynomial_regression(X_train, y_train, feature_column, degree=degree)

    linear_pred = linear_model.predict(X_test[[feature_column]])
    poly_pred = poly_model.predict(X_test[[feature_column]])

    return {
        "linear": evaluate_regression(y_test, linear_pred),
        "polynomial": evaluate_regression(y_test, poly_pred),
        "degree": degree,
        "feature_column": feature_column,
    }


def run_practical9(data=None):
    """
    Run practical 9 end-to-end: simple regression (car_age alone) and
    multiple regression (all features), both evaluated with R^2/MAE/MSE.
    `data` should be the dict from unit3_ml_intro.prepare_car_data_for_modeling()
    if you already have one (e.g. to reuse across practicals 9/10/11+
    without re-splitting each time); generates its own if not given.
    """
    data = data or ml_intro.prepare_car_data_for_modeling()

    simple_model = train_simple_linear_regression(data["X_train"], data["y_train"], "car_age")
    simple_pred = simple_model.predict(data["X_test"][["car_age"]])

    multiple_model = train_multiple_linear_regression(data["X_train"], data["y_train"])
    multiple_pred = multiple_model.predict(data["X_test"])

    return {
        "simple_linear": {"model": simple_model, "metrics": evaluate_regression(data["y_test"], simple_pred)},
        "multiple_linear": {"model": multiple_model, "metrics": evaluate_regression(data["y_test"], multiple_pred)},
    }


def run_practical10(data=None, degree=2):
    """Run practical 10 end-to-end: linear vs polynomial comparison on car_age."""
    data = data or ml_intro.prepare_car_data_for_modeling()
    return compare_linear_vs_polynomial(
        data["X_train"], data["X_test"], data["y_train"], data["y_test"],
        feature_column="car_age", degree=degree,
    )
