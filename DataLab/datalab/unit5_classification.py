"""
Unit 5: Classification - Model Training and Evaluation.

Covers syllabus topics 5.1-5.3 (kNN, Decision Tree, Random Forest, SVM,
evaluation via confusion matrix - accuracy/error rate/sensitivity/
specificity) and implements practicals 11-14:

  Practical 11: kNN classifier - confusion matrix, accuracy.
  Practical 12: Decision Tree (entropy criterion) - visualize + evaluate.
  Practical 13: Random Forest and SVM - compare accuracy.
  Practical 14: confusion matrix, accuracy, error rate, sensitivity,
                specificity for a classification dataset.

Uses students.csv (Pass/Fail is a genuine binary classification target,
unlike car_data.csv which is a regression target) - a different
dataset than units 3/4, since those units are built around predicting
a continuous price, and classification needs a real categorical label.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .generate_datasets import generate_students

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_HERE, "outputs")


def _ensure_output_dir():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


# ---------------------------------------------------------------------
# Shared prep: students.csv -> features/labels/split
# ---------------------------------------------------------------------

def prepare_students_for_classification(students_df=None, test_size=0.2, random_state=42):
    """
    Load (or accept) students.csv, encode gender + result, split into
    train/test. Shared by every classifier in this unit, same reasoning
    as unit3_ml_intro.prepare_car_data_for_modeling() for regression -
    built once here rather than duplicated per practical.
    """
    df = students_df if students_df is not None else generate_students()

    gender_encoder = LabelEncoder()
    result_encoder = LabelEncoder()  # Fail=0, Pass=1 (alphabetical)

    features = pd.DataFrame({
        "gender_encoded": gender_encoder.fit_transform(df["gender"]),
        "study_hours_per_day": df["study_hours_per_day"],
        "attendance_percent": df["attendance_percent"],
        "previous_score": df["previous_score"],
    })
    labels = result_encoder.fit_transform(df["result"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "result_encoder": result_encoder, "gender_encoder": gender_encoder,
        "feature_columns": list(features.columns),
    }


# ---------------------------------------------------------------------
# 5.3 - Evaluation: confusion matrix, accuracy, error rate,
#       sensitivity, specificity  (used by every practical below)
# ---------------------------------------------------------------------

def evaluate_classifier(y_true, y_pred):
    """
    Practical 14: confusion matrix plus every metric the syllabus names
    by name - accuracy, error rate, sensitivity, specificity. Assumes
    binary classification (Pass/Fail-style), which is what "sensitivity/
    specificity" as named terms specifically mean (they don't generalize
    to >2 classes without picking a "positive" class per-class, which
    is a different, more involved technique this practical isn't asking for).

    Returns the raw 2x2 confusion matrix plus every derived metric, so
    a caller can see exactly how each metric was derived from the
    counts, not just the final numbers.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError(
            f"evaluate_classifier expects binary classification (2x2 confusion "
            f"matrix), got shape {cm.shape} - sensitivity/specificity aren't "
            f"well-defined for more than 2 classes without a stated positive class."
        )

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    error_rate = 1 - accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # a.k.a. recall / true positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # true negative rate

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positive": int(tp), "true_negative": int(tn),
        "false_positive": int(fp), "false_negative": int(fn),
    }


def plot_confusion_matrix(cm, class_names=("Fail", "Pass"), title="Confusion Matrix"):
    """Practical 12/14: render a confusion matrix as a labeled heatmap."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scale_features(X_train, X_test):
    """
    Standardize features (zero mean, unit variance) - required for
    distance-based models (kNN, SVM) since they're sensitive to the
    raw scale of each feature; without this, a feature like
    attendance_percent (40-100) would dominate a feature like
    study_hours_per_day (0-14) in the distance calculation purely
    because of its larger numeric range, not because it's actually
    more informative. Fit on train only, applied to both, to avoid
    leaking test-set statistics into training.

    Tree-based models (Decision Tree, Random Forest) don't need this -
    they split on raw thresholds per feature independently, so scale
    doesn't affect them - which is why only train_knn()/train_svm()
    below use scaled features while the tree models use the raw ones.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------
# 5.1 - kNN, Decision Tree (entropy)
# ---------------------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5):
    """Practical 11: k-Nearest Neighbours classifier. Expects
    already-scaled features (see scale_features()) since kNN is
    distance-based."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=4):
    """Practical 12: Decision Tree using entropy (information gain) as
    the splitting criterion - the syllabus specifically names entropy,
    not the scikit-learn default (gini)."""
    model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def visualize_decision_tree(model, feature_names, class_names=("Fail", "Pass")):
    """Practical 12: visualize the trained tree."""
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(model, feature_names=feature_names, class_names=list(class_names),
              filled=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 5.2 - Random Forest, SVM
# ---------------------------------------------------------------------

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=4):
    """Practical 13: Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf"):
    """Practical 13: Support Vector Machine classifier. Expects
    already-scaled features (see scale_features()) - SVM is also
    distance/margin-based, same reasoning as kNN above."""
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    return model


def compare_classifier_accuracy(models_and_names, X_test, y_test):
    """
    Practical 13's "compare model accuracy" step: given a list of
    (fitted_model, name) pairs, evaluate every one on the SAME test
    set and return a dict of {name: accuracy} - a fair, direct
    comparison since every model sees identical held-out data.
    """
    results = {}
    for model, name in models_and_names:
        predictions = model.predict(X_test)
        results[name] = accuracy_score(y_test, predictions)
    return results


# ---------------------------------------------------------------------
# Practical runners - one call per practical, tying the above together
# ---------------------------------------------------------------------

def run_practical11(data=None):
    """kNN classifier, confusion matrix, accuracy. Uses scaled features
    (see scale_features()) since kNN is distance-based."""
    data = data or prepare_students_for_classification()
    X_train_scaled, X_test_scaled, _ = scale_features(data["X_train"], data["X_test"])
    model = train_knn(X_train_scaled, data["y_train"])
    predictions = model.predict(X_test_scaled)
    metrics = evaluate_classifier(data["y_test"], predictions)
    return {"model": model, "metrics": metrics}


def run_practical12(data=None):
    """Decision Tree (entropy), visualize, evaluate. Uses raw (unscaled)
    features - trees are scale-invariant, so scaling would add
    complexity with zero benefit here."""
    data = data or prepare_students_for_classification()
    model = train_decision_tree(data["X_train"], data["y_train"])
    predictions = model.predict(data["X_test"])
    metrics = evaluate_classifier(data["y_test"], predictions)
    tree_fig = visualize_decision_tree(model, feature_names=data["feature_columns"])
    return {"model": model, "metrics": metrics, "tree_figure": tree_fig}


def run_practical13(data=None):
    """Random Forest (raw features - scale-invariant) and SVM (scaled
    features - distance-based), compare accuracy on the same test set."""
    data = data or prepare_students_for_classification()
    X_train_scaled, X_test_scaled, _ = scale_features(data["X_train"], data["X_test"])

    rf_model = train_random_forest(data["X_train"], data["y_train"])
    svm_model = train_svm(X_train_scaled, data["y_train"])

    rf_predictions = rf_model.predict(data["X_test"])
    svm_predictions = svm_model.predict(X_test_scaled)

    comparison = {
        "random_forest": accuracy_score(data["y_test"], rf_predictions),
        "svm": accuracy_score(data["y_test"], svm_predictions),
    }
    return {"random_forest": rf_model, "svm": svm_model, "accuracy_comparison": comparison}


def run_practical14(model, X_test, y_test):
    """
    Full confusion-matrix-based evaluation (accuracy, error rate,
    sensitivity, specificity) for any already-fitted model - practical
    14 is explicitly "for A classification dataset", i.e. reuse
    whichever model you already have, not train a new one.

    IMPORTANT: pass the SAME kind of X_test the model was trained on -
    scaled features for a kNN/SVM model (see scale_features()), raw
    features for a Decision Tree/Random Forest model. Passing raw
    features to a model trained on scaled ones (or vice versa) won't
    raise an error, it'll just silently produce meaningless predictions,
    since the model has no way to know its input wasn't preprocessed
    the same way - there's no automatic check for this, so it's on the
    caller to keep the two consistent.
    """
    predictions = model.predict(X_test)
    return evaluate_classifier(y_test, predictions)
