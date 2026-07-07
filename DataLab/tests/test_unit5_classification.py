import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from datalab import unit5_classification as clf
from datalab.generate_datasets import generate_students


def _data():
    return clf.prepare_students_for_classification()


def test_prepare_students_for_classification_splits_and_encodes():
    data = _data()
    assert set(data["X_train"].columns) == {"gender_encoded", "study_hours_per_day", "attendance_percent", "previous_score"}
    assert set(np.unique(data["y_train"])) <= {0, 1}
    assert len(data["X_train"]) + len(data["X_test"]) == len(generate_students())


def test_evaluate_classifier_matches_manual_confusion_matrix_math():
    # 3 true negatives, 2 false positives, 1 false negative, 4 true positives
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    metrics = clf.evaluate_classifier(y_true, y_pred)

    assert metrics["true_negative"] == 3
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 4
    assert metrics["accuracy"] == (3 + 4) / 10
    assert metrics["error_rate"] == 1 - metrics["accuracy"]
    assert metrics["sensitivity"] == 4 / (4 + 1)  # TP / (TP+FN)
    assert metrics["specificity"] == 3 / (3 + 2)  # TN / (TN+FP)


def test_evaluate_classifier_rejects_non_binary_input():
    raised = False
    try:
        clf.evaluate_classifier([0, 1, 2], [0, 1, 2])
    except ValueError:
        raised = True
    assert raised, "should refuse a >2-class confusion matrix rather than silently mis-defining sensitivity/specificity"


def test_scale_features_produces_zero_mean_unit_variance_on_train():
    data = _data()
    X_train_scaled, X_test_scaled, scaler = clf.scale_features(data["X_train"], data["X_test"])

    # fit was on train, so train (not test) should come out ~standardized
    assert abs(X_train_scaled.mean().mean()) < 0.1
    assert abs(X_train_scaled.std().mean() - 1.0) < 0.15
    assert list(X_train_scaled.columns) == list(data["X_train"].columns)


def test_knn_with_scaling_outperforms_or_matches_unscaled_on_this_dataset():
    # Real bug this guards against: kNN is distance-based, and
    # attendance_percent (40-100) would dominate study_hours_per_day
    # (0-14) in raw distance purely due to scale, not information
    # content. Scaled kNN must not be WORSE than unscaled kNN here.
    data = _data()

    unscaled_model = clf.train_knn(data["X_train"], data["y_train"])
    unscaled_acc = (unscaled_model.predict(data["X_test"]) == data["y_test"]).mean()

    X_train_scaled, X_test_scaled, _ = clf.scale_features(data["X_train"], data["X_test"])
    scaled_model = clf.train_knn(X_train_scaled, data["y_train"])
    scaled_acc = (scaled_model.predict(X_test_scaled) == data["y_test"]).mean()

    assert scaled_acc >= unscaled_acc


def test_train_decision_tree_uses_entropy_criterion():
    data = _data()
    model = clf.train_decision_tree(data["X_train"], data["y_train"])
    assert model.criterion == "entropy"


def test_visualize_decision_tree_returns_a_real_figure():
    data = _data()
    model = clf.train_decision_tree(data["X_train"], data["y_train"])
    fig = clf.visualize_decision_tree(model, feature_names=data["feature_columns"])
    assert len(fig.axes) == 1


def test_run_practical11_kNN_end_to_end():
    result = clf.run_practical11()
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["metrics"]["confusion_matrix"].shape == (2, 2)


def test_run_practical12_decision_tree_end_to_end():
    result = clf.run_practical12()
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["tree_figure"] is not None


def test_run_practical13_random_forest_vs_svm_same_test_set():
    result = clf.run_practical13()
    assert "random_forest" in result["accuracy_comparison"]
    assert "svm" in result["accuracy_comparison"]
    assert 0.0 <= result["accuracy_comparison"]["random_forest"] <= 1.0
    assert 0.0 <= result["accuracy_comparison"]["svm"] <= 1.0


def test_run_practical14_reuses_a_given_model_correctly():
    data = _data()
    p11 = clf.run_practical11(data)
    X_train_scaled, X_test_scaled, _ = clf.scale_features(data["X_train"], data["X_test"])

    result = clf.run_practical14(p11["model"], X_test_scaled, data["y_test"])
    # reusing the exact same model+data as practical 11 must reproduce
    # the exact same metrics - proves it's really reusing, not retraining
    assert result["accuracy"] == p11["metrics"]["accuracy"]


def test_plot_confusion_matrix_returns_a_real_figure():
    import numpy as np
    cm = np.array([[10, 2], [3, 15]])
    fig = clf.plot_confusion_matrix(cm)
    assert len(fig.axes) == 2  # heatmap + colorbar


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
