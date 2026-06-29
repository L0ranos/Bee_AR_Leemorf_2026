import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Set font to Times New Roman for all plots
plt.rcParams["font.family"] = "Times New Roman"

FEAT_NAMES = [
    "mean",
    "median",
    "std",
    "variance",
    "min",
    "max",
    "range",
    "skewness",
    "kurtosis",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "iqr",
    "mad",
    "median_abs_dev",
    "mean_abs",
    "rms",
    "entropy",
    "hist_mean",
    "hist_std",
    "mode_approx",
    "peak_to_rms",
    "crest_factor",
    "jitter",
    "zerocross",
    "length",
    "outlier_fraction",
    "seglen_std",
]

NICE_NAMES = [
    "Mean",
    "Median",
    "Standard Deviation",
    "Variance",
    "Minimum",
    "Maximum",
    "Range",
    "Skewness",
    "Kurtosis",
    "10th Percentile",
    "25th Percentile",
    "50th Percentile (Median)",
    "75th Percentile",
    "90th Percentile",
    "Interquartile Range (IQR)",
    "Median Absolute Deviation (MAD)",
    "Median Absolute Deviation",
    "Mean Absolute Value",
    "Root Mean Square (RMS)",
    "Entropy",
    "Histogram Mean",
    "Histogram Standard Deviation",
    "Mode Approximation",
    "Peak to RMS Ratio",
    "Crest Factor",
    "Jitter",
    "Zero Crossing Rate",
    "Length of Event",
    "Outlier Fraction",
    "Segment Length Standard Deviation",
]

def get_nice_name(feat_name):
    """Convert feature name to readable display name."""
    addition = ""
    if feat_name.startswith("DIFF_"):
        base_name = feat_name.replace("DIFF_", "")
        addition = " (Diff)"
    else:
        base_name = feat_name

    try:
        idx = FEAT_NAMES.index(base_name)
        return NICE_NAMES[idx] + addition
    except (ValueError, IndexError):
        return feat_name

def run_logo_pipeline(dataset_name, dataset_dir, output_root):
    print(f"\n===== Running dataset: {dataset_name} =====")

    features_csv = dataset_dir / "SEGMENT_FEATURES.csv"
    diffs_csv = dataset_dir / "SEGMENT_DIFFS_FEATURES.csv"

    if not features_csv.exists() or not diffs_csv.exists():
        print("Missing required CSV files, skipping this dataset.")
        return None

    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    df_diffs = pd.read_csv(diffs_csv)

    df.drop(index=0, inplace=True)
    df_diffs.drop(index=0, inplace=True)

    # Add DIFF_ prefix while preserving class/specimen columns.
    df_diffs.rename(
        columns={
            col: f"DIFF_{col}"
            for col in df_diffs.columns
            if col not in ["class", "specimen_idx"]
        },
        inplace=True,
    )

    y = np.array(df["class"])
    specimens = np.array(df["specimen_idx"])

    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 1

    feature_names = list(df.drop(columns=["class", "specimen_idx"]).columns) + list(
        df_diffs.drop(columns=["class", "specimen_idx"]).columns
    )

    X = np.hstack(
        (
            np.array(df.drop(columns=["class", "specimen_idx"])),
            np.array(df_diffs.drop(columns=["class", "specimen_idx"])),
        )
    )

    print(np.shape(X), np.shape(y), np.shape(specimens))
    #Get row indices of rows where any feature is NaN or Inf, and remove those rows from X, y, and specimens
    invalid_rows = np.where(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))[0]
    if len(invalid_rows) > 0:
        print(f"Removing {len(invalid_rows)} rows with NaN or Inf values.")
        X = np.delete(X, invalid_rows, axis=0)
        y = np.delete(y, invalid_rows, axis=0)
        specimens = np.delete(specimens, invalid_rows, axis=0)

    nice_feature_names = [get_nice_name(f) for f in feature_names]
        #remove a list of features from the feature_names and X
    features_to_remove = [
    "mean",
    "median",
    "std",
    "min",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "mad",
    "median_abs_dev",
    "mean_abs",
    "rms",
    "hist_mean",
    "hist_std",
    "mode_approx",
    "peak_to_rms",
    "crest_factor",
    "cumsum",
    ]
    #remove the features
    for feature in features_to_remove:
        if feature in feature_names:
            idx = feature_names.index(feature)
            feature_names.pop(idx)
            nice_feature_names.pop(idx)
            X = np.delete(X, idx, axis=1)

    # Remove features with high correlation (r > 0.8)
    correlation_matrix = np.corrcoef(X, rowvar=False)
    features_to_remove = set()
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix[i, j]) > 0.8:
                features_to_remove.add(j)

    features_to_keep = [i for i in range(len(feature_names)) if i not in features_to_remove]
    X = X[:, features_to_keep]
    feature_names = [feature_names[i] for i in features_to_keep]
    nice_feature_names = [nice_feature_names[i] for i in features_to_keep]
    print(f"Removed {len(features_to_remove)} highly correlated features. Remaining features: {len(feature_names)}")


    rf_kfold = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    svc = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    lr = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        l1_ratio=0,
        solver="lbfgs",
    )

    worker_specimens = np.unique(specimens[y == 0])
    drone_specimens = np.unique(specimens[y == 1])
    pairwise_permutations = list(itertools.product(worker_specimens, drone_specimens))

    print(f"Total pairwise holdout combinations: {len(pairwise_permutations)}")
    if len(pairwise_permutations) == 0:
        print("No valid worker-drone specimen pairs, skipping this dataset.")
        return None

    accuracies = []
    accuracies_svc = []
    accuracies_lr = []
    cm_rf_fold = []
    cm_svc_fold = []
    cm_lr_fold = []

    all_feature_importances = []
    all_shap_values = []
    all_shap_feature_values = []
    all_results = []

    for fold, (holdout_worker, holdout_drone) in enumerate(pairwise_permutations, 1):
        test_mask = (specimens == holdout_worker) | (specimens == holdout_drone)
        train_index = np.where(~test_mask)[0]
        test_index = np.where(test_mask)[0]

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        rf_kfold.fit(X_train, y_train)
        y_pred = rf_kfold.predict(X_test)
        accuracies.append(balanced_accuracy_score(y_test, y_pred))
        cm_rf_fold.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        all_feature_importances.append(rf_kfold.feature_importances_)

        tn, fp, fn, tp = cm_rf_fold[-1].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        all_results.append(
            {
                "Fold": fold,
                "Worker_Holdout": holdout_worker,
                "Drone_Holdout": holdout_drone,
                "Balanced Accuracy": accuracies[-1],
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
                "True Positives": tp,
            }
        )

        explainer = shap.TreeExplainer(rf_kfold)
        shap_values = explainer(X_test)
        shap_vals_drone = shap_values.values[:, :, 1]
        all_shap_values.append(shap_vals_drone)
        all_shap_feature_values.append(X_test)

        svc.fit(X_train, y_train)
        y_pred_svc = svc.predict(X_test)
        balanced_accuracy_svc = balanced_accuracy_score(y_test, y_pred_svc)
        accuracies_svc.append(balanced_accuracy_svc)
        cm_svc = confusion_matrix(y_test, y_pred_svc, labels=[0, 1])
        cm_svc_fold.append(cm_svc)
        tn_svc, fp_svc, fn_svc, tp_svc = cm_svc.ravel()
        precision_svc = tp_svc / (tp_svc + fp_svc) if (tp_svc + fp_svc) > 0 else 0
        recall_svc = tp_svc / (tp_svc + fn_svc) if (tp_svc + fn_svc) > 0 else 0
        f1_svc = (
            2 * (precision_svc * recall_svc) / (precision_svc + recall_svc)
            if (precision_svc + recall_svc) > 0
            else 0
        )

        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        balanced_accuracy_lr = balanced_accuracy_score(y_test, y_pred_lr)
        accuracies_lr.append(balanced_accuracy_lr)
        cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[0, 1])
        cm_lr_fold.append(cm_lr)
        tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
        precision_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0
        recall_lr = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0
        f1_lr = (
            2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
            if (precision_lr + recall_lr) > 0
            else 0
        )

        all_results[-1].update(
            {
                "Balanced Accuracy SVC": balanced_accuracy_svc,
                "Precision SVC": precision_svc,
                "Recall SVC": recall_svc,
                "F1-Score SVC": f1_svc,
                "True Negatives SVC": tn_svc,
                "False Positives SVC": fp_svc,
                "False Negatives SVC": fn_svc,
                "True Positives SVC": tp_svc,
                "Balanced Accuracy LR": balanced_accuracy_lr,
                "Precision LR": precision_lr,
                "Recall LR": recall_lr,
                "F1-Score LR": f1_lr,
                "True Negatives LR": tn_lr,
                "False Positives LR": fp_lr,
                "False Negatives LR": fn_lr,
                "True Positives LR": tp_lr,
            }
        )

    metrics_df = pd.DataFrame(all_results)
    summary_cm_rf = np.sum(cm_rf_fold, axis=0)
    summary_tn, summary_fp, summary_fn, summary_tp = summary_cm_rf.ravel()
    summary_precision = (
        summary_tp / (summary_tp + summary_fp)
        if (summary_tp + summary_fp) > 0
        else 0
    )
    summary_recall = (
        summary_tp / (summary_tp + summary_fn)
        if (summary_tp + summary_fn) > 0
        else 0
    )
    summary_specificity = (
        summary_tn / (summary_tn + summary_fp)
        if (summary_tn + summary_fp) > 0
        else 0
    )
    summary_balanced_accuracy = (summary_recall + summary_specificity) / 2
    summary_f1 = (
        2 * (summary_precision * summary_recall) / (summary_precision + summary_recall)
        if (summary_precision + summary_recall) > 0
        else 0
    )
    summary_cm_svc = np.sum(cm_svc_fold, axis=0)
    summary_tn_svc, summary_fp_svc, summary_fn_svc, summary_tp_svc = summary_cm_svc.ravel()
    summary_precision_svc = (
        summary_tp_svc / (summary_tp_svc + summary_fp_svc)
        if (summary_tp_svc + summary_fp_svc) > 0
        else 0
    )
    summary_recall_svc = (
        summary_tp_svc / (summary_tp_svc + summary_fn_svc)
        if (summary_tp_svc + summary_fn_svc) > 0
        else 0
    )
    summary_specificity_svc = (
        summary_tn_svc / (summary_tn_svc + summary_fp_svc)
        if (summary_tn_svc + summary_fp_svc) > 0
        else 0
    )
    summary_balanced_accuracy_svc = (summary_recall_svc + summary_specificity_svc) / 2
    summary_f1_svc = (
        2 * (summary_precision_svc * summary_recall_svc)
        / (summary_precision_svc + summary_recall_svc)
        if (summary_precision_svc + summary_recall_svc) > 0
        else 0
    )
    summary_cm_lr = np.sum(cm_lr_fold, axis=0)
    summary_tn_lr, summary_fp_lr, summary_fn_lr, summary_tp_lr = summary_cm_lr.ravel()
    summary_precision_lr = (
        summary_tp_lr / (summary_tp_lr + summary_fp_lr)
        if (summary_tp_lr + summary_fp_lr) > 0
        else 0
    )
    summary_recall_lr = (
        summary_tp_lr / (summary_tp_lr + summary_fn_lr)
        if (summary_tp_lr + summary_fn_lr) > 0
        else 0
    )
    summary_specificity_lr = (
        summary_tn_lr / (summary_tn_lr + summary_fp_lr)
        if (summary_tn_lr + summary_fp_lr) > 0
        else 0
    )
    summary_balanced_accuracy_lr = (summary_recall_lr + summary_specificity_lr) / 2
    summary_f1_lr = (
        2 * (summary_precision_lr * summary_recall_lr)
        / (summary_precision_lr + summary_recall_lr)
        if (summary_precision_lr + summary_recall_lr) > 0
        else 0
    )
    avg_cm_rf = np.mean(cm_rf_fold, axis=0)
    avg_cm_svc = np.mean(cm_svc_fold, axis=0)
    avg_cm_lr = np.mean(cm_lr_fold, axis=0)

    print(
        f"Summary Balanced Accuracy from CV: {summary_balanced_accuracy:.2f}"
    )
    print(
        f"Summary Precision/Recall/F1: {summary_precision:.2f} / {summary_recall:.2f} / {summary_f1:.2f}"
    )
    print(f"Summary Confusion Matrix (Extra Trees):\\n{summary_cm_rf}")

    print(
        f"Summary Balanced Accuracy from SVC: {summary_balanced_accuracy_svc:.2f}"
    )
    print(
        f"Summary Precision/Recall/F1 from SVC: {summary_precision_svc:.2f} / {summary_recall_svc:.2f} / {summary_f1_svc:.2f}"
    )
    print(f"Summary Confusion Matrix (SVC):\\n{summary_cm_svc}")

    print(
        f"Summary Balanced Accuracy from Logistic Regression: {summary_balanced_accuracy_lr:.2f}"
    )
    print(
        f"Summary Precision/Recall/F1 from Logistic Regression: {summary_precision_lr:.2f} / {summary_recall_lr:.2f} / {summary_f1_lr:.2f}"
    )
    print(f"Summary Confusion Matrix (Logistic Regression):\\n{summary_cm_lr}")

    summary_row = pd.DataFrame(
        [
            {
                "Fold": "Summary",
                "Worker_Holdout": "ALL",
                "Drone_Holdout": "ALL",
                "Balanced Accuracy": summary_balanced_accuracy,
                "Precision": summary_precision,
                "Recall": summary_recall,
                "F1-Score": summary_f1,
                "True Negatives": summary_tn,
                "False Positives": summary_fp,
                "False Negatives": summary_fn,
                "True Positives": summary_tp,
            }
        ]
    )
    avg_row = pd.DataFrame(
        [
            {
                "Fold": "Average",
                "Worker_Holdout": "ALL",
                "Drone_Holdout": "ALL",
                "Balanced Accuracy": np.mean(accuracies),
                "Precision": metrics_df["Precision"].mean(),
                "Recall": metrics_df["Recall"].mean(),
                "F1-Score": metrics_df["F1-Score"].mean(),
                "True Negatives": metrics_df["True Negatives"].mean(),
                "False Positives": metrics_df["False Positives"].mean(),
                "False Negatives": metrics_df["False Negatives"].mean(),
                "True Positives": metrics_df["True Positives"].mean(),
                "Balanced Accuracy SVC": np.mean(accuracies_svc),
                "Precision SVC": metrics_df["Precision SVC"].mean(),
                "Recall SVC": metrics_df["Recall SVC"].mean(),
                "F1-Score SVC": metrics_df["F1-Score SVC"].mean(),
                "True Negatives SVC": metrics_df["True Negatives SVC"].mean(),
                "False Positives SVC": metrics_df["False Positives SVC"].mean(),
                "False Negatives SVC": metrics_df["False Negatives SVC"].mean(),
                "True Positives SVC": metrics_df["True Positives SVC"].mean(),
                "Balanced Accuracy LR": np.mean(accuracies_lr),
                "Precision LR": metrics_df["Precision LR"].mean(),
                "Recall LR": metrics_df["Recall LR"].mean(),
                "F1-Score LR": metrics_df["F1-Score LR"].mean(),
                "True Negatives LR": metrics_df["True Negatives LR"].mean(),
                "False Positives LR": metrics_df["False Positives LR"].mean(),
                "False Negatives LR": metrics_df["False Negatives LR"].mean(),
                "True Positives LR": metrics_df["True Positives LR"].mean(),
            }
        ]
    )
    std_row = pd.DataFrame(
        [
            {
                "Fold": "Std Dev",
                "Worker_Holdout": "ALL",
                "Drone_Holdout": "ALL",
                "Balanced Accuracy": np.std(accuracies),
                "Precision": metrics_df["Precision"].std(),
                "Recall": metrics_df["Recall"].std(),
                "F1-Score": metrics_df["F1-Score"].std(),
                "True Negatives": metrics_df["True Negatives"].std(),
                "False Positives": metrics_df["False Positives"].std(),
                "False Negatives": metrics_df["False Negatives"].std(),
                "True Positives": metrics_df["True Positives"].std(),
                "Balanced Accuracy SVC": np.std(accuracies_svc),
                "Precision SVC": metrics_df["Precision SVC"].std(),
                "Recall SVC": metrics_df["Recall SVC"].std(),
                "F1-Score SVC": metrics_df["F1-Score SVC"].std(),
                "True Negatives SVC": metrics_df["True Negatives SVC"].std(),
                "False Positives SVC": metrics_df["False Positives SVC"].std(),
                "False Negatives SVC": metrics_df["False Negatives SVC"].std(),
                "True Positives SVC": metrics_df["True Positives SVC"].std(),
                "Balanced Accuracy LR": np.std(accuracies_lr),
                "Precision LR": metrics_df["Precision LR"].std(),
                "Recall LR": metrics_df["Recall LR"].std(),
                "F1-Score LR": metrics_df["F1-Score LR"].std(),
                "True Negatives LR": metrics_df["True Negatives LR"].std(),
                "False Positives LR": metrics_df["False Positives LR"].std(),
                "False Negatives LR": metrics_df["False Negatives LR"].std(),
                "True Positives LR": metrics_df["True Positives LR"].std(),
            }
        ]
    )
    metrics_df = pd.concat([metrics_df, summary_row, avg_row, std_row], ignore_index=True)
    metrics_df.to_excel(out_dir / "pairwise_validation_metrics.xlsx", index=False)

    row_sums = avg_cm_rf.sum(axis=1, keepdims=True)
    cm_rf_norm = np.divide(
        avg_cm_rf.astype("float"),
        row_sums,
        out=np.zeros_like(avg_cm_rf, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_rf_norm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix - Extra Trees Classifier")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks([0, 1], ["Worker", "Drone"])
    plt.yticks([0, 1], ["Worker", "Drone"])
    plt.colorbar()
    for i in range(cm_rf_norm.shape[0]):
        for j in range(cm_rf_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_rf_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_rf_norm[i, j] > cm_rf_norm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_rf.pdf")
    plt.close()

    seed_shap_importances = [
        np.mean(np.abs(shap_vals), axis=0) for shap_vals in all_shap_values
    ]
    avg_shap_importance = np.mean(seed_shap_importances, axis=0)
    shap_indices = np.argsort(avg_shap_importance)
    sorted_shap_feature_names = [nice_feature_names[i] for i in shap_indices]
    sorted_shap_importance = avg_shap_importance[shap_indices]

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_shap_importance) * 0.3)))
    ax.barh(
        range(len(sorted_shap_importance)),
        sorted_shap_importance,
        color="orange",
        alpha=0.7,
    )
    ax.set_yticks(range(len(sorted_shap_importance)))
    ax.set_yticklabels(sorted_shap_feature_names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Average SHAP Feature Importance - Extra Trees")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_feature_importance_pairwise.pdf")
    plt.close()

    combined_shap_values = np.vstack(all_shap_values)
    combined_feature_values = np.vstack(all_shap_feature_values)
    shap.summary_plot(
        combined_shap_values,
        features=combined_feature_values,
        feature_names=nice_feature_names,
        show=False,
        max_display=len(nice_feature_names),
    )
    plt.title("SHAP Summary Plot - Feature Impact on Drone Classification")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_impact_pairwise.pdf")
    plt.close()

    avg_feature_importances = np.mean(all_feature_importances, axis=0)
    mdi_indices = np.argsort(avg_feature_importances)[-len(feature_names):]
    sorted_feature_names = [nice_feature_names[i] for i in mdi_indices]
    sorted_mdi = avg_feature_importances[mdi_indices]

    plt.figure(figsize=(12, max(6, len(sorted_mdi) * 0.3)))
    plt.barh(range(len(sorted_mdi)), sorted_mdi, color="red", alpha=0.7)
    plt.yticks(range(len(sorted_mdi)), sorted_feature_names)
    plt.xlabel("Mean Decrease in Impurity")
    plt.title("Average Feature Importance - Extra Trees")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mdi_feature_importance_pairwise.pdf")
    plt.close()

    top_6_indices = np.argsort(avg_feature_importances)[-6:]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for idx, feature_idx in enumerate(top_6_indices):
        worker_data = X[y == 0, feature_idx]
        drone_data = X[y == 1, feature_idx]

        data_to_plot = [worker_data, drone_data]
        bp = axes[idx].boxplot(
            data_to_plot,
            positions=[0, 1],
            patch_artist=True,
            tick_labels=["Worker", "Drone"],
            flierprops=dict(
                marker="o",
                markerfacecolor="red",
                markersize=5,
                linestyle="none",
                alpha=0,
            ),
        )
        colors = ["blue", "green"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i, 0.04, size=len(data))
            axes[idx].scatter(x, data, alpha=0.3, s=20, color=colors[i])

        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["Worker", "Drone"])
        axes[idx].set_ylabel("Value")
        axes[idx].set_title(f"{nice_feature_names[feature_idx]}")
        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "top_6_features_distribution_pairwise.pdf")
    plt.close()

    return {
        "Dataset": dataset_name,
        "Pairwise_Folds": len(pairwise_permutations),
        "Balanced_Accuracy_RF_Mean": np.mean(accuracies),
        "Balanced_Accuracy_RF_Std": np.std(accuracies),
        "Balanced_Accuracy_RF_Summary": summary_balanced_accuracy,
        "Precision_RF_Summary": summary_precision,
        "Recall_RF_Summary": summary_recall,
        "F1_Score_RF_Summary": summary_f1,
        "True_Negatives_RF_Summary": summary_tn,
        "False_Positives_RF_Summary": summary_fp,
        "False_Negatives_RF_Summary": summary_fn,
        "True_Positives_RF_Summary": summary_tp,
        "Balanced_Accuracy_SVC_Mean": np.mean(accuracies_svc),
        "Balanced_Accuracy_SVC_Summary": summary_balanced_accuracy_svc,
        "Precision_SVC_Summary": summary_precision_svc,
        "Recall_SVC_Summary": summary_recall_svc,
        "F1_Score_SVC_Summary": summary_f1_svc,
        "True_Negatives_SVC_Summary": summary_tn_svc,
        "False_Positives_SVC_Summary": summary_fp_svc,
        "False_Negatives_SVC_Summary": summary_fn_svc,
        "True_Positives_SVC_Summary": summary_tp_svc,
        "Balanced_Accuracy_LR_Mean": np.mean(accuracies_lr),
        "Balanced_Accuracy_LR_Summary": summary_balanced_accuracy_lr,
        "Precision_LR_Summary": summary_precision_lr,
        "Recall_LR_Summary": summary_recall_lr,
        "F1_Score_LR_Summary": summary_f1_lr,
        "True_Negatives_LR_Summary": summary_tn_lr,
        "False_Positives_LR_Summary": summary_fp_lr,
        "False_Negatives_LR_Summary": summary_fn_lr,
        "True_Positives_LR_Summary": summary_tp_lr,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "DATA_OUT"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_folders = ["LEEMORF", "STFT", "AUTOCORR", "YIN"]
    overview_rows = []

    for dataset_name in dataset_folders:
        dataset_dir = base_dir / dataset_name
        run_summary = run_logo_pipeline(dataset_name, dataset_dir, output_root)
        if run_summary is not None:
            overview_rows.append(run_summary)

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        overview_df.to_excel(output_root / "overview_metrics.xlsx", index=False)
        print(f"\nSaved overview metrics to: {output_root / 'overview_metrics.xlsx'}")
    else:
        print("\nNo datasets were processed successfully.")


if __name__ == "__main__":
    main()