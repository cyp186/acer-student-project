"""Regenerate Part 1.3 metrics, insights, and figures from processed tables."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import (
    evaluate_models,
    extract_model_insights,
    make_models,
    plot_threshold_curves,
    plot_top_features,
    save_confusion_matrices,
    tune_threshold_for_f1,
)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    processed = ROOT / "data" / "processed"
    figures = ROOT / "figures"
    results = ROOT / "results"
    for directory in [figures, results]:
        directory.mkdir(parents=True, exist_ok=True)

    uci = pd.read_csv(processed / "uci_enrolment_model.csv")
    oulad = pd.read_csv(processed / "oulad_day30_model.csv")
    print("loaded", uci.shape, oulad.shape)

    summary = pd.DataFrame(
        {
            "dataset": ["UCI enrolment", "OULAD day 30"],
            "observations": [len(uci), len(oulad)],
            "positive_cases": [
                uci["target_dropout"].sum(),
                oulad["target_withdrawn"].sum(),
            ],
            "positive_rate": [
                uci["target_dropout"].mean(),
                oulad["target_withdrawn"].mean(),
            ],
        }
    )
    plot_data = summary.assign(positive_rate_pct=summary["positive_rate"] * 100)
    ax = sns.barplot(
        data=plot_data, x="dataset", y="positive_rate_pct", color="#1769aa"
    )
    ax.set(
        xlabel="",
        ylabel="Positive class (%)",
        title="Class balance after leakage-safe cohort construction",
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    plt.tight_layout()
    plt.savefig(figures / "class_balance.png", dpi=180)
    plt.close()

    uci_categorical = [
        "marital_status",
        "application_mode",
        "course",
        "daytime_evening_attendance",
        "previous_qualification",
        "nationality",
        "mother_s_qualification",
        "father_s_qualification",
        "mother_s_occupation",
        "father_s_occupation",
        "displaced",
        "educational_special_needs",
        "debtor",
        "tuition_fees_up_to_date",
        "gender",
        "scholarship_holder",
        "international",
    ]
    uci_numeric = [
        "application_order",
        "previous_qualification_grade",
        "admission_grade",
        "age_at_enrollment",
        "unemployment_rate",
        "inflation_rate",
        "gdp",
    ]
    x_uci = uci[uci_categorical + uci_numeric]
    y_uci = uci["target_dropout"]
    x_uci_train, x_uci_test, y_uci_train, y_uci_test = train_test_split(
        x_uci, y_uci, test_size=0.20, stratify=y_uci, random_state=42
    )
    x_uci_fit, x_uci_val, y_uci_fit, y_uci_val = train_test_split(
        x_uci_train,
        y_uci_train,
        test_size=0.20,
        stratify=y_uci_train,
        random_state=42,
    )

    uci_thresholds: dict[str, float] = {}
    for name, model in make_models(uci_categorical, uci_numeric).items():
        model.fit(x_uci_fit, y_uci_fit)
        best_threshold, _scan = tune_threshold_for_f1(
            y_uci_val, model.predict_proba(x_uci_val)[:, 1]
        )
        uci_thresholds[name] = best_threshold
    print("UCI thresholds", uci_thresholds)

    uci_metrics_default, _, uci_matrices_default = evaluate_models(
        make_models(uci_categorical, uci_numeric),
        x_uci_train,
        y_uci_train,
        x_uci_test,
        y_uci_test,
    )
    uci_metrics_default.insert(0, "dataset", "UCI enrolment")
    uci_metrics_default.insert(2, "threshold_rule", "default_0.5")

    uci_metrics, uci_fitted, uci_matrices = evaluate_models(
        make_models(uci_categorical, uci_numeric),
        x_uci_train,
        y_uci_train,
        x_uci_test,
        y_uci_test,
        thresholds=uci_thresholds,
    )
    uci_metrics.insert(0, "dataset", "UCI enrolment")
    uci_metrics.insert(2, "threshold_rule", "f1_max_on_validation")

    uci_compare = pd.concat([uci_metrics_default, uci_metrics], ignore_index=True)
    uci_compare.to_csv(results / "uci_baseline_metrics.csv", index=False)
    save_confusion_matrices("UCI", uci_matrices, figures, suffix="tuned")

    uci_insight_frames = []
    for name, model in uci_fitted.items():
        insights = extract_model_insights(
            model, top_n=12, x=x_uci_train, y=y_uci_train
        )
        insights.insert(0, "dataset", "UCI enrolment")
        insights.insert(1, "model", name)
        uci_insight_frames.append(insights)
    plot_top_features(
        uci_insight_frames[1],
        "UCI top contributions: Neural Network",
        figures / "uci_neural_network_top_features.png",
    )
    uci_insights = pd.concat(uci_insight_frames, ignore_index=True)
    uci_insights.to_csv(results / "uci_model_insights.csv", index=False)
    print("UCI done")
    print(uci_compare.round(3).to_string(index=False))

    oulad_categorical = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    oulad_excluded = {"id_student", "target_withdrawn", *oulad_categorical}
    oulad_numeric = [col for col in oulad.columns if col not in oulad_excluded]
    x_oulad = oulad[oulad_categorical + oulad_numeric]
    y_oulad = oulad["target_withdrawn"]
    groups = oulad["id_student"]

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(group_split.split(x_oulad, y_oulad, groups=groups))
    x_oulad_train, x_oulad_test = x_oulad.iloc[train_idx], x_oulad.iloc[test_idx]
    y_oulad_train, y_oulad_test = y_oulad.iloc[train_idx], y_oulad.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    val_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    fit_idx, val_idx = next(
        val_split.split(x_oulad_train, y_oulad_train, groups=groups_train)
    )
    x_oulad_fit, x_oulad_val = x_oulad_train.iloc[fit_idx], x_oulad_train.iloc[val_idx]
    y_oulad_fit, y_oulad_val = y_oulad_train.iloc[fit_idx], y_oulad_train.iloc[val_idx]

    oulad_thresholds: dict[str, float] = {}
    oulad_threshold_scans: dict[str, pd.DataFrame] = {}
    for name, model in make_models(oulad_categorical, oulad_numeric).items():
        print("tuning", name)
        model.fit(x_oulad_fit, y_oulad_fit)
        best_threshold, scan = tune_threshold_for_f1(
            y_oulad_val, model.predict_proba(x_oulad_val)[:, 1]
        )
        oulad_thresholds[name] = best_threshold
        oulad_threshold_scans[name] = scan
    plot_threshold_curves(
        oulad_threshold_scans["Neural Network"],
        "OULAD validation threshold scan: Neural Network",
        figures / "oulad_neural_network_threshold_scan.png",
        oulad_thresholds["Neural Network"],
    )
    print("OULAD thresholds", oulad_thresholds)

    print("eval default")
    oulad_metrics_default, _, oulad_matrices_default = evaluate_models(
        make_models(oulad_categorical, oulad_numeric),
        x_oulad_train,
        y_oulad_train,
        x_oulad_test,
        y_oulad_test,
    )
    oulad_metrics_default.insert(0, "dataset", "OULAD day 30")
    oulad_metrics_default.insert(2, "threshold_rule", "default_0.5")

    print("eval tuned")
    oulad_metrics, oulad_fitted, oulad_matrices = evaluate_models(
        make_models(oulad_categorical, oulad_numeric),
        x_oulad_train,
        y_oulad_train,
        x_oulad_test,
        y_oulad_test,
        thresholds=oulad_thresholds,
    )
    oulad_metrics.insert(0, "dataset", "OULAD day 30")
    oulad_metrics.insert(2, "threshold_rule", "f1_max_on_validation")

    oulad_compare = pd.concat(
        [oulad_metrics_default, oulad_metrics], ignore_index=True
    )
    oulad_compare.to_csv(results / "oulad_baseline_metrics.csv", index=False)
    save_confusion_matrices(
        "OULAD",
        {"Neural Network": oulad_matrices_default["Neural Network"]},
        figures,
        suffix="default_0.5",
    )
    save_confusion_matrices("OULAD", oulad_matrices, figures, suffix="tuned")

    oulad_insight_frames = []
    for name, model in oulad_fitted.items():
        insights = extract_model_insights(
            model, top_n=12, x=x_oulad_train, y=y_oulad_train
        )
        insights.insert(0, "dataset", "OULAD day 30")
        insights.insert(1, "model", name)
        oulad_insight_frames.append(insights)
    plot_top_features(
        oulad_insight_frames[1],
        "OULAD top contributions: Neural Network",
        figures / "oulad_neural_network_top_features.png",
    )
    oulad_insights = pd.concat(oulad_insight_frames, ignore_index=True)
    oulad_insights.to_csv(results / "oulad_model_insights.csv", index=False)

    all_metrics = pd.concat([uci_compare, oulad_compare], ignore_index=True)
    all_metrics.to_csv(results / "all_baseline_metrics.csv", index=False)
    all_insights = pd.concat([uci_insights, oulad_insights], ignore_index=True)
    all_insights.to_csv(results / "all_model_insights.csv", index=False)

    tuned = all_metrics.loc[
        all_metrics["threshold_rule"] == "f1_max_on_validation"
    ].copy()
    comparison_plot = tuned.melt(
        id_vars=["dataset", "model"],
        value_vars=["recall", "precision", "f1", "pr_auc_average_precision"],
        var_name="metric",
        value_name="score",
    )
    g = sns.catplot(
        data=comparison_plot,
        x="metric",
        y="score",
        hue="model",
        col="dataset",
        kind="bar",
        height=4.2,
        aspect=1.15,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Score")
    g.set(ylim=(0, 1))
    for axes in g.axes.flat:
        axes.tick_params(axis="x", rotation=25)
    g.fig.suptitle("Tuned-threshold test metrics by dataset", y=1.03)
    g.savefig(figures / "tuned_metrics_comparison.png", dpi=180)
    plt.close("all")

    findings = []
    for dataset_name, default_frame, tuned_frame in [
        ("UCI enrolment", uci_metrics_default, uci_metrics),
        ("OULAD day 30", oulad_metrics_default, oulad_metrics),
    ]:
        findings.append(
            {
                "dataset": dataset_name,
                "best_recall_model_tuned": tuned_frame.loc[
                    tuned_frame["recall"].idxmax(), "model"
                ],
                "best_f1_model_tuned": tuned_frame.loc[
                    tuned_frame["f1"].idxmax(), "model"
                ],
                "best_pr_auc_model": tuned_frame.loc[
                    tuned_frame["pr_auc_average_precision"].idxmax(), "model"
                ],
                "nn_recall_default_0.5": default_frame.loc[
                    default_frame["model"] == "Neural Network"
                ].iloc[0]["recall"],
                "nn_recall_tuned": tuned_frame.loc[
                    tuned_frame["model"] == "Neural Network"
                ].iloc[0]["recall"],
                "lr_recall_tuned": tuned_frame.loc[
                    tuned_frame["model"] == "Logistic Regression"
                ].iloc[0]["recall"],
                "lr_precision_tuned": tuned_frame.loc[
                    tuned_frame["model"] == "Logistic Regression"
                ].iloc[0]["precision"],
                "nn_precision_tuned": tuned_frame.loc[
                    tuned_frame["model"] == "Neural Network"
                ].iloc[0]["precision"],
            }
        )
    findings_frame = pd.DataFrame(findings)
    findings_frame.to_csv(results / "cross_dataset_findings.csv", index=False)
    print("OULAD done")
    print(oulad_compare.round(3).to_string(index=False))
    print(findings_frame.round(3).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
