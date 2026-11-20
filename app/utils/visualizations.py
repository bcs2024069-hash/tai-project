"""
Interactive Plotly Visualizations for the Trustworthy Diabetes Decision System.
Uses a cohesive, professional healthcare/biomedical color palette.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# Palette
PRIMARY_BLUE = "#1e40af"
ACCENT_CYAN = "#0ea5e9"
ALERT_RED = "#dc2626"
SAFE_GREEN = "#16a34a"
MUTED_SLATE = "#64748b"
WARM_AMBER = "#d97706"


def plot_confusion_matrix(cm: List[List[int]], model_name: str) -> go.Figure:
    """Generates an annotated 2x2 confusion matrix heatmap."""
    z = np.array(cm)
    total = z.sum()
    labels = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"],
    ]
    text = [
        [f"{labels[0][0]}<br><b>{z[0,0]:,}</b><br>({z[0,0]/total:.1%})",
         f"{labels[0][1]}<br><b>{z[0,1]:,}</b><br>({z[0,1]/total:.1%})"],
        [f"{labels[1][0]}<br><b>{z[1,0]:,}</b><br>({z[1,0]/total:.1%})",
         f"{labels[1][1]}<br><b>{z[1,1]:,}</b><br>({z[1,1]/total:.1%})"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted: No Diabetes (0)", "Predicted: Diabetes (1)"],
            y=["Actual: No Diabetes (0)", "Actual: Diabetes (1)"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "black"},
            colorscale=[[0, "#f8fafc"], [1, "#93c5fd"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title=f"<b>Confusion Matrix: {model_name}</b>",
        title_font={"size": 15},
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def plot_roc_curves(eval_metrics: Dict[str, Any]) -> go.Figure:
    """Plots overlaid ROC curves for Model 1 and Model 2 with identical ROC-AUC."""
    fig = go.Figure()

    # Diagonal chance line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
            name="Chance Baseline (AUC = 0.500)",
            hoverinfo="none",
        )
    )

    # Model 1
    m1 = eval_metrics["baseline_model"]
    m1_roc = m1["roc_curve"]
    fig.add_trace(
        go.Scatter(
            x=m1_roc["fpr"],
            y=m1_roc["tpr"],
            mode="lines",
            line=dict(color=PRIMARY_BLUE, width=2.5),
            name=f"Model 1 (Baseline LR) — AUC = {m1['roc_auc']:.4f}",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    # Model 2
    m2 = eval_metrics["trustworthy_model"]
    m2_roc = m2["roc_curve"]
    fig.add_trace(
        go.Scatter(
            x=m2_roc["fpr"],
            y=m2_roc["tpr"],
            mode="lines",
            line=dict(color=ALERT_RED, dash="dot", width=2.5),
            name=f"Model 2 (Calibrated LR) — AUC = {m2['roc_auc']:.4f}",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Overlaid Receiver Operating Characteristic (ROC) Curves</b>",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        xaxis=dict(range=[-0.01, 1.01], gridcolor="#f1f5f9"),
        yaxis=dict(range=[-0.01, 1.01], gridcolor="#f1f5f9"),
        legend=dict(x=0.45, y=0.15, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
    )
    return fig


def plot_reliability_curve(df_cal: pd.DataFrame, summary_metrics: Dict[str, Any]) -> go.Figure:
    """Plots reliability curves comparing Model 1 and Model 2 against perfect calibration."""
    fig = go.Figure()

    # Perfect calibration reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
            name="Perfect Calibration (y = x)",
            hoverinfo="none",
        )
    )

    # Model 1
    m1_ece = summary_metrics["baseline_model"]["ece"]
    m1_brier = summary_metrics["baseline_model"]["brier_score"]
    fig.add_trace(
        go.Scatter(
            x=df_cal["baseline_mean_pred"],
            y=df_cal["baseline_frac_pos"],
            mode="lines+markers",
            marker=dict(size=8, color=PRIMARY_BLUE),
            line=dict(color=PRIMARY_BLUE, width=2),
            name=f"Model 1 (Baseline) — ECE={m1_ece:.4f}, Brier={m1_brier:.4f}",
            hovertemplate="Mean Pred: %{x:.3f}<br>Fraction Pos: %{y:.3f}<extra></extra>",
        )
    )

    # Model 2
    m2_ece = summary_metrics["trustworthy_model"]["ece"]
    m2_brier = summary_metrics["trustworthy_model"]["brier_score"]
    fig.add_trace(
        go.Scatter(
            x=df_cal["trustworthy_mean_pred"],
            y=df_cal["trustworthy_frac_pos"],
            mode="lines+markers",
            marker=dict(size=8, symbol="diamond", color=WARM_AMBER),
            line=dict(color=WARM_AMBER, width=2, dash="dash"),
            name=f"Model 2 (Calibrated) — ECE={m2_ece:.4f}, Brier={m2_brier:.4f}",
            hovertemplate="Mean Pred: %{x:.3f}<br>Fraction Pos: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Reliability Diagram (Calibration Curves, 10 Bins)</b>",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Empirical Fraction of Positives (Observed Frequency)",
        xaxis=dict(range=[-0.02, 1.02], gridcolor="#f1f5f9"),
        yaxis=dict(range=[-0.02, 1.02], gridcolor="#f1f5f9"),
        legend=dict(x=0.05, y=0.92, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
    )
    return fig


def plot_subgroup_fairness(
    group_metrics: List[Dict[str, Any]],
    attr_name: str,
    metric_keys: Optional[List[str]] = None,
) -> go.Figure:
    """Grouped bar chart for demographic subgroup audit metrics."""
    if metric_keys is None:
        metric_keys = ["tpr", "fpr", "selection_rate"]

    metric_names = {
        "tpr": "True Positive Rate (Sensitivity)",
        "fpr": "False Positive Rate",
        "fnr": "False Negative Rate (Miss Rate)",
        "selection_rate": "Selection Rate (Predicted Pos)",
        "accuracy": "Accuracy",
    }
    colors = {
        "tpr": SAFE_GREEN,
        "fpr": ALERT_RED,
        "fnr": WARM_AMBER,
        "selection_rate": PRIMARY_BLUE,
        "accuracy": ACCENT_CYAN,
    }

    groups = [f"{g['group_name']}<br>(N={g['n']:,})" for g in group_metrics]

    fig = go.Figure()
    for mk in metric_keys:
        vals = [g[mk] for g in group_metrics]
        fig.add_trace(
            go.Bar(
                name=metric_names.get(mk, mk),
                x=groups,
                y=vals,
                marker_color=colors.get(mk, MUTED_SLATE),
                text=[f"{v:.1%}" for v in vals],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="<b>%{x}</b><br>" + metric_names.get(mk, mk) + ": %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"<b>Fairness Subgroup Performance Audit: {attr_name}</b>",
        barmode="group",
        yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="#f1f5f9"),
        xaxis=dict(tickangle=-15 if len(groups) > 5 else 0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=50),
        height=420,
    )
    return fig


def plot_mitigation_comparison(mitigation_data: Dict[str, Any]) -> go.Figure:
    """Shows the fairness disparity vs accuracy trade-off before and after post-processing."""
    fig = go.Figure()

    categories = ["Equalized Odds Difference<br>(Disparity - Lower is Better)", "Overall Accuracy<br>(Utility - Higher is Better)"]
    before_vals = [
        mitigation_data["equalized_odds_diff_before"],
        mitigation_data["accuracy_before"],
    ]
    after_vals = [
        mitigation_data["equalized_odds_diff_after"],
        mitigation_data["accuracy_after"],
    ]

    fig.add_trace(
        go.Bar(
            name="Before Mitigation (Standard 0.50 Threshold)",
            x=categories,
            y=before_vals,
            marker_color=ALERT_RED,
            text=[f"{v:.4f}" for v in before_vals],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="After ThresholdOptimizer (Group-Specific Thresholds)",
            x=categories,
            y=after_vals,
            marker_color=SAFE_GREEN,
            text=[f"{v:.4f}" for v in after_vals],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Fairness Mitigation Trade-Off (Sensitive Attribute: Income)</b>",
        barmode="group",
        yaxis=dict(range=[0, 0.85], gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_global_shap(df_shap_global: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for global SHAP mean absolute value."""
    top_df = df_shap_global.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df["mean_abs_shap"],
            y=top_df["feature"],
            orientation="h",
            marker=dict(
                color=top_df["mean_abs_shap"],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in top_df["mean_abs_shap"]],
            textposition="outside",
            hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Global Feature Importance (Top {top_n} Features via Mean |SHAP|)</b>",
        xaxis_title="Mean |SHAP Value| (Average Impact on Model Log-Odds)",
        yaxis_title="Survey Feature",
        xaxis=dict(gridcolor="#f1f5f9"),
        margin=dict(l=140, r=40, t=50, b=40),
        height=460,
    )
    return fig


def plot_local_shap_horizontal(shap_values_row: pd.Series, max_display: int = 10) -> go.Figure:
    """
    Renders horizontal diverging bar chart for local case explanation.
    Features pushing towards diabetes (positive log-odds) in red/coral,
    features pushing towards no diabetes (negative log-odds) in blue/teal.
    """
    # Exclude case_id
    feats = [c for c in shap_values_row.index if c != "case_id"]
    vals = pd.Series({f: float(shap_values_row[f]) for f in feats})

    # Sort by absolute impact
    top_feats = vals.abs().sort_values(ascending=False).head(max_display).index
    plot_vals = vals[top_feats].sort_values(ascending=True)

    colors = [ALERT_RED if v > 0 else PRIMARY_BLUE for v in plot_vals.values]

    fig = go.Figure(
        go.Bar(
            x=plot_vals.values,
            y=plot_vals.index,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in plot_vals.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP attribution: %{x:+.4f} log-odds<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Local Feature Attributions (SHAP Log-Odds Contributions)</b>",
        xaxis_title="Decreases Diabetes Risk | Increases Diabetes Risk",
        yaxis_title="Feature",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black", gridcolor="#f1f5f9"),
        margin=dict(l=140, r=60, t=50, b=40),
        height=380,
    )
    return fig


def plot_robustness_shifts(df_rob_summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for perturbation impact: mean probability shift across models."""
    fig = go.Figure()

    perturbation_labels = {
        "bmi_plus1": "BMI (+1 unit)",
        "bmi_minus1": "BMI (-1 unit)",
        "genhlth_worse": "GenHlth (+1 worse)",
        "genhlth_better": "GenHlth (-1 better)",
        "physhlth_plus1": "PhysHlth (+1 day)",
        "menthlth_plus1": "MentHlth (+1 day)",
    }

    baseline_data = df_rob_summary[df_rob_summary["model"] == "Baseline"].copy()
    trustworthy_data = df_rob_summary[df_rob_summary["model"] == "Trust-Aware"].copy()

    # Map labels
    baseline_data["label"] = baseline_data["perturbation"].map(perturbation_labels)
    trustworthy_data["label"] = trustworthy_data["perturbation"].map(perturbation_labels)

    # Sort consistently
    order = list(perturbation_labels.values())
    baseline_data["sort_order"] = baseline_data["label"].map(lambda x: order.index(x) if x in order else 99)
    trustworthy_data["sort_order"] = trustworthy_data["label"].map(lambda x: order.index(x) if x in order else 99)
    baseline_data = baseline_data.sort_values("sort_order")
    trustworthy_data = trustworthy_data.sort_values("sort_order")

    fig.add_trace(
        go.Bar(
            name="Model 1 (Baseline LR)",
            x=baseline_data["label"],
            y=baseline_data["mean_abs_prob_change"],
            marker_color=PRIMARY_BLUE,
            text=[f"{v:.4f}" for v in baseline_data["mean_abs_prob_change"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Baseline Mean |Δp|: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Model 2 (Calibrated LR)",
            x=trustworthy_data["label"],
            y=trustworthy_data["mean_abs_prob_change"],
            marker_color=WARM_AMBER,
            text=[f"{v:.4f}" for v in trustworthy_data["mean_abs_prob_change"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Trust-Aware Mean |Δp|: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Mean Absolute Probability Change (|Δp|) under Controlled Perturbations</b>",
        barmode="group",
        xaxis_title="Perturbation Scenario",
        yaxis_title="Mean Absolute Probability Change (|Δp|)",
        yaxis=dict(gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig
