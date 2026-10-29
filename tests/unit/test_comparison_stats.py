import pandas as pd
import pytest

from components.comparison_charts import architecture_colors, fold_strip_chart
from services.comparison_stats import (
    architecture_summary,
    higher_is_better,
    metric_label,
    pairwise_matrix,
    per_target_matrix,
)


@pytest.fixture(scope="module")
def real_fold_df():
    return pd.read_csv("data/fold_results.csv")


def test_summary_reproduces_paper_table3(real_fold_df):
    summary = architecture_summary(real_fold_df).set_index("architecture_id")

    assert list(summary.index)[0] == "convnext_tiny"
    assert list(summary.index)[-1] == "resnet50"
    assert summary.loc["convnext_tiny", "mean"] == pytest.approx(0.339, abs=5e-4)
    assert summary.loc["convnext_tiny", "sd"] == pytest.approx(0.099, abs=5e-4)
    assert summary.loc["resnet50", "mean"] == pytest.approx(-0.133, abs=5e-4)
    assert summary.loc["resnet50", "sd"] == pytest.approx(0.156, abs=5e-4)


def test_error_metrics_rank_lower_first(real_fold_df):
    assert higher_is_better("weighted_r2") and not higher_is_better("GDM_g_mae")
    summary = architecture_summary(real_fold_df, "Dry_Total_g_mae")

    assert summary["mean"].is_monotonic_increasing


def test_per_target_matrix_matches_paper_table3(real_fold_df):
    order = architecture_summary(real_fold_df)["architecture_id"].tolist()
    r2 = per_target_matrix(real_fold_df, "r2", order)

    assert list(r2.index) == order
    assert r2.loc["convnext_tiny", "Dry_Green_g"] == pytest.approx(0.442, abs=5e-4)


def test_pairwise_matrix_is_symmetric_with_empty_diagonal():
    df = pd.DataFrame({"architecture_a": ["a"], "architecture_b": ["b"], "p_holm": [0.04]})
    matrix = pairwise_matrix(df, ["a", "b"], "p_holm")

    assert matrix.loc["a", "b"] == matrix.loc["b", "a"] == 0.04
    assert pd.isna(matrix.loc["a", "a"])


def test_metric_label():
    assert metric_label("weighted_r2") == "R² ponderado"
    assert metric_label("GDM_g_mae") == "Materia seca verde (GDM) · MAE"


def test_strip_chart_never_stacks_folds(real_fold_df):
    """Regresión: las barras apiladas sumaban el R² de los 5 folds (~1.7 para ConvNeXt)."""
    summary = architecture_summary(real_fold_df)
    colors = architecture_colors(list(summary["architecture_id"]))
    fig = fold_strip_chart(real_fold_df, summary, "weighted_r2", "R² ponderado", colors)

    plotted = [v for trace in fig.data for v in trace.y]
    assert max(plotted) == pytest.approx(real_fold_df["weighted_r2"].max())
    assert min(plotted) == pytest.approx(real_fold_df["weighted_r2"].min())
    assert all(trace.type == "scatter" for trace in fig.data)
