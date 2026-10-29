"""Gráficos de significancia por pares y de costo computacional (pestañas 3 y 4)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from components.comparison_charts import (
    GRID,
    INK,
    POSITIVE_BLUE,
    annotated_heatmap,
    style_figure,
)
from components.metric_chart import MUTED_INK

SIGNIFICANCE_SCALE = [[0.0, "#eceae5"], [1.0, POSITIVE_BLUE]]


def pairwise_heatmap(
    p_values: pd.DataFrame,
    significant: pd.DataFrame,
    title: str,
    labels: dict[str, str],
    hover_label: str,
) -> go.Figure:
    """Matriz de p-values por pares; azul = significativo (α=0.05), gris = no.

    El color solo marca la decisión; el número escrito es el p-value real.
    """
    shade = significant.astype(float).where(p_values.notna())
    return annotated_heatmap(
        shade, p_values, SIGNIFICANCE_SCALE, title, labels, labels, hover_label,
    )


def cost_bars(
    df: pd.DataFrame,
    relative_col: str,
    title: str,
    labels: dict[str, str],
) -> go.Figure:
    """Costo relativo al más rápido (1.00×).

    Solo se muestra el factor: el hardware de Colab no es fijo y la CPU local es ruidosa,
    así que el valor absoluto no es portable, el factor entre arquitecturas sí.
    """
    names = [labels.get(a, a) for a in df["architecture_id"]]
    fig = go.Figure(
        go.Bar(
            x=df[relative_col],
            y=names,
            orientation="h",
            marker=dict(color=POSITIVE_BLUE),
            text=[f"{r:.2f}×" for r in df[relative_col]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.2f}× el más rápido<extra></extra>",
        )
    )
    fig.add_vline(x=1, line=dict(color=MUTED_INK, width=1, dash="dash"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(
        title_text="Factor relativo al más rápido (1.00× = referencia)",
        range=[0, float(df[relative_col].max()) * 1.35],
        gridcolor=GRID,
        showgrid=True,
    )
    fig.update_layout(bargap=0.35, font=dict(color=INK))
    return style_figure(fig, title, height=360)
