"""Gráficos Plotly de la pestaña de comparación (rendimiento por fold y por variable).

Cada función recibe DataFrames/matrices reales ya agregadas por `services.comparison_stats`
y devuelve una figura; no carga datos ni usa layout de Streamlit. El color de cada
arquitectura es fijo (no depende de su ranking) para que no cambie al filtrar/ordenar.
"""

from __future__ import annotations

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from components.metric_chart import (
    CATEGORICAL_PALETTE,
    MUTED_INK,
    SEQUENTIAL_BLUE_DARK,
)

INK = "#1f1e1c"
GRID = "#e6e4df"
POSITIVE_BLUE = "#2a78d6"
NEGATIVE_ORANGE = "#eb6834"
NEUTRAL = "#f1f0ee"
DIVERGING = [[0.0, NEGATIVE_ORANGE], [0.5, NEUTRAL], [1.0, POSITIVE_BLUE]]
SEQUENTIAL = [[0.0, "#eaf2fc"], [1.0, SEQUENTIAL_BLUE_DARK]]
ZERO_NOTE = "R² = 0: equivale a predecir siempre el promedio"


def architecture_colors(architecture_ids: list[str]) -> dict[str, str]:
    """Color fijo por arquitectura, en el orden en que aparecen en los datos."""
    return {
        a: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        for i, a in enumerate(architecture_ids)
    }


def style_figure(
    fig: go.Figure, title: str, height: int = 430, top_margin: int = 60
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)),
        height=height,
        margin=dict(l=10, r=10, t=top_margin, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK),
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def _is_r2(metric: str) -> bool:
    return metric.lower().endswith("r2")


def _zero_line(fig: go.Figure, x_range: list) -> None:
    """Línea punteada en R² = 0, con entrada de leyenda (no tapa puntos con texto)."""
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[0, 0],
            mode="lines",
            name=ZERO_NOTE,
            line=dict(color=MUTED_INK, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )


def fold_strip_chart(
    fold_df: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    metric_label: str,
    colors: dict[str, str],
) -> go.Figure:
    """Un punto por fold + media ± SD (ddof=0) por arquitectura; `summary` fija el orden.

    Reemplaza las barras apiladas: la métrica no se suma entre folds, se compara su
    distribución. Cada fold tiene un desplazamiento fijo para que sus puntos no se tapen.
    """
    fig = go.Figure()
    folds = sorted(fold_df["fold_number"].unique())
    offsets = {f: -0.24 + 0.48 * i / max(len(folds) - 1, 1) for i, f in enumerate(folds)}
    for pos, arch in enumerate(summary["architecture_id"]):
        rows = fold_df[fold_df["architecture_id"] == arch]
        fig.add_trace(
            go.Scatter(
                x=[pos + offsets[f] for f in rows["fold_number"]],
                y=rows[metric],
                mode="markers",
                marker=dict(color=colors[arch], size=10, line=dict(color="white", width=2)),
                customdata=rows[["fold_number", "n_samples_val"]],
                hovertemplate=(
                    f"<b>{rows['variant_name'].iloc[0]}</b><br>Fold %{{customdata[0]}} "
                    f"(n=%{{customdata[1]}})<br>{metric_label}: %{{y:.4f}}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(summary))),
            y=summary["mean"],
            mode="markers",
            name="Media ± SD entre folds (ddof=0)",
            marker=dict(symbol="diamond", color=INK, size=11, line=dict(color="white", width=1.5)),
            error_y=dict(type="data", array=summary["sd"], color=INK, thickness=1.5, width=6),
            customdata=summary[["sd", "min", "max"]],
            hovertemplate=(
                "Media: %{y:.4f}<br>SD: %{customdata[0]:.4f}<br>Mín: %{customdata[1]:.4f} "
                "· Máx: %{customdata[2]:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(tickvals=list(range(len(summary))), ticktext=list(summary["variant_name"]))
    fig.update_yaxes(title_text=metric_label)
    if _is_r2(metric):
        _zero_line(fig, [-0.5, len(summary) - 0.5])
    return style_figure(fig, f"{metric_label} por arquitectura — un punto por fold")


def fold_paired_lines(
    fold_df: pd.DataFrame, metric: str, metric_label: str, colors: dict[str, str]
) -> go.Figure:
    """Una línea por arquitectura a lo largo de los folds: muestra el diseño pareado."""
    fig = go.Figure()
    sizes = fold_df.drop_duplicates("fold_number").set_index("fold_number")["n_samples_val"]
    ticks = {f: f"Fold {f}<br>(n={sizes[f]})" for f in sorted(sizes.index)}
    for arch, rows in fold_df.sort_values("fold_number").groupby("architecture_id", sort=False):
        fig.add_trace(
            go.Scatter(
                x=[ticks[f] for f in rows["fold_number"]],
                y=rows[metric],
                mode="lines+markers",
                name=rows["variant_name"].iloc[0],
                line=dict(color=colors[arch], width=2),
                marker=dict(size=8, line=dict(color="white", width=2)),
                hovertemplate=(
                    f"%{{x}}<br>{metric_label}: %{{y:.4f}}"
                    f"<extra>{rows['variant_name'].iloc[0]}</extra>"
                ),
            )
        )
    fig.update_xaxes(categoryorder="array", categoryarray=list(ticks.values()))
    fig.update_yaxes(title_text=metric_label)
    if _is_r2(metric):
        _zero_line(fig, [ticks[f] for f in (min(ticks), max(ticks))])
    return style_figure(fig, f"{metric_label} por fold — los 6 modelos ven los mismos datos")


def _text_color(colorscale: list, t: float) -> str:
    """Blanco sobre celdas oscuras, tinta sobre claras (contraste legible)."""
    r, g, b = pc.unlabel_rgb(pc.sample_colorscale(colorscale, [t])[0])
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 140 else INK


def annotated_heatmap(
    z_color: pd.DataFrame,
    z_text: pd.DataFrame,
    colorscale: list,
    title: str,
    labels: dict[str, str],
    column_labels: dict[str, str],
    hover_label: str,
    text_fmt: str = "{:.3f}",
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> go.Figure:
    """Heatmap con el valor real escrito en cada celda. `z_color` (ya normalizado a
    [zmin, zmax]) decide el color; `z_text` es lo que se lee, para no maquillar cifras."""
    y_names = [labels.get(i, i) for i in z_text.index]
    x_names = [column_labels.get(c, c) for c in z_text.columns]
    fig = go.Figure(
        go.Heatmap(
            z=z_color.values,
            x=x_names,
            y=y_names,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            xgap=2,
            ygap=2,
            customdata=z_text.values,
            hovertemplate=f"%{{y}} · %{{x}}<br>{hover_label}: %{{customdata:.4f}}<extra></extra>",
        )
    )
    for i, arch in enumerate(z_text.index):
        for j, col in enumerate(z_text.columns):
            value, shade = z_text.iloc[i, j], z_color.iloc[i, j]
            if pd.isna(value):
                continue
            t = (shade - zmin) / (zmax - zmin)
            fig.add_annotation(
                x=x_names[j], y=y_names[i], text=text_fmt.format(value), showarrow=False,
                font=dict(size=12, color=_text_color(colorscale, min(max(t, 0), 1))),
            )
    fig.update_yaxes(autorange="reversed", gridcolor="white")
    fig.update_xaxes(side="top")
    return style_figure(fig, title, height=380, top_margin=90)


def r2_heatmap(r2: pd.DataFrame, labels: dict, column_labels: dict) -> go.Figure:
    bound = float(max(abs(r2.values.min()), abs(r2.values.max())))
    return annotated_heatmap(
        r2, r2, DIVERGING, "R² medio por variable objetivo (azul = mejor que el promedio)",
        labels, column_labels, "R² medio", zmin=-bound, zmax=bound,
    )


def mae_heatmap(mae: pd.DataFrame, labels: dict, column_labels: dict) -> go.Figure:
    """Color normalizado por columna (cada variable tiene otra escala en gramos); el
    texto es el MAE real en g. Más oscuro = más error."""
    span = (mae.max() - mae.min()).replace(0, 1)
    shade = (mae - mae.min()) / span
    return annotated_heatmap(
        shade, mae, SEQUENTIAL,
        "Error absoluto medio (g) — más oscuro = más error dentro de cada variable",
        labels, column_labels, "MAE medio (g)", text_fmt="{:.2f}",
    )
