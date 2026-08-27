"""Diagramas pedagógicos para la sección de Conceptos de Deep Learning.

A diferencia de `metric_chart.py` (que grafica DataFrames reales del proyecto), estas
funciones dibujan ilustraciones sintéticas de conceptos (curvas de pérdida, diagramas
de arquitectura). Se etiquetan siempre como simulación/ilustración pedagógica en la
página que las usa -- nunca se presentan como resultados reales (Principio VI).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from components.metric_chart import CATEGORICAL_PALETTE, MUTED_INK, SEQUENTIAL_BLUE

TRAIN_COLOR = SEQUENTIAL_BLUE
VAL_COLOR = CATEGORICAL_PALETTE[1]
HIGHLIGHT_COLOR = CATEGORICAL_PALETTE[7]
TRAIN_SEGMENT_COLOR = "#e1e0d9"


def epoch_curve_chart(max_epoch: int, current_epoch: int) -> tuple[go.Figure, float, float]:
    """Curvas de pérdida de entrenamiento/validación (forma en U) + marcador en la
    época elegida. Devuelve la figura y los valores de pérdida en ese punto, para que
    el caller arme un mensaje corto sobre sub/sobreajuste."""
    epochs = np.arange(1, max_epoch + 1)
    train_loss = 9.5 * np.exp(-0.12 * epochs) + 0.25
    val_loss = 9.5 * np.exp(-0.10 * epochs) + 0.35 + 0.015 * np.clip(epochs - 18, 0, None) ** 1.4

    idx = max(0, min(current_epoch - 1, max_epoch - 1))

    fig = go.Figure()
    fig.add_scatter(
        x=epochs, y=train_loss, mode="lines", name="Entrenamiento",
        line=dict(color=TRAIN_COLOR, width=2.5),
    )
    fig.add_scatter(
        x=epochs, y=val_loss, mode="lines", name="Validación",
        line=dict(color=VAL_COLOR, width=2.5),
    )
    fig.add_vline(x=current_epoch, line_dash="dash", line_color=MUTED_INK)
    fig.add_scatter(
        x=[current_epoch, current_epoch],
        y=[train_loss[idx], val_loss[idx]],
        mode="markers",
        marker=dict(size=10, color=[TRAIN_COLOR, VAL_COLOR], line=dict(color="white", width=1)),
        showlegend=False,
        hoverinfo="skip",
    )
    fig.update_layout(
        title="Pérdida simulada por época",
        xaxis_title="Época",
        yaxis_title="Pérdida (simulación pedagógica)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
    )
    return fig, float(train_loss[idx]), float(val_loss[idx])


def fold_strip_chart(fold_sizes: list[int], total: int, selected_fold: int) -> go.Figure:
    """Franja de 5 folds: cada fila es 100% del dataset partido en validación
    (tamaño real) + entrenamiento; el fold elegido se resalta."""
    fold_labels = [f"Fold {i}" for i in range(1, len(fold_sizes) + 1)]
    train_sizes = [total - v for v in fold_sizes]
    val_colors = [
        HIGHLIGHT_COLOR if (i + 1) == selected_fold else MUTED_INK
        for i in range(len(fold_sizes))
    ]

    fig = go.Figure()
    fig.add_bar(
        y=fold_labels, x=fold_sizes, name="Validación", orientation="h",
        marker=dict(color=val_colors), text=fold_sizes, textposition="inside",
        insidetextfont=dict(color="white"),
    )
    fig.add_bar(
        y=fold_labels, x=train_sizes, name="Entrenamiento", orientation="h",
        marker=dict(color=TRAIN_SEGMENT_COLOR), text=train_sizes, textposition="inside",
        insidetextfont=dict(color="#52514e"),
    )
    fig.update_layout(
        barmode="stack",
        title="Partición real de cada fold (muestras)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
    )
    return fig


def connectivity_diagram(family: str) -> go.Figure:
    """Qué parches de una imagen 've' un parche central, según la familia de
    arquitectura: vecindario local (CNN), todos los parches (Transformer), o
    vecindario local + algunas conexiones de largo alcance (Híbrido). Es una
    simplificación ilustrativa del principio, no un mapa de atención real."""
    grid = [(c, r) for r in range(4) for c in range(4)]
    center = (1, 1)

    def local_neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
        c, r = pt
        return [
            (c + dc, r + dr)
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if not (dc == 0 and dr == 0) and 0 <= c + dc < 4 and 0 <= r + dr < 4
        ]

    if family == "CNN":
        edges = local_neighbors(center)
        subtitle = 'Cada filtro solo "ve" un vecindario local de parches cercanos.'
    elif family == "Transformer":
        edges = [p for p in grid if p != center]
        subtitle = "La atención relaciona el parche con todos los demás a la vez."
    else:
        edges = local_neighbors(center) + [(3, 3), (3, 0), (0, 3)]
        subtitle = "Combina vecindario local (como una CNN) con conexiones de largo alcance."

    xs: list[float | None] = []
    ys: list[float | None] = []
    for c, r in edges:
        xs += [center[0], c, None]
        ys += [center[1], r, None]

    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=ys, mode="lines", line=dict(color=SEQUENTIAL_BLUE, width=1.5),
        opacity=0.55, showlegend=False, hoverinfo="skip",
    )
    fig.add_scatter(
        x=[p[0] for p in grid], y=[p[1] for p in grid], mode="markers",
        marker=dict(size=26, color="#e1e0d9", line=dict(color=MUTED_INK, width=1), symbol="square"),
        showlegend=False, hoverinfo="skip",
    )
    fig.add_scatter(
        x=[center[0]], y=[center[1]], mode="markers",
        marker=dict(
            size=30, color=CATEGORICAL_PALETTE[1], line=dict(color="white", width=2),
            symbol="square",
        ),
        showlegend=False, hoverinfo="skip",
    )
    fig.update_xaxes(visible=False, range=[-0.6, 3.6])
    fig.update_yaxes(visible=False, range=[-0.6, 3.6], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f'Qué "ve" un parche en una arquitectura {family}',
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
        annotations=[
            dict(
                text=subtitle, xref="paper", yref="paper", x=0.5, y=-0.12,
                showarrow=False, font=dict(size=12, color=MUTED_INK),
            )
        ],
    )
    return fig


def pipeline_diagram(
    backbone_frozen: bool,
    output_labels: list[str],
    highlighted_output: str | None = None,
) -> go.Figure:
    """Diagrama imagen -> backbone -> cabeza de regresión -> salidas. Reutilizado por
    los conceptos de backbone congelado (resalta el backbone) y cabeza de regresión
    (resalta la salida elegida)."""
    fig = go.Figure()

    def add_box(x0, x1, y0, y1, text, fillcolor, linecolor, textcolor="#0b0b0b"):
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=fillcolor, line=dict(color=linecolor, width=2),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text, showarrow=False,
            font=dict(size=12, color=textcolor),
        )

    add_box(0.0, 1.0, -0.4, 0.4, "🖼️ Imagen", "#f9f9f7", MUTED_INK)

    if backbone_frozen:
        add_box(1.5, 3.0, -0.4, 0.4, "🔒 Backbone<br>(congelado)", "#e1e0d9", MUTED_INK)
    else:
        add_box(
            1.5, 3.0, -0.4, 0.4, "🔓 Backbone<br>(entrenable)", "#fde2d3", CATEGORICAL_PALETTE[1]
        )

    add_box(3.5, 5.0, -0.4, 0.4, "Cabeza de<br>regresión", "#dbe8fb", SEQUENTIAL_BLUE)

    for x0, x1 in [(1.0, 1.5), (3.0, 3.5)]:
        fig.add_annotation(
            x=x1, y=0, ax=x0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=MUTED_INK,
        )

    n = len(output_labels)
    y_positions = [((n - 1) / 2 - i) * 0.9 for i in range(n)]
    if y_positions:
        fig.add_shape(
            type="line", x0=5.0, x1=5.0, y0=min(y_positions), y1=max(y_positions),
            line=dict(color=MUTED_INK, width=1),
        )
    for label, y in zip(output_labels, y_positions):
        fig.add_shape(type="line", x0=5.0, x1=5.5, y0=y, y1=y, line=dict(color=MUTED_INK, width=1))
        is_highlighted = highlighted_output == label
        fill = HIGHLIGHT_COLOR if is_highlighted else "#f3f2ee"
        line_color = "#8a1f1f" if is_highlighted else MUTED_INK
        text_color = "white" if is_highlighted else "#0b0b0b"
        add_box(5.5, 6.8, y - 0.3, y + 0.3, label, fill, line_color, text_color)

    fig.update_xaxes(visible=False, range=[-0.3, 7.2])
    fig.update_yaxes(visible=False, range=[-2.0, 2.0])
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white")
    return fig
