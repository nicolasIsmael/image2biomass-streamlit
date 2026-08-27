"""Wrapper Plotly parametrizable para visualizaciones reutilizables del recorrido.

Se mantiene genérico (recibe DataFrames y nombres de columna) para que cada página
solo pase sus propios datos reales, sin acoplar este componente a un dataset
específico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Paleta categórica validada (orden fijo, seguro para daltonismo) -- ver skill dataviz.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aqua
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # rojo
]
MUTED_INK = "#898781"
SEQUENTIAL_BLUE = "#2a78d6"
SEQUENTIAL_BLUE_DARK = "#104281"


def render(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def assign_categorical_colors(
    categories: list[str], other_prefix: str = "Otras"
) -> dict[str, str]:
    """Asigna colores del set categórico validado (orden fijo) a categorías ya
    ordenadas por importancia. Cualquier categoría que empiece con `other_prefix`
    (un cajón "Otras...") recibe el tono neutro en vez de un color de serie.
    """
    color_map: dict[str, str] = {}
    palette_cycle = iter(CATEGORICAL_PALETTE)
    for category in categories:
        if category.startswith(other_prefix):
            color_map[category] = MUTED_INK
        else:
            color_map[category] = next(palette_cycle, MUTED_INK)
    return color_map


def histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    nbins: int = 20,
    color: str = SEQUENTIAL_BLUE,
    marginal: str | None = None,
):
    fig = px.histogram(df, x=column, nbins=nbins, title=title, marginal=marginal)
    fig.update_traces(marker_color=color, marker_line_color="white", marker_line_width=0.5)
    return fig


def categorical_bar(
    df: pd.DataFrame,
    category_col: str,
    count_col: str,
    title: str,
    color_map: dict[str, str] | None = None,
    orientation: str = "h",
):
    """Barras de conteo por categoría, coloreadas por categoría (paleta fija)."""
    if orientation == "h":
        fig = px.bar(
            df,
            x=count_col,
            y=category_col,
            orientation="h",
            color=category_col,
            color_discrete_map=color_map,
            title=title,
            text=count_col,
        )
        fig.update_yaxes(categoryorder="total ascending")
    else:
        fig = px.bar(
            df,
            x=category_col,
            y=count_col,
            color=category_col,
            color_discrete_map=color_map,
            title=title,
            text=count_col,
        )
        fig.update_xaxes(categoryorder="total descending")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def grouped_boxplot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    color_map: dict[str, str] | None = None,
):
    fig = px.box(
        df,
        x=category_col,
        y=value_col,
        color=category_col,
        color_discrete_map=color_map,
        title=title,
    )
    fig.update_layout(showlegend=False)
    return fig


def scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    marker_color: str = SEQUENTIAL_BLUE,
    trend_color: str = SEQUENTIAL_BLUE_DARK,
):
    """Dispersión de dos variables reales + línea de tendencia (regresión lineal
    simple sobre los propios datos, no un valor inventado)."""
    clean = df[[x_col, y_col]].dropna()
    fig = px.scatter(clean, x=x_col, y=y_col, title=title, opacity=0.7)
    fig.update_traces(marker=dict(color=marker_color, size=8, line=dict(width=0)))
    if len(clean) >= 2:
        slope, intercept = np.polyfit(clean[x_col], clean[y_col], 1)
        x_range = np.linspace(clean[x_col].min(), clean[x_col].max(), 50)
        fig.add_scatter(
            x=x_range,
            y=slope * x_range + intercept,
            mode="lines",
            name="Tendencia (regresión lineal)",
            line=dict(color=trend_color, width=2),
        )
    return fig


def geo_scatter(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    title: str,
    hover_name: str | None = None,
):
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        hover_name=hover_name,
        zoom=3,
        height=450,
        title=title,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    return fig


def fold_comparison_bar(
    df: pd.DataFrame,
    architecture_col: str,
    metric_col: str,
    fold_col: str,
    title: str,
):
    """Barras agrupadas por arquitectura, una barra por fold (User Story 3)."""
    return px.bar(
        df,
        x=architecture_col,
        y=metric_col,
        color=fold_col,
        barmode="group",
        title=title,
    )
