"""Wrapper Plotly parametrizable para visualizaciones reutilizables del recorrido.

Se mantiene genérico (recibe DataFrames y nombres de columna) para que cada página
solo pase sus propios datos reales, sin acoplar este componente a un dataset
específico.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render(fig) -> None:
    st.plotly_chart(fig, use_container_width=True)


def histogram(df: pd.DataFrame, column: str, title: str, nbins: int = 20):
    return px.histogram(df, x=column, nbins=nbins, title=title)


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
