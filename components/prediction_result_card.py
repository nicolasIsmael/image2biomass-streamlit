"""Tarjeta que muestra, lado a lado, la predicción del modelo de mejor y de peor
desempeño para una misma imagen (User Story 4).
"""

from __future__ import annotations

import streamlit as st

from services.inference import PredictionOutput


def render(
    best: PredictionOutput,
    worst: PredictionOutput,
    best_label: str,
    worst_label: str,
) -> None:
    col_best, col_worst = st.columns(2)

    with col_best:
        st.markdown(f"### 🏆 {best_label}")
        _render_values(best.predicted_values)

    with col_worst:
        st.markdown(f"### 📉 {worst_label}")
        _render_values(worst.predicted_values)


def _render_values(values: dict[str, float]) -> None:
    for target, value in values.items():
        st.metric(target, f"{value:.2f}")
