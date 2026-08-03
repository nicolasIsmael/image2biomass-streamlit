"""Tarjeta de concepto de DL: explicación + control interactivo + vínculo con el
proyecto real. La visualización reactiva específica de cada concepto la resuelve el
caller (pages/3_conceptos.py) vía `on_render_visualization`, para mantener este
componente genérico y reutilizable.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from services.concepts_content import DLConcept


def render(concept: DLConcept, on_render_visualization: Callable) -> None:
    st.subheader(concept.title)
    st.markdown(concept.explanation)

    value = _render_control(concept)

    st.caption(f"🔗 En este proyecto: {concept.real_project_binding}")
    on_render_visualization(value)


def _render_control(concept: DLConcept):
    key = f"control_{concept.concept_id}"

    if concept.control_type == "slider":
        min_v, max_v, step, default = concept.control_options
        return st.slider(concept.control_label, min_v, max_v, default, step, key=key)

    if concept.control_type == "select":
        return st.selectbox(concept.control_label, concept.control_options, key=key)

    if concept.control_type == "button":
        if key not in st.session_state:
            st.session_state[key] = concept.control_options[0]
        if st.button(f"{concept.control_label}: {st.session_state[key]}", key=f"btn_{key}"):
            options = concept.control_options
            next_index = (options.index(st.session_state[key]) + 1) % len(options)
            st.session_state[key] = options[next_index]
        return st.session_state[key]

    raise ValueError(f"Tipo de control no soportado: {concept.control_type}")
