"""Indicador de progreso/ubicación del recorrido (Principio V de la constitución).

Fuente única de verdad del orden narrativo de las páginas: app.py construye la
navegación (st.navigation/st.Page) a partir de PAGES, y cada página llama a render()
con su propia key para mostrar en qué punto del recorrido está el usuario.
"""

import streamlit as st

PAGES = [
    {"key": "portada", "title": "Introducción", "icon": "📋", "path": "pages/1_portada.py"},
    {"key": "dataset", "title": "Dataset", "icon": "📊", "path": "pages/2_dataset.py"},
    {"key": "conceptos", "title": "Conceptos DL", "icon": "🧠", "path": "pages/3_conceptos.py"},
    {"key": "comparacion", "title": "Comparación", "icon": "⚖️", "path": "pages/4_comparacion.py"},
    {
        "key": "prediccion",
        "title": "Predicción en vivo",
        "icon": "🔮",
        "path": "pages/5_prediccion_en_vivo.py",
    },
    {
        "key": "conclusiones",
        "title": "Conclusiones",
        "icon": "🎓",
        "path": "pages/6_conclusiones.py",
    },
]

_ACTIVE_COLOR = "#4CAF50"
_INACTIVE_COLOR = "#DDDDDD"


def render(current_key: str) -> None:
    """Renderiza una barra de pasos horizontal marcando la página activa.

    La navegación en sí (para saltar entre páginas) la provee la sidebar nativa de
    st.navigation; este componente es solo el indicador visual de ubicación.
    """
    columns = st.columns(len(PAGES))
    for column, page in zip(columns, PAGES):
        is_active = page["key"] == current_key
        color = _ACTIVE_COLOR if is_active else _INACTIVE_COLOR
        weight = "700" if is_active else "400"
        with column:
            st.markdown(
                f"<div style='text-align:center; padding:6px 2px; "
                f"border-bottom:3px solid {color}; font-weight:{weight}; "
                f"font-size:0.85rem;'>{page['icon']}<br/>{page['title']}</div>",
                unsafe_allow_html=True,
            )
    st.divider()
