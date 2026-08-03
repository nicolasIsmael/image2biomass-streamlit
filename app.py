"""Punto de entrada de la app. Define la navegación multipage explícita
(st.navigation/st.Page) en el orden narrativo del recorrido de sustentación.
"""

import streamlit as st

from components.progress_indicator import PAGES

st.set_page_config(
    page_title="Biomasa de Pasturas — Recorrido de Tesis",
    page_icon="🌱",
    layout="wide",
)

nav_pages = [st.Page(page["path"], title=page["title"], icon=page["icon"]) for page in PAGES]
navigation = st.navigation(nav_pages, position="sidebar")
navigation.run()
