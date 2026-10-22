"""Visor de un concepto de DL: incrusta el HTML interactivo (carpeta htmls/) en un
iframe de una sola ventana y debajo muestra el vínculo con el proyecto real.

Los HTML se diseñaron como un escenario fijo 1280x720 que se auto-escala al iframe, así
que basta con darle ancho completo y una altura fija (sin scroll interno).
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from services.concepts_content import DLConcept

HTML_DIR = Path(__file__).resolve().parent.parent / "htmls"
IFRAME_HEIGHT = 720


def force_light_theme(html: str) -> str:
    """Los HTML siguen prefers-color-scheme; la app es tema claro fijo (config.toml),
    así que se fuerza data-theme="light" para que no se vean oscuros dentro de ella."""
    return re.sub(r"<html\b", '<html data-theme="light"', html, count=1)


@st.cache_data(show_spinner=False)
def load_concept_html(html_file: str, html_dir: str = str(HTML_DIR)) -> str | None:
    path = Path(html_dir) / html_file
    if not path.exists():
        return None
    return force_light_theme(path.read_text(encoding="utf-8"))


def render(concept: DLConcept, binding: str) -> None:
    html = load_concept_html(concept.html_file)
    if html is None:
        st.warning(
            f"La visualización interactiva «{concept.title}» no está disponible "
            f"(falta `htmls/{concept.html_file}`)."
        )
    else:
        components.html(html, height=IFRAME_HEIGHT, scrolling=False)

    st.caption(f"🔗 En este proyecto: {binding}")
