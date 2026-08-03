"""Smoke test de cada página: corre el script con streamlit.testing.v1.AppTest y
confirma que no lanza excepciones sin manejar, incluso sin artefactos reales de datos
disponibles (deben degradar a un estado "no disponible", no fallar).
"""

import pytest
from streamlit.testing.v1 import AppTest

PAGE_PATHS = [
    "pages/1_portada.py",
    "pages/2_dataset.py",
    "pages/3_conceptos.py",
    "pages/4_comparacion.py",
    "pages/5_prediccion_en_vivo.py",
    "pages/6_conclusiones.py",
]


@pytest.mark.parametrize("page_path", PAGE_PATHS)
def test_page_runs_without_unhandled_exceptions(page_path):
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
