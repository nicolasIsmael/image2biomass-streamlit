import streamlit as st

from components.limitations_panel import (
    STATIC_QUALITATIVE_LIMITATIONS,
    build_fold_imbalance_limitation,
)
from components.limitations_panel import render as render_limitations
from components.progress_indicator import render as render_progress
from services.results_loader import load_results

render_progress(current_key="conclusiones")

st.title("🎓 Conclusiones")

st.caption(
    "Esta sección prioriza **claridad pedagógica** (Principio I): un cierre que "
    "conecta lo recorrido con el paper de la tesis."
)

st.markdown(
    """
### Lo que recorriste

- **El problema**: estimar biomasa de pasturas a partir de una sola foto top-view,
  en vez de un método destructivo y lento.
- **Los conceptos**: qué es una época, por qué se usa validación cruzada, y la
  diferencia entre una CNN, un Transformer y un modelo Híbrido.
- **La comparación**: cuál de las 6 arquitecturas reales tuvo mejor desempeño, y si
  esa diferencia es estadísticamente real o pudo haber sido casualidad.
- **La demo en vivo**: viste, con tu propia imagen, la diferencia práctica entre el
  modelo de mejor y de peor desempeño.

### Cómo conectar esto con el paper de la tesis

Cada número que viste en la sección de Comparación (métricas por fold, p-value,
significancia estadística) es el mismo que vas a encontrar citado en el documento
escrito de la tesis — no es un resumen simplificado con otros valores. Si al leer el
paper te encontrás con un término que no recordás (época, fold, backbone), podés
volver a la sección de Conceptos en cualquier momento: la navegación es libre.
"""
)

results = load_results()
fold_limitation = build_fold_imbalance_limitation(results)
limitations = ([fold_limitation] if fold_limitation else []) + STATIC_QUALITATIVE_LIMITATIONS
render_limitations(limitations, expander_label="📎 Limitaciones conocidas del estudio (resumen)")
