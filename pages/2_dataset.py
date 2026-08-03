import streamlit as st

from components.metric_chart import geo_scatter, histogram, render
from components.progress_indicator import render as render_progress
from services.dataset_loader import load_dataset

render_progress(current_key="dataset")

st.title("📊 El dataset: CSIRO Image2Biomass")

st.caption(
    "Esta sección combina claridad pedagógica con datos reales del estudio "
    "(Principio I): el texto es accesible, pero cada número proviene del "
    "artefacto real del dataset."
)

st.markdown(
    """
El dataset usado en este proyecto proviene de **CSIRO Image2Biomass** (Liao et al.,
Kaggle 2025): fotografías top-view de pasturas, cada una emparejada con mediciones
reales de biomasa por componente (ej. pasto verde, trébol, material muerto).
"""
)

result = load_dataset()

if not result.available:
    st.warning(
        "🔧 **Espacio preparado, artefacto pendiente.** Esta sección mostrará el "
        "conteo real de muestras y su distribución en cuanto `data/dataset_metadata.csv` "
        "esté disponible en el repositorio. No se muestran datos simulados "
        f"(Principio VI). Detalle técnico: {result.reason}"
    )
else:
    df = result.samples
    st.metric("Muestras en el dataset", len(df))

    if {"latitude", "longitude"}.issubset(df.columns):
        st.subheader("Distribución geográfica de las muestras")
        render(geo_scatter(df, "latitude", "longitude", "Ubicación de las muestras"))

    if result.target_columns:
        st.subheader("Distribución de las variables objetivo")
        chosen = st.selectbox("Variable objetivo a explorar", result.target_columns)
        render(histogram(df, chosen, f"Distribución de {chosen}"))
