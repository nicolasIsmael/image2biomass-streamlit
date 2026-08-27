from pathlib import Path

import streamlit as st

from components.metric_chart import histogram, render
from components.progress_indicator import render as render_progress
from services.dataset_loader import load_dataset

THUMBNAILS_DIR = Path("data/dataset_thumbnails")
GALLERY_COLUMNS = 6
DEFAULT_GALLERY_LIMIT = 24

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

    st.subheader("Galería del dataset")
    if not THUMBNAILS_DIR.is_dir() or not any(THUMBNAILS_DIR.glob("*.jpg")):
        st.info(
            "🔧 **Espacio preparado, artefacto pendiente.** Esta galería mostrará "
            "miniaturas reales de las 357 muestras en cuanto `data/dataset_thumbnails/` "
            "esté disponible. No se muestran imágenes simuladas (Principio VI)."
        )
    else:
        st.caption(
            "Miniaturas comprimidas de las 357 fotografías reales de entrenamiento "
            "(la predicción en vivo usa las imágenes en resolución completa). Cada "
            "una muestra su `Dry_Total_g` real medido."
        )
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            states_chosen = st.multiselect(
                "Filtrar por estado", sorted(df["State"].dropna().unique())
            )
        with filter_col2:
            species_chosen = st.multiselect(
                "Filtrar por especie", sorted(df["Species"].dropna().unique())
            )

        gallery_df = df
        if states_chosen:
            gallery_df = gallery_df[gallery_df["State"].isin(states_chosen)]
        if species_chosen:
            gallery_df = gallery_df[gallery_df["Species"].isin(species_chosen)]

        show_all = st.checkbox(
            f"Mostrar las {len(gallery_df)} muestras que cumplen el filtro "
            "(puede tardar más en cargar)"
        )
        limit = len(gallery_df) if show_all else min(DEFAULT_GALLERY_LIMIT, len(gallery_df))
        gallery_df = gallery_df.head(limit)

        st.caption(f"Mostrando {limit} miniatura(s).")
        cols = st.columns(GALLERY_COLUMNS)
        for i, row in enumerate(gallery_df.itertuples()):
            thumb_path = THUMBNAILS_DIR / f"{row.sample_id}.jpg"
            if not thumb_path.exists():
                continue
            with cols[i % GALLERY_COLUMNS]:
                st.image(
                    str(thumb_path),
                    caption=f"{row.sample_id} · {row.Dry_Total_g:.1f} g",
                    width="stretch",
                )

    if "State" in df.columns:
        st.subheader("Distribución de las muestras por estado")
        st.caption(
            "El dataset real no incluye coordenadas lat/long, sino el estado "
            "australiano donde se tomó cada muestra."
        )
        render(histogram(df, "State", "Muestras por estado"))

    if "Species" in df.columns:
        st.subheader("Distribución de las muestras por especie/mezcla forrajera")
        render(histogram(df, "Species", "Muestras por especie"))

    if result.target_columns:
        st.subheader("Distribución de las variables objetivo")
        chosen = st.selectbox("Variable objetivo a explorar", result.target_columns)
        render(histogram(df, chosen, f"Distribución de {chosen}"))
