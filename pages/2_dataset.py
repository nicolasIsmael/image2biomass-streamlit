from pathlib import Path

import streamlit as st

from components.metric_chart import (
    assign_categorical_colors,
    categorical_bar,
    grouped_boxplot,
    histogram,
    render,
    scatter_with_trend,
)
from components.progress_indicator import render as render_progress
from services.dataset_loader import load_dataset, species_component_counts

TARGET_LABELS_ORDER = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
RELATION_FEATURES = {
    "Altura promedio de la pastura (Height_Ave_cm)": "Height_Ave_cm",
    "Índice NDVI pre-corte (Pre_GSHH_NDVI)": "Pre_GSHH_NDVI",
}

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

    feature_options = [
        label
        for label, present in (
            ("Estado", "State" in df.columns),
            ("Especie/mezcla forrajera", "Species" in df.columns),
        )
        if present
    ]
    if feature_options:
        st.subheader("Composición del dataset")
        feature = st.radio("Explorar por", feature_options, horizontal=True)

        if feature == "Estado":
            st.caption(
                "El dataset real no incluye coordenadas lat/long, sino el estado "
                "australiano donde se tomó cada muestra."
            )
            counts = (
                df["State"].dropna().value_counts().rename_axis("Estado").reset_index(name="count")
            )
            color_map = assign_categorical_colors(counts["Estado"].tolist())
            render(
                categorical_bar(counts, "Estado", "count", "Muestras por estado", color_map)
            )
        else:
            st.caption(
                "Muchos valores de `Species` son en realidad mezclas forrajeras "
                "(ej. `Ryegrass_Clover`). En vez de graficar cada combinación como una "
                "categoría opaca, esta vista descompone cada mezcla en las especies "
                "individuales que la componen y cuenta cuántas muestras incluye cada "
                "una -- así se ve qué hay realmente en el dataset."
            )
            species_counts = species_component_counts(df, top_n=8)
            color_map = assign_categorical_colors(species_counts["Especie"].tolist())
            render(
                categorical_bar(
                    species_counts,
                    "Especie",
                    "count",
                    "Apariciones por especie (mezclas descompuestas)",
                    color_map,
                )
            )
            with st.expander("Ver las mezclas exactas tal como aparecen en el dataset"):
                raw_counts = (
                    df["Species"]
                    .dropna()
                    .value_counts()
                    .rename_axis("Mezcla")
                    .reset_index(name="count")
                )
                render(
                    categorical_bar(
                        raw_counts,
                        "Mezcla",
                        "count",
                        f"{len(raw_counts)} mezclas únicas registradas",
                        color_map={m: "#898781" for m in raw_counts["Mezcla"]},
                    )
                )

    if result.target_columns:
        st.subheader("Variables objetivo: componentes reales de la biomasa")
        st.caption(
            "El estudio mide 5 variables objetivo por muestra: 3 componentes de "
            "biomasa seca (verde, trébol, material muerto) y 2 agregados (GDM y "
            "total)."
        )

        ordered_targets = [c for c in TARGET_LABELS_ORDER if c in result.target_columns]
        ordered_targets += [c for c in result.target_columns if c not in ordered_targets]

        long_df = df.melt(
            id_vars=["sample_id"],
            value_vars=ordered_targets,
            var_name="Variable",
            value_name="Gramos",
        )
        box_color_map = assign_categorical_colors(ordered_targets)
        render(
            grouped_boxplot(
                long_df,
                "Variable",
                "Gramos",
                "Comparación de las 5 variables objetivo",
                color_map=box_color_map,
            )
        )

        chosen = st.selectbox("Explorar una variable en detalle", ordered_targets)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Media", f"{df[chosen].mean():.1f} g")
        metric_col2.metric("Mediana", f"{df[chosen].median():.1f} g")
        metric_col3.metric("Máximo", f"{df[chosen].max():.1f} g")
        render(
            histogram(
                df, chosen, f"Distribución de {chosen}", marginal="box",
                color=box_color_map.get(chosen),
            )
        )

    available_relation_features = {
        label: col for label, col in RELATION_FEATURES.items() if col in df.columns
    }
    if available_relation_features and "Dry_Total_g" in df.columns:
        st.subheader("¿Alcanza con altura o NDVI para estimar la biomasa?")
        st.caption(
            "Estas son variables tabulares simples, fáciles de medir en campo. La "
            "dispersión real (y qué tan lejos caen los puntos de la línea de "
            "tendencia) ayuda a entender por qué el proyecto usa un modelo de "
            "imagen en vez de una fórmula directa sobre estas variables."
        )
        relation_label = st.selectbox(
            "Variable a comparar con Dry_Total_g", list(available_relation_features.keys())
        )
        relation_col = available_relation_features[relation_label]
        corr = df[[relation_col, "Dry_Total_g"]].dropna().corr().iloc[0, 1]
        st.metric("Correlación de Pearson con Dry_Total_g", f"{corr:.2f}")
        render(
            scatter_with_trend(
                df, relation_col, "Dry_Total_g", f"{relation_col} vs. Dry_Total_g"
            )
        )
