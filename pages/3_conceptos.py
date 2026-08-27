import streamlit as st

from components.concept_card import render as render_concept
from components.dl_diagrams import (
    connectivity_diagram,
    epoch_curve_chart,
    fold_strip_chart,
    pipeline_diagram,
)
from components.metric_chart import render as render_chart
from components.progress_indicator import render as render_progress
from services.architecture_registry import load_architecture_registry
from services.concepts_content import CONCEPTS
from services.dataset_loader import load_dataset
from services.results_loader import load_results

FAMILY_LABEL_TO_REAL = {"CNN": "CNN", "Transformer": "Transformer", "Híbrido": "Hybrid"}
REGRESSION_OUTPUTS = ["Green", "Clover", "Dead"]

render_progress(current_key="conceptos")

st.title("🧠 Conceptos de Deep Learning")
st.caption(
    "Esta sección prioriza **claridad pedagógica** (Principio I): cada concepto se "
    "explica en lenguaje simple y se conecta con los datos/hechos reales del "
    "proyecto cuando están disponibles."
)

dataset_result = load_dataset()
architecture_result = load_architecture_registry()
results_data = load_results()


def _visualize(concept, value) -> None:
    if concept.concept_id == "epoca":
        fig, train_now, val_now = epoch_curve_chart(max_epoch=50, current_epoch=value)
        render_chart(fig)
        gap = val_now - train_now
        if gap < 0.5:
            msg = "entrenamiento y validación siguen juntas: el modelo todavía está aprendiendo."
        elif gap < 2:
            msg = "empieza a abrirse una brecha entre entrenamiento y validación: zona razonable."
        else:
            msg = "la brecha entre entrenamiento y validación ya es grande: señal de sobreajuste."
        st.caption(f"En la época {value}: {msg}")

    elif concept.concept_id == "validacion_cruzada":
        if results_data.available:
            fold_df = results_data.fold_results
            fold_sizes = fold_df.groupby("fold_number")["n_samples_val"].first().sort_index()
            total = int(fold_sizes.sum())
            n_architectures = fold_df["architecture_id"].nunique()
            n_folds = fold_df["fold_number"].nunique()
            render_chart(fold_strip_chart(fold_sizes.tolist(), total, selected_fold=value))
            st.write(
                f"Fold {value} de 5 usaría **{int(fold_sizes.loc[value])} de las {total} "
                "muestras reales** como validación (el resto, entrenamiento). En total: "
                f"**{n_architectures} arquitecturas × {n_folds} folds = {len(fold_df)} "
                "entrenamientos reales**."
            )
        else:
            st.info(
                "El tamaño real de cada fold se mostrará cuando `data/fold_results.csv` "
                "esté disponible."
            )

    elif concept.concept_id == "cnn_vs_transformer":
        render_chart(connectivity_diagram(value))
        if architecture_result.available:
            real_family = FAMILY_LABEL_TO_REAL.get(value, value)
            matches = [a for a in architecture_result.architectures if a.family == real_family]
            names = ", ".join(a.variant_name for a in matches) if matches else "ninguna"
            st.write(
                f"{len(matches)} de las 6 arquitecturas reales pertenecen a "
                f"**{value}**: {names}."
            )
        else:
            st.info(
                "La cantidad real de arquitecturas por familia se mostrará cuando los "
                "resultados estén disponibles."
            )

    elif concept.concept_id == "backbone_congelado":
        frozen = value == "Congelado"
        render_chart(pipeline_diagram(backbone_frozen=frozen, output_labels=REGRESSION_OUTPUTS))
        st.write(f"Estado seleccionado: **{value}**.")

    elif concept.concept_id == "cabeza_regresion":
        render_chart(
            pipeline_diagram(
                backbone_frozen=True, output_labels=REGRESSION_OUTPUTS, highlighted_output=value
            )
        )
        others = [o for o in REGRESSION_OUTPUTS if o != value]
        st.write(
            f"El modelo estimaría **{value}** junto con {' y '.join(others)}, "
            "en una sola pasada."
        )


for concept in CONCEPTS:
    render_concept(concept, lambda value, c=concept: _visualize(c, value))
    st.divider()
