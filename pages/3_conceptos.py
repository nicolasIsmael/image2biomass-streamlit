import streamlit as st

from components.concept_card import render as render_concept
from components.progress_indicator import render as render_progress
from services.architecture_registry import load_architecture_registry
from services.concepts_content import CONCEPTS
from services.dataset_loader import load_dataset

render_progress(current_key="conceptos")

st.title("🧠 Conceptos de Deep Learning")
st.caption(
    "Esta sección prioriza **claridad pedagógica** (Principio I): cada concepto se "
    "explica en lenguaje simple y se conecta con los datos/hechos reales del "
    "proyecto cuando están disponibles."
)

dataset_result = load_dataset()
architecture_result = load_architecture_registry()


def _visualize(concept, value) -> None:
    if concept.concept_id == "epoca":
        st.line_chart({"pérdida simulada": [max(0.1, 10 / v) for v in range(1, value + 1)]})

    elif concept.concept_id == "validacion_cruzada":
        if dataset_result.available:
            n = len(dataset_result.samples)
            fold_size = n // 5
            st.write(
                f"Fold {value} de 5 usaría aproximadamente **{fold_size} de las "
                f"{n} muestras reales** como partición de validación."
            )
        else:
            st.info(
                "El tamaño real de cada fold se mostrará cuando `data/fold_results.csv` "
                "esté disponible."
            )

    elif concept.concept_id == "cnn_vs_transformer":
        if architecture_result.available:
            matches = [a for a in architecture_result.architectures if a.family == value]
            st.write(f"{len(matches)} de las 6 arquitecturas reales pertenecen a **{value}**.")
        else:
            st.info(
                "La cantidad real de arquitecturas por familia se mostrará cuando los "
                "resultados estén disponibles."
            )

    elif concept.concept_id == "backbone_congelado":
        st.write(f"Estado seleccionado: **{value}**.")

    elif concept.concept_id == "cabeza_regresion":
        st.write(f"El modelo estimaría **{value}** junto con las demás variables, en una sola vez.")


for concept in CONCEPTS:
    render_concept(concept, lambda value, c=concept: _visualize(c, value))
    st.divider()
