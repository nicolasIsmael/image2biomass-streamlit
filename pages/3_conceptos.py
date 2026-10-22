import streamlit as st

from components.concept_viewer import render as render_concept
from components.progress_indicator import render as render_progress
from services.concepts_content import CONCEPTS, get_concept, resolve_binding
from services.results_loader import load_results

CURRENT_KEY = "concepto_actual"
CONCEPT_IDS = [c.concept_id for c in CONCEPTS]

render_progress(current_key="conceptos")

st.title("🧠 Conceptos de Deep Learning")
st.caption(
    "10 conceptos interactivos, de la neurona al R², en el orden en que se construyen. "
    "Cada uno cabe en una sola ventana: juega con los controles antes de pasar al siguiente."
)

st.session_state.setdefault(CURRENT_KEY, CONCEPT_IDS[0])
results_data = load_results()


def _go(delta: int) -> None:
    index = CONCEPT_IDS.index(st.session_state[CURRENT_KEY]) + delta
    st.session_state[CURRENT_KEY] = CONCEPT_IDS[min(max(index, 0), len(CONCEPT_IDS) - 1)]


def _label(concept_id: str) -> str:
    concept = get_concept(concept_id)
    return f"{concept.number:02d} · {concept.title}"


current_index = CONCEPT_IDS.index(st.session_state[CURRENT_KEY])
col_prev, col_select, col_next = st.columns([1, 4, 1])
col_prev.button(
    "← Anterior", on_click=_go, args=(-1,), disabled=current_index == 0, width="stretch"
)
col_select.selectbox(
    "Concepto",
    CONCEPT_IDS,
    key=CURRENT_KEY,
    format_func=_label,
    label_visibility="collapsed",
)
col_next.button(
    "Siguiente →",
    on_click=_go,
    args=(1,),
    disabled=current_index == len(CONCEPT_IDS) - 1,
    width="stretch",
)

concept = get_concept(st.session_state[CURRENT_KEY])
fold_results = results_data.fold_results if results_data.available else None
render_concept(concept, resolve_binding(concept, fold_results))
