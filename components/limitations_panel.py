"""Panel discreto de limitaciones conocidas del estudio (Principio II).

Se renderiza como expander/nota colapsable -- nunca como alerta prominente -- para no
opacar ni contradecir el mensaje principal de la sección donde se usa. La limitación
de desbalance de folds se calcula dinámicamente a partir de datos reales
(`results_loader`); las limitaciones cualitativas confirmadas por el autor del estudio
(zero-inflation en Clover, baseline NDVI competitivo en Green) se completan en
`STATIC_QUALITATIVE_LIMITATIONS` (User Story 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class KnownLimitation:
    limitation_id: str
    title: str
    description: str


STATIC_QUALITATIVE_LIMITATIONS: list[KnownLimitation] = [
    KnownLimitation(
        limitation_id="zero_inflation_clover",
        title="Zero-inflation en Clover",
        description=(
            "La variable objetivo **Clover** tiene una alta proporción de muestras "
            "con valor cero (parcelas sin trébol presente). Esto dificulta que el "
            "modelo aprenda a distinguir valores bajos-pero-no-cero de valores "
            "genuinamente cero, y puede inflar artificialmente el desempeño "
            "aparente en esa variable."
        ),
    ),
    KnownLimitation(
        limitation_id="baseline_ndvi_green",
        title="Un baseline simple (NDVI) fue competitivo en Green",
        description=(
            "Para la variable **Green**, un índice simple derivado de la imagen "
            "(NDVI, sin Deep Learning) obtuvo un desempeño competitivo frente a los "
            "modelos entrenados. Esto sugiere que, para esa variable en particular, "
            "parte de la señal es capturable con métodos mucho más simples."
        ),
    ),
]


def render(
    limitations: list[KnownLimitation],
    expander_label: str = "📎 Limitaciones conocidas del estudio",
) -> None:
    if not limitations:
        return
    with st.expander(expander_label, expanded=False):
        for limitation in limitations:
            st.markdown(f"**{limitation.title}**")
            st.markdown(limitation.description)
            st.markdown("---")


def build_fold_imbalance_limitation(results) -> KnownLimitation | None:
    """Deriva la limitación de desbalance de folds a partir de n_samples_val real.

    `results` es un services.results_loader.ResultsData. Devuelve None si los datos
    no están disponibles o no muestran un desbalance relevante -- nunca inventa cifras.
    """
    if not results.available:
        return None

    sizes = results.fold_results[["fold_number", "n_samples_val"]].drop_duplicates()
    if sizes.empty:
        return None

    smallest = sizes.loc[sizes["n_samples_val"].idxmin()]
    largest = sizes.loc[sizes["n_samples_val"].idxmax()]
    if smallest["n_samples_val"] == largest["n_samples_val"]:
        return None

    return KnownLimitation(
        limitation_id="desbalance_folds",
        title="Desbalance en el tamaño de los folds",
        description=(
            f"El Fold {int(smallest['fold_number'])} tiene solo "
            f"{int(smallest['n_samples_val'])} muestras de validación, frente a "
            f"{int(largest['n_samples_val'])} en el Fold {int(largest['fold_number'])}. "
            "Un fold con menos muestras puede producir métricas más ruidosas para "
            "ese fold específico."
        ),
    )
