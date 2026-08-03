import streamlit as st

from components.limitations_panel import (
    STATIC_QUALITATIVE_LIMITATIONS,
    build_fold_imbalance_limitation,
)
from components.limitations_panel import render as render_limitations
from components.metric_chart import fold_comparison_bar, render
from components.progress_indicator import render as render_progress
from services.architecture_registry import load_architecture_registry
from services.results_loader import load_results

render_progress(current_key="comparacion")

st.title("⚖️ Comparación de arquitecturas")

st.caption(
    "Esta sección prioriza **rigor estadístico** (Principio I): el lenguaje se "
    "simplifica para ser entendible, pero ningún número se redondea ni se altera "
    "de forma que cambie su interpretación."
)

architecture_result = load_architecture_registry()
results = load_results()

if not results.available or not architecture_result.available:
    reasons = " / ".join(
        r for r in [architecture_result.reason, results.reason] if r
    )
    st.warning(
        "🔧 **Espacio preparado, artefactos pendientes.** Esta sección mostrará las "
        "6 arquitecturas reales, sus métricas por los 5 folds, y el resultado del "
        "análisis estadístico en cuanto `data/fold_results.csv` y "
        "`data/statistical_comparison.csv` estén disponibles. No se muestran datos "
        f"simulados (Principio VI). Detalle técnico: {reasons}"
    )
else:
    st.subheader("Desempeño por arquitectura y fold")
    metric = st.selectbox("Métrica a comparar", results.metric_columns)
    render(
        fold_comparison_bar(
            results.fold_results,
            architecture_col="architecture_id",
            metric_col=metric,
            fold_col="fold_number",
            title=f"{metric} por arquitectura y fold",
        )
    )

    st.subheader("¿La diferencia es real o pudo ser casualidad?")
    summary_row = results.statistical_comparison.iloc[0]
    st.markdown(summary_row["plain_language_summary"])
    st.caption(
        f"Prueba estadística: {summary_row['test_name']} · "
        f"p-value real: {summary_row['p_value']} · "
        f"¿Significativo?: {'Sí' if summary_row['is_significant'] else 'No'}"
    )

    best = next((a for a in architecture_result.architectures if a.is_best), None)
    worst = next((a for a in architecture_result.architectures if a.is_worst), None)
    if best and worst:
        st.info(
            f"🏆 Mejor desempeño: **{best.variant_name}** ({best.family}) · "
            f"📉 Peor desempeño: **{worst.variant_name}** ({worst.family}) — "
            "estos dos son los que podrás comparar en la sección de predicción en vivo."
        )

    fold_limitation = build_fold_imbalance_limitation(results)
    limitations = ([fold_limitation] if fold_limitation else []) + STATIC_QUALITATIVE_LIMITATIONS
    render_limitations(limitations)
