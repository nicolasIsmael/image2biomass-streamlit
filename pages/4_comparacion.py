import streamlit as st

from components.comparison_charts import architecture_colors
from components.comparison_tabs import (
    render_cost_tab,
    render_significance_tab,
    render_summary_tab,
    render_target_tab,
)
from components.limitations_panel import (
    STATIC_QUALITATIVE_LIMITATIONS,
    build_fold_imbalance_limitation,
)
from components.limitations_panel import render as render_limitations
from components.progress_indicator import render as render_progress
from services.architecture_registry import load_architecture_registry
from services.comparison_stats import architecture_summary
from services.results_loader import load_cost_metrics, load_pairwise, load_results

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
    labels = {a.architecture_id: a.variant_name for a in architecture_result.architectures}
    colors = architecture_colors(list(labels))
    order = architecture_summary(results.fold_results)["architecture_id"].tolist()

    tab_summary, tab_targets, tab_stats, tab_cost = st.tabs(
        ["📊 Desempeño", "🎯 Por variable", "🔬 ¿Es real la diferencia?", "⏱️ Costo"]
    )
    with tab_summary:
        render_summary_tab(results, colors)
    with tab_targets:
        render_target_tab(results, labels)
    with tab_stats:
        render_significance_tab(results, load_pairwise(), labels, order)
    with tab_cost:
        render_cost_tab(load_cost_metrics(), labels)

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
