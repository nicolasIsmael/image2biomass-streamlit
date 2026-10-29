"""Contenido de cada pestaña de la página de comparación (Streamlit + gráficos).

La lógica de agregación vive en `services.comparison_stats`; aquí solo se compone el
layout. Sección con prioridad de rigor estadístico (Principio I): se muestran las
cifras reales con 4 decimales y se rotula la convención de la SD.
"""

from __future__ import annotations

import streamlit as st

from components.comparison_charts import (
    fold_paired_lines,
    fold_strip_chart,
    mae_heatmap,
    r2_heatmap,
)
from components.metric_chart import render
from components.significance_charts import cost_bars, pairwise_heatmap
from services.comparison_stats import (
    PRIMARY_METRIC,
    TARGET_LABELS,
    architecture_summary,
    metric_label,
    pairwise_matrix,
    per_target_matrix,
)

SD_NOTE = "SD = desviación estándar entre los 5 folds, poblacional (ddof=0), igual que en el paper."


def _pending(reason: str | None, what: str) -> None:
    st.info(f"🔧 {what} aún no está disponible. Detalle técnico: {reason}")


def render_summary_tab(results, colors: dict[str, str]) -> None:
    fold_df = results.fold_results
    ranking = architecture_summary(fold_df)
    best, worst = ranking.iloc[0], ranking.iloc[-1]

    col_best, col_worst, col_folds = st.columns(3)
    col_best.metric(f"🏆 Mejor: {best.variant_name}", f"{best['mean']:.4f}", help=SD_NOTE)
    col_best.caption(f"R² ponderado medio · SD {best.sd:.4f}")
    col_worst.metric(f"📉 Peor: {worst.variant_name}", f"{worst['mean']:.4f}", help=SD_NOTE)
    col_worst.caption(f"R² ponderado medio · SD {worst.sd:.4f}")
    sizes = fold_df.drop_duplicates("fold_number")["n_samples_val"]
    col_folds.metric("Validación cruzada", f"{fold_df['fold_number'].nunique()} folds")
    col_folds.caption(
        f"{int(sizes.sum())} muestras · {int(sizes.min())}–{int(sizes.max())} por fold"
    )

    st.dataframe(
        ranking.rename(
            columns={"variant_name": "Arquitectura", "family": "Familia", "mean": "Media",
                     "sd": "SD (ddof=0)", "min": "Mínimo", "max": "Máximo"}
        )[["Arquitectura", "Familia", "Media", "SD (ddof=0)", "Mínimo", "Máximo"]],
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.4f")
                       for c in ["Media", "SD (ddof=0)", "Mínimo", "Máximo"]},
    )
    st.caption(f"R² ponderado por arquitectura, de mejor a peor media. {SD_NOTE}")

    metric = st.selectbox(
        "Métrica a comparar",
        results.metric_columns,
        index=results.metric_columns.index(PRIMARY_METRIC),
        format_func=metric_label,
    )
    summary = architecture_summary(fold_df, metric)
    render(fold_strip_chart(fold_df, summary, metric, metric_label(metric), colors))
    render(fold_paired_lines(fold_df, metric, metric_label(metric), colors))


def render_target_tab(results, labels: dict[str, str]) -> None:
    fold_df = results.fold_results
    order = architecture_summary(fold_df)["architecture_id"].tolist()
    r2 = per_target_matrix(fold_df, "r2", order)
    mae = per_target_matrix(fold_df, "mae", order)
    st.markdown(
        "Un mismo modelo no rinde igual en todas las variables: el R² medio (entre folds) "
        "muestra en cuáles acierta más que el promedio y en cuáles no."
    )
    render(r2_heatmap(r2, labels, TARGET_LABELS))
    render(mae_heatmap(mae, labels, TARGET_LABELS))
    st.caption(
        "El color del MAE se normaliza dentro de cada columna (cada variable tiene otra "
        "escala en gramos); el número escrito es el MAE real."
    )


def _honest_reading(df, row, labels: dict[str, str]) -> str:
    """Texto derivado de los p-values reales; no afirma robustez que los datos no den."""
    n_pairs = len(df)
    text = (
        f"**Lectura honesta:** con el t pareado (Holm) {int(df['is_significant'].sum())} de "
        f"{n_pairs} pares son significativos; con Friedman + Nemenyi, más conservador, "
        f"{int(df['is_significant_nemenyi'].sum())} de {n_pairs}. Con solo 5 folds es esperable "
        "que la prueba basada en rangos sea menos sensible a las diferencias pequeñas."
    )
    ids = {row["best_architecture_id"], row["worst_architecture_id"]}
    pair = df[df.apply(lambda r: {r.architecture_a, r.architecture_b} == ids, axis=1)]
    if pair.empty:
        return text
    best, worst = labels[row["best_architecture_id"]], labels[row["worst_architecture_id"]]
    p = pair.iloc[0]
    if p["is_significant"] and p["is_significant_nemenyi"]:
        return f"{text} La ventaja de {best} sobre {worst} se sostiene en ambas pruebas."
    return f"{text} La diferencia {best} vs. {worst} no se sostiene en ambas pruebas."


def render_significance_tab(results, pairwise, labels: dict[str, str], order: list[str]) -> None:
    row = results.statistical_comparison.iloc[0]
    st.markdown(row["plain_language_summary"])
    st.caption(
        f"Prueba: {row['test_name']} · p-value (mejor vs. peor, ajustado): "
        f"{row['p_value']:.4f} · ¿Significativo?: {'Sí' if row['is_significant'] else 'No'}"
    )
    if not pairwise.available:
        return _pending(pairwise.reason, "La matriz de comparaciones por pares")

    df = pairwise.data
    st.markdown(_honest_reading(df, row, labels))
    left, right = st.columns(2)
    for col, p_col, sig_col, title, hover in [
        (left, "p_holm", "is_significant", "t pareado + Holm", "p ajustado (Holm)"),
        (right, "p_nemenyi", "is_significant_nemenyi", "Friedman + Nemenyi", "p (Nemenyi)"),
    ]:
        with col:
            render(pairwise_heatmap(
                pairwise_matrix(df, order, p_col), pairwise_matrix(df, order, sig_col),
                title, labels, hover,
            ))
    st.caption("Azul = significativo (α = 0.05), gris = no. El número es el p-value real.")


def render_cost_tab(cost, labels: dict[str, str]) -> None:
    if not cost.available:
        return _pending(cost.reason, "La información de costo computacional")
    df = cost.data
    st.markdown(
        "Mejor R² no implica mayor costo: se compara cuánto tarda cada arquitectura en "
        "entrenar y en predecir una imagen."
    )
    render(cost_bars(df, "train_relative_to_fastest", "Tiempo de entrenamiento (5 folds)", labels))
    render(cost_bars(df.sort_values("infer_median_ms"), "infer_relative_to_fastest",
                     "Latencia de inferencia en CPU (mediana)", labels))
    st.caption(
        "Se muestra solo el factor relativo porque el entrenamiento corrió en Google Colab "
        "(GPU no garantizada entre sesiones) y la CPU de inferencia no estaba aislada: los "
        "valores absolutos no son portables; el factor entre arquitecturas sí."
    )

