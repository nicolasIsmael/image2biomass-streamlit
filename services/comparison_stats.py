"""Agregaciones puras sobre data/fold_results.csv para la pestaña de comparación.

Sin Streamlit ni Plotly: solo pandas, para poder testear que las cifras que muestra la
app reproducen las del paper. La SD es poblacional (ddof=0) para coincidir exactamente
con las tablas del paper (resultados/para paper/tables/table3_model_ranking).
"""

from __future__ import annotations

import pandas as pd

PRIMARY_METRIC = "weighted_r2"

# Orden de las variables objetivo igual al del paper (TARGET_ORDER).
TARGET_ORDER = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
TARGET_LABELS = {
    "Dry_Green_g": "Biomasa verde",
    "Dry_Clover_g": "Trébol",
    "Dry_Dead_g": "Biomasa muerta",
    "GDM_g": "Materia seca verde (GDM)",
    "Dry_Total_g": "Biomasa total",
}


def metric_label(column: str) -> str:
    """Nombre legible de una columna de métrica (`GDM_g_mae` -> "GDM · MAE")."""
    if column == PRIMARY_METRIC:
        return "R² ponderado"
    for target in TARGET_ORDER:
        if column.startswith(f"{target}_"):
            kind = column[len(target) + 1 :].upper().replace("R2", "R²")
            return f"{TARGET_LABELS[target]} · {kind}"
    return column


def higher_is_better(metric: str) -> bool:
    """R² mejora al subir; MAE/RMSE mejoran al bajar."""
    return not metric.lower().endswith(("mae", "rmse"))


def architecture_summary(fold_df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Media, SD (ddof=0), mínimo y máximo de `metric` entre folds, de mejor a peor media
    (descendente para R², ascendente para métricas de error)."""
    grouped = fold_df.groupby("architecture_id")[metric]
    summary = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "sd": grouped.std(ddof=0),
            "min": grouped.min(),
            "max": grouped.max(),
        }
    )
    identity = fold_df.drop_duplicates("architecture_id").set_index("architecture_id")[
        ["family", "variant_name"]
    ]
    ranked = summary.join(identity).sort_values("mean", ascending=not higher_is_better(metric))
    return ranked.reset_index()


def per_target_matrix(fold_df: pd.DataFrame, suffix: str, order: list[str]) -> pd.DataFrame:
    """Media entre folds de `<target>_<suffix>`: filas = arquitectura (en `order`),
    columnas = variable objetivo (en TARGET_ORDER). `suffix` es "r2" o "mae"."""
    columns = {f"{t}_{suffix}": t for t in TARGET_ORDER if f"{t}_{suffix}" in fold_df.columns}
    means = fold_df.groupby("architecture_id")[list(columns)].mean().rename(columns=columns)
    return means.loc[order]


def pairwise_matrix(pairwise_df: pd.DataFrame, order: list[str], column: str) -> pd.DataFrame:
    """Matriz simétrica arquitectura × arquitectura de `column` (p. ej. `p_holm`);
    la diagonal queda en NaN (un modelo no se compara consigo mismo)."""
    matrix = pd.DataFrame(index=order, columns=order, dtype=float)
    for row in pairwise_df.itertuples():
        value = float(getattr(row, column))
        matrix.loc[row.architecture_a, row.architecture_b] = value
        matrix.loc[row.architecture_b, row.architecture_a] = value
    return matrix


def fold_sizes(fold_df: pd.DataFrame) -> dict[int, int]:
    sizes = fold_df.drop_duplicates("fold_number").set_index("fold_number")["n_samples_val"]
    return {int(k): int(v) for k, v in sizes.sort_index().items()}
