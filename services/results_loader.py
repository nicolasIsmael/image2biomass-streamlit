"""Carga y cachea los CSVs de resultados por fold y comparación estadística.

Ver specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md para el
esquema esperado. Estado "no disponible" explícito si falta cualquiera de los dos
artefactos -- nunca datos simulados (Principio VI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

FOLD_RESULTS_CSV_PATH = Path("data/fold_results.csv")
STATISTICAL_COMPARISON_CSV_PATH = Path("data/statistical_comparison.csv")

_REQUIRED_FOLD_COLUMNS = {
    "architecture_id",
    "family",
    "variant_name",
    "fold_number",
    "n_samples_val",
}
_REQUIRED_STATS_COLUMNS = {
    "test_name",
    "p_value",
    "is_significant",
    "plain_language_summary",
    "best_architecture_id",
    "worst_architecture_id",
}


@dataclass
class ResultsData:
    available: bool
    fold_results: pd.DataFrame | None = None
    statistical_comparison: pd.DataFrame | None = None
    metric_columns: list[str] | None = None
    reason: str | None = None


@st.cache_data(show_spinner=False)
def load_results(
    fold_results_csv: str = str(FOLD_RESULTS_CSV_PATH),
    statistical_comparison_csv: str = str(STATISTICAL_COMPARISON_CSV_PATH),
) -> ResultsData:
    fold_path = Path(fold_results_csv)
    if not fold_path.exists():
        return ResultsData(available=False, reason=f"No se encontró {fold_results_csv}.")

    fold_df = pd.read_csv(fold_path)
    missing = _REQUIRED_FOLD_COLUMNS - set(fold_df.columns)
    if missing:
        return ResultsData(
            available=False,
            reason=f"Faltan columnas requeridas en {fold_results_csv}: {sorted(missing)}.",
        )

    stats_path = Path(statistical_comparison_csv)
    if not stats_path.exists():
        return ResultsData(available=False, reason=f"No se encontró {statistical_comparison_csv}.")

    stats_df = pd.read_csv(stats_path)
    missing_stats = _REQUIRED_STATS_COLUMNS - set(stats_df.columns)
    if missing_stats:
        return ResultsData(
            available=False,
            reason=(
                f"Faltan columnas requeridas en {statistical_comparison_csv}: "
                f"{sorted(missing_stats)}."
            ),
        )

    metric_columns = [c for c in fold_df.columns if c not in _REQUIRED_FOLD_COLUMNS]

    return ResultsData(
        available=True,
        fold_results=fold_df,
        statistical_comparison=stats_df,
        metric_columns=metric_columns,
    )
