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

PAIRWISE_CSV_PATH = Path("data/pairwise_significance.csv")
COST_METRICS_CSV_PATH = Path("data/cost_metrics.csv")

_REQUIRED_PAIRWISE_COLUMNS = {
    "architecture_a",
    "architecture_b",
    "p_raw",
    "p_holm",
    "is_significant",
    "p_nemenyi",
    "is_significant_nemenyi",
}
_REQUIRED_COST_COLUMNS = {
    "architecture_id",
    "train_total_min",
    "train_relative_to_fastest",
    "infer_median_ms",
    "infer_relative_to_fastest",
    "params_millions",
}
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


@dataclass
class OptionalTable:
    available: bool
    data: pd.DataFrame | None = None
    reason: str | None = None


def _load_optional_table(csv_path: str, required: set[str]) -> OptionalTable:
    path = Path(csv_path)
    if not path.exists():
        return OptionalTable(available=False, reason=f"No se encontró {csv_path}.")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        return OptionalTable(
            available=False,
            reason=f"Faltan columnas requeridas en {csv_path}: {sorted(missing)}.",
        )
    return OptionalTable(available=True, data=df)


@st.cache_data(show_spinner=False)
def load_pairwise(csv_path: str = str(PAIRWISE_CSV_PATH)) -> OptionalTable:
    """15 pares de arquitecturas: t pareado + Holm y chequeo Nemenyi (tabla 4 del paper)."""
    return _load_optional_table(csv_path, _REQUIRED_PAIRWISE_COLUMNS)


@st.cache_data(show_spinner=False)
def load_cost_metrics(csv_path: str = str(COST_METRICS_CSV_PATH)) -> OptionalTable:
    """Tiempo de entrenamiento e inferencia por arquitectura (tablas 5 y 6 del paper)."""
    return _load_optional_table(csv_path, _REQUIRED_COST_COLUMNS)
