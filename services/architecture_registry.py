"""Registro de las 6 arquitecturas de modelo comparadas en el estudio.

Se deriva de data/fold_results.csv (identidad + familia) y
data/statistical_comparison.csv (cuál es la de mejor/peor desempeño) -- nunca de una
lista de nombres inventados. Si los artefactos aún no existen, se devuelve un estado
"no disponible" explícito (Principio VI). Ver
specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

FOLD_RESULTS_CSV_PATH = Path("data/fold_results.csv")
STATISTICAL_COMPARISON_CSV_PATH = Path("data/statistical_comparison.csv")
_IDENTITY_COLUMNS = {"architecture_id", "family", "variant_name"}


@dataclass
class ModelArchitecture:
    architecture_id: str
    family: str
    variant_name: str
    is_best: bool
    is_worst: bool


@dataclass
class ArchitectureRegistryResult:
    available: bool
    architectures: list[ModelArchitecture] | None = None
    reason: str | None = None


@st.cache_data(show_spinner=False)
def load_architecture_registry(
    fold_results_csv: str = str(FOLD_RESULTS_CSV_PATH),
    statistical_comparison_csv: str = str(STATISTICAL_COMPARISON_CSV_PATH),
) -> ArchitectureRegistryResult:
    fold_path = Path(fold_results_csv)
    if not fold_path.exists():
        return ArchitectureRegistryResult(
            available=False, reason=f"No se encontró {fold_results_csv}."
        )

    fold_df = pd.read_csv(fold_path)
    missing = _IDENTITY_COLUMNS - set(fold_df.columns)
    if missing:
        return ArchitectureRegistryResult(
            available=False,
            reason=f"Faltan columnas requeridas en {fold_results_csv}: {sorted(missing)}.",
        )

    identity_df = fold_df[list(_IDENTITY_COLUMNS)].drop_duplicates(subset="architecture_id")

    best_id, worst_id = _load_best_worst_ids(statistical_comparison_csv)

    architectures = [
        ModelArchitecture(
            architecture_id=row.architecture_id,
            family=row.family,
            variant_name=row.variant_name,
            is_best=(row.architecture_id == best_id),
            is_worst=(row.architecture_id == worst_id),
        )
        for row in identity_df.itertuples()
    ]
    return ArchitectureRegistryResult(available=True, architectures=architectures)


def _load_best_worst_ids(statistical_comparison_csv: str) -> tuple[str | None, str | None]:
    stats_path = Path(statistical_comparison_csv)
    if not stats_path.exists():
        return None, None

    stats_df = pd.read_csv(stats_path)
    required = {"best_architecture_id", "worst_architecture_id"}
    if stats_df.empty or not required.issubset(stats_df.columns):
        return None, None

    first_row = stats_df.iloc[0]
    return first_row["best_architecture_id"], first_row["worst_architecture_id"]
