"""Carga y cachea el CSV de metadatos del dataset (data/dataset_metadata.csv).

Ver specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md para el
esquema esperado. Si el artefacto real aún no existe o está incompleto, se devuelve un
estado "no disponible" explícito -- nunca datos simulados (Principio VI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

DATASET_CSV_PATH = Path("data/dataset_metadata.csv")
METADATA_COLUMNS = {
    "sample_id",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
}


@dataclass
class DatasetResult:
    available: bool
    samples: pd.DataFrame | None = None
    target_columns: list[str] | None = None
    reason: str | None = None


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str = str(DATASET_CSV_PATH)) -> DatasetResult:
    """Lee dataset_metadata.csv y separa columnas de metadata vs variables objetivo."""
    path = Path(csv_path)
    if not path.exists():
        return DatasetResult(available=False, reason=f"No se encontró {csv_path}.")

    df = pd.read_csv(path)
    if df.empty:
        return DatasetResult(available=False, reason=f"{csv_path} está vacío.")

    missing = METADATA_COLUMNS - set(df.columns)
    if missing:
        return DatasetResult(
            available=False,
            reason=f"Faltan columnas requeridas en {csv_path}: {sorted(missing)}.",
        )

    target_columns = [c for c in df.columns if c not in METADATA_COLUMNS]
    return DatasetResult(available=True, samples=df, target_columns=target_columns)
