"""Descarga y cachea los 2 checkpoints (mejor y peor arquitectura) desde Hugging Face
Hub, reconstruyendo el modelo con timm a partir de la metadata publicada junto a cada
checkpoint. Ver
specs/001-recorrido-guiado-biomasa/contracts/checkpoint-contract.md.

Cacheado con st.cache_resource (compartido a nivel de proceso, no por sesión) y un TTL
configurable, para respetar el límite de ~1GB de RAM con hasta ~5 sesiones
concurrentes liberando modelos no usados recientemente (research.md §3/§4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st
import timm
import torch
from huggingface_hub import hf_hub_download

from services.config import load_config


class ModelLoadError(Exception):
    """Error manejable al descargar o reconstruir un checkpoint (FR-016)."""


@dataclass
class LoadedModel:
    architecture_id: str
    model: torch.nn.Module
    input_size: tuple[int, int]
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    output_targets: list[str]


def _resolve_ttl_seconds() -> int:
    return load_config().model_cache_ttl_minutes * 60


@st.cache_resource(ttl=_resolve_ttl_seconds(), show_spinner=False)
def load_checkpoint(architecture_id: str) -> LoadedModel:
    """Descarga (o sirve desde caché de proceso) y reconstruye un checkpoint."""
    config = load_config()
    if not config.hf_repo_id:
        raise ModelLoadError("HF_REPO_ID no está configurado (ver .env.example).")

    try:
        checkpoint_path = hf_hub_download(
            repo_id=config.hf_repo_id, filename=f"{architecture_id}.pt"
        )
        metadata_path = hf_hub_download(
            repo_id=config.hf_repo_id, filename=f"{architecture_id}_metadata.json"
        )
    except Exception as exc:  # huggingface_hub lanza distintos tipos según el fallo
        raise ModelLoadError(
            f"No se pudo descargar el checkpoint de '{architecture_id}': {exc}"
        ) from exc

    return _build_model(architecture_id, checkpoint_path, metadata_path)


def _build_model(architecture_id: str, checkpoint_path: str, metadata_path: str) -> LoadedModel:
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        model = timm.create_model(
            metadata["timm_model_name"],
            pretrained=False,
            num_classes=len(metadata["output_targets"]),
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(
            f"No se pudo reconstruir el modelo '{architecture_id}' desde el checkpoint: {exc}"
        ) from exc

    return LoadedModel(
        architecture_id=architecture_id,
        model=model,
        input_size=tuple(metadata["input_size"]),
        normalization_mean=tuple(metadata["normalization_mean"]),
        normalization_std=tuple(metadata["normalization_std"]),
        output_targets=metadata["output_targets"],
    )
