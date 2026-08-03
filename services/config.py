"""Configuración de entorno para la predicción en vivo (User Story 4).

Ver .env.example para las variables esperadas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL_CACHE_TTL_MINUTES = 15


@dataclass
class AppConfig:
    hf_repo_id: str | None
    model_cache_ttl_minutes: int


def load_config() -> AppConfig:
    hf_repo_id = os.environ.get("HF_REPO_ID") or None
    ttl_raw = os.environ.get("MODEL_CACHE_TTL_MINUTES", str(DEFAULT_MODEL_CACHE_TTL_MINUTES))
    try:
        ttl_minutes = int(ttl_raw)
    except ValueError:
        ttl_minutes = DEFAULT_MODEL_CACHE_TTL_MINUTES
    return AppConfig(hf_repo_id=hf_repo_id, model_cache_ttl_minutes=ttl_minutes)
