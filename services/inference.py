"""Preprocesamiento + inferencia + postprocesamiento para la predicción en vivo.

Usa exclusivamente la metadata publicada junto a cada checkpoint (input_size,
normalización, output_targets) -- nunca valores hardcodeados -- para que cada modelo
se alimente exactamente como fue entrenado (contracts/checkpoint-contract.md).

El modelo predice en el espacio transformado de entrenamiento (log(1+y) + z-score por
fold, ver TargetScaler en el pipeline de entrenamiento), no en gramos. `predict()`
invierte esa transformación con target_scaler_mean/target_scaler_std de la metadata
antes de devolver los valores.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from services.model_registry import LoadedModel


@dataclass
class PredictionOutput:
    architecture_id: str
    predicted_values: dict[str, float]
    elapsed_seconds: float


def predict(loaded_model: LoadedModel, image: Image.Image) -> PredictionOutput:
    start = time.monotonic()

    preprocess = transforms.Compose(
        [
            transforms.Resize(loaded_model.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=loaded_model.normalization_mean,
                std=loaded_model.normalization_std,
            ),
        ]
    )
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = loaded_model.model(input_tensor)

    scaled_values = output.squeeze(0).tolist()
    values = [
        max(0.0, math.expm1(scaled * std + mean))
        for scaled, mean, std in zip(
            scaled_values, loaded_model.target_scaler_mean, loaded_model.target_scaler_std
        )
    ]
    predicted_values = dict(zip(loaded_model.output_targets, values))

    return PredictionOutput(
        architecture_id=loaded_model.architecture_id,
        predicted_values=predicted_values,
        elapsed_seconds=time.monotonic() - start,
    )
