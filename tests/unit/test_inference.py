import math

import pytest
import torch
from PIL import Image

from services.inference import predict
from services.model_registry import LoadedModel


class _FakeModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.tensor([[1.5, 2.5]] * batch_size)


def test_predict_inverts_target_scaler_and_maps_to_names_in_metadata_order():
    loaded_model = LoadedModel(
        architecture_id="arch_fake",
        model=_FakeModel(),
        input_size=(32, 32),
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        output_targets=["green_g", "clover_g"],
        target_scaler_mean=(0.0, 0.0),
        target_scaler_std=(1.0, 1.0),
    )
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))

    result = predict(loaded_model, image)

    assert result.architecture_id == "arch_fake"
    expected = {"green_g": math.expm1(1.5), "clover_g": math.expm1(2.5)}
    assert result.predicted_values.keys() == expected.keys()
    for key, value in expected.items():
        assert result.predicted_values[key] == pytest.approx(value)
    assert result.elapsed_seconds >= 0


def test_predict_clips_negative_values_to_zero():
    loaded_model = LoadedModel(
        architecture_id="arch_fake",
        model=_FakeModel(),
        input_size=(32, 32),
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        output_targets=["green_g", "clover_g"],
        # mean muy negativa fuerza expm1(...) < 0 para ambos targets
        target_scaler_mean=(-10.0, -10.0),
        target_scaler_std=(1.0, 1.0),
    )
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))

    result = predict(loaded_model, image)

    assert result.predicted_values == {"green_g": 0.0, "clover_g": 0.0}
