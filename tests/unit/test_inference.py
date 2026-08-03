import torch
from PIL import Image

from services.inference import predict
from services.model_registry import LoadedModel


class _FakeModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.tensor([[1.5, 2.5]] * batch_size)


def test_predict_maps_output_to_target_names_in_metadata_order():
    loaded_model = LoadedModel(
        architecture_id="arch_fake",
        model=_FakeModel(),
        input_size=(32, 32),
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        output_targets=["green_g", "clover_g"],
    )
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))

    result = predict(loaded_model, image)

    assert result.architecture_id == "arch_fake"
    assert result.predicted_values == {"green_g": 1.5, "clover_g": 2.5}
    assert result.elapsed_seconds >= 0
