import json

import pytest

from services import model_registry
from services.config import AppConfig


def test_missing_hf_repo_id_raises_model_load_error(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "load_config",
        lambda: AppConfig(hf_repo_id=None, model_cache_ttl_minutes=15),
    )

    with pytest.raises(model_registry.ModelLoadError, match="HF_REPO_ID"):
        model_registry.load_checkpoint("arch_missing_repo")


def test_download_failure_raises_model_load_error(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "load_config",
        lambda: AppConfig(hf_repo_id="user/repo", model_cache_ttl_minutes=15),
    )

    def failing_download(repo_id, filename):
        raise RuntimeError("network error")

    monkeypatch.setattr(model_registry, "hf_hub_download", failing_download)

    with pytest.raises(model_registry.ModelLoadError, match="No se pudo descargar"):
        model_registry.load_checkpoint("arch_download_fail")


def test_successful_checkpoint_build_uses_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_registry,
        "load_config",
        lambda: AppConfig(hf_repo_id="user/repo", model_cache_ttl_minutes=15),
    )

    metadata = {
        "timm_model_name": "resnet18",
        "input_size": [224, 224],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "output_targets": ["green_g", "clover_g"],
        "target_scaler_mean": [2.5, 1.2],
        "target_scaler_std": [1.1, 1.3],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    checkpoint_path = tmp_path / "checkpoint.pt"

    # Checkpoint real: mismo wrapper backbone+head que reconstruye _build_model, para
    # verificar que el state_dict carga con strict=True sin mockear la arquitectura.
    real_backbone = model_registry.timm.create_model(
        metadata["timm_model_name"], pretrained=False, num_classes=0
    )
    real_head = model_registry.torch.nn.Sequential(
        model_registry.torch.nn.Linear(real_backbone.num_features, 128),
        model_registry.torch.nn.ReLU(),
        model_registry.torch.nn.Dropout(p=0.3),
        model_registry.torch.nn.Linear(128, len(metadata["output_targets"])),
    )
    real_model = model_registry.BiomassModel(real_backbone, real_head)
    model_registry.torch.save(real_model.state_dict(), checkpoint_path)

    def fake_hf_hub_download(repo_id, filename):
        return str(metadata_path) if filename.endswith("_metadata.json") else str(checkpoint_path)

    monkeypatch.setattr(model_registry, "hf_hub_download", fake_hf_hub_download)

    result = model_registry.load_checkpoint("arch_ok_success")

    assert result.architecture_id == "arch_ok_success"
    assert result.output_targets == ["green_g", "clover_g"]
    assert result.input_size == (224, 224)
    assert result.target_scaler_mean == (2.5, 1.2)
    assert result.target_scaler_std == (1.1, 1.3)
    assert isinstance(result.model, model_registry.BiomassModel)
    assert result.model.training is False  # eval() fue llamado
