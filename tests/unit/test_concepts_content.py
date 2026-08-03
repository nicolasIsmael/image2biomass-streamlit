import pytest

from services.architecture_registry import load_architecture_registry
from services.concepts_content import CONCEPTS, get_concept

REQUIRED_CONCEPT_IDS = {
    "epoca",
    "validacion_cruzada",
    "cnn_vs_transformer",
    "backbone_congelado",
    "cabeza_regresion",
}


def test_all_required_concepts_present_with_unique_ids():
    ids = [c.concept_id for c in CONCEPTS]
    assert set(ids) == REQUIRED_CONCEPT_IDS
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("concept", CONCEPTS, ids=lambda c: c.concept_id)
def test_concept_has_required_fields(concept):
    assert concept.title
    assert concept.explanation
    assert concept.control_type in {"slider", "select", "button"}
    assert len(concept.control_options) >= 2
    assert concept.real_project_binding


def test_get_concept_returns_matching_entry():
    concept = get_concept("validacion_cruzada")
    assert concept.concept_id == "validacion_cruzada"


def test_get_concept_raises_for_unknown_id():
    with pytest.raises(KeyError):
        get_concept("no_existe")


def test_architecture_registry_missing_file_is_not_available(tmp_path):
    result = load_architecture_registry(
        fold_results_csv=str(tmp_path / "missing.csv"),
        statistical_comparison_csv=str(tmp_path / "missing_stats.csv"),
    )

    assert result.available is False


def test_architecture_registry_marks_at_most_one_best_and_one_worst(tmp_path):
    fold_csv = tmp_path / "fold_results.csv"
    fold_csv.write_text(
        "architecture_id,family,variant_name,fold_number\n"
        "arch_a,CNN,Arch A,1\n"
        "arch_b,Transformer,Arch B,1\n"
        "arch_c,Hybrid,Arch C,1\n"
    )
    stats_csv = tmp_path / "statistical_comparison.csv"
    stats_csv.write_text(
        "best_architecture_id,worst_architecture_id\narch_a,arch_c\n"
    )

    result = load_architecture_registry(
        fold_results_csv=str(fold_csv),
        statistical_comparison_csv=str(stats_csv),
    )

    assert result.available is True
    best = [a for a in result.architectures if a.is_best]
    worst = [a for a in result.architectures if a.is_worst]
    assert len(best) == 1 and best[0].architecture_id == "arch_a"
    assert len(worst) == 1 and worst[0].architecture_id == "arch_c"
