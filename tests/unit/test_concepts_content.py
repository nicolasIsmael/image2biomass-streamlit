import pandas as pd
import pytest

from components.concept_viewer import HTML_DIR, force_light_theme, load_concept_html
from services.architecture_registry import load_architecture_registry
from services.concepts_content import CONCEPTS, get_concept, resolve_binding

EXPECTED_ORDER = [
    "neurona",
    "capas_profundidad",
    "error",
    "colina_error",
    "sobreajuste",
    "validacion_honesta",
    "convolucion",
    "atencion",
    "transfer_learning",
    "r2_vs_promedio",
]


def _fold_results() -> pd.DataFrame:
    rows = []
    for arch_id, family, name, r2 in [
        ("a", "CNN", "Arch A", [-0.2, -0.1]),
        ("b", "Transformer", "Arch B", [0.3, 0.4]),
    ]:
        for fold, score in zip((1, 2), r2):
            rows.append(
                {
                    "architecture_id": arch_id,
                    "family": family,
                    "variant_name": name,
                    "fold_number": fold,
                    "n_samples_val": 70 + fold,
                    "weighted_r2": score,
                }
            )
    return pd.DataFrame(rows)


def test_concepts_are_the_ten_in_pedagogical_order():
    assert [c.concept_id for c in CONCEPTS] == EXPECTED_ORDER
    assert [c.number for c in CONCEPTS] == list(range(1, 11))


@pytest.mark.parametrize("concept", CONCEPTS, ids=lambda c: c.concept_id)
def test_concept_has_required_fields_and_existing_html(concept):
    assert concept.title
    assert concept.real_project_binding
    assert (HTML_DIR / concept.html_file).is_file()


def test_get_concept_returns_matching_entry():
    assert get_concept("validacion_honesta").number == 6


def test_get_concept_raises_for_unknown_id():
    with pytest.raises(KeyError):
        get_concept("no_existe")


def test_force_light_theme_marks_html_root_only_once():
    html = "<!doctype html><html><head></head><body><html></body></html>"

    themed = force_light_theme(html)

    assert themed.count('<html data-theme="light"') == 1


def test_load_concept_html_forces_light_theme_and_handles_missing_file(tmp_path):
    (tmp_path / "x.html").write_text("<html><body>hola</body></html>", encoding="utf-8")

    assert 'data-theme="light"' in load_concept_html("x.html", str(tmp_path))
    assert load_concept_html("falta.html", str(tmp_path)) is None


def test_binding_falls_back_to_static_text_without_results():
    concept = get_concept("r2_vs_promedio")

    assert resolve_binding(concept, None) == concept.real_project_binding


def test_binding_uses_real_fold_sizes():
    binding = resolve_binding(get_concept("validacion_honesta"), _fold_results())

    assert "2 folds" in binding and "71" in binding and "72" in binding


def test_binding_lists_architectures_of_the_concept_family():
    cnn = resolve_binding(get_concept("convolucion"), _fold_results())
    transformer = resolve_binding(get_concept("atencion"), _fold_results())

    assert "Arch A" in cnn and "Arch B" not in cnn
    assert "Arch B" in transformer and "Arch A" not in transformer


def test_binding_r2_reports_range_and_flags_below_average_model():
    binding = resolve_binding(get_concept("r2_vs_promedio"), _fold_results())

    assert "-0.15 (Arch A)" in binding and "0.35 (Arch B)" in binding
    assert "por debajo de simplemente predecir el promedio" in binding


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
