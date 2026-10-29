from services.results_loader import load_results


def _write_valid_fold_results(path):
    path.write_text(
        "architecture_id,family,variant_name,fold_number,n_samples_val,mae\n"
        "arch_a,CNN,Arch A,1,60,4.2\n"
        "arch_a,CNN,Arch A,2,40,5.1\n"
        "arch_b,Transformer,Arch B,1,60,3.8\n"
        "arch_b,Transformer,Arch B,2,40,4.9\n"
    )


def _write_valid_statistical_comparison(path):
    path.write_text(
        "test_name,p_value,is_significant,plain_language_summary,"
        "best_architecture_id,worst_architecture_id\n"
        "paired_test,0.01,True,La diferencia es real,arch_b,arch_a\n"
    )


def test_missing_fold_results_is_not_available(tmp_path):
    result = load_results(
        fold_results_csv=str(tmp_path / "missing.csv"),
        statistical_comparison_csv=str(tmp_path / "missing_stats.csv"),
    )

    assert result.available is False
    assert "No se encontró" in result.reason


def test_missing_statistical_comparison_is_not_available(tmp_path):
    fold_csv = tmp_path / "fold_results.csv"
    _write_valid_fold_results(fold_csv)

    result = load_results(
        fold_results_csv=str(fold_csv),
        statistical_comparison_csv=str(tmp_path / "missing_stats.csv"),
    )

    assert result.available is False


def test_missing_required_columns_is_not_available(tmp_path):
    fold_csv = tmp_path / "incomplete.csv"
    fold_csv.write_text("architecture_id,family\narch_a,CNN\n")
    stats_csv = tmp_path / "statistical_comparison.csv"
    _write_valid_statistical_comparison(stats_csv)

    result = load_results(
        fold_results_csv=str(fold_csv),
        statistical_comparison_csv=str(stats_csv),
    )

    assert result.available is False
    assert "Faltan columnas" in result.reason


def test_valid_csvs_are_available_and_extract_metric_columns(tmp_path):
    fold_csv = tmp_path / "fold_results.csv"
    _write_valid_fold_results(fold_csv)
    stats_csv = tmp_path / "statistical_comparison.csv"
    _write_valid_statistical_comparison(stats_csv)

    result = load_results(
        fold_results_csv=str(fold_csv),
        statistical_comparison_csv=str(stats_csv),
    )

    assert result.available is True
    assert result.metric_columns == ["mae"]
    assert len(result.fold_results) == 4


def test_optional_tables_missing_file_is_not_available(tmp_path):
    from services.results_loader import load_cost_metrics, load_pairwise

    assert load_pairwise(str(tmp_path / "missing.csv")).available is False
    assert load_cost_metrics(str(tmp_path / "missing.csv")).available is False


def test_real_optional_tables_are_available():
    from services.results_loader import load_cost_metrics, load_pairwise

    pairwise = load_pairwise()
    cost = load_cost_metrics()

    assert pairwise.available and len(pairwise.data) == 15
    assert cost.available and len(cost.data) == 6


def test_optional_table_missing_columns_is_not_available(tmp_path):
    from services.results_loader import load_pairwise

    csv = tmp_path / "pairwise.csv"
    csv.write_text("architecture_a,architecture_b\na,b\n")

    result = load_pairwise(str(csv))

    assert result.available is False
    assert "Faltan columnas" in result.reason
