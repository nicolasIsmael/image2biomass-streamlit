from services.dataset_loader import load_dataset


def test_missing_file_is_not_available(tmp_path):
    result = load_dataset(str(tmp_path / "does_not_exist.csv"))

    assert result.available is False
    assert "No se encontró" in result.reason


def test_empty_file_is_not_available(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "sample_id,image_path,Sampling_Date,State,Species,Pre_GSHH_NDVI,Height_Ave_cm\n"
    )

    result = load_dataset(str(csv_path))

    assert result.available is False
    assert "vacío" in result.reason


def test_missing_required_columns_is_not_available(tmp_path):
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text("sample_id,image_path\nA1,img/a1.jpg\n")

    result = load_dataset(str(csv_path))

    assert result.available is False
    assert "Faltan columnas" in result.reason


def test_valid_csv_is_available_and_extracts_target_columns(tmp_path):
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(
        "sample_id,image_path,Sampling_Date,State,Species,Pre_GSHH_NDVI,"
        "Height_Ave_cm,green_g,clover_g\n"
        "A1,img/a1.jpg,2015/9/4,Tas,Ryegrass,0.62,4.6,120.5,15.0\n"
        "A2,img/a2.jpg,2015/9/1,WA,Lucerne,0.55,16.0,98.0,4.2\n"
    )

    result = load_dataset(str(csv_path))

    assert result.available is True
    assert len(result.samples) == 2
    assert sorted(result.target_columns) == ["clover_g", "green_g"]
