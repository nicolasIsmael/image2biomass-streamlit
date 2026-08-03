import io

from PIL import Image

from services.image_validation import validate_image


def _make_image_bytes(width=200, height=200, fmt="JPEG") -> bytes:
    image = Image.new("RGB", (width, height), color=(0, 128, 0))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_valid_jpeg_passes():
    result = validate_image(_make_image_bytes(fmt="JPEG"))

    assert result.valid is True
    assert result.image is not None
    assert result.error_message is None


def test_valid_png_passes():
    result = validate_image(_make_image_bytes(fmt="PNG"))

    assert result.valid is True


def test_corrupt_file_produces_understandable_error():
    result = validate_image(b"not-a-real-image-just-random-bytes")

    assert result.valid is False
    assert result.image is None
    assert "no parece ser una imagen" in result.error_message


def test_image_too_small_is_rejected():
    result = validate_image(_make_image_bytes(width=10, height=10))

    assert result.valid is False
    assert "demasiado pequeña" in result.error_message


def test_image_too_large_is_rejected():
    result = validate_image(_make_image_bytes(width=9000, height=9000))

    assert result.valid is False
    assert "demasiado grande" in result.error_message
