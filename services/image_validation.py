"""Validación de imágenes subidas por el usuario (User Story 4, FR-010).

Usa Pillow para confirmar que el archivo es una imagen legible antes de pasarla al
preprocesamiento de inferencia. Nunca deja pasar una excepción sin manejar al usuario.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_DIMENSION_PX = 64
MAX_DIMENSION_PX = 8000


@dataclass
class ImageValidationResult:
    valid: bool
    image: Image.Image | None = None
    error_message: str | None = None


def validate_image(file_bytes: bytes) -> ImageValidationResult:
    try:
        probe = Image.open(io.BytesIO(file_bytes))
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return ImageValidationResult(
            valid=False,
            error_message="El archivo no parece ser una imagen válida. Probá con un JPG o PNG.",
        )

    # Image.verify() deja el objeto inutilizable para más operaciones; se reabre.
    image = Image.open(io.BytesIO(file_bytes))

    if image.format not in ALLOWED_FORMATS:
        return ImageValidationResult(
            valid=False,
            error_message=f"Formato no soportado ({image.format}). Usá JPG o PNG.",
        )

    width, height = image.size
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        return ImageValidationResult(
            valid=False,
            error_message="La imagen es demasiado pequeña. Subí una foto de mayor resolución.",
        )
    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        return ImageValidationResult(
            valid=False,
            error_message="La imagen es demasiado grande. Subí una foto de menor resolución.",
        )

    return ImageValidationResult(valid=True, image=image.convert("RGB"))
