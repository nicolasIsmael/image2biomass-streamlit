from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import streamlit as st

from components.prediction_result_card import render as render_prediction
from components.progress_indicator import render as render_progress
from services.architecture_registry import load_architecture_registry
from services.image_validation import validate_image
from services.inference import predict
from services.model_registry import ModelLoadError, load_checkpoint

TIMEOUT_SECONDS = 45
EXAMPLE_IMAGES_DIR = Path("data/example_images")


def _predict_with(architecture_id: str, image):
    loaded = load_checkpoint(architecture_id)
    return predict(loaded, image)


render_progress(current_key="prediccion")

st.title("🔮 Predicción en vivo")
st.caption(
    "Elegí una imagen de ejemplo o subí la tuya. Vas a ver, lado a lado, la "
    "predicción del modelo de **mejor** y de **peor** desempeño de este estudio "
    "(la diferencia que sí resultó significativa en el análisis)."
)

architecture_result = load_architecture_registry()

if not architecture_result.available:
    st.warning(
        "🔧 **Espacio preparado, artefacto pendiente.** La predicción en vivo "
        "necesita `data/fold_results.csv` y `data/statistical_comparison.csv` para "
        f"saber qué 2 modelos cargar. Detalle técnico: {architecture_result.reason}"
    )
    st.stop()

best = next((a for a in architecture_result.architectures if a.is_best), None)
worst = next((a for a in architecture_result.architectures if a.is_worst), None)

if not best or not worst:
    st.warning(
        "🔧 `data/statistical_comparison.csv` aún no indica cuál arquitectura es la "
        "de mejor/peor desempeño (`best_architecture_id`/`worst_architecture_id`)."
    )
    st.stop()

source = st.radio("Fuente de la imagen", ["Imagen de ejemplo", "Subir mi propia imagen"])

image_bytes = None
if source == "Imagen de ejemplo":
    example_files = sorted(EXAMPLE_IMAGES_DIR.glob("*.jpg")) + sorted(
        EXAMPLE_IMAGES_DIR.glob("*.png")
    )
    if not example_files:
        st.info(
            "🔧 Espacio preparado: aún no hay imágenes de ejemplo en "
            f"`{EXAMPLE_IMAGES_DIR}/` (ver data/example_images/README.md)."
        )
    else:
        chosen = st.selectbox("Elegí una imagen", example_files, format_func=lambda p: p.name)
        image_bytes = chosen.read_bytes()
        st.image(image_bytes, caption=chosen.name, width=300)
else:
    uploaded = st.file_uploader("Subí una imagen top-view de pasto", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        st.image(image_bytes, caption="Tu imagen", width=300)

if image_bytes is None:
    st.stop()

validation = validate_image(image_bytes)
if not validation.valid:
    st.error(f"⚠️ {validation.error_message}")
    st.stop()

if st.button("Generar predicción"):
    with st.spinner("Cargando modelos por primera vez (puede tardar hasta 45s)..."):
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_best = executor.submit(
                    _predict_with, best.architecture_id, validation.image
                )
                future_worst = executor.submit(
                    _predict_with, worst.architecture_id, validation.image
                )
                best_prediction = future_best.result(timeout=TIMEOUT_SECONDS)
                worst_prediction = future_worst.result(timeout=TIMEOUT_SECONDS)
        except FutureTimeoutError:
            st.error(
                "⏱️ La predicción está tardando más de lo esperado (>45s). Probá de "
                "nuevo en unos segundos — puede deberse a la carga en frío de los "
                "modelos."
            )
            st.stop()
        except ModelLoadError as exc:
            st.error(f"⚠️ No se pudo generar la predicción: {exc}")
            st.stop()

    render_prediction(
        best_prediction,
        worst_prediction,
        best_label=f"{best.variant_name} (mejor desempeño)",
        worst_label=f"{worst.variant_name} (peor desempeño)",
    )

    if image_bytes is not None:
        # La imagen subida solo vive en memoria para este request; no se persiste ni
        # se registra en logs (FR-018).
        image_bytes = None
