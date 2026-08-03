# Image2Biomass — Recorrido de Sustentación

App Streamlit que acompaña la sustentación de una tesis de pregrado: evaluación
comparativa de arquitecturas de Deep Learning (CNN, Transformer e Híbridas) para
estimación multisalida de biomasa de pasturas a partir de imágenes top-view (dataset
CSIRO Image2Biomass, Liao et al., Kaggle 2025).

Es un recorrido guiado — no un dashboard técnico — pensado para un jurado sin
formación en Deep Learning.

## Setup rápido

```bash
python -m venv .venv
source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
cp .env.example .env        # completar HF_REPO_ID cuando los checkpoints existan
streamlit run app.py
```

## Estado de los artefactos de datos

Este repositorio contiene el **código completo de la app**, pero los artefactos reales
del estudio (`data/dataset_metadata.csv`, `data/fold_results.csv`,
`data/statistical_comparison.csv`, `data/example_images/*`, y los 2 checkpoints en
Hugging Face Hub) todavía no están cargados. Mientras eso ocurre, cada página muestra
un espacio preparado ("🔧 artefacto pendiente") en vez de datos simulados. El esquema
exacto que cada artefacto debe cumplir está documentado en
[`data/README.md`](data/README.md).

## Desarrollo

```bash
ruff check .
pytest tests/unit
python tests/ci/smoke_test.py   # arranca la app real y verifica que responde
```

## Estructura

- `app.py` — navegación multipage (`st.navigation`).
- `pages/` — una página por sección del recorrido.
- `components/` — UI reutilizable (gráficos, tarjetas, indicador de progreso, panel de
  limitaciones).
- `services/` — carga de datos/modelos, inferencia, validación de imágenes.
- `data/` — artefactos versionados (CSVs + imágenes de ejemplo); los checkpoints van en
  Hugging Face Hub, no aquí.

## Nota sobre el repositorio

Este repositorio contiene únicamente el código de la aplicación. Las carpetas de
herramientas de desarrollo asistido usadas para planificar y generar este proyecto
(`.claude/`, `.specify/`, `specs/`) están excluidas vía `.gitignore` y no se suben —
no forman parte del producto ni son necesarias para ejecutarlo.
