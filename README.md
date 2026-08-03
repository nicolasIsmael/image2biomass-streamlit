# Image2Biomass

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