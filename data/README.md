# `data/` — Artefactos versionados del proyecto

Ver el contrato completo en
[`specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md`](../specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md).
Los checkpoints de modelo **no** van aquí — se alojan en Hugging Face Hub (ver
`.env.example` → `HF_REPO_ID`).

## `dataset_metadata.csv`

Una fila por muestra del dataset (357 filas, derivadas de `train.csv` del artefacto
Kaggle CSIRO Image2Biomass — `test.csv` se descarta porque no trae mediciones reales,
solo la imagen). Columnas:

- `sample_id` — identificador único de la muestra (stem del nombre de archivo).
- `image_path` — ruta relativa a la imagen top-view dentro del artefacto original
  (`train/<sample_id>.jpg`); las imágenes en sí no se versionan en este repo (1.1 GB).
- `Sampling_Date`, `State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm` — metadata real
  de la muestra. El dataset no incluye lat/long; `State` (estado australiano) es el
  equivalente real de ubicación geográfica.
- `Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `Dry_Total_g`, `GDM_g` — las 5 variables
  objetivo reales del estudio (componentes de biomasa seca + biomasa verde
  disponible/GDM), en formato ancho (una columna por variable; el CSV original de
  Kaggle viene en formato largo, una fila por combinación muestra×variable).

`services/dataset_loader.py` trata cualquier columna que no sea metadata
(`sample_id`, `image_path`, `Sampling_Date`, `State`, `Species`, `Pre_GSHH_NDVI`,
`Height_Ave_cm`) como variable objetivo automáticamente.

## `fold_results.csv`

Una fila por combinación arquitectura × fold (30 filas esperadas: 6 arquitecturas ×
5 folds). Columnas requeridas por `services/results_loader.py` y
`services/architecture_registry.py`:

- `architecture_id`, `family`, `variant_name` — identidad de la arquitectura (se
  deduplica por `architecture_id` para construir el registro de arquitecturas).
- `fold_number` — número de fold (1-5).
- `n_samples_val` — tamaño real de la partición de validación de ese fold (usado
  para detectar y explicar el desbalance del Fold 2, User Story 5).
- Una o más columnas de métrica de error real por variable objetivo (ej. `mae`,
  `rmse` — nombre exacto TBD); cualquier columna que no sea una de las anteriores se
  trata automáticamente como columna de métrica.

## `statistical_comparison.csv`

Una fila por comparación estadística reportada en el estudio. Columnas requeridas:

- `test_name` — nombre real de la prueba estadística usada.
- `p_value` — valor real reportado por el análisis.
- `is_significant` — booleano ya derivado por el pipeline de análisis (la app no
  recalcula el test).
- `plain_language_summary` — traducción a lenguaje simple del resultado, sin alterar
  la conclusión estadística.
- `best_architecture_id`, `worst_architecture_id` — `architecture_id` (coincidente
  con `fold_results.csv`) de mejor y peor desempeño; determinan los 2 checkpoints
  usados en la predicción en vivo (User Story 4).

Metodología (igual que el paper): ANOVA de medidas repetidas (modelo como factor
intra-sujeto, fold como sujeto) + t pareado con corrección de Holm; `p_value` es el
p ajustado de mejor vs. peor. Columnas opcionales (`omnibus_*`, `robustness_*`,
`n_pairs_*`) documentan el ómnibus y el chequeo Friedman + Nemenyi.

## `pairwise_significance.csv`

Los 15 pares de arquitecturas: `architecture_a`, `architecture_b`, `t_statistic`,
`p_raw`, `p_holm`, `is_significant` (α=0.05, t pareado + Holm), `p_nemenyi`,
`is_significant_nemenyi`. Equivale a `table4_pairwise_significance` del paper.

## `cost_metrics.csv`

Una fila por arquitectura: tiempo de entrenamiento total/medio/SD por fold (min, de
Google Colab; solo el factor relativo `train_relative_to_fastest` es comparable) y
latencia de inferencia en CPU (mediana/media/SD en ms, `infer_relative_to_fastest`),
más `params_millions`. Equivale a `table5`/`table6` del paper.

## Regenerar los CSV de resultados

`statistical_comparison.csv`, `pairwise_significance.csv` y `cost_metrics.csv` se
generan con `resultados/build_app_data.py` (reutiliza el pipeline estadístico de
`resultados/build_paper_figures.py`), no se editan a mano.

## Convención de la desviación estándar

La app muestra la SD entre folds como **poblacional (ddof=0)**, igual que las tablas
del paper (p. ej. ConvNeXt-Tiny 0.099; con ddof=1 sería 0.110).

## `dataset_thumbnails/`

Miniaturas JPEG comprimidas (320px de ancho, calidad 65) de las 357 imágenes reales
de `train/`, una por `sample_id`, usadas por la galería de la página de dataset
(`pages/2_dataset.py`). ~6 MB en total frente a los ~1.1 GB del artefacto original en
resolución completa — por eso no se versiona el dataset de imágenes completo, solo
estas miniaturas de exhibición. Generadas con un script puntual (no versionado) que
redimensiona cada imagen de `train/` con Pillow; si el dataset original cambia, hay
que regenerarlas.

## `example_images/`

Ver `data/example_images/README.md`. A diferencia de `dataset_thumbnails/`, estas 5
imágenes se guardan en **resolución completa** porque alimentan la predicción en vivo
(el modelo espera imágenes reales, no miniaturas).

---

Mientras un artefacto no exista, el loader correspondiente (`services/`) devuelve un
estado "no disponible" explícito y la página muestra un espacio preparado — nunca
datos simulados (Principio VI de la constitución del proyecto).
