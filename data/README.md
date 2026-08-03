# `data/` — Artefactos versionados del proyecto

Ver el contrato completo en
[`specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md`](../specs/001-recorrido-guiado-biomasa/contracts/data-artifacts-contract.md).
Los checkpoints de modelo **no** van aquí — se alojan en Hugging Face Hub (ver
`.env.example` → `HF_REPO_ID`).

## `dataset_metadata.csv` (pendiente)

Una fila por muestra del dataset (357 filas esperadas). Columnas requeridas:

- `sample_id` — identificador único de la muestra.
- `image_path` — ruta relativa a la imagen top-view.
- `latitude`, `longitude` — ubicación geográfica de la muestra.
- Una columna por cada variable objetivo real del estudio (ej. componentes de
  biomasa como Green, Clover, Dead — nombres exactos a confirmar contra el
  artefacto real del pipeline de entrenamiento).

`services/dataset_loader.py` trata cualquier columna que no sea `sample_id`,
`image_path`, `latitude` o `longitude` como variable objetivo automáticamente.

## `fold_results.csv` (pendiente)

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

## `statistical_comparison.csv` (pendiente)

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

## `example_images/` (pendiente)

Ver `data/example_images/README.md`.

---

Mientras un artefacto no exista, el loader correspondiente (`services/`) devuelve un
estado "no disponible" explícito y la página muestra un espacio preparado — nunca
datos simulados (Principio VI de la constitución del proyecto).
