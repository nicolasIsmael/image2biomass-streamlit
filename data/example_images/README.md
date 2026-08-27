# `data/example_images/` — Imágenes de ejemplo

5 imágenes top-view de pasto en **resolución completa** (mismas dimensiones que el
artefacto original, ~2000×1000, 2-3 MB c/u), usadas como opciones rápidas en la
sección de predicción en vivo (`pages/5_prediccion_en_vivo.py`). Se usa resolución
completa (no miniaturas) porque el modelo predice sobre imágenes reales, no
comprimidas.

`pages/5_prediccion_en_vivo.py` lista automáticamente todos los `.jpg`/`.png` de esta
carpeta — no hace falta registrar nombres de archivo en ningún otro lugar.

Cada imagen se nombra igual que su `sample_id` en `data/dataset_metadata.csv` (ver
`../README.md`).

## Criterio de selección

Elegidas a partir de `data/dataset_metadata.csv` para maximizar la variedad visual y
de composición de biomasa (una fila por variable objetivo real, sin valores
inventados):

| sample_id | Motivo | Dry_Clover_g | Dry_Dead_g | Dry_Green_g | Dry_Total_g | State / Species |
|---|---|---|---|---|---|---|
| `ID1831254380` | Máximo `Dry_Clover_g` del dataset | 71.79 | 15.76 | 0.88 | 88.42 | Tas / Clover |
| `ID384648061` | Máximo `Dry_Green_g` (y máximo `Dry_Total_g`) | 0.00 | 27.72 | 157.98 | 185.70 | NSW / Fescue |
| `ID1139866256` | Máximo `Dry_Dead_g` | 0.00 | 83.84 | 74.06 | 157.90 | NSW / Fescue |
| `ID746335827` | Alto `Dry_Total_g`, distinto perfil al de green máximo | 0.00 | 53.26 | 112.84 | 166.10 | NSW / Fescue |
| `ID1963715583` | Mínimo `Dry_Total_g` (parcela casi sin biomasa) | 0.34 | 0.00 | 0.70 | 1.04 | WA / Clover |

Mientras esta carpeta esté vacía, la página muestra un espacio preparado en vez de
imágenes simuladas (Principio VI de la constitución del proyecto).
