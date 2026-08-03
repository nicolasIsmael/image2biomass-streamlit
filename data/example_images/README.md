# `data/example_images/` — Imágenes de ejemplo (pendiente)

Exactamente **5 imágenes** top-view de pasto (formato `.jpg` o `.png`),
representativas del dataset, usadas como opciones rápidas en la sección de
predicción en vivo (`pages/5_prediccion_en_vivo.py`).

`pages/5_prediccion_en_vivo.py` lista automáticamente todos los `.jpg`/`.png` de esta
carpeta — no hace falta registrar nombres de archivo en ningún otro lugar.

Si se desea, cada imagen puede nombrarse igual que su `sample_id` en
`data/dataset_metadata.csv` (ver `../README.md`) para poder mostrar su valor real de
biomasa conocido en una iteración futura (ver Assumptions en spec.md).

Mientras esta carpeta esté vacía, la página muestra un espacio preparado en vez de
imágenes simuladas (Principio VI de la constitución del proyecto).
