"""Catálogo de los 10 conceptos de Deep Learning, cada uno un HTML interactivo de una
sola ventana (carpeta htmls/) en orden pedagógico: neurona -> R² vs promedio.

No es un artefacto experimental (no vive en data/): es contenido curado a mano. Cada
concepto trae un vínculo con el proyecto real (FR-004). Los hechos que dependen de
resultados (folds, familias, R²) se calculan desde data/fold_results.csv cuando está
disponible; si no, se muestra un texto genérico sin cifras -- nunca datos simulados
(Principio VI).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DLConcept:
    concept_id: str
    number: int
    title: str
    html_file: str
    real_project_binding: str


CONCEPTS: list[DLConcept] = [
    DLConcept(
        concept_id="neurona",
        number=1,
        title="La neurona",
        html_file="01 La neurona.html",
        real_project_binding=(
            "Cada una de las 6 arquitecturas comparadas es, en el fondo, una red de "
            "unidades como esta, en una versión mucho más grande."
        ),
    ),
    DLConcept(
        concept_id="capas_profundidad",
        number=2,
        title="Capas y profundidad",
        html_file="02 Capas y profundidad.html",
        real_project_binding=(
            "Las 6 arquitecturas comparadas son redes profundas: apilan muchas capas "
            "para reconocer patrones cada vez más complejos en la foto del potrero."
        ),
    ),
    DLConcept(
        concept_id="error",
        number=3,
        title="El error (pérdida)",
        html_file="03 El error.html",
        real_project_binding=(
            "El estudio mide el error de cada modelo con MAE y RMSE por variable "
            "(Green, Dead, Clover, GDM y Total); los valores reales están en la "
            "sección de Comparación."
        ),
    ),
    DLConcept(
        concept_id="colina_error",
        number=4,
        title="Aprender: bajar la colina del error",
        html_file="04 Colina del error.html",
        real_project_binding=(
            "Entrenar cada una de las 6 arquitecturas es ajustar sus parámetros, paso "
            "a paso, para que el error sobre las muestras de entrenamiento baje."
        ),
    ),
    DLConcept(
        concept_id="sobreajuste",
        number=5,
        title="Sobreajuste",
        html_file="05 Sobreajuste.html",
        real_project_binding=(
            "Con solo 357 muestras reales el riesgo de memorizar es alto; por eso cada "
            "modelo se evalúa siempre sobre muestras que no vio al entrenar."
        ),
    ),
    DLConcept(
        concept_id="validacion_honesta",
        number=6,
        title="Validación honesta (fuga de datos)",
        html_file="06 Validación honesta.html",
        real_project_binding=(
            "Las 357 muestras reales se dividieron en 5 folds; cada modelo se evalúa "
            "en el fold que no usó para entrenar."
        ),
    ),
    DLConcept(
        concept_id="convolucion",
        number=7,
        title="Convolución: mirar una imagen con una ventana",
        html_file="07 La convolución.html",
        real_project_binding=(
            "Las arquitecturas de la familia CNN del estudio aprenden filtros como "
            "estos; ver la sección de Comparación para sus nombres y resultados."
        ),
    ),
    DLConcept(
        concept_id="atencion",
        number=8,
        title="Atención: decidir a qué mirar",
        html_file="08 Atención.html",
        real_project_binding=(
            "Las arquitecturas de la familia Transformer del estudio usan atención; "
            "ver la sección de Comparación para sus nombres y resultados."
        ),
    ),
    DLConcept(
        concept_id="transfer_learning",
        number=9,
        title="Transfer learning: no empezar de cero",
        html_file="09 Transfer learning.html",
        real_project_binding=(
            "Con solo 357 muestras reales, reutilizar un backbone preentrenado y "
            "congelado ayuda a no memorizar el dataset; la cabeza de regresión, "
            "multisalida, estima Green, Clover y Dead en una sola pasada (Total y GDM "
            "se derivan de esas 3)."
        ),
    ),
    DLConcept(
        concept_id="r2_vs_promedio",
        number=10,
        title="R²: ¿mejor que adivinar el promedio?",
        html_file="10 R² vs promedio.html",
        real_project_binding=(
            "Los valores reales de Weighted R² de cada arquitectura están en la "
            "sección de Comparación."
        ),
    ),
]


def get_concept(concept_id: str) -> DLConcept:
    for concept in CONCEPTS:
        if concept.concept_id == concept_id:
            return concept
    raise KeyError(f"Concepto desconocido: {concept_id}")


def _variants_of_family(fold_results: pd.DataFrame, family: str) -> list[str]:
    rows = fold_results[fold_results["family"] == family]
    return sorted(rows["variant_name"].unique())


def resolve_binding(concept: DLConcept, fold_results: pd.DataFrame | None) -> str:
    """Vínculo con el proyecto real; usa cifras de fold_results solo si están disponibles."""
    if fold_results is None or fold_results.empty:
        return concept.real_project_binding

    if concept.concept_id == "validacion_honesta":
        sizes = fold_results.groupby("fold_number")["n_samples_val"].first()
        return (
            f"Las {int(sizes.sum())} muestras reales se dividieron en {len(sizes)} folds "
            f"(entre {int(sizes.min())} y {int(sizes.max())} muestras de validación "
            "cada uno); cada modelo se evalúa en el fold que no usó para entrenar."
        )

    if concept.concept_id in {"convolucion", "atencion"}:
        family = "CNN" if concept.concept_id == "convolucion" else "Transformer"
        names = _variants_of_family(fold_results, family)
        if names:
            verb = "aprenden filtros como estos" if family == "CNN" else "usan atención"
            return (
                f"{len(names)} de las {fold_results['architecture_id'].nunique()} "
                f"arquitecturas del estudio son {family} y {verb}: {', '.join(names)}."
            )

    if concept.concept_id == "r2_vs_promedio" and "weighted_r2" in fold_results.columns:
        by_arch = fold_results.groupby("variant_name")["weighted_r2"].mean().sort_values()
        worst, best = by_arch.index[0], by_arch.index[-1]
        text = (
            f"Weighted R² medio por arquitectura (5 folds): de {by_arch.iloc[0]:.2f} "
            f"({worst}) a {by_arch.iloc[-1]:.2f} ({best})."
        )
        if by_arch.iloc[0] < 0:
            text += f" {worst} queda por debajo de simplemente predecir el promedio."
        return text

    return concept.real_project_binding
