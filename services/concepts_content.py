"""Contenido editorial de los conceptos de Deep Learning interactivos (User Story 2).

No es un artefacto experimental (no vive en data/): es contenido curado a mano, cada
uno con su control interactivo y su conexión explícita a los datos/hechos reales del
proyecto (FR-004). Los controles que simulan un valor (ej. épocas) se etiquetan como
simulación pedagógica -- nunca se presentan como el valor real usado en el
entrenamiento, salvo que ese valor provenga de un artefacto real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ControlType = Literal["slider", "select", "button"]


@dataclass
class DLConcept:
    concept_id: str
    title: str
    explanation: str
    control_type: ControlType
    control_label: str
    control_options: tuple
    real_project_binding: str


CONCEPTS: list[DLConcept] = [
    DLConcept(
        concept_id="epoca",
        title="¿Qué es una época?",
        explanation=(
            "Una **época** es una pasada completa del modelo por todos los datos de "
            "entrenamiento. Entrenar varias épocas le da al modelo más oportunidades "
            "de ajustar sus parámetros, pero demasiadas pueden hacer que memorice en "
            "vez de generalizar (sobreajuste)."
        ),
        control_type="slider",
        control_label="Número de épocas (simulación pedagógica)",
        control_options=(1, 50, 1, 10),
        real_project_binding=(
            "El número real de épocas usado para cada arquitectura de este estudio "
            "está documentado en el pipeline de entrenamiento, no en este control "
            "interactivo, que solo ilustra el concepto."
        ),
    ),
    DLConcept(
        concept_id="validacion_cruzada",
        title="¿Qué es la validación cruzada (folds)?",
        explanation=(
            "En vez de separar los datos una sola vez en entrenamiento/prueba, la "
            "**validación cruzada** los divide en varias partes (folds) y repite el "
            "proceso, usando cada parte como prueba una vez. Esto da una evaluación "
            "más confiable del desempeño real del modelo."
        ),
        control_type="slider",
        control_label="Fold a inspeccionar",
        control_options=(1, 5, 1, 1),
        real_project_binding=(
            "Las 357 imágenes reales de este estudio se dividieron exactamente en "
            "5 folds; mueve el control para ver la partición real de cada uno en la "
            "sección de Comparación."
        ),
    ),
    DLConcept(
        concept_id="cnn_vs_transformer",
        title="CNN vs Transformer vs Híbrido",
        explanation=(
            "Una **CNN** aprende patrones locales (bordes, texturas) mediante "
            "filtros que se deslizan por la imagen. Un **Transformer** analiza la "
            "imagen completa de una vez, relacionando todas sus partes entre sí. Un "
            "modelo **Híbrido** combina ambos enfoques."
        ),
        control_type="select",
        control_label="Familia de arquitectura",
        control_options=("CNN", "Transformer", "Híbrido"),
        real_project_binding=(
            "Las 6 arquitecturas comparadas en este estudio se agrupan exactamente "
            "en estas 3 familias (ver la sección de Comparación para sus nombres y "
            "resultados reales)."
        ),
    ),
    DLConcept(
        concept_id="backbone_congelado",
        title="¿Qué es un backbone congelado?",
        explanation=(
            "Un **backbone** es la parte del modelo que ya aprendió a reconocer "
            "patrones visuales generales (entrenada previamente con millones de "
            "imágenes). 'Congelarlo' significa no modificar esos pesos durante el "
            "entrenamiento, ahorrando tiempo y evitando sobreajuste con pocos datos."
        ),
        control_type="button",
        control_label="Alternar: backbone congelado / entrenable",
        control_options=("Congelado", "Entrenable"),
        real_project_binding=(
            "Con solo 357 muestras reales disponibles, congelar el backbone ayuda a "
            "que el modelo no memorice el dataset de entrenamiento."
        ),
    ),
    DLConcept(
        concept_id="cabeza_regresion",
        title="¿Qué es la cabeza de regresión?",
        explanation=(
            "La **cabeza de regresión** es la última parte del modelo: toma lo que "
            "el backbone 'vio' en la imagen y lo convierte en números — en este "
            "caso, valores de biomasa por componente."
        ),
        control_type="select",
        control_label="Variable objetivo a estimar (ejemplo)",
        control_options=("Green", "Clover", "Dead"),
        real_project_binding=(
            "Es multisalida: una sola cabeza produce las 3 variables reales del "
            "modelo (Green, Clover, Dead) en una sola pasada; Total y GDM se "
            "derivan de esas 3. La lista completa de variables del dataset se "
            "muestra en la sección de Dataset."
        ),
    ),
]


def get_concept(concept_id: str) -> DLConcept:
    for concept in CONCEPTS:
        if concept.concept_id == concept_id:
            return concept
    raise KeyError(f"Concepto desconocido: {concept_id}")
