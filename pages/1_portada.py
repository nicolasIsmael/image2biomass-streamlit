import streamlit as st

from components.progress_indicator import render as render_progress

render_progress(current_key="portada")

st.title("🌱 Estimar la biomasa de un potrero, con una sola foto")

st.caption(
    "Esta sección prioriza **claridad pedagógica** sobre rigor estadístico "
    "(Principio I de la constitución del proyecto): el objetivo es que cualquier "
    "persona, sin formación en agronomía ni en Deep Learning, entienda el problema."
)

st.markdown(
    """
### ¿Por qué medir la biomasa de una pastura?

La **biomasa** de una pastura es, en términos simples, cuánto pasto hay disponible
para el ganado en un momento dado. Saber esto le permite a un productor decidir
cuándo mover los animales a otro potrero, cuánto forraje sembrar, o si un terreno
está siendo sub o sobre-utilizado.

El método tradicional para medirla es **destructivo y lento**: alguien corta una
muestra de pasto en una parcela pequeña, la seca en un horno y la pesa. Es preciso,
pero no es viable hacerlo en cientos de hectáreas de forma frecuente.

### La idea: estimarla a partir de una foto

¿Y si, en vez de cortar y pesar pasto, pudiéramos **tomar una foto desde arriba**
(vista "top-view") de un pedazo de potrero y que un modelo de Deep Learning
estimara la biomasa directamente de esa imagen? Eso es lo que este proyecto de tesis
evalúa: distintas arquitecturas de redes neuronales, entrenadas con fotos reales y
sus mediciones reales de biomasa, para ver **cuál aprende a hacer esta estimación
mejor**.

### Lo que vas a recorrer

1. **El dataset** — las fotos y mediciones reales usadas en este estudio.
2. **Conceptos de Deep Learning** — de forma interactiva, sin jerga innecesaria.
3. **La comparación** — qué arquitectura funcionó mejor, y si esa diferencia es
   real o pudo haber sido casualidad.
4. **Una demo en vivo** — vas a poder probar el modelo con una foto tuya.

Usa el menú lateral (o los botones de cada página) para moverte libremente por el
recorrido, en el orden que prefieras.
"""
)
