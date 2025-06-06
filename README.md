# Ensamble Secuencial de Modelos Predictivos

**Curso:** Inteligencia Artificial (IA) – 2024/25  
**Autores:** Carlos Martín de Prado Barragán y Marco Padilla Gómez  
**Convocatoria:** Primera convocatoria – Junio 2025  
**Tipo de tarea:** Regresión  
**Lenguaje:** Python + Scikit-Learn  
**Entorno:** Jupyter Notebooks

---

## 📌 Descripción general

Este proyecto implementa un **meta-modelo de ensamble secuencial**, diseñado para tareas de regresión, utilizando técnicas de aprendizaje aditivo. La idea central es construir una secuencia de modelos que, de forma iterativa, se especialicen en corregir los errores de sus predecesores.

Se trabaja con dos algoritmos base:
- `DecisionTreeRegressor` (estimador obligatorio)
- `LinearRegression` (estimador de libre elección)

Y con dos conjuntos de datos proporcionados por la asignatura:
- `parkinsons.csv` – Información biomédica relacionada con la enfermedad de Parkinson.
- `house_prices.csv` – Atributos de viviendas y sus precios de venta.

---

## 🧠 Objetivos principales

- Implementar un ensamble secuencial de modelos de regresión.
- Aplicar el modelo sobre dos conjuntos de datos reales.
- Explorar hiperparámetros como:
  - Número de modelos (`n_estimators`)
  - Proporción de muestra (`sample_size`)
  - Tasa de aprendizaje (`learning_rate`)
  - Parámetros específicos del estimador (`max_depth`, etc.)
- Comparar el rendimiento con modelos base simples.
- Evaluar el rendimiento con **validación cruzada** y métricas como **R²**.

---

## ⚙️ Requisitos

- Python ≥ 3.8  
- Scikit-learn ≥ 1.0  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn (opcional, para visualización)

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo ejecutar

1. Abre los notebooks en Jupyter o VSCode.
2. Asegúrate de que los archivos CSV están en la carpeta `data/`.
3. Ejecuta las celdas en orden para entrenar, predecir y evaluar el modelo.
4. Puedes modificar los hiperparámetros al final de cada notebook para observar cómo afectan al rendimiento.

---

## 📈 Evaluación y resultados

Cada conjunto de experimentos incluye:

- Separación de datos en entrenamiento y prueba.
- Entrenamiento del meta-modelo con `SequentialEnsembleRegressor`.
- Comparación con un modelo base sin ensamble.
- Visualización de predicciones vs. valores reales.
- Métricas como el **R²** y el **error cuadrático medio (MSE)**.
- Validación cruzada sobre configuraciones de hiperparámetros.

---

## 📄 Documentación técnica

La clase `SequentialEnsembleRegressor` está definida en `sequential_ensemble.py` e implementa:

- Entrenamiento secuencial con residuales.
- Predicción agregada con tasa de aprendizaje.
- Compatibilidad total con `Scikit-Learn` (`fit`, `predict`, `score`).
- Uso flexible con cualquier regresor de `sklearn`.

---

## 🧪 Trabajo futuro

- Implementación de parada temprana (early stopping).
- Extensión a tareas de clasificación binaria.
- Inclusión de nuevos algoritmos base como `KNeighborsRegressor` o `SVR`.
- Optimización de hiperparámetros con técnicas automáticas (e.g. GridSearchCV).

---

## 🧾 Licencia

Uso exclusivamente académico, bajo los términos de la asignatura Inteligencia Artificial (IA), Universidad de Sevilla, 2025.
