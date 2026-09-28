# Ensamble secuencial de modelos predictivos

Proyecto de aprendizaje automático que implementa un regresor por ensamble: entrena varios modelos en secuencia para que cada uno aprenda a corregir los errores acumulados de los anteriores.

Se estudia su comportamiento con árboles de decisión y regresión lineal, sobre precios de viviendas y medidas de progresión del Parkinson. El objetivo es comparar configuraciones y comprender cuándo el ensamble mejora un modelo individual.

Trabajo en equipo de Inteligencia Artificial, Universidad de Sevilla, curso 2024/25. Autores: Carlos Martín de Prado Barragán y Marco Padilla Gómez.

## Qué incluye

- Implementación de `SequentialEnsembleRegressor` con `fit`, `predict` y `score`.
- Muestreo sin reemplazo en cada iteración y ponderación de predicciones mediante una tasa de aprendizaje.
- Doce notebooks: entrenamiento, experimentación con hiperparámetros y validación cruzada para cada combinación de dataset y regresor.
- Preprocesamiento de variables, gráficas y comparación con modelos individuales.
- Datos CSV y memoria del trabajo.

## Cómo funciona

1. La predicción acumulada comienza en cero.
2. Se calculan los residuos: diferencia entre los valores reales y la predicción actual.
3. Un nuevo regresor aprende esos residuos sobre una muestra del conjunto de entrenamiento.
4. Su predicción, ponderada por `lr`, se suma al ensamble.
5. Se repite el proceso hasta alcanzar `n_estimators`.

El estimador recibe la **clase** del regresor base, no una instancia. Trabaja con matrices NumPy; convertir un DataFrame antes de llamar a `fit`.

## Tecnologías

Python, NumPy, pandas, scikit-learn, Matplotlib y Seaborn. Los experimentos se ejecutan en Jupyter.

## Instalación

Usar un entorno de Python compatible con las dependencias. Los notebooks utilizan `OneHotEncoder(sparse_output=False)`, que requiere scikit-learn 1.2 o posterior.

```sh
git clone https://github.com/carlosmdpb/Ensamble_Secuencial.git
cd Ensamble_Secuencial
python -m venv .venv
```

Activar el entorno:

```powershell
# Windows / PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / macOS
source .venv/bin/activate
```

```sh
python -m pip install -r requirements.txt
python -m pip install "scikit-learn>=1.2" jupyter
python -m jupyter notebook
```

## Recorrido por los experimentos

Cada subcarpeta de [notebooks/](notebooks/) contiene un notebook de entrenamiento (`notebook_*`), uno de hiperparámetros (`experimentos_hyperparam_*`) y otro de validación cruzada (`validacion_cruzada_*`).

| Carpeta | Datos | Regresor base |
| --- | --- | --- |
| `house_prices_decisionTreeRegressor/` | Viviendas; objetivo `SalePrice` | Árbol de decisión |
| `house_prices_linearRegression/` | Viviendas; objetivo `SalePrice` | Regresión lineal |
| `parkinsons_decisionTreeRegressor/` | Parkinson; objetivo `total_UPDRS` | Árbol de decisión |
| `parkinsons_linearRegression/` | Parkinson; objetivo `total_UPDRS` | Regresión lineal |

Abrir primero un notebook de entrenamiento y ejecutar sus celdas en orden. Las rutas `../../src` y `../../data` asumen que el directorio de trabajo del kernel es la subcarpeta donde está el notebook.

## Ejemplo de uso

Desde la raíz del repositorio, en un script o sesión de Python:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from src.sequential_ensemble import SequentialEnsembleRegressor

X, y = make_regression(n_samples=300, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SequentialEnsembleRegressor(
    base_estimator=DecisionTreeRegressor,
    n_estimators=50,
    lr=0.1,
    sample_size=0.8,
    est_params={"max_depth": 3, "random_state": 42},
    random_state=42,
)
model.fit(X_train, y_train)
print("R²:", model.score(X_test, y_test))
```

Los parámetros principales son `n_estimators`, `lr`, `sample_size`, `est_params` y `random_state`. La semilla del ensamble controla el muestreo; la del árbol se configura por separado en `est_params`.

## Resultados de los experimentos

Valores de R² obtenidos en los experimentos de entrenamiento y recogidos en los notebooks `notebook_*`:

| Dataset | Regresor base del ensamble | R² en test |
| --- | --- | --- |
| Viviendas | Árbol de decisión | 0.7274 |
| Viviendas | Regresión lineal | 0.7902 |
| Parkinson | Árbol de decisión | 0.7700 |
| Parkinson | Regresión lineal | 0.1453 |

Los notebooks recogen las configuraciones utilizadas, la comparación con modelos individuales y los resultados de validación cruzada.

## Estructura

```text
src/sequential_ensemble.py  Implementación del regresor
notebooks/                 Entrenamiento, hiperparámetros y validación
data/                      Dos datasets CSV
img/                       Recursos de la memoria
```

## Licencia

[MIT](LICENSE).
