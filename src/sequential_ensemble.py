import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

class SequentialEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, n_estimators=10, lr=0.1, sample_size=0.8, est_params=None, random_state=None):
        """
        Inicializa el meta-modelo secuencial.

        Parámetros:
        - base_estimator: clase del modelo base (e.g. DecisionTreeRegressor)
        - n_estimators: número total de modelos a entrenar en el ensamble
        - lr: tasa de aprendizaje para ponderar la contribución de cada modelo
        - sample_size: proporción (0,1] del conjunto de datos usada para entrenar cada modelo
        - est_params: diccionario con parámetros para el estimador base
        - random_state: semilla para reproducibilidad del muestreo aleatorio
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.lr = lr
        self.sample_size = sample_size
        self.est_params = est_params if est_params is not None else {}
        self.random_state = random_state
        self.models = []  # Lista para almacenar los modelos entrenados

    def fit(self, X, y):
        """
        Entrena el meta-modelo secuencial.

        Parámetros:
        - X: matriz de características (num_samples x num_features)
        - y: vector objetivo (num_samples,)

        Proceso:
        - Inicializa la predicción acumulada como ceros.
        - Por cada iteración:
            - Calcula el residuo (error actual)
            - Extrae una muestra aleatoria sin reemplazo del conjunto de entrenamiento
            - Entrena un modelo base para predecir el residuo
            - Actualiza la predicción acumulada con la predicción ponderada del nuevo modelo
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)  # Generador aleatorio reproducible

        pred = np.zeros(n_samples)  # Predicción inicial: todos ceros
        self.models = []  # Limpiar lista de modelos por si se vuelve a entrenar

        for i in range(self.n_estimators):
            residual = y - pred  # Residuo: error actual del ensamble
            # Selección aleatoria sin reemplazo para entrenar este modelo
            idx = rng.choice(n_samples, int(n_samples * self.sample_size), replace=False)
            X_sample = X[idx]
            y_sample = residual[idx]

            # Crear y entrenar el modelo base con la muestra
            model = self.base_estimator(**self.est_params)
            model.fit(X_sample, y_sample)
            self.models.append(model)  # Guardar el modelo entrenado

            # Actualizar la predicción acumulada con la nueva predicción ponderada
            pred += self.lr * model.predict(X)

        return self  # Para permitir encadenar métodos (fit().predict())

    def predict(self, X):
        """
        Genera predicciones para nuevos datos usando el meta-modelo entrenado.

        Parámetros:
        - X: matriz de características (num_samples x num_features)

        Retorna:
        - pred: vector con predicciones acumuladas (num_samples,)
        """
        pred = np.zeros(X.shape[0])  # Inicializar predicción acumulada
        for model in self.models:
            pred += self.lr * model.predict(X)  # Sumar contribución de cada modelo ponderada por lr
        return pred

    def score(self, X, y):
        """
        Evalúa el rendimiento del modelo utilizando el coeficiente de determinación R².

        Parámetros:
        - X: matriz de características para evaluar
        - y: valores reales de la variable objetivo

        Retorna:
        - r2: valor del coeficiente R²
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
