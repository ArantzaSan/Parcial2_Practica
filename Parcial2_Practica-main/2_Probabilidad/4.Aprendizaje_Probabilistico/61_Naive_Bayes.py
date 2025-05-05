import numpy as np
from collections import Counter

class NaiveBayes:
    def fit(self, X, y):
        """
        Entrena el clasificador Naive Bayes.

        Args:
            X (array-like): Matriz de características de entrenamiento (n_muestras, n_características).
            y (array-like): Vector de etiquetas de clase de entrenamiento (n_muestras).
        """
        self.n_muestras, self.n_características = X.shape
        self._clases = np.unique(y)
        self.n_clases = len(self._clases)

        # Inicializar diccionarios para almacenar probabilidades
        self._prob_clase = {}  # P(clase)
        self._medias = {}      # Media de cada característica por clase
        self._varianzas = {}   # Varianza de cada característica por clase

        for c in self._clases:
            X_c = X[y == c]
            self._prob_clase[c] = X_c.shape[0] / self.n_muestras
            self._medias[c] = np.mean(X_c, axis=0)
            self._varianzas[c] = np.var(X_c, axis=0)

    def predict(self, X):
        """
        Realiza predicciones para un conjunto de muestras.

        Args:
            X (array-like): Matriz de características de prueba (n_muestras, n_características).

        Returns:
            np.ndarray: Vector de etiquetas de clase predichas (n_muestras).
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predice la clase para una sola muestra.

        Args:
            x (array-like): Vector de características de la muestra.

        Returns:
            any: La etiqueta de clase predicha.
        """
        probabilidades_posteriores = {}
        for c in self._clases:
            prior = np.log(self._prob_clase[c])
            likelihood = np.sum(np.log(self._probabilidad_gaussiana(x, self._medias[c], self._varianzas[c])))
            probabilidades_posteriores[c] = prior + likelihood

        return max(probabilidades_posteriores, key=probabilidades_posteriores.get)

    def _probabilidad_gaussiana(self, x, media, var):
        """
        Calcula la probabilidad (densidad de probabilidad) de una característica
        dado una media y varianza (asumiendo distribución gaussiana).
        """
        eps = 1e-8  # Para evitar división por cero en caso de varianza cero
        numerador = np.exp(-(x - media)**2 / (2 * (var + eps)))
        denominador = np.sqrt(2 * np.pi * (var + eps))
        return numerador / denominador

# Ejemplo de uso
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification

    # Generar datos de ejemplo
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, random_state=42)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear e entrenar el clasificador Naive Bayes
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = nb.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del clasificador Naive Bayes: {accuracy:.2f}")

    # Visualización de la frontera de decisión (solo para 2 características)
    if X.shape[1] == 2:
        import matplotlib.pyplot as plt
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor='k')
        plt.title('Frontera de Decisión de Naive Bayes')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.show()
