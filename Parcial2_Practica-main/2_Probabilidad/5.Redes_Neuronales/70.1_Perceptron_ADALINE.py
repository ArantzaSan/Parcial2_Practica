import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Perceptrón ---

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.01, n_iter=100):
        self.lr = tasa_aprendizaje
        self.n_iter = n_iter
        self.pesos = np.zeros(1 + num_entradas) # +1 para el bias

    def predecir(self, X):
        """Devuelve la etiqueta de clase después del paso unitario."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """Calcula la entrada neta ponderada."""
        return np.dot(X, self.pesos[1:]) + self.pesos[0] # pesos[0] es el bias

    def fit(self, X, y):
        """Entrena el modelo del perceptrón usando los datos de entrenamiento."""
        self.pesos = np.zeros(1 + X.shape[1])
        errores = []

        for _ in range(self.n_iter):
            error_epoch = 0
            for xi, target in zip(X, y):
                prediccion = self.predecir(xi)
                actualizacion = self.lr * (target - prediccion)
                self.pesos[1:] += actualizacion * xi
                self.pesos[0] += actualizacion
                error_epoch += int(actualizacion != 0.0)
            errores.append(error_epoch)
        return errores

# --- ADALINE (Adaptive Linear Neuron) ---

class AdalineGD:
    """Clasificador ADALINE basado en descenso de gradiente."""
    def __init__(self, tasa_aprendizaje=0.01, n_iter=100):
        self.lr = tasa_aprendizaje
        self.n_iter = n_iter
        self.pesos = None
        self.costo = []

    def net_input(self, X):
        """Calcula la entrada neta ponderada."""
        return np.dot(X, self.pesos[1:]) + self.pesos[0]

    def activacion(self, X):
        """Calcula la activación lineal."""
        return self.net_input(X)

    def predecir(self, X):
        """Devuelve la etiqueta de clase después del paso unitario."""
        return np.where(self.activacion(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        """Entrena el modelo ADALINE usando los datos de entrenamiento."""
        self.pesos = np.zeros(1 + X.shape[1])
        self.costo = []

        for i in range(self.n_iter):
            salida = self.net_input(X)
            errores = (y - salida)
            self.pesos[1:] += self.lr * X.T.dot(errores) / X.shape[0]
            self.pesos[0] += self.lr * errores.sum() / X.shape[0]
            costo = (errores**2).sum() / (2.0 * X.shape[0])
            self.costo.append(costo)
        return self.costo

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo para clasificación binaria
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y_labels = np.where(y == 0, -1, 1) # Etiquetas para Perceptrón y ADALINE

    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)

    # --- Entrenar y evaluar el Perceptrón ---
    perceptron = Perceptron(num_entradas=X_train.shape[1], tasa_aprendizaje=0.1, n_iter=50)
    errores_perceptron = perceptron.fit(X_train, y_train)
    y_pred_perceptron = perceptron.predecir(X_test)
    accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
    print("--- Perceptrón ---")
    print(f"Precisión en el conjunto de prueba: {accuracy_perceptron:.2f}")

    # Graficar la frontera de decisión del Perceptrón
    def plot_decision_boundary_perceptron(X, y, clf):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = clf.predecir(np.c_[xx.ravel(), yy.ravel()])
        Z = np.where(Z == 1, 1, 0) # Para el contourf
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, 1, 0), s=40, cmap=plt.cm.RdBu)
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Frontera de Decisión del Perceptrón')
        plt.grid(True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_decision_boundary_perceptron(X_train, y_train, perceptron)
    plt.title('Frontera de Decisión del Perceptrón (Entrenamiento)')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(errores_perceptron) + 1), errores_perceptron, marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Número de Errores')
    plt.title('Número de Errores del Perceptrón por Época')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Entrenar y evaluar ADALINE ---
    adaline = AdalineGD(tasa_aprendizaje=0.01, n_iter=50)
    costos_adaline = adaline.fit(X_train, y_train)
    y_pred_adaline = adaline.predecir(X_test)
    accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
    print("\n--- ADALINE ---")
    print(f"Precisión en el conjunto de prueba: {accuracy_adaline:.2f}")

    # Graficar la función de costo de ADALINE
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(costos_adaline) + 1), costos_adaline, marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Costo')
    plt.title('Función de Costo de ADALINE por Época')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Graficar la frontera de decisión de ADALINE
    def plot_decision_boundary_adaline(X, y, clf):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() +
