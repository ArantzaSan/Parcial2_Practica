import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdalineNeuron:
    """Implementación de una neurona ADALINE."""
    def __init__(self, num_entradas, tasa_aprendizaje=0.01, n_iter=100):
        self.lr = tasa_aprendizaje
        self.n_iter = n_iter
        self.pesos = np.zeros(1 + num_entradas) # +1 para el bias
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
        """Entrena la neurona ADALINE usando los datos de entrenamiento."""
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

class MADALINE:
    """Implementación de una red MADALINE (Multiple ADALINE)."""
    def __init__(self, num_entradas, num_adaline_units, tasa_aprendizaje=0.01, n_iter=100):
        self.num_entradas = num_entradas
        self.num_adaline_units = num_adaline_units
        self.lr = tasa_aprendizaje
        self.n_iter = n_iter
        self.adaline_units = [AdalineNeuron(num_entradas, tasa_aprendizaje, n_iter) for _ in range(num_adaline_units)]

    def forward(self, X):
        """Realiza la propagación hacia adelante a través de las unidades ADALINE."""
        outputs = np.array([unit.activacion(X) for unit in self.adaline_units]).T
        return outputs

    def predict(self, X):
        """Predice la clase basándose en la salida de las unidades ADALINE (regla de la mayoría)."""
        outputs = self.forward(X)
        return np.sign(np.sum(outputs, axis=1))

    def fit(self, X, y):
        """Entrena la red MADALINE."""
        for epoch in range(self.n_iter):
            outputs = self.forward(X)
            final_predictions = np.sign(np.sum(outputs, axis=1))
            errors = y - final_predictions

            # Actualizar los pesos de las unidades ADALINE que contribuyen al error
            for i, error in enumerate(errors):
                if error != 0:
                    # Encontrar la unidad ADALINE cuya salida está más cerca de la decisión correcta
                    if y[i] == 1:
                        closest_unit_index = np.argmin(outputs[i])
                    else:  # y[i] == -1
                        closest_unit_index = np.argmax(outputs[i])

                    # Entrenar solo la unidad ADALINE seleccionada para corregir el error
                    self.adaline_units[closest_unit_index].fit(X[i].reshape(1, -1), y[i].reshape(1))

            # Evaluar el rendimiento después de cada época (opcional)
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            print(f"Epoch {epoch + 1}/{self.n_iter}, Accuracy: {accuracy:.4f}")

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo no linealmente separables
    X, y = make_blobs(n_samples=200, centers=3, random_state=42, cluster_std=1.5)
    y_labels = np.where(y == 0, -1, 1) # Convertir a etiquetas -1 y 1 (para simplificar binario por ahora)
    y_labels[y == 2] = 1 # Forzar a un problema binario más complejo

    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)

    # Hiperparámetros de MADALINE
    num_entradas = X_train.shape[1]
    num_adaline_units = 3
    tasa_aprendizaje = 0.01
    n_iter = 100

    # Crear y entrenar la red MADALINE
    madaline = MADALINE(num_entradas, num_adaline_units, tasa_aprendizaje, n_iter)
    madaline.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred_madaline = madaline.predict(X_test)
    accuracy_madaline = accuracy_score(y_test, y_pred_madaline)
    print(f"\nPrecisión de MADALINE en el conjunto de prueba: {accuracy_madaline:.4f}")

    # Visualizar la frontera de decisión de MADALINE (para datos bidimensionales)
    if num_entradas == 2:
        def plot_decision_boundary_madaline(X, y, model, resolution=0.02):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                                 np.arange(y_min, y_max, resolution))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.where(Z == 1, 1, 0)
            plt.contourf(xx, yy, Z, alpha=0.3)
            plt.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, 1, 0), s=40, cmap=plt.cm.RdBu, edgecolors='k')
            plt.xlabel('Característica 1')
            plt.ylabel('Característica 2')
            plt.title('Frontera de Decisión de MADALINE')
            plt.grid(True)

        plt.figure(figsize=(8, 6))
        plot_decision_boundary_madaline(X_train, y_train, madaline)
        plt.title('Frontera de Decisión de MADALINE (Entrenamiento)')
        plt.show()
