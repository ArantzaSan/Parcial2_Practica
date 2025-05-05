import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RedNeuronalBasica:
    def __init__(self, num_entradas, num_ocultas, num_salidas, tasa_aprendizaje=0.1, random_seed=None):
        """
        Inicializa una red neuronal básica de una capa oculta.

        Args:
            num_entradas (int): Número de nodos en la capa de entrada.
            num_ocultas (int): Número de nodos en la capa oculta.
            num_salidas (int): Número de nodos en la capa de salida.
            tasa_aprendizaje (float): Tasa de aprendizaje para el descenso de gradiente.
            random_seed (int): Semilla aleatoria para inicializar los pesos.
        """
        self.num_entradas = num_entradas
        self.num_ocultas = num_ocultas
        self.num_salidas = num_salidas
        self.tasa_aprendizaje = tasa_aprendizaje
        if random_seed is not None:
            np.random.seed(random_seed)
        self.pesos_entrada_oculta = np.random.rand(self.num_entradas, self.num_ocultas) - 0.5
        self.bias_oculta = np.zeros((1, self.num_ocultas))
        self.pesos_oculta_salida = np.random.rand(self.num_ocultas, self.num_salidas) - 0.5
        self.bias_salida = np.zeros((1, self.num_salidas))

    def sigmoide(self, x):
        """Función de activación sigmoide."""
        return 1 / (1 + np.exp(-x))

    def sigmoide_derivada(self, x):
        """Derivada de la función sigmoide."""
        s = self.sigmoide(x)
        return s * (1 - s)

    def softmax(self, x):
        """Función softmax para la capa de salida (clasificación multiclase)."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, entrada):
        """Propagación hacia adelante a través de la red."""
        self.capa_oculta_entrada = np.dot(entrada, self.pesos_entrada_oculta) + self.bias_oculta
        self.capa_oculta_activacion = self.sigmoide(self.capa_oculta_entrada)
        self.capa_salida_entrada = np.dot(self.capa_oculta_activacion, self.pesos_oculta_salida) + self.bias_salida
        self.capa_salida_activacion = self.softmax(self.capa_salida_entrada)
        return self.capa_salida_activacion

    def backward(self, entrada, etiquetas, salida):
        """Propagación hacia atrás para calcular los gradientes y actualizar los pesos."""
        m = entrada.shape[0]  # Número de ejemplos de entrenamiento

        # Gradiente de la capa de salida (usando la derivada de la función de pérdida softmax con cross-entropy)
        delta_salida = salida - etiquetas

        # Gradiente de los pesos entre la capa oculta y la capa de salida
        d_pesos_oculta_salida = (1 / m) * np.dot(self.capa_oculta_activacion.T, delta_salida)
        d_bias_salida = (1 / m) * np.sum(delta_salida, axis=0, keepdims=True)

        # Gradiente de la capa oculta
        delta_oculta = np.dot(delta_salida, self.pesos_oculta_salida.T) * self.sigmoide_derivada(self.capa_oculta_entrada)

        # Gradiente de los pesos entre la capa de entrada y la capa oculta
        d_pesos_entrada_oculta = (1 / m) * np.dot(entrada.T, delta_oculta)
        d_bias_oculta = (1 / m) * np.sum(delta_oculta, axis=0, keepdims=True)

        # Actualizar los pesos y los bias
        self.pesos_oculta_salida -= self.tasa_aprendizaje * d_pesos_oculta_salida
        self.bias_salida -= self.tasa_aprendizaje * d_bias_salida
        self.pesos_entrada_oculta -= self.tasa_aprendizaje * d_pesos_entrada_oculta
        self.bias_oculta -= self.tasa_aprendizaje * d_bias_oculta

    def entrenar(self, entrada_entrenamiento, etiquetas_entrenamiento, epochs, batch_size=32):
        """Entrena la red neuronal."""
        num_ejemplos = entrada_entrenamiento.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_ejemplos)
            entrada_aleatoria = entrada_entrenamiento[indices]
            etiquetas_aleatorias = etiquetas_entrenamiento[indices]
            for i in range(0, num_ejemplos, batch_size):
                batch_entrada = entrada_aleatoria[i:i + batch_size]
                batch_etiquetas = etiquetas_aleatorias[i:i + batch_size]
                salida = self.forward(batch_entrada)
                self.backward(batch_entrada, batch_etiquetas, salida)
            if (epoch + 1) % 10 == 0:
                perdida = self.calcular_perdida(entrada_entrenamiento, etiquetas_entrenamiento)
                precision = self.calcular_precision(entrada_entrenamiento, etiquetas_entrenamiento)
                print(f"Epoch {epoch + 1}/{epochs}, Pérdida: {perdida:.4f}, Precisión: {precision:.4f}")

    def predecir(self, entrada):
        """Realiza predicciones para una entrada dada."""
        salida = self.forward(entrada)
        return np.argmax(salida, axis=1)

    def calcular_perdida(self, entrada, etiquetas):
        """Calcula la pérdida de entropía cruzada."""
        m = entrada.shape[0]
        salida = self.forward(entrada)
        # Evitar logaritmo de cero
        log_prob = -np.log(salida[np.arange(m), np.argmax(etiquetas, axis=1)] + 1e-8)
        perdida = np.sum(log_prob) / m
        return perdida

    def calcular_precision(self, entrada, etiquetas):
        """Calcula la precisión de las predicciones."""
        predicciones = self.predecir(entrada)
        etiquetas_verdaderas = np.argmax(etiquetas, axis=1)
        precision = np.mean(predicciones == etiquetas_verdaderas)
        return precision

# Ejemplo de uso para clasificación binaria
if __name__ == "__main__":
    # Generar datos de ejemplo
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.0)
    y_one_hot = np.eye(2)[y]  # Convertir etiquetas a one-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

    # Hiperparámetros
    num_entradas = 2
    num_ocultas = 4
    num_salidas = 2
    tasa_aprendizaje = 0.1
    epochs = 100
    batch_size = 16

    # Crear y entrenar la red neuronal
    red_neuronal = RedNeuronalBasica(num_entradas, num_ocultas, num_salidas, tasa_
