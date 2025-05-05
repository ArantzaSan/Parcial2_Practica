import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RedNeuronalMulticapa:
    def __init__(self, num_entradas, num_ocultas, num_salidas, tasa_aprendizaje=0.1, random_seed=None):
        """
        Inicializa una red neuronal multicapa con una capa oculta.

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

        # Inicializar pesos y bias para la capa oculta
        self.pesos_entrada_oculta = np.random.rand(self.num_entradas, self.num_ocultas) - 0.5
        self.bias_oculta = np.zeros((1, self.num_ocultas))

        # Inicializar pesos y bias para la capa de salida
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
        # Capa oculta
        self.capa_oculta_entrada = np.dot(entrada, self.pesos_entrada_oculta) + self.bias_oculta
        self.capa_oculta_activacion = self.sigmoide(self.capa_oculta_entrada)

        # Capa de salida
        self.capa_salida_entrada = np.dot(self.capa_oculta_activacion, self.pesos_oculta_salida) + self.bias_salida
        self.capa_salida_activacion = self.softmax(self.capa_salida_entrada)
        return self.capa_salida_activacion

    def backward(self, entrada, etiquetas, salida):
        """Propagación hacia atrás para calcular los gradientes y actualizar los pesos."""
        m = entrada.shape[0]  # Número de ejemplos de entrenamiento

        # Gradiente de la capa de salida
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

# Ejemplo de uso para un problema no linealmente separable (make_moons)
if __name__ == "__main__":
    # Generar datos no linealmente separables
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    y_one_hot = np.eye(2)[y]  # Convertir etiquetas a one-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

    # Hiperparámetros
    num_entradas = 2
    num_ocultas = 8  # Más nodos ocultos para aprender la no linealidad
    num_salidas = 2
    tasa_aprendizaje = 0.01
    epochs = 200  # Más épocas para converger
    batch_size = 16

    # Crear y entrenar la red neuronal multicapa
    red_neuronal_multicapa = RedNeuronalMulticapa(num_entradas, num_ocultas, num_salidas, tasa_aprendizaje, random_seed=42)
    red_neuronal_multicapa.entrenar(X_train, y_train, epochs, batch_size)

    # Evaluar el modelo
    precision_entrenamiento = red_neuronal_multicapa.calcular_precision(X_train, y_train)
    precision_prueba = red_neuronal_multicapa.calcular_precision(X_test, y_test)
    print(f"\nPrecisión en el conjunto de entrenamiento: {precision_entrenamiento:.4f}")
    print(f"Precisión en el conjunto de prueba: {precision_prueba:.4f}")

    # Visualizar la frontera de decisión
    def plot_decision_boundary(X, y, model):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = model.predecir(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), s=40, cmap=plt.cm.Spectral, edgecolors='k')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Frontera de Decisión de la Red Neuronal Multicapa')
        plt.grid(True)
        plt.show()

    plot_decision_boundary(X, y_one_hot, red_neuronal_multicapa)
