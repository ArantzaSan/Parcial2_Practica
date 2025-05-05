import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class KohonenSOM:
    """
    Implementación de un Mapa Autoorganizado (SOM) de Kohonen.
    """
    def __init__(self, filas, columnas, num_caracteristicas, tasa_aprendizaje_inicial=0.1,
                 radio_inicial=None, iteraciones=100, random_seed=None):
        """
        Inicializa el SOM.

        Args:
            filas (int): Número de filas en la cuadrícula del mapa SOM.
            columnas (int): Número de columnas en la cuadrícula del mapa SOM.
            num_caracteristicas (int): Número de características en los datos de entrada.
            tasa_aprendizaje_inicial (float): Tasa de aprendizaje inicial.
            radio_inicial (float): Radio inicial del vecindario. Si es None, se establece a max(filas, columnas) / 2.
            iteraciones (int): Número de iteraciones de entrenamiento.
            random_seed (int): Semilla aleatoria para la inicialización de los pesos.
        """
        self.filas = filas
        self.columnas = columnas
        self.num_caracteristicas = num_caracteristicas
        self.tasa_aprendizaje_inicial = tasa_aprendizaje_inicial
        self.iteraciones = iteraciones
        self.random_state = np.random.RandomState(random_seed)
        self.pesos = self.random_state.rand(filas * columnas, num_caracteristicas) # Inicialización aleatoria
        self.radio_inicial = radio_inicial if radio_inicial is not None else max(filas, columnas) / 2

    def _distancia_euclidiana(self, vector1, vector2):
        """Calcula la distancia euclidiana entre dos vectores."""
        return np.sqrt(np.sum((vector1 - vector2) ** 2))

    def _encontrar_BMU(self, vector_entrada):
        """
        Encuentra la Unidad de Mejor Concordancia (BMU) para un vector de entrada dado.

        Args:
            vector_entrada (np.ndarray): Vector de entrada.

        Returns:
            tuple: Índice plano de la BMU y sus coordenadas (fila, columna) en la cuadrícula.
        """
        distancias = [self._distancia_euclidiana(vector_entrada, peso) for peso in self.pesos]
        indice_bmu = np.argmin(distancias)
        fila_bmu = indice_bmu // self.columnas
        columna_bmu = indice_bmu % self.columnas
        return indice_bmu, (fila_bmu, columna_bmu)

    def _calcular_vecindario(self, bmu_coords, radio_actual):
        """
        Calcula los índices de las neuronas dentro del vecindario de la BMU.

        Args:
            bmu_coords (tuple): Coordenadas (fila, columna) de la BMU.
            radio_actual (float): Radio actual del vecindario.

        Returns:
            list: Lista de índices planos de las neuronas en el vecindario.
        """
        vecindario = []
        fila_bmu, columna_bmu = bmu_coords
        for i in range(self.filas):
            for j in range(self.columnas):
                distancia_cuadrada = (i - fila_bmu) ** 2 + (j - columna_bmu) ** 2
                if distancia_cuadrada <= radio_actual ** 2:
                    vecindario.append(i * self.columnas + j)
        return vecindario

    def _funcion_aprendizaje(self, distancia_cuadrada, radio_actual):
        """
        Calcula el factor de aprendizaje basado en la distancia y el radio.
        Función Gaussiana típica.
        """
        return np.exp(-distancia_cuadrada / (2 * (radio_actual ** 2)))

    def entrenar(self, datos_entrenamiento):
        """
        Entrena el Mapa Autoorganizado de Kohonen.

        Args:
            datos_entrenamiento (np.ndarray): Datos de entrenamiento (num_muestras, num_caracteristicas).
        """
        num_muestras = datos_entrenamiento.shape[0]
        for iteracion in range(self.iteraciones):
            # Calcular la tasa de aprendizaje y el radio del vecindario actuales
            tasa_aprendizaje_actual = self.tasa_aprendizaje_inicial * (1 - iteracion / self.iteraciones)
            radio_actual = self.radio_inicial * np.exp(-iteracion / self.iteraciones)

            # Seleccionar aleatoriamente una muestra de entrenamiento
            indice_aleatorio = self.random_state.randint(0, num_muestras)
            vector_entrada = datos_entrenamiento[indice_aleatorio]

            # Encontrar la BMU
            indice_bmu, coords_bmu = self._encontrar_BMU(vector_entrada)

            # Calcular el vecindario de la BMU
            vecindario_indices = self._calcular_vecindario(coords_bmu, radio_actual)

            # Actualizar los pesos de las neuronas en el vecindario
            for indice_neurona in vecindario_indices:
                coords_neurona = (indice_neurona // self.columnas, indice_neurona % self.columnas)
                distancia_cuadrada = (coords_neurona[0] - coords_bmu[0]) ** 2 + (coords_neurona[1] - coords_bmu[1]) ** 2
                factor_aprendizaje = self._funcion_aprendizaje(distancia_cuadrada, radio_actual)
                self.pesos[indice_neurona] += tasa_aprendizaje_actual * factor_aprendizaje * (vector_entrada - self.pesos[indice_neurona])

            if (iteracion + 1) % (self.iteraciones // 10) == 0:
                print(f"Iteración {iteracion + 1}/{self.iteraciones}")

    def mapear_datos(self, datos):
        """
        Mapea los datos de entrada a las neuronas ganadoras en el mapa SOM.

        Args:
            datos (np.ndarray): Datos de entrada.

        Returns:
            list: Lista de coordenadas (fila, columna) de la BMU para cada vector de entrada.
        """
        mapeo = []
        for vector in datos:
            _, coords_bmu = self._encontrar_BMU(vector)
            mapeo.append(coords_bmu)
        return mapeo

    def visualizar_pesos(self, num_filas_subplot=None, num_columnas_subplot=None):
        """
        Visualiza los pesos de cada neurona en el mapa SOM como imágenes.

        Args:
            num_filas_subplot (int): Número de filas para los subplots.
            num_columnas_subplot (int): Número de columnas para los subplots.
        """
        if num_filas_subplot is None:
            num_filas_subplot = self.filas
        if num_columnas_subplot is None:
            num_columnas_subplot = self.columnas

        fig, axes = plt.subplots(num_filas_subplot, num_columnas_subplot, figsize=(10, 10))
        for i in range(self.filas):
            for j in range(self.columnas):
                indice_neurona = i * self.columnas + j
                peso = self.pesos[indice_neurona].reshape(1, -1) # Asegurar que sea 2D para imshow
                ax = axes[i, j] if self.filas > 1 or self.columnas > 1 else axes
                ax.imshow(peso, aspect='auto')
                ax.set_title(f'({i},{j})', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Gener
