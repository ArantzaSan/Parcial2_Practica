import numpy as np
import random
import matplotlib.pyplot as plt

class ProcesoMarkov:
    def __init__(self, estados, matriz_transicion):
        """
        Inicializa el Proceso de Markov.

        Args:
            estados (list): Una lista de los posibles estados del proceso.
            matriz_transicion (numpy.ndarray): Una matriz cuadrada donde
                                               matriz_transicion[i][j] es la
                                               probabilidad de transitar del
                                               estado estados[i] al estado estados[j].
        """
        self.estados = estados
        self.matriz_transicion = np.array(matriz_transicion)
        self._validar_matriz()

    def _validar_matriz(self):
        """Valida que la matriz de transición sea cuadrada y que las filas sumen 1."""
        n_estados = len(self.estados)
        if self.matriz_transicion.shape != (n_estados, n_estados):
            raise ValueError("La matriz de transición debe ser cuadrada.")
        for fila in self.matriz_transicion:
            if not np.isclose(np.sum(fila), 1.0):
                raise ValueError(f"Las probabilidades de transición desde un estado deben sumar 1. Fila: {fila}")

    def simular(self, estado_inicial, num_pasos):
        """
        Simula una trayectoria del Proceso de Markov.

        Args:
            estado_inicial: El estado inicial de la simulación. Debe estar en la lista de estados.
            num_pasos (int): El número de pasos a simular.

        Returns:
            list: Una lista de los estados visitados durante la simulación.
        """
        if estado_inicial not in self.estados:
            raise ValueError(f"El estado inicial '{estado_inicial}' no es válido.")

        trayectoria = [estado_inicial]
        estado_actual_indice = self.estados.index(estado_inicial)

        for _ in range(num_pasos):
            probabilidades_transicion = self.matriz_transicion[estado_actual_indice]
            siguiente_estado = random.choices(self.estados, weights=probabilidades_transicion, k=1)[0]
            trayectoria.append(siguiente_estado)
            estado_actual_indice = self.estados.index(siguiente_estado)

        return trayectoria

    def calcular_distribucion_estacionaria(self, tolerancia=1e-9, max_iteraciones=1000):
        """
        Calcula la distribución estacionaria del Proceso de Markov utilizando el método de la potencia.
        Solo funciona para cadenas de Markov ergódicas (irreducibles y aperiódicas).

        Args:
            tolerancia (float): La tolerancia para la convergencia.
            max_iteraciones (int): El número máximo de iteraciones.

        Returns:
            dict: Un diccionario donde las claves son los estados y los valores son las probabilidades
                  en la distribución estacionaria. Retorna None si no converge.
        """
        n_estados = len(self.estados)
        distribucion_actual = np.ones(n_estados) / n_estados  # Distribución inicial uniforme

        for _ in range(max_iteraciones):
            distribucion_siguiente = np.dot(distribucion_actual, self.matriz_transicion)
            if np.linalg.norm(distribucion_siguiente - distribucion_actual, ord=1) < tolerancia:
                return dict(zip(self.estados, distribucion_siguiente))
            distribucion_actual = distribucion_siguiente

        print("Advertencia: No se alcanzó la convergencia para la distribución estacionaria.")
        return None

if __name__ == "__main__":
    # Ejemplo: Modelo del clima con tres estados
    estados_clima = ["Soleado", "Lluvioso", "Nublado"]
    matriz_transicion_clima = [
        [0.7, 0.2, 0.1],  # De Soleado a Soleado, Lluvioso, Nublado
        [0.3, 0.4, 0.3],  # De Lluvioso a Soleado, Lluvioso, Nublado
        [0.2, 0.3, 0.5]   # De Nublado a Soleado, Lluvioso, Nublado
    ]

    proceso_clima = ProcesoMarkov(estados_clima, matriz_transicion_clima)

    # Simular una trayectoria del clima
    estado_inicial_clima = "Soleado"
    num_pasos_clima = 50
    trayectoria_clima = proceso_clima.simular(estado_inicial_clima, num_pasos_clima)
    print(f"Trayectoria del clima ({num_pasos_clima} pasos): {trayectoria_clima}")

    # Calcular la distribución estacionaria del clima
    distribucion_estacionaria_clima = proceso_clima.calcular_distribucion_estacionaria()
    if distribucion_estacionaria_clima:
        print("Distribución estacionaria del clima:", distribucion_estacionaria_clima)

    # Visualizar la frecuencia de los estados en una simulación larga
    num_simulaciones_larga = 1000
    trayectoria_larga = proceso_clima.simular(estado_inicial_clima, num_simimulaciones_larga)
    frecuencia_estados = {estado: trayectoria_larga.count(estado) / num_simimulaciones_larga for estado in estados_clima}
    print(f"Frecuencia de estados en {num_simulaciones_larga} pasos:", frecuencia_estados)

    plt.figure(figsize=(8, 6))
    plt.bar(frecuencia_estados.keys(), frecuencia_estados.values(), color='skyblue')
    plt.xlabel("Estado del Clima")
    plt.ylabel("Frecuencia")
    plt.title(f"Frecuencia de Estados del Clima en {num_simulaciones_larga} Pasos")
    plt.show()

    # Ejemplo: Caminata aleatoria en un grafo simple
    estados_caminata = ["A", "B", "C"]
    matriz_transicion_caminata = [
        [0.0, 0.5, 0.5],  # De A a B o C
        [0.5, 0.0, 0.5],  # De B a A o C
        [0.5, 0.5, 0.0]   # De C a A o B
    ]

    proceso_caminata = ProcesoMarkov(estados_caminata, matriz_transicion_caminata)
    trayectoria_caminata = proceso_caminata.simular("A", 20)
    print(f"\nTrayectoria de la caminata aleatoria (20 pasos): {trayectoria_caminata}")

    distribucion_estacionaria_caminata = proceso_caminata.calcular_distribucion_estacionaria()
    if distribucion_estacionaria_caminata:
        print("Distribución estacionaria de la caminata aleatoria:", distribucion_estacionaria_caminata)
