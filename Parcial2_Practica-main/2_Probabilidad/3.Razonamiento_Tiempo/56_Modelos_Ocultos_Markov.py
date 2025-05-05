import numpy as np
import random

class ModeloOcultoMarkov:
    def __init__(self, estados, observaciones, prob_inicial, prob_transicion, prob_emision):
        """
        Inicializa el Modelo Oculto de Markov (HMM).

        Args:
            estados (list): Lista de estados ocultos.
            observaciones (list): Lista de posibles observaciones.
            prob_inicial (dict): Probabilidad inicial de cada estado.
                                 Ejemplo: {'estado1': 0.6, 'estado2': 0.4}
            prob_transicion (dict): Probabilidad de transición entre estados.
                                     Ejemplo: {'estado1': {'estado1': 0.7, 'estado2': 0.3},
                                               'estado2': {'estado1': 0.4, 'estado2': 0.6}}
            prob_emision (dict): Probabilidad de emitir una observación dado un estado.
                                  Ejemplo: {'estado1': {'obs1': 0.9, 'obs2': 0.1},
                                            'estado2': {'obs1': 0.2, 'obs2': 0.8}}
        """
        self.estados = estados
        self.observaciones = observaciones
        self.prob_inicial = prob_inicial
        self.prob_transicion = prob_transicion
        self.prob_emision = prob_emision

    def simular(self, num_pasos):
        """
        Simula una secuencia de estados ocultos y observaciones.

        Args:
            num_pasos (int): Número de pasos en la simulación.

        Returns:
            tuple: Una tupla que contiene:
                - estados_ocultos (list): Secuencia de estados ocultos generados.
                - secuencia_observaciones (list): Secuencia de observaciones generadas.
        """
        estado_actual = random.choices(self.estados, weights=list(self.prob_inicial.values()), k=1)[0]
        estados_ocultos = [estado_actual]
        secuencia_observaciones = [random.choices(self.observaciones, weights=list(self.prob_emision[estado_actual].values()), k=1)[0]]

        for _ in range(num_pasos - 1):
            # Transición a un nuevo estado
            probs_transicion = self.prob_transicion[estado_actual]
            estado_actual = random.choices(self.estados, weights=list(probs_transicion.values()), k=1)[0]
            estados_ocultos.append(estado_actual)

            # Emisión de una observación
            probs_emision = self.prob_emision[estado_actual]
            observacion = random.choices(self.observaciones, weights=list(probs_emision.values()), k=1)[0]
            secuencia_observaciones.append(observacion)

        return estados_ocultos, secuencia_observaciones

    def probabilidad_secuencia(self, secuencia_observaciones):
        """
        Calcula la probabilidad de una secuencia de observaciones dada el modelo (usando el algoritmo hacia adelante).

        Args:
            secuencia_observaciones (list): La secuencia de observaciones.

        Returns:
            float: La probabilidad de la secuencia de observaciones.
        """
        T = len(secuencia_observaciones)
        N = len(self.estados)
        alfa = np.zeros((T, N))
        estados_indices = {estado: i for i, estado in enumerate(self.estados)}

        # Inicialización
        for i, estado in enumerate(self.estados):
            alfa[0, i] = self.prob_inicial[estado] * self.prob_emision[estado][secuencia_observaciones[0]]

        # Inducción
        for t in range(1, T):
            for j, estado_actual in enumerate(self.estados):
                suma = 0
                for i, estado_previo in enumerate(self.estados):
                    suma += alfa[t - 1, i] * self.prob_transicion[estado_previo][estado_actual]
                alfa[t, j] = suma * self.prob_emision[estado_actual][secuencia_observaciones[t]]

        # Terminación
        return np.sum(alfa[T - 1, :])

    def viterbi(self, secuencia_observaciones):
        """
        Encuentra la secuencia de estados ocultos más probable para una secuencia de observaciones dada (algoritmo de Viterbi).

        Args:
            secuencia_observaciones (list): La secuencia de observaciones.

        Returns:
            list: La secuencia de estados ocultos más probable.
        """
        T = len(secuencia_observaciones)
        N = len(self.estados)
        viterbi_matrix = np.zeros((T, N))
        ruta_previa = np.zeros((T, N), dtype=int)
        estados_indices = {estado: i for i, estado in enumerate(self.estados)}
        observaciones_indices = {obs: i for i, obs in enumerate(self.observaciones)}

        # Inicialización
        for i, estado in enumerate(self.estados):
            viterbi_matrix[0, i] = self.prob_inicial[estado] * self.prob_emision[estado][secuencia_observaciones[0]]
            ruta_previa[0, i] = 0

        # Recursión
        for t in range(1, T):
            for j, estado_actual in enumerate(self.estados):
                max_prob = 0
                mejor_estado_previo = 0
                for i, estado_previo in enumerate(self.estados):
                    prob = viterbi_matrix[t - 1, i] * self.prob_transicion[estado_previo][estado_actual]
                    if prob > max_prob:
                        max_prob = prob
                        mejor_estado_previo = i
                viterbi_matrix[t, j] = max_prob * self.prob_emision[estado_actual][secuencia_observaciones[t]]
                ruta_previa[t, j] = mejor_estado_previo

        # Terminación y retroceso
        mejor_ruta = [0] * T
        mejor_ruta[T - 1] = np.argmax(viterbi_matrix[T - 1, :])
        for t in range(T - 2, -1, -1):
            mejor_ruta[t] = ruta_previa[t + 1, mejor_ruta[t + 1]]

        return [self.estados[i] for i in mejor_ruta]

if __name__ == "__main__":
    # Definición de un HMM para un modelo del tiempo simplificado
    estados = ['Soleado', 'Lluvioso']
    observaciones = ['paraguas', 'no_paraguas']
    prob_inicial = {'Soleado': 0.6, 'Lluvioso': 0.4}
    prob_transicion = {
        'Soleado': {'Soleado': 0.7, 'Lluvioso': 0.3},
        'Lluvioso': {'Soleado': 0.4, 'Lluvioso': 0.6}
    }
    prob_emision = {
        'Soleado': {'paraguas': 0.1, 'no_paraguas': 0.9},
        'Lluvioso': {'paraguas': 0.8, 'no_paraguas': 0.2}
    }

    # Crear una instancia del HMM
    hmm = ModeloOcultoMarkov(estados, observaciones, prob_inicial, prob_transicion, prob_emision)

    # Simular una secuencia
    num_pasos = 10
    estados_ocultos, secuencia_observaciones = hmm.simular(num_pasos)
    print(f"Secuencia de estados ocultos simulada: {estados_ocultos}")
    print(f"Secuencia de observaciones simulada: {secuencia_observaciones}")

    # Calcular la probabilidad de una secuencia de observaciones
    prob_obs = hmm.probabilidad_secuencia(secuencia_observaciones)
    print(f"\nProbabilidad de la secuencia de observaciones '{secuencia_observaciones}': {prob_obs:.6f}")

    # Encontrar la secuencia de estados ocultos más probable usando Viterbi
    observaciones_ejemplo = ['paraguas', 'no_paraguas', 'paraguas', 'paraguas', 'no_paraguas']
    ruta_viterbi = hmm.viterbi(observaciones_ejemplo)
    print(f"\nSecuencia de observaciones: {observaciones_ejemplo}")
    print(f"Secuencia de estados ocultos más probable (Viterbi): {ruta_viterbi}")
