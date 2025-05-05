import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModeloOcultoMarkov:
    def __init__(self, estados, observaciones, prob_inicial, prob_transicion, prob_emision):
        """
        Inicializa el Modelo Oculto de Markov (HMM).

        Args:
            estados (list): Lista de posibles estados ocultos.
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
        self.prob_inicial = self._convertir_a_numpy(prob_inicial, estados)
        self.prob_transicion = self._convertir_a_numpy(prob_transicion, estados, estados)
        self.prob_emision = self._convertir_a_numpy(prob_emision, estados, observaciones)
        self.n_estados = len(estados)
        self.n_observaciones = len(observaciones)
        self._estado_a_index = {estado: i for i, estado in enumerate(estados)}
        self._observacion_a_index = {obs: i for i, obs in enumerate(observaciones)}

    def _convertir_a_numpy(self, estructura, filas, columnas=None):
        """Convierte diccionarios anidados a arrays NumPy."""
        if columnas is None:
            n_filas = len(filas)
            matriz = np.zeros(n_filas)
            for i, fila in enumerate(filas):
                matriz[i] = estructura[fila]
        else:
            n_filas = len(filas)
            n_columnas = len(columnas)
            matriz = np.zeros((n_filas, n_columnas))
            for i, fila in enumerate(filas):
                for j, columna in enumerate(columnas):
                    matriz[i, j] = estructura[fila][columna]
        return matriz

    def probabilidad_secuencia(self, secuencia_observaciones):
        """
        Calcula la probabilidad de una secuencia de observaciones dada el modelo (algoritmo hacia adelante).

        Args:
            secuencia_observaciones (list): La secuencia de observaciones.

        Returns:
            float: La probabilidad de la secuencia de observaciones.
        """
        T = len(secuencia_observaciones)
        alpha = np.zeros((T, self.n_estados))

        # Inicialización
        t = 0
        for i in range(self.n_estados):
            obs_index = self._observacion_a_index[secuencia_observaciones[t]]
            alpha[t, i] = self.prob_inicial[i] * self.prob_emision[i, obs_index]

        # Inducción
        for t in range(1, T):
            for j in range(self.n_estados):
                obs_index = self._observacion_a_index[secuencia_observaciones[t]]
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.prob_transicion[:, j]) * self.prob_emision[j, obs_index]

        # Terminación
        return np.sum(alpha[T - 1, :])

    def viterbi(self, secuencia_observaciones):
        """
        Encuentra la secuencia de estados ocultos más probable para una secuencia de observaciones dada (algoritmo de Viterbi).

        Args:
            secuencia_observaciones (list): La secuencia de observaciones.

        Returns:
            list: La secuencia de estados ocultos más probable.
        """
        T = len(secuencia_observaciones)
        viterbi_matrix = np.zeros((T, self.n_estados))
        ruta_previa = np.zeros((T, self.n_estados), dtype=int)

        # Inicialización
        t = 0
        for i in range(self.n_estados):
            obs_index = self._observacion_a_index[secuencia_observaciones[t]]
            viterbi_matrix[t, i] = np.log(self.prob_inicial[i] + 1e-9) + np.log(self.prob_emision[i, obs_index] + 1e-9)
            ruta_previa[t, i] = 0

        # Recursión
        for t in range(1, T):
            for j in range(self.n_estados):
                obs_index = self._observacion_a_index[secuencia_observaciones[t]]
                trans_prob = np.log(self.prob_transicion[:, j] + 1e-9)
                prob_actual = viterbi_matrix[t - 1, :] + trans_prob
                viterbi_matrix[t, j] = np.max(prob_actual) + np.log(self.prob_emision[j, obs_index] + 1e-9)
                ruta_previa[t, j] = np.argmax(prob_actual)

        # Terminación y retroceso
        mejor_ruta = [0] * T
        mejor_ruta[T - 1] = np.argmax(viterbi_matrix[T - 1, :])
        for t in range(T - 2, -1, -1):
            mejor_ruta[t] = ruta_previa[t + 1, mejor_ruta[t + 1]]

        return [self.estados[i] for i in mejor_ruta]

    def baum_welch(self, secuencia_observaciones, num_iteraciones=100, tol=1e-4):
        """
        Implementa el algoritmo de Baum-Welch para aprender los parámetros del HMM.

        Args:
            secuencia_observaciones (list): La secuencia de observaciones de entrenamiento.
            num_iteraciones (int): Número máximo de iteraciones.
            tol (float): Tolerancia para la convergencia.

        Returns:
            tuple: Las matrices de probabilidad inicial, transición y emisión aprendidas.
        """
        T = len(secuencia_observaciones)
        N = self.n_estados
        M = self.n_observaciones

        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))
        gamma = np.zeros((T, N))
        xi = np.zeros((T - 1, N, N))

        for iteracion in range(num_iteraciones):
            # Paso E: Calcular alpha, beta, gamma y xi
            alpha = self._forward(secuencia_observaciones)
            beta = self._backward(secuencia_observaciones, alpha)
            gamma = self._posterior(alpha, beta)
            xi = self._pairwise_posterior(secuencia_observaciones, alpha, beta)

            # Guardar los parámetros antiguos para la verificación de convergencia
            prob_inicial_antiguo = self.prob_inicial.copy()
            prob_transicion_antiguo = self.prob_transicion.copy()
            prob_emision_antiguo = self.prob_emision.copy()

            # Paso M: Actualizar los parámetros
            self.prob_inicial = gamma[0, :] / np.sum(gamma[0, :]) if np.sum(gamma[0, :]) > 0 else self.prob_inicial

            for i in range(N):
                for j in range(N):
                    numerador = np.sum(xi[:-1, i, j])
                    denominador = np.sum(gamma[:-1, i])
                    self.prob_transicion[i, j] = numerador / (denominador + 1e-9)

            for i in range(N):
                for k in range(M):
                    numerador = np.sum(gamma[np.array([self._observacion_a_index[secuencia_observaciones[t]] == k for t in range(T)]), i])
                    denominador = np.sum(gamma[:, i])
                    self.prob_emision[i, k] = numerador / (denominador + 1e-9)

            # Verificar convergencia
            cambio_inicial = np.sum(np.abs(self.prob_inicial - prob_inicial_antiguo))
            cambio_transicion = np.sum(np.abs(self.prob_transicion - prob_transicion_antiguo))
            cambio_emision = np.sum(np.abs(self.prob_emision - prob_emision_antiguo))

            if cambio_inicial < tol and cambio_transicion < tol and cambio_emision < tol:
                print(f"Convergencia alcanzada en la iteración {iteracion + 1}")
                break

        return self.estados, self.observaciones, self.prob_inicial, self.prob_transicion, self.prob_emision

    def _forward(self, secuencia_observaciones):
        T = len(secuencia_observaciones)
        N = self.n_estados
        alpha = np.zeros((T, N))
        for i in range(N):
            obs_index = self._observacion_a_index[secuencia_observaciones[0]]
            alpha[0, i] = self.prob_inicial[i] * self.prob_emision[i, obs_index]
        for t in range(1, T):
            for j in range(N):
                obs_index = self._observacion_a_index[secuencia_observaciones[t]]
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.prob_transicion[:, j]) * self.prob_emision[j, obs_index]
        return alpha

    def _backward(self, secuencia_observaciones, alpha):
        T = len(secuencia_observaciones)
        N = self.n_estados
        beta = np.zeros((T, N))
        beta[T - 1, :] = 1
        for t in range(T - 2, -1, -1):
            for i in range(N):
                obs_index_next = self._observacion_a_index[secuencia_observaciones[t + 1]]
                beta[t, i] = np.sum(beta[t + 1, :] * self.prob_transicion[i, :] * self.prob_emision[:, obs_index_next])
        return beta

    def _posterior(self, alpha, beta):
        T = alpha.shape[0]
        gamma = np.zeros_like(alpha)
        prob_observaciones = np.sum(alpha[T - 1, :])
        gamma = (alpha * beta) / (prob_observaciones + 1e-9)
        return gamma

    def _pairwise_posterior(self, secuencia_observaciones, alpha, beta):
        T = len(secuencia_observaciones)
        N = self.n_estados
        xi = np.zeros((T - 1, N, N))
        prob_observaciones = np.sum(alpha[T - 1, :])
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    obs_index_next = self._observacion_a_index[secuencia_observaciones[t + 1]]
                    xi[t, i, j] = (alpha[t, i] * self.prob_transicion[i, j] * self.prob_emision[j, obs_index_next] * beta[t + 1, j]) / (prob_observaciones + 1e-9)
        return xi

# Ejemplo de uso
if __name__ == "__main__":
    # Definición del HMM
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

    # Crear el modelo HMM
    hmm = ModeloOcultoMarkov(estados, observaciones, prob_inicial, prob_transicion, prob_emision)

    # Secuencia de observaciones
    secuencia = ['paraguas', 'no_paraguas', 'paraguas', 'paraguas', 'no_paraguas']

    # Calcular la probabilidad de la secuencia
    prob = hmm.probabilidad_secuencia(secuencia)
    print(f"Probabilidad de la secuencia '{secuencia}': {prob:.4f}")

    # Encontrar la secuencia de estados más probable (Viterbi)
    ruta_viterbi = hmm.viterbi(secuencia)
    print(f"Secuencia de estados más probable (Viterbi): {ruta_viterbi}")

    print("\n--- Aprendizaje de los parámetros del HMM (Baum-Welch) ---")
    # Secuencia de entrenamiento (más larga para aprender)
    secuencia_entrenamiento = ['paraguas', 'no_paraguas', 'paraguas', 'paraguas', 'no_paraguas',
                               'no_paraguas', 'no_paraguas', 'paraguas', 'lluvias'] # 'lluvias' no está en las obs iniciales

    # Ampliar el conjunto de observaciones si es necesario para el entrenamiento
    if 'lluvias' not in hmm.observaciones:
        hmm.observaciones.append('lluvias')
        n_obs_nuevo = len(hmm.observaciones)
        emission_nuevo = np.zeros((hmm.n_estados, n_obs_nuevo))
        emission_nuevo[:, :hmm.n_observaciones] = hmm.prob_emision
        # Inicializar la probabilidad de 'lluvias' para cada estado (puede requerir conocimiento del dominio)
        emission_nuevo[0, hmm._observacion_a_index['lluvias']] = 0.05
        emission_nuevo[1, hmm._observacion_a_index['lluvias']] = 0.95
        hmm.prob_emision = emission_nuevo
        hmm.n_observaciones = n_obs_nuevo
        hmm._observacion_a_index['lluvias'] = n_obs_nuevo - 1

    estados_aprendidos, observaciones_aprendidas, prob_inicial_aprendida, prob_transicion_aprendida, prob_emision_aprendida = \
        hmm.baum_welch(secuencia_entrenamiento, num_iteraciones=50)

    print("\nParámetros Aprendidos (Baum-Welch):")
    print("Probabilidad Inicial:")
    for i, estado in enumerate(estados_aprendidos):
        print(f"  {estado}: {prob_inicial_aprendida[i]:.4f}")
    print("\nMatriz de Transición:")
    sns.heatmap(prob_transicion_aprendida, annot=True, fmt=".2f", xticklabels=estados_aprendidos, yticklabels=estados_aprendidos, cmap="Blues")
    plt.title("Matriz de Transición Aprendida")
    plt.show()
    print("\nMatriz de Emisión:")
    sns.heatmap(prob_emision_aprendida, annot=True, fmt=".2f", xticklabels=observaciones_aprendidas, yticklabels=estados_aprendidos, cmap="Greens")
    plt.title("Matriz de Emisión Aprendida")
    plt.show()
