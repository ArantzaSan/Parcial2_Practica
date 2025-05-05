import numpy as np

def algoritmo_delante_atras(observaciones, estados, estado_inicial_prob, matriz_transicion, matriz_emision):
    """
    Implementación del algoritmo Delante-Atrás para modelos ocultos de Markov (HMM).

    Args:
        observaciones (list): Secuencia de observaciones.
        estados (list): Lista de posibles estados ocultos.
        estado_inicial_prob (dict): Probabilidad inicial de cada estado.
                                     Ejemplo: {'estado1': 0.6, 'estado2': 0.4}
        matriz_transicion (dict): Probabilidad de transición entre estados.
                                  Ejemplo: {'estado1': {'estado1': 0.7, 'estado2': 0.3},
                                            'estado2': {'estado1': 0.4, 'estado2': 0.6}}
        matriz_emision (dict): Probabilidad de emitir una observación dado un estado.
                                Ejemplo: {'estado1': {'obs1': 0.9, 'obs2': 0.1},
                                          'estado2': {'obs1': 0.2, 'obs2': 0.8}}

    Returns:
        tuple: Una tupla que contiene:
            - alfa (numpy.ndarray): Matriz de probabilidades hacia adelante (forward).
            - beta (numpy.ndarray): Matriz de probabilidades hacia atrás (backward).
            - gamma (numpy.ndarray): Matriz de probabilidades posteriores (posterior).
    """
    N = len(estados)
    T = len(observaciones)

    # Inicialización de alfa (probabilidades hacia adelante)
    alfa = np.zeros((T, N))
    for i, estado in enumerate(estados):
        alfa[0, i] = estado_inicial_prob[estado] * matriz_emision[estado][observaciones[0]]

    # Paso hacia adelante
    for t in range(1, T):
        for j, estado_actual in enumerate(estados):
            suma = 0
            for i, estado_previo in enumerate(estados):
                suma += alfa[t - 1, i] * matriz_transicion[estado_previo][estado_actual]
            alfa[t, j] = suma * matriz_emision[estado_actual][observaciones[t]]

    # Inicialización de beta (probabilidades hacia atrás)
    beta = np.zeros((T, N))
    beta[T - 1, :] = 1  # Al final, la probabilidad de estar en cualquier estado y ver el final es 1

    # Paso hacia atrás
    for t in range(T - 2, -1, -1):
        for i, estado_previo in enumerate(estados):
            suma = 0
            for j, estado_actual in enumerate(estados):
                suma += beta[t + 1, j] * matriz_transicion[estado_previo][estado_actual] * matriz_emision[estado_actual][observaciones[t + 1]]
            beta[t, i] = suma

    # Cálculo de gamma (probabilidades posteriores)
    gamma = np.zeros((T, N))
    prob_observaciones = np.sum(alfa[T - 1, :])  # Probabilidad total de la secuencia de observaciones
    if prob_observaciones == 0:
        raise ValueError("La probabilidad de la secuencia de observaciones es cero. Revisar los parámetros del HMM.")

    for t in range(T):
        for i in range(N):
            gamma[t, i] = (alfa[t, i] * beta[t, i]) / prob_observaciones

    return alfa, beta, gamma

if __name__ == "__main__":
    # Definición del HMM de ejemplo (modelo del tiempo simplificado)
    estados = ['Soleado', 'Lluvioso']
    observaciones = ['paraguas', 'no_paraguas', 'paraguas']
    estado_inicial_prob = {'Soleado': 0.6, 'Lluvioso': 0.4}
    matriz_transicion = {
        'Soleado': {'Soleado': 0.7, 'Lluvioso': 0.3},
        'Lluvioso': {'Soleado': 0.4, 'Lluvioso': 0.6}
    }
    matriz_emision = {
        'Soleado': {'paraguas': 0.1, 'no_paraguas': 0.9},
        'Lluvioso': {'paraguas': 0.8, 'no_paraguas': 0.2}
    }

    # Ejecutar el algoritmo Delante-Atrás
    alfa, beta, gamma = algoritmo_delante_atras(observaciones, estados, estado_inicial_prob, matriz_transicion, matriz_emision)

    print("Probabilidades hacia adelante (alfa):\n", alfa)
    print("\nProbabilidades hacia atrás (beta):\n", beta)
    print("\nProbabilidades posteriores (gamma):\n", gamma)

    # Interpretación de gamma:
    # gamma[t, i] es la probabilidad de estar en el estado estados[i] en el tiempo t,
    # dado la secuencia completa de observaciones.
    print("\nInterpretación de las probabilidades posteriores (gamma):")
    for t, obs in enumerate(observaciones):
        print(f"Tiempo {t}, Observación '{obs}':")
        for i, estado in enumerate(estados):
            print(f"  P(Estado = {estado} | Observaciones) = {gamma[t, i]:.4f}")
