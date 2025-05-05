import numpy as np
import matplotlib.pyplot as plt
import random

def filtro_kalman(z, x_inicial, P_inicial, F, H, R, Q):
    """
    Implementación del Filtro de Kalman.

    Args:
        z (np.ndarray): Vector de observaciones (medidas).
        x_inicial (np.ndarray): Estimado del estado inicial.
        P_inicial (np.ndarray): Matriz de covarianza del error del estado inicial.
        F (np.ndarray): Matriz de transición del estado.
        H (np.ndarray): Matriz de observación.
        R (np.ndarray): Matriz de covarianza del ruido de la observación.
        Q (np.ndarray): Matriz de covarianza del ruido del proceso.

    Returns:
        tuple: Una tupla que contiene:
            - x_estimado_trayectoria (list): Lista de los estados estimados en cada paso.
            - P_estimado_trayectoria (list): Lista de las matrices de covarianza del error del estado en cada paso.
    """
    n = x_inicial.shape[0]  # Dimensión del estado
    m = z.shape[1]        # Dimensión de la observación
    T = z.shape[0]        # Número de pasos de tiempo

    x_estimado = x_inicial
    P_estimado = P_inicial

    x_estimado_trayectoria = []
    P_estimado_trayectoria = []

    for t in range(T):
        # Predicción
        x_predicho = F @ x_estimado
        P_predicho = F @ P_estimado @ F.T + Q

        # Actualización
        innovacion = z[t].T - H @ x_predicho
        S = H @ P_predicho @ H.T + R
        K = P_predicho @ H.T @ np.linalg.inv(S)

        x_estimado = x_predicho + K @ innovacion
        P_estimado = (np.eye(n) - K @ H) @ P_predicho

        x_estimado_trayectoria.append(x_estimado.flatten().tolist())
        P_estimado_trayectoria.append(P_estimado.tolist())

    return np.array(x_estimado_trayectoria), np.array(P_estimado_trayectoria)

if __name__ == "__main__":
    # Ejemplo: Seguimiento de un objeto en 1D con ruido

    # Definición de las matrices del sistema
    dt = 1.0    # Intervalo de tiempo
    F = np.array([[1, dt],
                  [0, 1]])  # Matriz de transición del estado (posición, velocidad)
    H = np.array([[1, 0]])  # Matriz de observación (solo medimos la posición)
    R = np.array([[1]])    # Covarianza del ruido de la observación (posición)
    Q = np.array([[0.01, 0],
                  [0, 0.01]]) # Covarianza del ruido del proceso (posición, velocidad)

    # Generación de datos simulados con ruido
    tiempo = np.arange(0, 20, dt)
    posicion_real = 2 * tiempo + 5
    velocidad_real = 2 * np.ones_like(tiempo)
    ruido_observacion = np.random.normal(0, np.sqrt(R[0, 0]), len(tiempo))
    observaciones = posicion_real + ruido_observacion
    z = observaciones.reshape(-1, 1)

    # Inicialización del filtro de Kalman
    x_inicial = np.array([[0],   # Estimado inicial de la posición
                          [0]])   # Estimado inicial de la velocidad
    P_inicial = np.array([[10, 0],
                          [0, 10]]) # Covarianza inicial del error del estado

    # Ejecutar el filtro de Kalman
    x_estimado_trayectoria, P_estimado_trayectoria = filtro_kalman(z, x_inicial, P_inicial, F, H, R, Q)
    posicion_estimada = x_estimado_trayectoria[:, 0]
    velocidad_estimada = x_estimado_trayectoria[:, 1]

    # Visualización de los resultados
    plt.figure(figsize=(12, 6))
    plt.plot(tiempo, posicion_real, label='Posición Real', linestyle='--')
    plt.scatter(tiempo, observaciones, label='Observaciones (con ruido)', marker='o', s=15)
    plt.plot(tiempo, posicion_estimada, label='Posición Estimada (Kalman)', color='red')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición')
    plt.title('Seguimiento de Objeto 1D con Filtro de Kalman')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plt.plot(tiempo, velocidad_real, label='Velocidad Real', linestyle='--')
    plt.plot(tiempo, velocidad_estimada, label='Velocidad Estimada (Kalman)', color='green')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad')
    plt.title('Estimación de Velocidad con Filtro de Kalman')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
