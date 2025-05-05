import numpy as np
import matplotlib.pyplot as plt
import random

def filtro_particulas(observaciones, num_particulas, estado_inicial_estimado, sigma_inicial,
                      modelo_transicion, ruido_transicion_sigma,
                      modelo_observacion, ruido_observacion_sigma):
    """
    Implementación básica del Filtro de Partículas (Particle Filter).

    Args:
        observaciones (list): Secuencia de observaciones.
        num_particulas (int): Número de partículas a mantener.
        estado_inicial_estimado (float): Estimado inicial del estado.
        sigma_inicial (float): Desviación estándar de la distribución inicial de partículas.
        modelo_transicion (callable): Función que toma el estado anterior y devuelve el siguiente estado (sin ruido).
        ruido_transicion_sigma (float): Desviación estándar del ruido del proceso (transición).
        modelo_observacion (callable): Función que toma el estado actual y devuelve la observación esperada.
        ruido_observacion_sigma (float): Desviación estándar del ruido de la observación.

    Returns:
        tuple: Una tupla que contiene:
            - estados_estimados (list): Lista de las estimaciones del estado en cada paso de tiempo (media de las partículas).
            - trayectorias_particulas (list): Lista de las trayectorias de todas las partículas en cada paso de tiempo.
            - pesos_particulas_trayectoria (list): Lista de los pesos de las partículas en cada paso de tiempo.
    """
    particulas = np.random.normal(estado_inicial_estimado, sigma_inicial, num_particulas)
    pesos = np.ones(num_particulas) / num_particulas
    estados_estimados = [np.mean(particulas)]
    trayectorias_particulas = [particulas.copy()]
    pesos_particulas_trayectoria = [pesos.copy()]

    for observacion in observaciones:
        # Predicción
        particulas_predichas = np.array([modelo_transicion(p) + np.random.normal(0, ruido_transicion_sigma) for p in particulas])

        # Actualización (Cálculo de pesos)
        prob_observacion = np.array([np.exp(-(observacion - modelo_observacion(p))**2 / (2 * ruido_observacion_sigma**2))
                                     for p in particulas_predichas])
        pesos = pesos * prob_observacion
        pesos = pesos / np.sum(pesos)  # Normalizar los pesos

        # Estimación del estado (media ponderada de las partículas)
        estado_estimado = np.sum(particulas_predichas * pesos)
        estados_estimados.append(estado_estimado)

        # Resampleo (muestreo con reemplazo basado en los pesos)
        indices_resampleados = np.random.choice(range(num_particulas), num_particulas, p=pesos)
        particulas = particulas_predichas[indices_resampleados]
        pesos = np.ones(num_particulas) / num_particulas  # Resetear los pesos después del resampleo

        trayectorias_particulas.append(particulas.copy())
        pesos_particulas_trayectoria.append(pesos.copy())

    return estados_estimados, trayectorias_particulas, pesos_particulas_trayectoria

if __name__ == "__main__":
    # Definición del sistema y los parámetros del filtro
    def modelo_transicion_real(estado):
        return 0.8 * estado + 5

    def modelo_observacion_real(estado):
        return estado

    ruido_transicion_real_sigma = 2
    ruido_observacion_real_sigma = 3

    # Generación de datos simulados
    tiempo = range(50)
    estado_real = [10]
    observaciones = []
    for _ in tiempo:
        estado_real.append(modelo_transicion_real(estado_real[-1]) + np.random.normal(0, ruido_transicion_real_sigma))
        observaciones.append(modelo_observacion_real(estado_real[-1]) + np.random.normal(0, ruido_observacion_real_sigma))

    # Parámetros del filtro de partículas
    num_particulas = 1000
    estado_inicial_estimado = 15
    sigma_inicial = 5

    # Definición de los modelos utilizados por el filtro (podrían ser diferentes del real)
    def modelo_transicion_filtro(estado):
        return 0.7 * estado + 6

    def modelo_observacion_filtro(estado):
        return estado

    ruido_transicion_filtro_sigma = 2
    ruido_observacion_filtro_sigma = 3

    # Ejecutar el filtro de partículas
    estados_estimados, trayectorias_particulas, pesos_particulas_trayectoria = filtro_particulas(
        observaciones, num_particulas, estado_inicial_estimado, sigma_inicial,
        modelo_transicion_filtro, ruido_transicion_filtro_sigma,
        modelo_observacion_filtro, ruido_observacion_filtro_sigma
    )

    # Visualización de los resultados
    plt.figure(figsize=(12, 6))
    plt.plot(tiempo, estado_real[1:], label='Estado Real', linestyle='--')
    plt.scatter(tiempo, observaciones, label='Observaciones', marker='o', s=15)
    plt.plot(tiempo, estados_estimados[1:], label='Estimación Filtro de Partículas', color='red')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.title('Filtrado de Partículas')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualización de las partículas en algunos pasos de tiempo
    tiempos_visualizacion = [0, 10, 25, 49]
    plt.figure(figsize=(15, 5))
    for i, t in enumerate(tiempos_visualizacion):
        plt.subplot(1, len(tiempos_visualizacion), i + 1)
        plt.hist(trayectorias_particulas[t], bins=20, density=True, alpha=0.6, label='Partículas')
        plt.axvline(estados_estimados[t], color='red', linestyle='dashed', linewidth=1, label='Estimación')
        plt.axvline(estado_real[t + 1], color='green', linestyle='dashed', linewidth=1, label='Real')
        plt.title(f'Tiempo: {t}')
        plt.xlabel('Estado')
        plt.ylabel('Densidad')
        plt.legend()
    plt.tight_layout()
    plt.show()
