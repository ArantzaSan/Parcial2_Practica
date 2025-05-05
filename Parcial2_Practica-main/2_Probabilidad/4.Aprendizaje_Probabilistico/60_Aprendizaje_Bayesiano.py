import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def actualizar_creencia_bayesiana(creencia_anterior, likelihood, evidencia):
    """
    Actualiza una creencia probabilística utilizando el teorema de Bayes.

    Args:
        creencia_anterior (np.ndarray): Distribución de probabilidad previa (prior).
        likelihood (np.ndarray): Probabilidad de los datos dada la hipótesis.
        evidencia (float): Probabilidad total de los datos (normalización).

    Returns:
        np.ndarray: Distribución de probabilidad posterior.
    """
    numerador = creencia_anterior * likelihood
    posterior = numerador / evidencia
    return posterior

def calcular_evidencia(creencia_anterior, likelihood, espacio_hipotesis):
    """
    Calcula la evidencia (probabilidad total de los datos).

    Args:
        creencia_anterior (np.ndarray): Distribución de probabilidad previa.
        likelihood (np.ndarray): Probabilidad de los datos dada la hipótesis.
        espacio_hipotesis (np.ndarray): El espacio de posibles valores de la hipótesis.

    Returns:
        float: La evidencia.
    """
    return np.sum(creencia_anterior * likelihood) * np.diff(espacio_hipotesis)[0]

def likelihood_moneda(prob_cara, num_lanzamientos, num_caras):
    """
    Calcula la verosimilitud de observar un número de caras dado una probabilidad de cara.

    Args:
        prob_cara (float): Probabilidad de obtener cara en un lanzamiento.
        num_lanzamientos (int): Número total de lanzamientos.
        num_caras (int): Número de caras observadas.

    Returns:
        float: La verosimilitud.
    """
    from scipy.special import comb
    return comb(num_lanzamientos, num_caras) * (prob_cara ** num_caras) * ((1 - prob_cara) ** (num_lanzamientos - num_caras))

if __name__ == "__main__":
    # Problema de la moneda sesgada: estimar la probabilidad de obtener cara.
    espacio_probabilidades = np.linspace(0, 1, 200)  # Espacio de hipótesis (probabilidad de cara)

    # Creencia previa (prior): moneda justa
