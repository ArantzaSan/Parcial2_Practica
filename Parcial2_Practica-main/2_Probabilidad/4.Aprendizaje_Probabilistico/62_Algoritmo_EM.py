import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    """Función de densidad de probabilidad gaussiana."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def e_step(X, mu1, sigma1, mu2, sigma2, pi):
    """Paso E del algoritmo EM para una mezcla de dos gaussianas."""
    gamma1 = pi * gaussian(X, mu1, sigma1)
    gamma2 = (1 - pi) * gaussian(X, mu2, sigma2)
    responsabilidades1 = gamma1 / (gamma1 + gamma2 + 1e-9)  # Evitar división por cero
    responsabilidades2 = gamma2 / (gamma1 + gamma2 + 1e-9)
    return responsabilidades1, responsabilidades2

def m_step(X, responsabilidades1, responsabilidades2):
    """Paso M del algoritmo EM para una mezcla de dos gaussianas."""
    n = len(X)
    n1 = np.sum(responsabilidades1)
    n2 = np.sum(responsabilidades2)

    pi_nuevo = n1 / n
    mu1_nuevo = np.sum(responsabilidades1 * X) / n1 if n1 > 0 else 0
    mu2_nuevo = np.sum(responsabilidades2 * X) / n2 if n2 > 0 else 0
    sigma1_nuevo = np.sqrt(np.sum(responsabilidades1 * (X - mu1_nuevo) ** 2) / n1) if n1 > 0 else 1
    sigma2_nuevo = np.sqrt(np.sum(responsabilidades2 * (X - mu2_nuevo) ** 2) / n2) if n2 > 0 else 1

    return mu1_nuevo, sigma1_nuevo, mu2_nuevo, sigma2_nuevo, pi_nuevo

def algoritmo_em_gaussiana_mixta(X, num_iteraciones=100, inicializacion=None):
    """
    Implementa el algoritmo EM para ajustar una mezcla de dos distribuciones gaussianas a los datos.

    Args:
        X (np.ndarray): Datos unidimensionales.
        num_iteraciones (int): Número máximo de iteraciones del algoritmo EM.
        inicializacion (tuple, optional): Tupla con los valores iniciales de (mu1, sigma1, mu2, sigma2, pi).
                                          Si es None, se realiza una inicialización aleatoria.

    Returns:
        tuple: Los parámetros ajustados (mu1, sigma1, mu2, sigma2, pi) y la historia de los parámetros.
    """
    if inicializacion:
        mu1, sigma1, mu2, sigma2, pi = inicializacion
    else:
        # Inicialización aleatoria
        mu1 = np.random.normal(np.min(X), np.std(X))
        sigma1 = np.random.uniform(0.5, np.std(X))
        mu2 = np.random.normal(np.max(X), np.std(X))
        sigma2 = np.random.uniform(0.5, np.std(X))
        pi = np.random.uniform(0, 1)

    historia_parametros = [(mu1, sigma1, mu2, sigma2, pi)]

    for i in range(num_iteraciones):
        # Paso E
        responsabilidades1, responsabilidades2 = e_step(X, mu1, sigma1, mu2, sigma2, pi)

        # Paso M
        mu1, sigma1, mu2, sigma2, pi = m_step(X, responsabilidades1, responsabilidades2)
        historia_parametros.append((mu1, sigma1, mu2, sigma2, pi))

        # Criterio de convergencia (opcional): verificar si los parámetros cambian poco
        if i > 10:
            parametros_actual = historia_parametros[-1]
            parametros_previo = historia_parametros[-2]
            if np.allclose(parametros_actual, parametros_previo, atol=1e-4):
                print(f"Convergencia alcanzada en la iteración {i+1}")
                break

    return (mu1, sigma1, mu2, sigma2, pi), historia_parametros

if __name__ == "__main__":
    # Generar datos de ejemplo a partir de una mezcla de dos gaussianas
    np.random.seed(42)
    mu_real1, sigma_real1 = 10, 2
    mu_real2, sigma_real2 = 20, 3
    pi_real = 0.4
    n_samples = 500

    datos1 = np.random.normal(mu_real1, sigma_real1, int(pi_real * n_samples))
    datos2 = np.random.normal(mu_real2, sigma_real2, int((1 - pi_real) * n_samples))
    X = np.concatenate([datos1, datos2])
    np.random.shuffle(X)

    # Ejecutar el algoritmo EM
    parametros_ajustados, historia_parametros = algoritmo_em_gaussiana_mixta(X, num_iteraciones=200)
    mu1_ajustado, sigma1_ajustado, mu2_ajustado, sigma2_ajustado, pi_ajustado = parametros_ajustados

    print("\nParámetros Reales:")
    print(f"mu1: {mu_real1:.2f}, sigma1: {sigma_real1:.2f}, mu2: {mu_real2:.2f}, sigma2: {sigma_real2:.2f}, pi: {pi_real:.2f}")
    print("\nParámetros Ajustados por EM:")
    print(f"mu1: {mu1_ajustado:.2f}, sigma1: {sigma1_ajustado:.2f}, mu2: {mu2_ajustado:.2f}, sigma2: {sigma2_ajustado:.2f}, pi: {pi_ajustado:.2f}")

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=30, density=True, alpha=0.6, label='Datos')

    x_plot = np.linspace(X.min() - 2, X.max() + 2, 200)
    componente1_real = gaussian(x_plot, mu_real1, sigma_real1)
    componente2_real = gaussian(x_plot, mu_real2, sigma_real2)
    mezcla_real = pi_real * componente1_real + (1 - pi_real) * componente2_real
    plt.plot(x_plot, mezcla_real, 'k--', label='Mezcla Real')

    componente1_ajustado = gaussian(x_plot, mu1_ajustado, sigma1_ajustado)
    componente2_ajustado = gaussian(x_plot, mu2_ajustado, sigma2_ajustado)
    mezcla_ajustada = pi_ajustado * componente1_ajustado + (1 - pi_ajustado) * componente2_ajustado
    plt.plot(x_plot, mezcla_ajustada, 'r-', label='Mezcla Ajustada (EM)')

    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.title('Ajuste de Mezcla de Gaussianas con Algoritmo EM')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualizar la evolución de los parámetros
    historia_mu1 = [h[0] for h in historia_parametros]
    historia_sigma1 = [h[1] for h in historia_parametros]
    historia_mu2 = [h[2] for h in historia_parametros]
    historia_sigma2 = [h[3] for h in historia_parametros]
    historia_pi = [h[4] for h in historia_parametros]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(historia_mu1, label='mu1')
    plt.plot(historia_mu2, label='mu2')
    plt.axhline(mu_real1, color='k', linestyle='--', alpha=0.5)
    plt.axhline(mu_real2, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('Media')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(historia_sigma1, label='sigma1')
    plt.plot(historia_sigma2, label='sigma2')
    plt.axhline(sigma_real1, color='k', linestyle='--', alpha=0.5)
    plt.axhline(sigma_real2, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('Desviación Estándar')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(historia_pi, label='pi')
    plt.axhline(pi_real, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('Proporción (pi)')
    plt.xlabel('Iteración')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
