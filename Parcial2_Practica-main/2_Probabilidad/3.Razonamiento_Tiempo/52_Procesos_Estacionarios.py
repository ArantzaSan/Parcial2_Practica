import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def generar_proceso_estacionario_media_constante(media, desviacion_estandar, longitud):
    """
    Genera un proceso estacionario con media constante utilizando ruido blanco gaussiano.

    Args:
        media (float): La media del proceso.
        desviacion_estandar (float): La desviación estándar del ruido blanco.
        longitud (int): La longitud de la serie de tiempo a generar.

    Returns:
        numpy.ndarray: Una serie de tiempo estacionaria.
    """
    ruido = np.random.normal(0, desviacion_estandar, longitud)
    proceso = np.array([media] * longitud) + ruido
    return proceso

def generar_proceso_autorregresivo_estacionario(ar_coeficientes, desviacion_estandar, longitud, descartar=100):
    """
    Genera un proceso autorregresivo (AR) estacionario.

    Args:
        ar_coeficientes (list): Lista de coeficientes AR (ej., [0.5]).
                                 Asegurarse de que las raíces del polinomio característico estén fuera del círculo unitario
                                 para garantizar la estacionariedad.
        desviacion_estandar (float): La desviación estándar del ruido blanco.
        longitud (int): La longitud de la serie de tiempo a generar.
        descartar (int): Número de puntos iniciales a descartar para minimizar los efectos transitorios.

    Returns:
        numpy.ndarray: Una serie de tiempo AR estacionaria.
    """
    p = len(ar_coeficientes)
    ruido = np.random.normal(0, desviacion_estandar, longitud + descartar)
    proceso = np.zeros(longitud + descartar)

    for t in range(p, longitud + descartar):
        termino_ar = np.dot(ar_coeficientes, proceso[t-p:t][::-1])
        proceso[t] = termino_ar + ruido[t]

    return proceso[descartar:]

def analizar_estacionariedad(serie_tiempo, nombre="Serie de Tiempo"):
    """
    Realiza pruebas visuales y estadísticas para evaluar la estacionariedad de una serie de tiempo.

    Args:
        serie_tiempo (numpy.ndarray): La serie de tiempo a analizar.
        nombre (str): Nombre de la serie de tiempo para los gráficos.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(serie_tiempo)
    plt.title(f'Gráfico de {nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')

    plt.subplot(3, 1, 2)
    sm.graphics.tsa.plot_acf(serie_tiempo, lags=40, ax=plt.gca())
    plt.title(f'Función de Autocorrelación (ACF) de {nombre}')

    plt.subplot(3, 1, 3)
    sm.graphics.tsa.plot_pacf(serie_tiempo, lags=40, ax=plt.gca())
    plt.title(f'Función de Autocorrelación Parcial (PACF) de {nombre}')

    plt.tight_layout()
    plt.show()

    # Prueba de Dickey-Fuller aumentada
    adf_test = sm.tsa.stattools.adfuller(serie_tiempo)
    print(f'\nResultados de la Prueba ADF para {nombre}:')
    print(f'Estadístico ADF: {adf_test[0]:.3f}')
    print(f'Valor p: {adf_test[1]:.3f}')
    print('Valores críticos:')
    for key, value in adf_test[4].items():
        print(f'   {key}: {value:.3f}')

    if adf_test[1] <= 0.05:
        print(f'{nombre} parece ser estacionaria (rechazamos la hipótesis nula).')
    else:
        print(f'{nombre} puede no ser estacionaria (no rechazamos la hipótesis nula).')

if __name__ == "__main__":
    longitud_serie = 500

    # Ejemplo 1: Proceso estacionario con media constante
    media_constante = 5
    desviacion_ruido = 1
    proceso_mc = generar_proceso_estacionario_media_constante(media_constante, desviacion_ruido, longitud_serie)
    analizar_estacionariedad(proceso_mc, "Proceso con Media Constante")

    # Ejemplo 2: Proceso autorregresivo (AR(1)) estacionario
    coeficientes_ar1 = [0.7]  # La raíz (1/0.7) está fuera del círculo unitario (|1/0.7| > 1)
    desviacion_ruido_ar1 = 0.5
    proceso_ar1 = generar_proceso_autorregresivo_estacionario(coeficientes_ar1, desviacion_ruido_ar1, longitud_serie)
    analizar_estacionariedad(proceso_ar1, "Proceso AR(1) Estacionario")

    # Ejemplo 3: Proceso autorregresivo (AR(2)) estacionario
    # Coeficientes que aseguran estacionariedad (comprobar las raíces del polinomio característico)
    coeficientes_ar2 = [0.5, -0.2]
    desviacion_ruido_ar2 = 0.3
    proceso_ar2 = generar_proceso_autorregresivo_estacionario(coeficientes_ar2, desviacion_ruido_ar2, longitud_serie)
    analizar_estacionariedad(proceso_ar2, "Proceso AR(2) Estacionario")
