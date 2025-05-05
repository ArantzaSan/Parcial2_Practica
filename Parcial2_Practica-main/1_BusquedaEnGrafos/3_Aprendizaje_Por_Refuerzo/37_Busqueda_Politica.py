import numpy as np
import random

# Definir el entorno como una cuadrícula
# 0: celda normal, -1: obstáculo, 1: recompensa
terreno = np.array([
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0],
    [0, -1, 0, 0]
])

# Parámetros de la política
parametros_politica = np.random.rand(4)  # Parámetros de la política para cada acción (arriba, abajo, izquierda, derecha)

# Definir las acciones posibles
acciones_posibles = ['arriba', 'abajo', 'izquierda', 'derecha']

# Función para obtener la siguiente acción basada en la política
def obtener_accion(estado, parametros):
    fila, columna = estado
    caracteristicas_estado = np.array([fila, columna, abs(fila - 3), abs(columna - 3)])  # Características del estado
    probabilidades_accion = np.exp(parametros * caracteristicas_estado)
    probabilidades_accion /= np.sum(probabilidades_accion)  # Normalizar para obtener probabilidades
    return np.random.choice(acciones_posibles, p=probabilidades_accion)

# Función para obtener el siguiente estado
def obtener_estado_siguiente(estado_actual, accion):
    fila_actual, columna_actual = estado_actual
    num_filas, num_columnas = terreno.shape
    if accion == 'arriba' and fila_actual > 0:
        return (fila_actual - 1, columna_actual)
    elif accion == 'abajo' and fila_actual < num_filas - 1:
        return (fila_actual + 1, columna_actual)
    elif accion == 'izquierda' and columna_actual > 0:
        return (fila_actual, columna_actual - 1)
    elif accion == 'derecha' and columna_actual < num_columnas - 1:
        return (fila_actual, columna_actual + 1)
    return estado_actual

# Función para evaluar la política
def evaluar_politica_actual(parametros, num_episodios_evaluacion):
    recompensa_total = 0
    num_filas, num_columnas = terreno.shape
    for _ in range(num_episodios_evaluacion):
        estado = (0, 0)
        terminado = False
        while not terminado:
            accion = obtener_accion(estado, parametros)
            siguiente_estado = obtener_estado_siguiente(estado, accion)
            recompensa = terreno[siguiente_estado]
            recompensa_total += recompensa
            estado = siguiente_estado
            if recompensa == 1 or
