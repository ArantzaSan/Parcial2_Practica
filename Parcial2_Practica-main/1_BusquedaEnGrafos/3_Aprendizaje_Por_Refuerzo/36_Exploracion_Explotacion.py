import numpy as np
import random

# Definir el entorno como una cuadrícula
# 0: celda normal, -1: obstáculo, 1: recompensa
mapa_terreno = np.array([
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0],
    [0, -1, 0, 0]
])

# Parámetros de Q-aprendizaje
factor_alfa = 0.1  # Tasa de aprendizaje
factor_gamma = 0.9  # Factor de descuento
tasa_exploracion_inicial = 1.0  # Epsilon inicial
tasa_exploracion_minima = 0.01  # Epsilon mínimo
tasa_decremento_exploracion = 0.995  # Decaimiento de epsilon

# Inicializar la tabla Q con ceros
num_filas, num_columnas = mapa_terreno.shape
tabla_q_valores = np.zeros((num_filas, num_columnas, 4))  # 4 acciones posibles: arriba, abajo, izquierda, derecha

# Definir las acciones posibles
posibles_acciones = ['arriba', 'abajo', 'izquierda', 'derecha']

# Función para obtener la siguiente acción usando epsilon-greedy
def seleccionar_accion(estado, epsilon_actual):
    if random.uniform(0, 1) < epsilon_actual:
        return random.choice(posibles_acciones)  # Exploración: elegir una acción al azar
    else:
        return posibles_acciones[np.argmax(tabla_q_valores[estado])]  # Explotación: elegir la mejor acción conocida

# Función para obtener el siguiente estado
def calcular_siguiente_estado(estado_actual, accion):
    fila_actual, columna_actual = estado_actual
    if accion == 'arriba' and fila_actual > 0:
        return (fila_actual - 1, columna_actual)
    elif accion == 'abajo' and fila_actual < num_filas - 1:
        return (fila_actual + 1, columna_actual)
    elif accion == 'izquierda' and columna_actual > 0:
        return (fila_actual, columna_actual - 1)
    elif accion == 'derecha' and columna_actual < num_columnas - 1:
        return (fila_actual, columna_actual + 1)
    return estado_actual

# Entrenamiento del agente
num_intentos = 1000
epsilon_actual = tasa_exploracion_inicial
for intento in range(num_intentos):
    # Reiniciar el entorno y el estado inicial
    estado_actual = (0, 0)
    juego_terminado = False

    while not juego_terminado:
        accion_elegida = seleccionar_accion(estado_actual, epsilon_actual)
        siguiente_estado = calcular_siguiente_estado(estado_actual, accion_elegida)
        recompensa_obtenida = mapa_terreno[siguiente_estado]

        # Actualizar la tabla Q
        q_previo = tabla_q_valores[estado_actual][posibles_acciones.index(accion_elegida)]
        max_q_siguiente = np.max(tabla_q_valores[siguiente_estado])
        q_actualizado = (1 - factor_alfa) * q_previo + factor_alfa * (recompensa_obtenida + factor_gamma * max_q_siguiente)
        tabla_q_valores[estado_actual][posibles_acciones.index(accion_elegida)] = q_actualizado

        estado_actual = siguiente_estado

        if recompensa_obtenida == 1 or recompensa_obtenida == -1:
            juego_terminado =
