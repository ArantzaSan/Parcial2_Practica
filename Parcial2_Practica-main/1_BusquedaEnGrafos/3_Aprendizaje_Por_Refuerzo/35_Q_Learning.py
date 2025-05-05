import numpy as np
import random

# Definir el entorno como una cuadrícula
# 0: celda normal, -1: obstáculo, 1: recompensa
mapa_entorno = np.array([
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0],
    [0, -1, 0, 0]
])

# Parámetros de Q-aprendizaje
tasa_aprendizaje = 0.1
factor_descuento = 0.9
tasa_exploracion = 1.0
tasa_exploracion_maxima = 1.0
tasa_exploracion_minima = 0.01
tasa_decremento_exploracion = 0.01

# Inicializar la tabla Q con ceros
filas, columnas = mapa_entorno.shape
tabla_q = np.zeros((filas, columnas, 4))  # 4 acciones posibles: arriba, abajo, izquierda, derecha

# Definir las acciones posibles
acciones = ['arriba', 'abajo', 'izquierda', 'derecha']

# Función para obtener la siguiente acción
def obtener_siguiente_accion(estado, tasa_exploracion):
    if random.uniform(0, 1) < tasa_exploracion:
        return random.choice(acciones)
    else:
        return acciones[np.argmax(tabla_q[estado])]

# Función para obtener el siguiente estado
def obtener_siguiente_estado(estado, accion):
    fila, columna = estado
    if accion == 'arriba' and fila > 0:
        return (fila - 1, columna)
    elif accion == 'abajo' and fila < filas - 1:
        return (fila + 1, columna)
    elif accion == 'izquierda' and columna > 0:
        return (fila, columna - 1)
    elif accion == 'derecha' and columna < columnas - 1:
        return (fila, columna + 1)
    return estado

# Entrenamiento del agente
numero_episodios = 1000
for episodio in range(numero_episodios):
    # Reiniciar el entorno y el estado inicial
    estado = (0, 0)
    terminado = False

    while not terminado:
        accion = obtener_siguiente_accion(estado, tasa_exploracion)
        siguiente_estado = obtener_siguiente_estado(estado, accion)
        recompensa = mapa_entorno[siguiente_estado]

        # Actualizar la tabla Q
        q_actual = tabla_q[estado][acciones.index(accion)]
        max_q_futuro = np.max(tabla_q[siguiente_estado])
        q_nuevo = (1 - tasa_aprendizaje) * q_actual + tasa_aprendizaje * (recompensa + factor_descuento * max_q_futuro)
        tabla_q[estado][acciones.index(accion)] = q_nuevo

        estado = siguiente_estado

        if recompensa == 1 or recompensa == -1:
            terminado = True

    # Reducir la tasa de exploración
    tasa_exploracion = tasa_exploracion_minima + (tasa_exploracion_maxima - tasa_exploracion_minima) * np.exp(-tasa_decremento_exploracion * episodio)

# Imprimir la tabla Q final
print("Tabla Q final:")
print(tabla_q)
