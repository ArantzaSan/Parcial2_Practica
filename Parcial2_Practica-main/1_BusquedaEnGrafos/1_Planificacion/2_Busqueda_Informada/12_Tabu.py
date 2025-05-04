import random
from collections import deque

def evaluar_estado(estado):
    # Heurística: penalizar la distancia desde un estado "ideal"
    ideal_state = [0] * len(estado)
    return sum(abs(estado[i] - ideal_state[i]) for i in range(len(estado)))

def generar_movimientos(estado_actual, memoria_tabu):
    # Genera movimientos intercambiando elementos, evitando movimientos tabú recientes
    movimientos = []
    n = len(estado_actual)
    for i in range(n):
        for j in range(i + 1, n):
            movimiento = tuple(sorted(((i, estado_actual[i]), (j, estado_actual[j]))))
            if movimiento not in memoria_tabu:
                vecino = list(estado_actual)
                vecino[i], vecino[j] = vecino[j], vecino[i]
                movimientos.append(vecino)
    return movimientos

def busqueda_tabu_adaptativa(estado_inicial, max_iteraciones=150, tamano_tabu=7):
    estado_corriente = list(estado_inicial)
    valor_corriente = evaluar_estado(estado_corriente)
    mejor_estado = list(estado_corriente)
    mejor_valor = valor_corriente
    lista_tabu = deque(maxlen=tamano_tabu)

    for iteracion in range(max_iteraciones):
        movimientos_posibles = generar_movimientos(estado_corriente, lista_tabu)

        if not movimientos_posibles:
            break

        mejor_movimiento = None
        mejor_valor_movimiento = float('inf')

        for movimiento in movimientos_posibles:
            valor_movimiento = evaluar_estado(movimiento)
            if valor_movimiento < mejor_valor_movimiento:
                mejor_valor_movimiento = valor_movimiento
                mejor_movimiento = movimiento

        if mejor_movimiento:
            estado_corriente = mejor_movimiento
            valor_corriente = mejor_valor_movimiento

            # Actualizar el mejor estado global
            if valor_corriente < mejor_valor:
                mejor_estado = list(estado_corriente)
                mejor_valor = valor_corriente

            # Añadir el movimiento inverso a la lista tabú
            indices_cambiados = []
            valores_originales = []
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != mejor_movimiento[i]:
                    indices_cambiados.append(i)
                    valores_originales.append(estado_inicial[i])

            if len(indices_cambiados) == 2:
                movimiento_inverso = tuple(sorted(((indices_cambiados[0], mejor_movimiento[indices_cambiados[0]]), (indices_cambiados[1], mejor_movimiento[indices_cambiados[1]]))))
                lista_tabu.append(movimiento_inverso)
            elif len(indices_cambiados) == 1: # Para el caso del vecino generado con +/- 1 (si se usara)
                pass # No se define un movimiento inverso claro para +/- 1 en este esquema de vecinos

    return mejor_estado, mejor_valor

# Estado inicial
estado_inicial = [10, 5, 7, 3]

# Ejecutar la búsqueda tabú adaptativa
estado_final, valor_final = busqueda_tabu_adaptativa(estado_inicial)
print(f"Estado final encontrado: {estado_final} con valor de evaluación: {valor_final}")
