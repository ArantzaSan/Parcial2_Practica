import random

def funcion_evaluacion(estado):
    # Heurística alternativa: varianza de los elementos del estado (minimizarla)
    n = len(estado)
    if n == 0:
        return 0
    promedio = sum(estado) / n
    varianza = sum((x - promedio) ** 2 for x in estado) / n
    return varianza

def generar_vecindario(estado_actual):
    # Crea vecinos intercambiando pares de elementos del estado
    vecinos = []
    n = len(estado_actual)
    for i in range(n):
        for j in range(i + 1, n):
            vecino = list(estado_actual)
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
    return vecinos

def escalada_de_colinas(estado_inicial, max_iteraciones=1000):
    estado_actual = list(estado_inicial)
    valor_actual = funcion_evaluacion(estado_actual)

    for _ in range(max_iteraciones):
        vecinos = generar_vecindario(estado_actual)
        mejor_vecino = None
        mejor_valor = valor_actual

        for vecino in vecinos:
            valor_vecino = funcion_evaluacion(vecino)
            if valor_vecino < mejor_valor:
                mejor_valor = valor_vecino
                mejor_vecino = vecino

        # Moverse al mejor vecino si hay una mejora
        if mejor_vecino:
            estado_actual = mejor_vecino
            valor_actual = mejor_valor
        else:
            # Detener si no hay mejora en el vecindario
            break

    return estado_actual, valor_actual

# Estado inicial de ejemplo
estado_inicial = [10, 5, 7, 3]

# Ejecutar el algoritmo de escalada de colinas
estado_final, valor_final = escalada_de_colinas(estado_inicial)
print(f"Estado resultante: {estado_final} con valor de evaluación: {valor_final}")
