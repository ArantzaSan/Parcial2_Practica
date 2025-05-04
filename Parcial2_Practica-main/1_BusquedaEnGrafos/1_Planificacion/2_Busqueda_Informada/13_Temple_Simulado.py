import random
import math

def evaluar(solucion):
    # Función de costo: maximizar la desviación estándar de la solución
    n = len(solucion)
    if n == 0:
        return 0
    promedio = sum(solucion) / n
    desviacion_estandar = math.sqrt(sum((x - promedio) ** 2 for x in solucion) / n)
    return desviacion_estandar

def generar_sucesor(solucion_actual):
    # Genera un sucesor intercambiando dos elementos aleatorios
    sucesor = list(solucion_actual)
    i, j = random.sample(range(len(sucesor)), 2)
    sucesor[i], sucesor[j] = sucesor[j], sucesor[i]
    return sucesor

def probabilidad_aceptacion(costo_viejo, costo_nuevo, temperatura):
    if costo_nuevo > costo_viejo:
        return 1.0
    else:
        return math.exp((costo_nuevo - costo_viejo) / temperatura)

def recocido_simulado(estado_inicial, temp_inicial=5000, ratio_enfriamiento=0.001):
    estado_actual = list(estado_inicial)
    costo_actual = evaluar(estado_actual)
    mejor_solucion = list(estado_actual)
    mejor_costo = costo_actual
    temperatura = temp_inicial

    while temperatura > 0.1:
        nuevo_estado = generar_sucesor(estado_actual)
        nuevo_costo = evaluar(nuevo_estado)

        if probabilidad_aceptacion(costo_actual, nuevo_costo, temperatura) > random.random():
            estado_actual = nuevo_estado
            costo_actual = nuevo_costo

            if costo_actual > mejor_costo:
                mejor_solucion = list(estado_actual)
                mejor_costo = costo_actual

        # Enfriar la temperatura
        temperatura *= 1 - ratio_enfriamiento

    return mejor_solucion, mejor_costo

# Estado inicial
estado_inicial = [10, 5, 7, 3]

# Ejecutar el algoritmo de recocido simulado
solucion_final, valor_final = recocido_simulado(estado_inicial)
print(f"Solución final: {solucion_final} con valor de evaluación: {valor_final}")
