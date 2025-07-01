import random

# Algoritmos de búsqueda local aplicados a la seguridad de una empresa
# Este programa utiliza algoritmos de búsqueda local para optimizar la asignación de recursos de seguridad en una empresa.


# Función de evaluación: mide la calidad de una solución
# En este caso, la función evalúa la efectividad de la asignación de recursos de seguridad.
def evaluar_solucion(solucion):
    # Supongamos que cada recurso tiene un impacto en la seguridad (valores simulados)
    impacto_seguridad = [10, 20, 15, 25, 30]
    return sum(impacto_seguridad[i] for i in range(len(solucion)) if solucion[i] == 1)

# Genera una solución inicial aleatoria
def generar_solucion_inicial(n):
    return [random.choice([0, 1]) for _ in range(n)]

# Genera vecinos de una solución actual (modificando un recurso a la vez)
def generar_vecinos(solucion):
    vecinos = []
    for i in range(len(solucion)):
        vecino = solucion[:]
        vecino[i] = 1 - vecino[i]  # Cambia el estado del recurso (0 a 1 o 1 a 0)
        vecinos.append(vecino)
    return vecinos

# Algoritmo de búsqueda local: Hill Climbing
def hill_climbing(solucion_inicial):
    solucion_actual = solucion_inicial
    valor_actual = evaluar_solucion(solucion_actual)

    while True:
        vecinos = generar_vecinos(solucion_actual)
        mejor_vecino = max(vecinos, key=evaluar_solucion)
        mejor_valor = evaluar_solucion(mejor_vecino)

        # Si no hay mejora, se detiene el algoritmo
        if mejor_valor <= valor_actual:
            break

        # Actualiza la solución actual
        solucion_actual = mejor_vecino
        valor_actual = mejor_valor

    return solucion_actual, valor_actual

# Algoritmo de búsqueda local: Simulated Annealing
def simulated_annealing(solucion_inicial, temperatura_inicial, enfriamiento):
    solucion_actual = solucion_inicial
    valor_actual = evaluar_solucion(solucion_actual)
    temperatura = temperatura_inicial

    while temperatura > 0.1:
        vecino = random.choice(generar_vecinos(solucion_actual))
        valor_vecino = evaluar_solucion(vecino)

        # Acepta el vecino si es mejor o con cierta probabilidad si es peor
        if valor_vecino > valor_actual or random.random() < pow(2.718, (valor_vecino - valor_actual) / temperatura):
            solucion_actual = vecino
            valor_actual = valor_vecino

        # Reduce la temperatura
        temperatura *= enfriamiento

    return solucion_actual, valor_actual

# Configuración del problema
# Supongamos que hay 5 recursos de seguridad que se pueden activar (1) o desactivar (0)
n_recursos = 5

# Genera una solución inicial
solucion_inicial = generar_solucion_inicial(n_recursos)

# Ejecuta Hill Climbing
print("Ejecutando Hill Climbing...")
solucion_hc, valor_hc = hill_climbing(solucion_inicial)
print(f"Mejor solución (Hill Climbing): {solucion_hc}, Valor: {valor_hc}")

# Ejecuta Simulated Annealing
print("\nEjecutando Simulated Annealing...")
solucion_sa, valor_sa = simulated_annealing(solucion_inicial, temperatura_inicial=100, enfriamiento=0.95)
print(f"Mejor solución (Simulated Annealing): {solucion_sa}, Valor: {valor_sa}")
