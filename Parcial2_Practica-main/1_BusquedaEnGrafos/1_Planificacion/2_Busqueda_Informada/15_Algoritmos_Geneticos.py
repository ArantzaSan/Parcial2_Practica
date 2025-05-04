import random

def calcular_fitness(individuo):
    # Función de fitness: penalizar la distancia desde un vector objetivo
    vector_objetivo = [0] * len(individuo)
    return -sum(abs(individuo[i] - vector_objetivo[i]) for i in range(len(individuo)))

def crear_poblacion_inicial(tamano_poblacion, longitud_estado):
    # Genera una población inicial de individuos aleatorios
    return [
        [random.randint(-5, 5) for _ in range(longitud_estado)]
        for _ in range(tamano_poblacion)
    ]

def torneo_seleccion(poblacion, tamano_torneo=3):
    # Selecciona un individuo mediante torneo
    participantes = random.sample(poblacion, tamano_torneo)
    return max(participantes, key=calcular_fitness)

def recombinacion(padre1, padre2):
    # Realiza una recombinación uniforme entre dos padres
    hijo1 = []
    hijo2 = []
    for i in range(len(padre1)):
        if random.random() < 0.5:
            hijo1.append(padre1[i])
            hijo2.append(padre2[i])
        else:
            hijo1.append(padre2[i])
            hijo2.append(padre1[i])
    return hijo1, hijo2

def mutacion(individuo, tasa_mutacion=0.05):
    # Aplica mutación a un individuo
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            individuo[i] += random.randint(-2, 2)
    return individuo

def algoritmo_genetico_evolutivo(tamano_poblacion=15, num_generaciones=75, longitud_estado=4):
    # Inicializar la población
    poblacion = crear_poblacion_inicial(tamano_poblacion, longitud_estado)

    for generacion in range(num_generaciones):
        nueva_poblacion = []
        for _ in range(tamano_poblacion // 2):
            # Selección de padres mediante torneo
            madre = torneo_seleccion(poblacion)
            padre = torneo_seleccion(poblacion)
            # Recombinación
            descendiente1, descendiente2 = recombinacion(madre, padre)
            # Mutación
            nueva_poblacion.append(mutacion(descendiente1))
            nueva_poblacion.append(mutacion(descendiente2))
        poblacion = nueva_poblacion

    # Devolver el mejor individuo de la última generación
    mejor_individuo = max(poblacion, key=calcular_fitness)
    return mejor_individuo, calcular_fitness(mejor_individuo)

# Ejecutar el algoritmo genético evolutivo
solucion_final, valor_final = algoritmo_genetico_evolutivo()
print(f"Solución final: {solucion_final} con valor de fitness: {valor_final}")
