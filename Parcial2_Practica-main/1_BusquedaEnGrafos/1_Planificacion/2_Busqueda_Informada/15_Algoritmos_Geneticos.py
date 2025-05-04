import random

def heuristic(state):
    # Ejemplo de heurística: minimizar la suma de los elementos del estado
    return sum(state)

def initialize_population(pop_size, state_length):
    # Inicializa una población de estados aleatorios
    return [
        [random.randint(0, 10) for _ in range(state_length)]
        for _ in range(pop_size)
    ]

def selection(population):
    # Selecciona individuos basándose en la heurística (menor valor es mejor)
    sorted_population = sorted(population, key=heuristic)
    return sorted_population[:len(population) // 2]

def crossover(parent1, parent2):
    # Realiza un cruce de un punto entre dos padres
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(state, mutation_rate=0.1):
    # Realiza una mutación en el estado
    for i in range(len(state)):
        if random.random() < mutation_rate:
            state[i] = state[i] + random.choice([-1, 1])
    return state

def genetic_algorithm(pop_size=10, generations=50, state_length=4):
    # Inicializa la población
    population = initialize_population(pop_size, state_length)

    for _ in range(generations):
        # Selección
        selected = selection(population)

        # Crear la siguiente generación
        next_generation = []
        while len(next_generation) < pop_size:
            # Seleccionar padres
            parents = random.sample(selected, 2)
            # Cruce
            child1, child2 = crossover(parents[0], parents[1])
            # Mutación
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))

        # Actualizar la población
        population = next_generation

    # Devolver el mejor individuo de la última generación
    best_state = min(population, key=heuristic)
    return best_state, heuristic(best_state)

# Ejecutamos el algoritmo genético
final_state, final_value = genetic_algorithm()
print(f"Estado final: {final_state} con valor heurístico: {final_value}")
