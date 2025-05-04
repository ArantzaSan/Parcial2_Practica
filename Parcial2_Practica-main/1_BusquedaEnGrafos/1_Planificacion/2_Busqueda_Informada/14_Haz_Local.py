import random

def heuristic(state):
    # Ejemplo de heurística: minimizar la suma de los elementos del estado
    return sum(state)

def get_neighbors(state):
    # Genera vecinos modificando un elemento del estado
    neighbors = []
    for i in range(len(state)):
        neighbor = state.copy()
        neighbor[i] = neighbor[i] + random.choice([-1, 1])  # Modificación simple
        neighbors.append(neighbor)
    return neighbors

def local_beam_search(initial_state, beam_width=3, max_iterations=100):
    # Inicializamos el haz con el estado inicial
    beam = [initial_state]
    best_state = initial_state
    best_value = heuristic(initial_state)

    for _ in range(max_iterations):
        candidates = []
        for state in beam:
            neighbors = get_neighbors(state)
            candidates.extend(neighbors)

        # Seleccionamos los mejores estados del haz
        candidates.sort(key=heuristic)
        beam = candidates[:beam_width]

        # Actualizamos el mejor estado encontrado
        for state in beam:
            value = heuristic(state)
            if value < best_value:
                best_state = state
                best_value = value

    return best_state, best_value

# Ejemplo de estado inicial
initial_state = [10, 5, 7, 3]

# Ejecutamos la búsqueda de haz local
final_state, final_value = local_beam_search(initial_state)
print(f"Estado final: {final_state} con valor heurístico: {final_value}")
