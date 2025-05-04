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

def hill_climbing(initial_state, max_iterations=1000):
    current_state = initial_state
    current_value = heuristic(current_state)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current_state)
        next_state = min(neighbors, key=heuristic)
        next_value = heuristic(next_state)

        # Si encontramos un vecino mejor, nos movemos a ese estado
        if next_value < current_value:
            current_state = next_state
            current_value = next_value
        else:
            # Si no hay mejora, terminamos la búsqueda
            break

    return current_state, current_value

# Ejemplo de estado inicial
initial_state = [10, 5, 7, 3]

# Ejecutamos la búsqueda de ascensión de colinas
final_state, final_value = hill_climbing(initial_state)
print(f"Estado final: {final_state} con valor heurístico: {final_value}")
