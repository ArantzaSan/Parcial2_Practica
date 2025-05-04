import random

def heuristic(state):
    # Ejemplo de heurística: minimizar la suma de los elementos del estado
    return sum(state)

def get_neighbors(state, tabu_list):
    # Genera vecinos modificando un elemento del estado
    neighbors = []
    for i in range(len(state)):
        neighbor = state.copy()
        # Evitamos mover en la dirección tabú
        if (i, '+') not in tabu_list:
            neighbor[i] = neighbor[i] + 1
            neighbors.append(neighbor)
        if (i, '-') not in tabu_list:
            neighbor = state.copy()
            neighbor[i] = neighbor[i] - 1
            neighbors.append(neighbor)
    return neighbors

def tabu_search(initial_state, max_iterations=100, tabu_size=5):
    current_state = initial_state
    current_value = heuristic(current_state)
    best_state = current_state
    best_value = current_value
    tabu_list = []

    for _ in range(max_iterations):
        neighbors = get_neighbors(current_state, tabu_list)
        if not neighbors:
            break

        # Seleccionamos el mejor vecino no tabú
        next_state = min(neighbors, key=heuristic)
        next_value = heuristic(next_state)

        # Actualizamos el mejor estado encontrado
        if next_value < best_value:
            best_state = next_state
            best_value = next_value

        # Actualizamos el estado actual
        current_state = next_state
        current_value = next_value

        # Actualizamos la lista tabú
        for i in range(len(current_state)):
            if current_state[i] > initial_state[i]:
                tabu_list.append((i, '-'))
            elif current_state[i] < initial_state[i]:
                tabu_list.append((i, '+'))

        # Mantenemos el tamaño de la lista tabú
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_state, best_value

# Ejemplo de estado inicial
initial_state = [10, 5, 7, 3]

# Ejecutamos la búsqueda tabú
final_state, final_value = tabu_search(initial_state)
print(f"Estado final: {final_state} con valor heurístico: {final_value}")
