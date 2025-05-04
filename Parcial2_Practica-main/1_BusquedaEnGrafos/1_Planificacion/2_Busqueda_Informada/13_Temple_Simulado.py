import random
import math

def heuristic(state):
    # Ejemplo de heurística: minimizar la suma de los elementos del estado
    return sum(state)

def get_neighbor(state):
    # Genera un vecino modificando un elemento del estado
    neighbor = state.copy()
    index = random.randint(0, len(state) - 1)
    change = random.choice([-1, 1])
    neighbor[index] += change
    return neighbor

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)

def simulated_annealing(initial_state, initial_temperature=1000, cooling_rate=0.003):
    current_state = initial_state
    current_cost = heuristic(current_state)
    best_state = current_state
    best_cost = current_cost
    temperature = initial_temperature

    while temperature > 1:
        new_state = get_neighbor(current_state)
        new_cost = heuristic(new_state)

        if acceptance_probability(current_cost, new_cost, temperature) > random.random():
            current_state = new_state
            current_cost = new_cost

            if new_cost < best_cost:
                best_state = new_state
                best_cost = new_cost

        # Enfriamiento
        temperature *= 1 - cooling_rate

    return best_state, best_cost

# Ejemplo de estado inicial
initial_state = [10, 5, 7, 3]

# Ejecutamos la búsqueda de templado simulado
final_state, final_value = simulated_annealing(initial_state)
print(f"Estado final: {final_state} con valor heurístico: {final_value}")
