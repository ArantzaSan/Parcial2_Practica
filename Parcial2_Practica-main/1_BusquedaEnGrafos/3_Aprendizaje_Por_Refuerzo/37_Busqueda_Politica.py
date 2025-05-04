import numpy as np
import random

# Definir el entorno como una cuadrícula
# 0: celda normal, -1: obstáculo, 1: recompensa
environment = np.array([
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0],
    [0, -1, 0, 0]
])

# Parámetros de la política
policy_params = np.random.rand(4)  # Parámetros de la política para cada acción (arriba, abajo, izquierda, derecha)

# Definir las acciones posibles
actions = ['up', 'down', 'left', 'right']

# Función para obtener la siguiente acción basada en la política
def get_next_action(state, policy_params):
    row, col = state
    state_features = np.array([row, col, abs(row - 3), abs(col - 3)])  # Características del estado
    action_probabilities = np.exp(policy_params * state_features)
    action_probabilities /= np.sum(action_probabilities)  # Normalizar para obtener probabilidades
    return np.random.choice(actions, p=action_probabilities)

# Función para obtener el siguiente estado
def get_next_state(state, action):
    row, col = state
    if action == 'up' and row > 0:
        return (row - 1, col)
    elif action == 'down' and row < environment.shape[0] - 1:
        return (row + 1, col)
    elif action == 'left' and col > 0:
        return (row, col - 1)
    elif action == 'right' and col < environment.shape[1] - 1:
        return (row, col + 1)
    return state

# Función para evaluar la política
def evaluate_policy(policy_params, num_episodes):
    total_reward = 0
    for _ in range(num_episodes):
        state = (0, 0)
        done = False
        while not done:
            action = get_next_action(state, policy_params)
            next_state = get_next_state(state, action)
            reward = environment[next_state]
            total_reward += reward
            state = next_state
            if reward == 1 or reward == -1:
                done = True
    return total_reward / num_episodes

# Búsqueda de la política óptima
num_iterations = 100
num_episodes_per_evaluation = 50
best_policy = policy_params
best_reward = evaluate_policy(best_policy, num_episodes_per_evaluation)

for _ in range(num_iterations):
    # Pequeña perturbación en los parámetros de la política
    new_policy = best_policy + np.random.normal(0, 0.1, size=best_policy.shape)
    new_reward = evaluate_policy(new_policy, num_episodes_per_evaluation)

    if new_reward > best_reward:
        best_reward = new_reward
        best_policy = new_policy

print("Mejor política encontrada:", best_policy)
print("Recompensa promedio:", best_reward)
