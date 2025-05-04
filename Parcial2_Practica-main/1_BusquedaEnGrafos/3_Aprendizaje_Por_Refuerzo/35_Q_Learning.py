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

# Parámetros de Q-learning
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Inicializar la tabla Q con ceros
rows, cols = environment.shape
q_table = np.zeros((rows, cols, 4))  # 4 acciones posibles: arriba, abajo, izquierda, derecha

# Definir las acciones posibles
actions = ['up', 'down', 'left', 'right']

# Función para obtener la siguiente acción
def get_next_action(state, exploration_rate):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state])]

# Función para obtener el siguiente estado
def get_next_state(state, action):
    row, col = state
    if action == 'up' and row > 0:
        return (row - 1, col)
    elif action == 'down' and row < rows - 1:
        return (row + 1, col)
    elif action == 'left' and col > 0:
        return (row, col - 1)
    elif action == 'right' and col < cols - 1:
        return (row, col + 1)
    return state

# Entrenamiento del agente
num_episodes = 1000
for episode in range(num_episodes):
    # Reiniciar el entorno y el estado inicial
    state = (0, 0)
    done = False

    while not done:
        action = get_next_action(state, exploration_rate)
        next_state = get_next_state(state, action)
        reward = environment[next_state]

        # Actualizar la tabla Q
        current_q = q_table[state][actions.index(action)]
        max_future_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[state][actions.index(action)] = new_q

        state = next_state

        if reward == 1 or reward == -1:
            done = True

    # Reducir la tasa de exploración
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Imprimir la tabla Q final
print("Tabla Q final:")
print(q_table)
