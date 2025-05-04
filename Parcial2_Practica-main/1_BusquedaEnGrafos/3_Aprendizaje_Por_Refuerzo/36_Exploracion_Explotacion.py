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
epsilon = 1.0  # Tasa de exploración inicial
epsilon_min = 0.01  # Tasa de exploración mínima
epsilon_decay = 0.995  # Tasa de decaimiento de la exploración

# Inicializar la tabla Q con ceros
rows, cols = environment.shape
q_table = np.zeros((rows, cols, 4))  # 4 acciones posibles: arriba, abajo, izquierda, derecha

# Definir las acciones posibles
actions = ['up', 'down', 'left', 'right']

# Función para obtener la siguiente acción usando epsilon-greedy
def get_next_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploración: elegir una acción aleatoria
    else:
        return actions[np.argmax(q_table[state])]  # Explotación: elegir la mejor acción conocida

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
        action = get_next_action(state, epsilon)
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
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Imprimir la tabla Q final
print("Tabla Q final:")
print(q_table)
