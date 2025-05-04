import numpy as np
import random

class VoltageControlMDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9):
        self.states = states  # Estados posibles (niveles de voltaje)
        self.actions = actions  # Acciones posibles (ajustes de voltaje)
        self.transition_probabilities = transition_probabilities  # Probabilidades de transición
        self.rewards = rewards  # Recompensas inmediatas por acción en cada estado
        self.gamma = gamma  # Factor de descuento
        self.value_function = {state: 0 for state in states}  # Función de valor inicial
        self.policy = {state: random.choice(actions) for state in states}  # Política inicial aleatoria

    def evaluate_policy(self):
        # Evaluar la política actual
        while True:
            delta = 0
            for state in self.states:
                old_value = self.value_function[state]
                action = self.policy[state]
                new_value = self.rewards[state][action] + self.gamma * sum(
                    self.transition_probabilities[state][action][next_state] * self.value_function[next_state]
                    for next_state in self.states
                )
                self.value_function[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < 1e-3:
                break

    def improve_policy(self):
        # Mejorar la política basada en la función de valor actual
        policy_stable = True
        for state in self.states:
            old_action = self.policy[state]
            action_values = {}
            for action in self.actions:
                action_values[action] = self.rewards[state][action] + self.gamma * sum(
                    self.transition_probabilities[state][action][next_state] * self.value_function[next_state]
                    for next_state in self.states
                )
            self.policy[state] = max(action_values, key=action_values.get)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        # Iteración de políticas
        while True:
            self.evaluate_policy()
            if self.improve_policy():
                break

# Definir estados, acciones, recompensas y probabilidades de transición
states = ['low', 'normal', 'high']  # Niveles de voltaje
actions = ['decrease', 'maintain', 'increase']  # Ajustes de voltaje

# Recompensas inmediatas por acción en cada estado
rewards = {
    'low': {'decrease': -10, 'maintain': 0, 'increase': 5},
    'normal': {'decrease': 0, 'maintain': 5, 'increase': 0},
    'high': {'decrease': 5, 'maintain': 0, 'increase': -10}
}

# Probabilidades de transición de estado dada una acción
transition_probabilities = {
    'low': {
        'decrease': {'low': 0.9, 'normal': 0.1, 'high': 0.0},
        'maintain': {'low': 0.6, 'normal': 0.4, 'high': 0.0},
        'increase': {'low': 0.2, 'normal': 0.7, 'high': 0.1}
    },
    'normal': {
        'decrease': {'low': 0.3, 'normal': 0.6, 'high': 0.1},
        'maintain': {'low': 0.1, 'normal': 0.8, 'high': 0.1},
        'increase': {'low': 0.1, 'normal': 0.4, 'high': 0.5}
    },
    'high': {
        'decrease': {'low': 0.1, 'normal': 0.7, 'high': 0.2},
        'maintain': {'low': 0.0, 'normal': 0.4, 'high': 0.6},
        'increase': {'low': 0.0, 'normal': 0.1, 'high': 0.9}
    }
}

# Crear el controlador de voltaje y ejecutar la iteración de políticas
voltage_control = VoltageControlMDP(states, actions, transition_probabilities, rewards)
voltage_control.policy_iteration()

# Mostrar la política óptima
print("Política óptima de control de voltaje:")
for state, action in voltage_control.policy.items():
    print(f"Si el voltaje es {state}, la acción óptima es {action}.")
