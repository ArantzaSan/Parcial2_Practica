import random

class AstronomyPOMDP:
    def __init__(self, states, actions, observations, transition_probabilities, observation_probabilities, rewards, gamma=0.9):
        self.states = states  # Estados posibles (condiciones del cielo)
        self.actions = actions  # Acciones posibles (observar estrella o galaxia)
        self.observations = observations  # Observaciones posibles (clima aparente)
        self.transition_probabilities = transition_probabilities  # Probabilidades de transición
        self.observation_probabilities = observation_probabilities  # Probabilidades de observación
        self.rewards = rewards  # Recompensas inmediatas por acción en cada estado
        self.gamma = gamma  # Factor de descuento
        self.belief_state = {state: 1/len(states) for state in states}  # Estado de creencia inicial uniforme

    def update_belief_state(self, action, observation):
        # Actualizar el estado de creencia basado en la acción y la observación
        new_belief = {}
        for next_state in self.states:
            prob = 0
            for state in self.states:
                transition_prob = self.transition_probabilities[state][action][next_state]
                observation_prob = self.observation_probabilities[next_state][observation]
                prob += self.belief_state[state] * transition_prob * observation_prob
            new_belief[next_state] = prob

        # Normalizar el estado de creencia
        total = sum(new_belief.values())
        self.belief_state = {state: prob / total for state, prob in new_belief.items()}

    def choose_action(self):
        # Elegir la mejor acción basada en el estado de creencia actual
        action_values = {}
        for action in self.actions:
            value = sum(
                self.belief_state[state] * (self.rewards[state][action] + self.gamma * max(
                    sum(self.transition_probabilities[state][action][next_state] * self.belief_state[next_state]
                        for next_state in self.states),
                    0
                ))
                for state in self.states
            )
            action_values[action] = value
        return max(action_values, key=action_values.get)

    def simulate(self, steps=5):
        # Simular la toma de decisiones durante un número de pasos
        for _ in range(steps):
            action = self.choose_action()
            observation = random.choice(self.observations)  # Simular una observación
            print(f"Acción elegida: {action}, Observación: {observation}")
            self.update_belief_state(action, observation)
            print(f"Estado de creencia actual: {self.belief_state}")

# Definir estados, acciones, observaciones, recompensas y probabilidades
states = ['clear', 'cloudy', 'rainy']  # Condiciones del cielo
actions = ['observe_star', 'observe_galaxy']  # Acciones posibles
observations = ['clear_sky', 'cloudy_sky', 'rainy_sky']  # Observaciones posibles

# Recompensas inmediatas por acción en cada estado
rewards = {
    'clear': {'observe_star': 10, 'observe_galaxy': 8},
    'cloudy': {'observe_star': 5, 'observe_galaxy': 4},
    'rainy': {'observe_star': 1, 'observe_galaxy': 2}
}

# Probabilidades de transición de estado dada una acción
transition_probabilities = {
    'clear': {
        'observe_star': {'clear': 0.7, 'cloudy': 0.2, 'rainy': 0.1},
        'observe_galaxy': {'clear': 0.6, 'cloudy': 0.3, 'rainy': 0.1}
    },
    'cloudy': {
        'observe_star': {'clear': 0.3, 'cloudy': 0.5, 'rainy': 0.2},
        'observe_galaxy': {'clear': 0.2, 'cloudy': 0.6, 'rainy': 0.2}
    },
    'rainy': {
        'observe_star': {'clear': 0.1, 'cloudy': 0.3, 'rainy': 0.6},
        'observe_galaxy': {'clear': 0.1, 'cloudy': 0.2, 'rainy': 0.7}
    }
}

# Probabilidades de observación dado un estado
observation_probabilities = {
    'clear': {'clear_sky': 0.8, 'cloudy_sky': 0.15, 'rainy_sky': 0.05},
    'cloudy': {'clear_sky': 0.2, 'cloudy_sky': 0.7, 'rainy_sky': 0.1},
    'rainy': {'clear_sky': 0.05, 'cloudy_sky': 0.25, 'rainy_sky': 0.7}
}

# Crear el POMDP de astronomía y simular la toma de decisiones
astronomy_pomdp = AstronomyPOMDP(states, actions, observations, transition_probabilities, observation_probabilities, rewards)
astronomy_pomdp.simulate(steps=5)
