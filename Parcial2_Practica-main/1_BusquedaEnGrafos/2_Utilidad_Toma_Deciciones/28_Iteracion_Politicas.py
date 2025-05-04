class CompanyManagement:
    def __init__(self, states, actions, rewards, transition_probabilities, gamma=0.9):
        self.states = states  # Estados posibles (niveles de ingresos)
        self.actions = actions  # Acciones posibles (niveles de inversión en marketing)
        self.rewards = rewards  # Recompensas inmediatas por acción en cada estado
        self.transition_probabilities = transition_probabilities  # Probabilidades de transición
        self.gamma = gamma  # Factor de descuento
        self.policy = {state: actions[0] for state in states}  # Política inicial
        self.value_function = {state: 0 for state in states}  # Función de valor inicial

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
states = ['low', 'medium', 'high']  # Niveles de ingresos
actions = ['invest_low', 'invest_medium', 'invest_high']  # Niveles de inversión en marketing

# Recompensas inmediatas por acción en cada estado
rewards = {
    'low': {'invest_low': 10, 'invest_medium': 20, 'invest_high': 25},
    'medium': {'invest_low': 30, 'invest_medium': 40, 'invest_high': 45},
    'high': {'invest_low': 50, 'invest_medium': 55, 'invest_high': 60}
}

# Probabilidades de transición de estado dada una acción
transition_probabilities = {
    'low': {
        'invest_low': {'low': 0.7, 'medium': 0.2, 'high': 0.1},
        'invest_medium': {'low': 0.5, 'medium': 0.3, 'high': 0.2},
        'invest_high': {'low': 0.4, 'medium': 0.3, 'high': 0.3}
    },
    'medium': {
        'invest_low': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
        'invest_medium': {'low': 0.2, 'medium': 0.5, 'high': 0.3},
        'invest_high': {'low': 0.2, 'medium': 0.3, 'high': 0.5}
    },
    'high': {
        'invest_low': {'low': 0.2, 'medium': 0.3, 'high': 0.5},
        'invest_medium': {'low': 0.1, 'medium': 0.3, 'high': 0.6},
        'invest_high': {'low': 0.1, 'medium': 0.2, 'high': 0.7}
    }
}

# Crear el administrador de la empresa y ejecutar la iteración de políticas
company = CompanyManagement(states, actions, rewards, transition_probabilities)
company.policy_iteration()

# Mostrar la política óptima
print("Política óptima de inversión en marketing:")
for state, action in company.policy.items():
    print(f"Si los ingresos son {state}, la inversión óptima es {action}.")
