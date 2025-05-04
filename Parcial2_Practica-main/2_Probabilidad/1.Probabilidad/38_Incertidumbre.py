import numpy as np

class InvestmentUncertainty:
    def __init__(self, initial_investment, success_prob, failure_prob, success_return, failure_return):
        self.initial_investment = initial_investment
        self.success_prob = success_prob
        self.failure_prob = failure_prob
        self.success_return = success_return
        self.failure_return = failure_return

    def expected_value(self):
        # Calcular el valor esperado de la inversión
        expected_value = (self.success_prob * self.success_return) + (self.failure_prob * self.failure_return)
        return expected_value - self.initial_investment

    def simulate_investment(self, num_simulations=1000):
        # Simular múltiples escenarios de inversión
        outcomes = []
        for _ in range(num_simulations):
            if np.random.rand() < self.success_prob:
                outcome = self.success_return
            else:
                outcome = self.failure_return
            net_outcome = outcome - self.initial_investment
            outcomes.append(net_outcome)
        return outcomes

    def investment_risk(self, outcomes):
        # Evaluar el riesgo de la inversión
        outcomes = np.array(outcomes)
        mean_outcome = np.mean(outcomes)
        std_dev_outcome = np.std(outcomes)
        return mean_outcome, std_dev_outcome

# Parámetros de la inversión
initial_investment = 1000  # Inversión inicial
success_prob = 0.6  # Probabilidad de éxito
failure_prob = 0.4  # Probabilidad de fracaso
success_return = 2000  # Retorno en caso de éxito
failure_return = 500   # Retorno en caso de fracaso

# Crear el modelo de incertidumbre de inversión
investment = InvestmentUncertainty(initial_investment, success_prob, failure_prob, success_return, failure_return)

# Calcular el valor esperado de la inversión
expected_value = investment.expected_value()
print(f"Valor esperado de la inversión: ${expected_value:.2f}")

# Simular múltiples escenarios de inversión
outcomes = investment.simulate_investment(num_simulations=1000)

# Evaluar el riesgo de la inversión
mean_outcome, std_dev_outcome = investment.investment_risk(outcomes)
print(f"Resultado promedio de la inversión: ${mean_outcome:.2f}")
print(f"Desviación estándar de la inversión: ${std_dev_outcome:.2f}")
