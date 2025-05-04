class ElectionPrediction:
    def __init__(self, prior_prob_win, poll_support_prob, voter_turnout_prob):
        self.prior_prob_win = prior_prob_win  # Probabilidad a priori de que el candidato gane
        self.poll_support_prob = poll_support_prob  # Probabilidad de apoyo en encuestas
        self.voter_turnout_prob = voter_turnout_prob  # Probabilidad de participación de votantes

    def conditional_probability(self, poll_support, voter_turnout):
        # Calcular la probabilidad conjunta de ganar dado el apoyo en encuestas y la participación de votantes
        joint_prob_win = (self.prior_prob_win *
                          self.poll_support_prob[poll_support] *
                          self.voter_turnout_prob[voter_turnout])

        joint_prob_lose = ((1 - self.prior_prob_win) *
                           (1 - self.poll_support_prob[poll_support]) *
                           (1 - self.voter_turnout_prob[voter_turnout]))

        # Calcular la probabilidad total (normalización)
        total_prob = joint_prob_win + joint_prob_lose

        # Calcular la probabilidad condicional de ganar
        conditional_prob_win = joint_prob_win / total_prob if total_prob != 0 else 0
        return conditional_prob_win

# Probabilidades a priori y condicionales
prior_prob_win = 0.5  # Probabilidad a priori de que el candidato gane

# Probabilidades de apoyo en encuestas
poll_support_prob = {
    'high': 0.7,   # Alta probabilidad de apoyo en encuestas
    'medium': 0.5, # Probabilidad media de apoyo en encuestas
    'low': 0.3     # Baja probabilidad de apoyo en encuestas
}

# Probabilidades de participación de votantes
voter_turnout_prob = {
    'high': 0.6,   # Alta probabilidad de participación de votantes
    'medium': 0.4, # Probabilidad media de participación de votantes
    'low': 0.2     # Baja probabilidad de participación de votantes
}

# Crear el modelo de predicción de elecciones
election_prediction = ElectionPrediction(prior_prob_win, poll_support_prob, voter_turnout_prob)

# Calcular la probabilidad condicional de ganar dado el apoyo en encuestas y la participación de votantes
poll_support = 'high'  # Observación: alto apoyo en encuestas
voter_turnout = 'medium'  # Observación: participación media de votantes

conditional_prob_win = election_prediction.conditional_probability(poll_support, voter_turnout)
print(f"Probabilidad condicional de que el candidato gane: {conditional_prob_win:.2f}")
