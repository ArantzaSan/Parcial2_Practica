class MarbleRace:
    def __init__(self, prior_prob_win, likelihood_win_given_past, likelihood_win_given_not_past):
        self.prior_prob_win = prior_prob_win  # Probabilidad a priori de que la canica gane
        self.likelihood_win_given_past = likelihood_win_given_past  # Probabilidad de ganar dado el rendimiento pasado
        self.likelihood_win_given_not_past = likelihood_win_given_not_past  # Probabilidad de ganar sin rendimiento pasado

    def update_probability(self, won_last_race):
        # Aplicar la regla de Bayes para actualizar la probabilidad de ganar
        if won_last_race:
            likelihood = self.likelihood_win_given_past
        else:
            likelihood = self.likelihood_win_given_not_past

        # Calcular la probabilidad posterior
        posterior_prob_win = (likelihood * self.prior_prob_win) / (
            (likelihood * self.prior_prob_win) +
            ((1 - likelihood) * (1 - self.prior_prob_win))
        )

        # Actualizar la probabilidad a priori para la siguiente iteración
        self.prior_prob_win = posterior_prob_win

        return posterior_prob_win

# Probabilidades iniciales
prior_prob_win = 0.5  # Probabilidad a priori de que la canica gane
likelihood_win_given_past = 0.8  # Probabilidad de ganar dado que ganó la última carrera
likelihood_win_given_not_past = 0.4  # Probabilidad de ganar dado que no ganó la última carrera

# Crear el modelo de carrera de canicas
marble_race = MarbleRace(prior_prob_win, likelihood_win_given_past, likelihood_win_given_not_past)

# Simular el resultado de la última carrera
won_last_race = True  # Supongamos que la canica ganó la última carrera

# Actualizar la probabilidad de ganar
posterior_prob_win = marble_race.update_probability(won_last_race)
print(f"Probabilidad actualizada de que la canica gane: {posterior_prob_win:.2f}")
