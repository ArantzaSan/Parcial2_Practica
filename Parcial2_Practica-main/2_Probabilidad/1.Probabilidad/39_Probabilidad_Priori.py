class EconomicPolicyEvaluation:
    def __init__(self, prior_prob_success, prior_prob_failure):
        self.prior_prob_success = prior_prob_success  # Probabilidad a priori de éxito
        self.prior_prob_failure = prior_prob_failure  # Probabilidad a priori de fracaso

    def update_belief(self, evidence_success_prob, evidence_failure_prob, observed_success):
        # Actualizar la creencia basada en la evidencia observada
        # Usar el teorema de Bayes para actualizar las probabilidades
        likelihood_success = evidence_success_prob if observed_success else (1 - evidence_success_prob)
        likelihood_failure = evidence_failure_prob if observed_success else (1 - evidence_failure_prob)

        # Calcular las probabilidades posteriores
        posterior_prob_success = (likelihood_success * self.prior_prob_success) / (
            (likelihood_success * self.prior_prob_success) + (likelihood_failure * self.prior_prob_failure)
        )
        posterior_prob_failure = 1 - posterior_prob_success

        # Actualizar las probabilidades a priori para la siguiente iteración
        self.prior_prob_success = posterior_prob_success
        self.prior_prob_failure = posterior_prob_failure

        return posterior_prob_success, posterior_prob_failure

# Probabilidades a priori iniciales
prior_prob_success = 0.5  # Creencia inicial de que la política tendrá éxito
prior_prob_failure = 0.5  # Creencia inicial de que la política fracasará

# Crear el evaluador de políticas económicas
policy_evaluation = EconomicPolicyEvaluation(prior_prob_success, prior_prob_failure)

# Evidencia observada
evidence_success_prob = 0.7  # Probabilidad de observar éxito si la política es efectiva
evidence_failure_prob = 0.3  # Probabilidad de observar éxito si la política no es efectiva
observed_success = True  # Observación: la política tuvo éxito

# Actualizar la creencia basada en la evidencia observada
posterior_prob_success, posterior_prob_failure = policy_evaluation.update_belief(
    evidence_success_prob, evidence_failure_prob, observed_success
)

print(f"Probabilidad posterior de éxito: {posterior_prob_success:.2f}")
print(f"Probabilidad posterior de fracaso: {posterior_prob_failure:.2f}")
