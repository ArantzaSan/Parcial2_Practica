class EvaluacionPoliticaEconomica:
    def __init__(self, probabilidad_prior_exito, probabilidad_prior_fracaso):
        self.probabilidad_prior_exito = probabilidad_prior_exito  # Probabilidad a priori de éxito
        self.probabilidad_prior_fracaso = probabilidad_prior_fracaso  # Probabilidad a priori de fracaso

    def actualizar_creencia(self, probabilidad_evidencia_exito, probabilidad_evidencia_fracaso, exito_observado):
        # Actualizar la creencia basada en la evidencia observada
        # Usar el teorema de Bayes para actualizar las probabilidades
        verosimilitud_exito = probabilidad_evidencia_exito if exito_observado else (1 - probabilidad_evidencia_exito)
        verosimilitud_fracaso = probabilidad_evidencia_fracaso if exito_observado else (1 - probabilidad_evidencia_fracaso)

        # Calcular las probabilidades posteriores
        probabilidad_posterior_exito = (verosimilitud_exito * self.probabilidad_prior_exito) / (
            (verosimilitud_exito * self.probabilidad_prior_exito) + (verosimilitud_fracaso * self.probabilidad_prior_fracaso)
        )
        probabilidad_posterior_fracaso = 1 - probabilidad_posterior_exito

        # Actualizar las probabilidades a priori para la siguiente iteración
        self.probabilidad_prior_exito = probabilidad_posterior_exito
        self.probabilidad_prior_fracaso = probabilidad_posterior_fracaso

        return probabilidad_posterior_exito, probabilidad_posterior_fracaso

# Probabilidades a priori iniciales
probabilidad_prior_exito_inicial = 0.5  # Creencia inicial de que la política tendrá éxito
probabilidad_prior_fracaso_inicial = 0.5  # Creencia inicial de que la política fracasará

# Crear el evaluador de políticas económicas
evaluador_politica = EvaluacionPoliticaEconomica(probabilidad_prior_exito_inicial, probabilidad_prior_fracaso_inicial)

# Evidencia observada
probabilidad_evidencia_exito_observado = 0.7  # Probabilidad de observar éxito si la política es efectiva
probabilidad_evidencia_fracaso_observado = 0.3  # Probabilidad de observar éxito si la política no es efectiva
observacion_exito = True  # Observación: la política tuvo éxito

# Actualizar la creencia basada en la evidencia observada
probabilidad_posterior_exito_calculada, probabilidad_posterior_fracaso_calculada = evaluador_politica.actualizar_creencia(
    probabilidad_evidencia_exito_observado, probabilidad_evidencia_fracaso_observado, observacion_exito
)

print(f"Probabilidad posterior de éxito: {probabilidad_posterior_exito_calculada:.2f}")
print(f"Probabilidad posterior de fracaso: {probabilidad_posterior_fracaso_calculada:.2f}")
