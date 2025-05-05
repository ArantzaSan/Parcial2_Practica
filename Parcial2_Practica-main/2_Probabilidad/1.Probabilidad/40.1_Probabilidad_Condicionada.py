class PrediccionElecciones:
    def __init__(self, probabilidad_prior_victoria, probabilidad_apoyo_encuestas, probabilidad_participacion_votantes):
        self.probabilidad_prior_victoria = probabilidad_prior_victoria  # Probabilidad a priori de que el candidato gane
        self.probabilidad_apoyo_encuestas = probabilidad_apoyo_encuestas  # Probabilidad de apoyo en encuestas
        self.probabilidad_participacion_votantes = probabilidad_participacion_votantes  # Probabilidad de participación de votantes

    def probabilidad_condicional(self, apoyo_encuestas, participacion_votantes):
        # Calcular la probabilidad condicional de ganar dado el apoyo en encuestas y la participación de votantes
        probabilidad_conjunta_victoria = (self.probabilidad_prior_victoria *
                                          self.probabilidad_apoyo_encuestas[apoyo_encuestas] *
                                          self.probabilidad_participacion_votantes[participacion_votantes])

        probabilidad_conjunta_derrota = ((1 - self.probabilidad_prior_victoria) *
                                          (1 - self.probabilidad_apoyo_encuestas[apoyo_encuestas]) *
                                          (1 - self.probabilidad_participacion_votantes[participacion_votantes]))

        probabilidad_total = probabilidad_conjunta_victoria + probabilidad_conjunta_derrota

        probabilidad_condicional_victoria = probabilidad_conjunta_victoria / probabilidad_total
        return probabilidad_condicional_victoria

# Probabilidades a priori y condicionales
probabilidad_prior_victoria_inicial = 0.5  # Probabilidad a priori de que el candidato gane

# Probabilidades de apoyo en encuestas
probabilidad_apoyo_encuestas_diccionario = {
    'alto': 0.7,  # Alta probabilidad de apoyo en encuestas
    'medio': 0.5,  # Probabilidad media de apoyo en encuestas
    'bajo': 0.3  # Baja probabilidad de apoyo en encuestas
}

# Probabilidades de participación de votantes
probabilidad_participacion_votantes_diccionario = {
    'alto': 0.6,  # Alta probabilidad de participación de votantes
    'medio': 0.4,  # Probabilidad media de participación de votantes
    'bajo': 0.2  # Baja probabilidad de participación de votantes
}

# Crear el modelo de predicción de elecciones
modelo_prediccion = PrediccionElecciones(probabilidad_prior_victoria_inicial, probabilidad_apoyo_encuestas_diccionario, probabilidad_participacion_votantes_diccionario)

# Calcular la probabilidad condicional de ganar dado el apoyo en encuestas y la participación de votantes
apoyo_encuestas_observado = 'alto'  # Observación: alto apoyo en encuestas
participacion_votantes_observada = 'medio'  # Observación: participación media de votantes

probabilidad_condicional_victoria_calculada = modelo_prediccion.probabilidad_condicional(apoyo_encuestas_observado, participacion_votantes_observada)
print(f"Probabilidad condicional de que el candidato gane: {probabilidad_condicional_victoria_calculada:.2f}")
