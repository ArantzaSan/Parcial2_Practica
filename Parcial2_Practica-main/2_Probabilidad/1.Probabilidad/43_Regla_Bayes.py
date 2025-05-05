class CarreraCanicas:
    def __init__(self, probabilidad_previa_victoria, verosimilitud_victoria_dado_pasado, verosimilitud_victoria_sin_pasado):
        self.probabilidad_previa_victoria = probabilidad_previa_victoria  # Probabilidad a priori de que la canica gane
        self.verosimilitud_victoria_dado_pasado = verosimilitud_victoria_dado_pasado  # Probabilidad de ganar dado el rendimiento pasado
        self.verosimilitud_victoria_sin_pasado = verosimilitud_victoria_sin_pasado  # Probabilidad de ganar sin rendimiento pasado

    def actualizar_probabilidad(self, gano_ultima_carrera):
        # Aplicar la regla de Bayes para actualizar la probabilidad de ganar
        if gano_ultima_carrera:
            verosimilitud = self.verosimilitud_victoria_dado_pasado
        else:
            verosimilitud = self.verosimilitud_victoria_sin_pasado

        # Calcular la probabilidad posterior
        probabilidad_posterior_victoria = (verosimilitud * self.probabilidad_previa_victoria) / (
            (verosimilitud * self.probabilidad_previa_victoria) +
            ((1 - verosimilitud) * (1 - self.probabilidad_previa_victoria))
        )

        # Actualizar la probabilidad a priori para la siguiente iteración
        self.probabilidad_previa_victoria = probabilidad_posterior_victoria

        return probabilidad_posterior_victoria

# Probabilidades iniciales
probabilidad_previa_victoria_inicial = 0.5  # Probabilidad a priori de que la canica gane
verosimilitud_victoria_con_pasado = 0.8  # Probabilidad de ganar dado que ganó la última carrera
verosimilitud_victoria_sin_pasado = 0.4  # Probabilidad de ganar dado que no ganó la última carrera

# Crear el modelo de carrera de canicas
modelo_carrera = CarreraCanicas(probabilidad_previa_victoria_inicial, verosimilitud_victoria_con_pasado, verosimilitud_victoria_sin_pasado)

# Simular el resultado de la última carrera
gano_ultima_carrera_simulada = True  # Supongamos que la canica ganó la última carrera

# Actualizar la probabilidad de ganar
probabilidad_posterior_victoria_calculada = modelo_carrera.actualizar_probabilidad(gano_ultima_carrera_simulada)
print(f"Probabilidad actualizada de que la canica gane: {probabilidad_posterior_victoria_calculada:.2f}")
