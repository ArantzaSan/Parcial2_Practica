import numpy as np

class IncertidumbreInversion:
    def __init__(self, inversion_inicial, prob_exito, prob_fracaso, retorno_exito, retorno_fracaso):
        self.inversion_inicial = inversion_inicial
        self.prob_exito = prob_exito
        self.prob_fracaso = prob_fracaso
        self.retorno_exito = retorno_exito
        self.retorno_fracaso = retorno_fracaso

    def valor_esperado(self):
        # Calcular el valor esperado de la inversión
        valor_esperado = (self.prob_exito * self.retorno_exito) + (self.prob_fracaso * self.retorno_fracaso)
        return valor_esperado - self.inversion_inicial

    def simular_inversion(self, num_simulaciones=1000):
        # Simular múltiples escenarios de inversión
        resultados = []
        for _ in range(num_simulaciones):
            if np.random.rand() < self.prob_exito:
                resultado = self.retorno_exito
            else:
                resultado = self.retorno_fracaso
            resultado_neto = resultado - self.inversion_inicial
            resultados.append(resultado_neto)
        return resultados

    def riesgo_inversion(self, resultados):
        # Evaluar el riesgo de la inversión
        resultados_np = np.array(resultados)
        promedio_resultado = np.mean(resultados_np)
        desviacion_estandar_resultado = np.std(resultados_np)
        return promedio_resultado, desviacion_estandar_resultado

# Parámetros de la inversión
inversion_inicial = 1000  # Inversión inicial
prob_exito = 0.6  # Probabilidad de éxito
prob_fracaso = 0.4  # Probabilidad de fracaso
retorno_exito = 2000  # Retorno en caso de éxito
retorno_fracaso = 500   # Retorno en caso de fracaso

# Crear el modelo de incertidumbre de inversión
inversion = IncertidumbreInversion(inversion_inicial, prob_exito, prob_fracaso, retorno_exito, retorno_fracaso)

# Calcular el valor esperado de la inversión
valor_esperado = inversion.valor_esperado()
print(f"Valor esperado de la inversión: ${valor_esperado:.2f}")

# Simular múltiples escenarios de inversión
resultados = inversion.simular_inversion(num_simulaciones=1000)

# Evaluar el riesgo de la inversión
promedio_resultado, desviacion_estandar_resultado = inversion.riesgo_inversion(resultados)
print(f"Resultado promedio de la inversión: ${promedio_resultado:.2f}")
print(f"Desviación estándar de la inversión: ${desviacion_estandar_resultado:.2f}")
