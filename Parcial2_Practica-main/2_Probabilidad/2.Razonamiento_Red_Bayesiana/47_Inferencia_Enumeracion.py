from itertools import product

def inferencia_por_enumeracion(espacio_variables, distribucion_probabilidades, evidencia_observada):
    """
    Realiza inferencia por enumeración para calcular probabilidades condicionales.

    :param espacio_variables: Diccionario de variables con sus posibles valores.
    :param distribucion_probabilidades: Diccionario con probabilidades conjuntas indexadas por tuplas de valores.
    :param evidencia_observada: Diccionario con la evidencia observada (variable: valor).
    :return: Diccionario de probabilidades condicionales normalizadas dada la evidencia.
    """
    def consistente_con_evidencia(evento, evidencia):
        return all(evento[variable] == valor for variable, valor in evidencia.items())

    # Obtener la lista de variables
    lista_variables = list(espacio_variables.keys())

    # Enumerar todas las combinaciones posibles de valores para las variables
    todos_los_eventos = list(product(*espacio_variables.values()))

    # Calcular la probabilidad total de los eventos consistentes con la evidencia
    probabilidad_total_evidencia = 0
    for evento in todos_los_eventos:
        evento_diccionario = dict(zip(lista_variables, evento))
        if consistente_con_evidencia(evento_diccionario, evidencia_observada):
            probabilidad_total_evidencia += distribucion_probabilidades[evento]

    # Normalizar las probabilidades de los eventos consistentes con la evidencia
    probabilidades_normalizadas = {}
    if probabilidad_total_evidencia > 0:
        for evento in todos_los_eventos:
            evento_diccionario = dict(zip(lista_variables, evento))
            if consistente_con_evidencia(evento_diccionario, evidencia_observada):
                probabilidades_normalizadas[evento] = distribucion_probabilidades[evento] / probabilidad_total_evidencia
    else:
        print("Advertencia: La evidencia es inconsistente con las probabilidades dadas.")

    return probabilidades_normalizadas


# Ejemplo de uso
espacio_variables = {
    "Apuesta": ["Ganar", "Perder"],
    "Clima": ["Soleado", "Lluvioso"]
}

# Probabilidades conjuntas
distribucion_probabilidades = {
    ("Ganar", "Soleado"): 0.3,
    ("Ganar", "Lluvioso"): 0.2,
    ("Perder", "Soleado"): 0.4,
    ("Perder", "Lluvioso"): 0.1
}

# Evidencia observada
evidencia = {"Clima": "Soleado"}

# Calcular inferencia
resultado = inferencia_por_enumeracion(espacio_variables, distribucion_probabilidades, evidencia)
print("Probabilidades condicionales dada la evidencia:", resultado)
