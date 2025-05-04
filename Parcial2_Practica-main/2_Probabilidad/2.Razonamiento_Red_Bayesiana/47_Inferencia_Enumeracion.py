from itertools import product

def enumeration_inference(variables, probabilities, evidence):
    """
    Realiza inferencia por enumeración para calcular probabilidades condicionales.

    :param variables: Lista de variables en el espacio de probabilidad.
    :param probabilities: Diccionario con probabilidades conjuntas.
    :param evidence: Diccionario con evidencia observada.
    :return: Probabilidad normalizada basada en la evidencia.
    """
    def consistent_with_evidence(event, evidence):
        return all(event[var] == val for var, val in evidence.items())

    # Enumerar todas las combinaciones posibles de valores para las variables
    all_events = list(product(*[variables[var] for var in variables]))

    # Calcular la probabilidad total consistente con la evidencia
    total_prob = 0
    for event in all_events:
        event_dict = {var: val for var, val in zip(variables.keys(), event)}
        if consistent_with_evidence(event_dict, evidence):
            total_prob += probabilities[tuple(event_dict.values())]

    # Normalizar las probabilidades
    normalized_probs = {}
    for event in all_events:
        event_dict = {var: val for var, val in zip(variables.keys(), event)}
        if consistent_with_evidence(event_dict, evidence):
            normalized_probs[tuple(event_dict.values())] = probabilities[tuple(event_dict.values())] / total_prob

    return normalized_probs


# Ejemplo de uso
variables = {
    "Apuesta": ["Ganar", "Perder"],
    "Clima": ["Soleado", "Lluvioso"]
}

# Probabilidades conjuntas
probabilities = {
    ("Ganar", "Soleado"): 0.3,
    ("Ganar", "Lluvioso"): 0.2,
    ("Perder", "Soleado"): 0.4,
    ("Perder", "Lluvioso"): 0.1
}

# Evidencia observada
evidence = {"Clima": "Soleado"}

# Calcular inferencia
result = enumeration_inference(variables, probabilities, evidence)
print("Probabilidades condicionales dadas la evidencia:", result)