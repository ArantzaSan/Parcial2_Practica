# Función para calcular la probabilidad conjunta y condicional usando la regla de la cadena y el teorema de Bayes

def calcular_probabilidades(probabilidad_a, probabilidad_b_si_a, probabilidad_b_si_no_a):
    """
    Calcula la probabilidad conjunta P(A ∩ B) y la probabilidad condicional P(A | B)
    empleando la regla de la cadena y el teorema de Bayes.

    Args:
        probabilidad_a (float): Probabilidad del evento A, P(A).
        probabilidad_b_si_a (float): Probabilidad del evento B dado que ocurrió A, P(B | A).
        probabilidad_b_si_no_a (float): Probabilidad del evento B dado que no ocurrió A, P(B | ¬A).

    Returns:
        tuple: (P(A ∩ B), P(A | B))
    """
    # Probabilidad de la negación de A
    probabilidad_no_a = 1 - probabilidad_a

    # Probabilidad total de B (aplicando el teorema de la probabilidad total)
    probabilidad_b = (probabilidad_b_si_a * probabilidad_a) + (probabilidad_b_si_no_a * probabilidad_no_a)

    # Probabilidad de la intersección de A y B, P(A ∩ B), usando la regla de la cadena
    probabilidad_a_y_b = probabilidad_b_si_a * probabilidad_a

    # Probabilidad condicional de A dado B, P(A | B), aplicando el teorema de Bayes
    probabilidad_a_dado_b = probabilidad_a_y_b / probabilidad_b

    return probabilidad_a_y_b, probabilidad_a_dado_b


# Ejemplo de aplicación
prob_a = 0.3      # P(A)
prob_b_dado_a = 0.8  # P(B | A)
prob_b_dado_neg_a = 0.2 # P(B | ¬A)

prob_ayb, prob_adob = calcular_probabilidades(prob_a, prob_b_dado_a, prob_b_dado_neg_a)

print(f"P(A ∩ B): {prob_ayb:.4f}")
print(f"P(A | B): {prob_adob:.4f}")
